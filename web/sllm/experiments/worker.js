import { SessionRangeLoader } from '../range-loader.js';
import { installGpuTracking } from '../gpu-device.js';
import { saveCheckpoint } from '../diagnostics.js';
import { OpfsWeightStore } from '../opfs-store.js';
import { loadOrt } from '../ort-runtime.js';

function shardResponse(shards, resolve) {
  let reader, index = 0;
  return new Response(new ReadableStream({
    async pull(controller) {
      for (;;) {
        if (!reader) {
          if (index === shards.length) { controller.close(); return; }
          const response = await fetch(resolve(shards[index++].url));
          if (!response.ok || !response.body) throw new Error('Experiment shard download failed');
          reader = response.body.getReader();
        }
        const { done, value } = await reader.read();
        if (!done) { controller.enqueue(value); return; }
        reader.releaseLock(); reader = null;
      }
    },
    async cancel() { await reader?.cancel(); reader?.releaseLock(); },
  }));
}

self.onmessage = async ({ data: config }) => {
  let session, loader, tracked, store;
  const result = { ...config.experiment, success: false, startedAt: Date.now(), pageReloadObserved: false };
  try {
    const mode = config.mode || (config.experiment.kind === 'stock' ? 'stock' : 'asyncify');
    const { ort, build } = await loadOrt(mode);
    result.releaseId = build.releaseId;
    const response = await fetch(config.source);
    if (!response.ok) throw new Error(`Experiment manifest HTTP ${response.status}`);
    const description = await response.json();
    const resolve = value => new URL(value, config.source).href;
    const manifest = await (await fetch(resolve(description.manifest))).json();
    const graph = new Uint8Array(await (await fetch(resolve(description.graph))).arrayBuffer());
    const hash = Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256', graph)), b => b.toString(16).padStart(2, '0')).join('');
    if (hash !== manifest.graphSha256) throw new Error('Graph hash mismatch');
    const files = description.variants[String(config.experiment.fileMiB)];
    if (!files) throw new Error(`Missing ${config.experiment.fileMiB} MiB physical files`);
    const externalData = [];
    if (config.storage === 'opfs') store = await OpfsWeightStore.open(manifest);
    const stockMetrics = { cpuStagingPeak: 0, wasmHeapPeak: null, loadedInitializerCount: 0, rangeReadCount: 0 };
    // Baselines use the repository's original BlobFile technique, without its
    // forbidden eager retry. null metrics mean unmeasured, never zero.
    class StockBlobFile extends Uint8Array {
      constructor(blob, location) { super(0); this.blob = blob; this.location = location; this.reader = new FileReaderSync(); }
      get byteLength() { return this.blob.size; }
      subarray(begin, end) {
        const init = manifest.initializers.find(t => t.location === this.location && t.offset === begin && t.bytes === end - begin);
        result.lastInitializer = init?.name;
        stockMetrics.cpuStagingPeak = Math.max(stockMetrics.cpuStagingPeak, end - begin);
        stockMetrics.loadedInitializerCount++;
        stockMetrics.rangeReadCount++;
        self.postMessage({ type: 'progress', result: { ...result, stage: 'stock-range-read' } });
        return new Uint8Array(this.reader.readAsArrayBuffer(this.blob.slice(begin, end)));
      }
    }
    let cache;
    try { cache = await caches.open('didimdol-experiment-shards-v1'); } catch {}
    for (const file of files) {
      if (store) {
        const metadata = manifest.files.find(entry => entry.location === file.location);
        await store.prepare(metadata, { openResponse: () => shardResponse(file.shards, resolve) });
        externalData.push({ path: file.location, data: store.descriptor(metadata) });
        continue;
      }
      const parts = [];
      for (const shard of file.shards) {
        const url = resolve(shard.url);
        let response = await cache?.match(url);
        if (!response) {
          response = await fetch(url);
          if (!response.ok) throw new Error(`Weight HTTP ${response.status}`);
          if (cache) {
            try {
              await cache.put(url, response);
              response = await cache.match(url);
              if (!response) response = await fetch(url);
            } catch (error) {
              result.cacheFallback = String(error);
              cache = null;
              response = await fetch(url);
            }
          }
        }
        const blob = await response.blob();
        if (blob.size !== shard.bytes) throw new Error('Truncated transport shard');
        parts.push(blob);
      }
      const blob = new Blob(parts);
      if (blob.size !== file.bytes) throw new Error('Logical file size mismatch');
      externalData.push({ path: file.location, data: mode === 'stock' ? new StockBlobFile(blob, file.location) : blob });
    }
    store?.finishPreparation();
    tracked = await installGpuTracking(manifest.largestInitializerBytes);
    if (mode !== 'stock') {
      loader = new SessionRangeLoader({ manifest, stagingMiB: config.experiment.stagingMiB,
        checkpoint: record => saveCheckpoint({ ...record, experiment: result.id }, `experiment-${result.id}`),
        emit: record => {
          if (record.initializerName) result.lastInitializer = record.initializerName;
          if (record.stage === 'initializer-complete') self.postMessage({ type: 'progress', result: { ...result, ...record } });
        },
      });
      globalThis.__ortExternalTensorLoader = loader;
    }
    const start = performance.now();
    await saveCheckpoint({ stage: 'session-create', id: result.id }, `experiment-${result.id}`);
    session = await ort.InferenceSession.create(graph, {
      executionProviders: ['webgpu'], externalData,
      graphOptimizationLevel: 'disabled', enableCpuMemArena: false, enableMemPattern: false,
      ...(config.verifyGemma ? { preferredOutputLocation: Object.fromEntries(
        Array.from({ length: 18 }, (_, i) => ['key', 'value'].map(k => [`present.${i}.${k}`, 'gpu-buffer'])).flat()) } : {}),
    });
    result.sessionCreateMs = performance.now() - start;
    const drainStart = performance.now();
    await tracked.device.queue.onSubmittedWorkDone();
    result.uploadDrainMs = performance.now() - drainStart;
    result.metrics = loader?.close(true) || stockMetrics;
    store?.close();
    if (store) result.storage = { ...store.metrics };
    externalData.length = 0;
    result.sessionGpuLedger = { ...tracked.ledger };
    result.success = true;
    // Fixture verification runs twice AFTER sealing the loader, proving no
    // external-data reads during repeated inference. Gemma uses the app worker.
    if (config.verifyFixture) {
      const before = result.metrics.rangeReadCount;
      for (let i = 0; i < 2; i++) {
        const outputs = await session.run({ X: new ort.Tensor('float32', new Float32Array(1024).fill(1), [1, 1024]) });
        if (outputs.Y.data.length !== 2560 || outputs.Y.data.some(v => v !== 1024.5)) throw new Error('Wrong FP32 inference output');
        outputs.Y.dispose();
      }
      if (loader && loader.metrics.rangeReadCount !== before) throw new Error('Weights reloaded during inference');
      result.inferenceVerified = true;
    }
    if (config.verifyGemma) {
      const reads = loader.metrics.rangeReadCount;
      let past = Object.fromEntries(Array.from({ length: 18 }, (_, i) => ['key', 'value'].map(k =>
        [`past_key_values.${i}.${k}`, new ort.Tensor('float32', new Float32Array(0), [1, 1, 0, 256])])).flat());
      let token = 2;
      result.decode = [];
      for (let step = 0; step < 2; step++) {
        const start = performance.now();
        const outputs = await session.run({ ...past,
          input_ids: new ort.Tensor('int64', BigInt64Array.of(BigInt(token)), [1, 1]),
          attention_mask: new ort.Tensor('int64', new BigInt64Array(step + 1).fill(1n), [1, step + 1]),
        });
        const logits = outputs.logits.data;
        if (logits.length !== 262144 || logits.some(v => !Number.isFinite(v))) throw new Error('Invalid Gemma FP32 logits');
        token = 0;
        for (let i = 1; i < logits.length; i++) if (logits[i] > logits[token]) token = i;
        result.decode.push({ step, token, ms: performance.now() - start });
        outputs.logits.dispose();
        for (const tensor of Object.values(past)) tensor.dispose();
        past = Object.fromEntries(Object.entries(outputs).filter(([name]) => name.startsWith('present.'))
          .map(([name, tensor]) => [name.replace('present.', 'past_key_values.'), tensor]));
        if (Object.values(past).some(t => t.location !== 'gpu-buffer')) throw new Error('KV readback: cache is not on GPU');
      }
      for (const tensor of Object.values(past)) tensor.dispose();
      if (loader.metrics.rangeReadCount !== reads) throw new Error('Weights reloaded during Gemma decode');
      result.inferenceVerified = true;
    }
  } catch (error) {
    result.success = false;
    result.error = String(error.stack || error);
    if (loader && !loader.closed) result.metrics = loader.close(false);
  } finally {
    store?.close();
    tracked?.restore();
    if (tracked) result.gpuLedger = { ...tracked.ledger };
    if (session) await session.release();
    tracked?.device?.destroy();
    result.completedAt = Date.now();
    await saveCheckpoint(result, `experiment-${result.id}`);
    self.postMessage({ type: 'result', result });
  }
};
