import { StreamedWeights } from './streamed-weights.js';

/** Projections per session whose chunks are timed individually (about 1.5 KB each). */
export const TIMELINE_PROJECTIONS = 2;

export function bodyManifest(source, descriptor) {
  if (descriptor.format !== 'didimdol-streamed-fp32-v1' || descriptor.sourceGraphSha256 !== source.graphSha256 ||
      descriptor.sourceRevision !== source.revision) throw new Error('Streamed model/source mismatch');
  const names = new Set(descriptor.bodyInitializerNames);
  const initializers = source.initializers.filter(value => names.has(value.name));
  const bytes = initializers.filter(value => value.location).reduce((sum, value) => sum + value.bytes, 0);
  if (initializers.length !== names.size || bytes !== descriptor.bodyWeightBytes ||
      initializers.some(value => value.name.startsWith('embed_tokens.chunk'))) throw new Error('Invalid streamed body manifest');
  if (descriptor.chunks.length !== 16 || descriptor.chunkBytes !== 40 * 2**20 || descriptor.chunkRows !== 16384 ||
      descriptor.hiddenSize !== 640 || descriptor.vocabSize !== 262144) throw new Error('Invalid streamed embedding layout');
  for (const [index, chunk] of descriptor.chunks.entries()) {
    const original = source.initializers.find(value => value.name === `embed_tokens.chunk${index}`);
    if (!original || ['name', 'location', 'offset', 'bytes', 'dtype'].some(key => chunk[key] !== original[key]) ||
        original.bytes !== descriptor.chunkBytes || original.dtype !== 'FLOAT') throw new Error('Streamed embedding/source mismatch');
  }
  return { ...source, graphSha256: descriptor.graphs.body.sha256, initializers,
    totalExternalTensorBytes: bytes, largestInitializerBytes: Math.max(...initializers.map(value => value.bytes)) };
}

export async function verifiedStreamedGraph(descriptor, name, baseURL) {
  const graph = descriptor.graphs[name];
  if (graph?.file !== `${name}.onnx`) throw new Error('Invalid streamed graph path');
  const response = await fetch(new URL(graph.file, baseURL));
  if (!response.ok) throw new Error(`Streamed graph HTTP ${response.status}`);
  const bytes = new Uint8Array(await response.arrayBuffer());
  const hash = Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256', bytes)), b => b.toString(16).padStart(2, '0')).join('');
  if (bytes.length !== graph.bytes || hash !== graph.sha256) throw new Error(`Streamed ${name} graph hash mismatch`);
  return bytes;
}

/** ORT session facade: preserves Transformers' generation/cache/sampling machinery. */
export class StreamedSession {
  constructor({ ort, body, head, descriptor, store, manifest, tracker, signal, checkpoint = async () => {}, scratchBytes,
    outputBuffers, stepCheckpointEvery = 1 }) {
    Object.assign(this, { ort, body, head, descriptor, checkpoint });
    // Durable `streamed-step-complete` records are written every N projections.
    // Inference samples already carry the streaming totals every few seconds, so
    // a 100-row evaluation does not need one IndexedDB transaction per token.
    this.stepCheckpointEvery = stepCheckpointEvery;
    // Per-chunk timelines of the first projections, attached to the next checkpoint.
    this.timelines = [];
    this.weights = new StreamedWeights({ ort, device: tracker.device, tracker, store, manifest, descriptor, signal, scratchBytes, outputBuffers });
    this.inputNames = [...new Set(['input_ids', ...body.inputNames.filter(name => name !== 'streamed_embeddings')])];
    this.inputMetadata = this.inputNames.map(name => body.inputMetadata.find(value => value.name === name) ||
      { name, type: 'int64', shape: ['batch_size', 'sequence_length'], isTensor: true });
    this.outputNames = ['logits', ...body.outputNames.filter(name => name !== descriptor.hiddenOutput)];
    this.outputMetadata = this.outputNames.map(name => body.outputMetadata.find(value => value.name === name) ||
      { name, type: 'float32', shape: [1, 1, descriptor.vocabSize], isTensor: true });
  }
  async run(feeds) {
    if (this.busy || this.released) throw new Error('Streamed session is busy or released');
    this.busy = true;
    let embeddings, outputs, returned = false;
    const start = performance.now();
    try {
      embeddings = await this.weights.embeddings(feeds.input_ids);
      this.weights.check();
      const bodyFeeds = Object.fromEntries(this.body.inputNames.map(name => [name, name === 'streamed_embeddings' ? embeddings : feeds[name]]));
      outputs = await this.body.run(bodyFeeds);
      this.weights.check();
      const hidden = outputs[this.descriptor.hiddenOutput];
      const logits = new Float32Array(this.descriptor.vocabSize);
      const chunks = this.descriptor.chunks.length, overlap = this.weights.buffers.length > 1;
      const projectionStart = performance.now();
      // The first projections of a session keep a per-chunk timeline (run
      // submit and return, readback submit and completion, upload span) so a
      // phone export shows where an overlapped chunk waits. Later projections
      // carry the totals only; the timelines ride on the next checkpoint.
      const timeline = this.weights.metrics.projections < TIMELINE_PROJECTIONS ? [] : null;
      const ms = value => Math.round(value * 10) / 10;
      const entry = i => timeline && (timeline[i] ||= { chunk: i, slot: this.weights.slot(i) });
      const mark = (i, name) => { const e = entry(i); if (e) e[name] = ms(performance.now() - projectionStart); };
      const uploaded = (i, timing) => {
        const e = entry(i);
        if (e) Object.assign(e, { uploadStart: ms(timing.start - projectionStart), uploadEnd: ms(timing.end - projectionStart), readMs: ms(timing.readMs), waitMs: ms(timing.waitMs) });
        return null;
      };
      uploaded(0, await this.weights.upload(0));
      for (let i = 0; i < chunks; i++) {
        this.weights.check();
        const slot = this.weights.slot(i);
        this.weights.computing = { chunk: i, slot };
        this.weights.position = { operation: 'output-compute', chunk: i, slot };
        const computeStart = performance.now();
        mark(i, 'runSubmit');
        const pending = this.head.run({ hidden, weight: this.weights.tensors[slot] });
        let result, failure, next = null;
        try {
          result = await pending;
          mark(i, 'runResolved');
          // `getData()` submits the readback copy synchronously (the runtime's
          // downloader encodes copyBufferToBuffer and submits before awaiting
          // mapAsync), so it is issued before the next chunk's writes. With two
          // buffers the next chunk is then read from OPFS and written into the
          // other buffer while this projection and its readback run on the GPU;
          // the queue order stays compute(i), readback(i), writes(i+1), so the
          // readback never waits behind a 40 MiB upload. The buffer being
          // rewritten held projection i-1, whose readback completed below.
          const download = result.chunk_logits.getData();
          mark(i, 'readbackSubmit');
          if (overlap && i + 1 < chunks) next = this.weights.upload(i + 1, { queueWait: false }).then(timing => uploaded(i + 1, timing), error => error);
          const values = await download;
          mark(i, 'readbackResolved');
          logits.set(values, i * this.descriptor.chunkRows);
        } catch (error) { failure = error; }
        finally { result?.chunk_logits.dispose(); }
        this.weights.metrics.outputComputeMs += performance.now() - computeStart;
        // The in-flight upload is always settled before this call fails or
        // proceeds, so disposal never races a read into the scratch.
        const uploadFailure = next ? await next : null;
        if (failure) throw failure;
        if (uploadFailure) throw uploadFailure;
        // One buffer: the readback above completed after this projection in the
        // in-order queue, and the upload's own queue wait follows the rewrite.
        if (!overlap && i + 1 < chunks) uploaded(i + 1, await this.weights.upload(i + 1));
      }
      this.weights.computing = null;
      this.weights.metrics.projectionMs += performance.now() - projectionStart;
      this.weights.check();
      this.weights.metrics.projections++;
      if (timeline) this.timelines.push({ projection: this.weights.metrics.projections, outputBuffers: this.weights.buffers.length, chunks: timeline });
      const every = Math.max(1, Math.floor(this.stepCheckpointEvery) || 1);
      if (this.weights.metrics.projections % every === 0) {
        await this.checkpoint({ stage: 'streamed-step-complete', durationMs: performance.now() - start,
          stepCheckpointEvery: every, streaming: { ...this.weights.metrics },
          ...(this.timelines.length ? { chunkTimelines: this.timelines.splice(0) } : {}) });
      }
      hidden.dispose(); delete outputs[this.descriptor.hiddenOutput];
      outputs.logits = new this.ort.Tensor('float32', logits, [1, 1, this.descriptor.vocabSize]);
      returned = true;
      return outputs;
    } catch (error) {
      await this.checkpoint({ stage: 'streamed-error', message: String(error), position: this.weights.position,
        computing: this.weights.computing ?? null, streaming: { ...this.weights.metrics } });
      throw error;
    } finally {
      embeddings?.dispose();
      if (!returned && outputs) for (const tensor of Object.values(outputs)) tensor.dispose();
      this.busy = false;
    }
  }
  async release() {
    if (this.released) return;
    this.released = true;
    try { await this.body.release(); }
    finally { try { await this.head.release(); } finally { this.weights.dispose(); } }
  }
}
