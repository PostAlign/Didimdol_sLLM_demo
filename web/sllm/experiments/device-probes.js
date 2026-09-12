import { RunDiagnostics, newRunId, buildIdentity, errorDetails, gpuOperationContext } from '../diagnostics.js';
import { SessionRangeLoader, STAGING_MIB } from '../range-loader.js';
import { OpfsWeightStore } from '../opfs-store.js';
import { installGpuTracking } from '../gpu-device.js';
import { FIXTURE } from './fixture.js';
import { loadOrt, runtimeRelease } from '../ort-runtime.js';
import { executionSettings, isResident } from './results.js';

/** Use every uploaded element while retaining all weight buffers until the end. No ORT import. */
export async function residentProbe(manifest, checkpoint, stagingMiB = 8, store = null) {
  if (!STAGING_MIB.includes(stagingMiB)) throw new Error('Invalid staging size');
  const persist = checkpoint;
  let current = {}, tracked;
  tracked = await installGpuTracking(manifest.largestInitializerBytes, record => {
    if (['gpu-error', 'gpu-uncaptured-error', 'device-lost'].includes(record.stage)) {
      return persist({ ...record, gpuLedger: tracked?.ledger });
    }
  }, { context: () => gpuOperationContext(current) });
  checkpoint = async record => {
    current = record;
    await tracked.flush();
    await persist({ ...record, gpuLedger: tracked.ledger, ...(store ? { storage: { ...store.metrics } } : {}) });
    if (tracked.ledger.lastError) throw new Error(`WebGPU: ${tracked.ledger.lastError}`);
  };
  try {
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) throw new Error('WebGPU adapter unavailable');
    const largest = manifest.largestInitializerBytes;
    const device = await adapter.requestDevice({ requiredLimits: {
      maxBufferSize: Math.max(268435456, largest), maxStorageBufferBindingSize: Math.max(134217728, largest),
    } });
    const buffers = [];
    let allocated = 0, uploaded = 0, writeReturned = 0, lost, closed = false, lossSaved = Promise.resolve();
    device.lost.then(info => {
      if (closed && info.reason === 'destroyed') return;
      lost = { reason: info.reason, message: info.message };
      lossSaved = checkpoint({ stage: 'device-lost', info: lost, gpuWeightAllocated: allocated, gpuWeightUploaded: uploaded });
    });
    device.addEventListener('uncapturederror', event => {
      void checkpoint({ stage: 'gpu-uncaptured-error', ...errorDetails(event.error) });
    });
    device.pushErrorScope('out-of-memory');
    device.pushErrorScope('validation');
    let scopes = 2;
    try {
      await checkpoint({ stage: 'resident-start', expectedInitializerCount: manifest.initializers.filter(t => t.location).length,
        expectedGpuResidentBytes: manifest.totalExternalTensorBytes, inputSource: store ? 'opfs-cache' : 'synthetic',
        allocationOrder: 'manifest', verification: 'u32-fnv1a-64-lanes-v1' });
      const pipeline = await device.createComputePipelineAsync({ layout: 'auto', compute: { entryPoint: 'main', module: device.createShaderModule({ code: `
        @group(0) @binding(0) var<storage, read> weights: array<u32>;
        @group(0) @binding(1) var<storage, read_write> sums: array<u32>;
        @compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id: vec3<u32>) {
          var sum = 2166136261u;
          for (var i = id.x; i < arrayLength(&weights); i += 64u) { sum = (sum ^ weights[i]) * 16777619u; }
          sums[id.x] = sum;
        }` }) } });
      const output = device.createBuffer({ size: 256, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
      const readback = device.createBuffer({ size: 256, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
      buffers.push(output, readback);
      const scratch = new Float32Array(stagingMiB * 2**20 / 4).fill(1);
      const scratchBytes = new Uint8Array(scratch.buffer), words = new Uint32Array(scratch.buffer);
      const expected = new Uint32Array(64);
      const files = new Map(manifest.files?.map(file => [file.location, file]));
      for (const [index, init] of manifest.initializers.filter(t => t.location).entries()) {
        expected.fill(2166136261);
        await checkpoint({ stage: 'resident-allocate', initializerName: init.name, length: init.bytes, gpuWeightAllocated: allocated,
          gpuWeightUploaded: uploaded, gpuWriteReturnedBytes: writeReturned, loadedInitializerCount: index, gpuWeightBufferCount: index });
        const buffer = device.createBuffer({ size: init.bytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
        tracked.observeBuffer(device, buffer);
        buffers.push(buffer); allocated += init.bytes;
        for (let offset = 0; offset < init.bytes; offset += scratch.byteLength) {
          const chunkBytes = Math.min(scratch.byteLength, init.bytes - offset);
          const range = { initializerName: init.name, destinationOffset: offset, chunkBytes, gpuWeightAllocated: allocated,
            location: init.location, fileOffset: init.offset + offset,
            gpuWeightUploaded: uploaded, gpuWriteReturnedBytes: writeReturned,
            gpuWeightBufferCount: index + 1, loadedInitializerCount: index };
          if (store) {
            await checkpoint({ stage: 'resident-read', ...range });
            await store.readRangeInto(files.get(init.location), init.offset + offset, chunkBytes, scratchBytes.subarray(0, chunkBytes));
          }
          // Hash the same raw FP32 bits on CPU/GPU for both sources. No tolerance,
          // sampled reads or full-file readback; every word contributes to a lane.
          for (let i = 0; i < chunkBytes / 4; i++) {
            const lane = (offset / 4 + i) % 64;
            expected[lane] = Math.imul(expected[lane] ^ words[i], 16777619) >>> 0;
          }
          await checkpoint({ stage: 'resident-write', ...range });
          const writeStart = performance.now();
          device.queue.writeBuffer(buffer, offset, scratch, 0, chunkBytes / 4);
          writeReturned += chunkBytes;
          const gpuWriteMs = performance.now() - writeStart;
          await checkpoint({ stage: 'resident-wait', ...range, gpuWriteMs, gpuWriteReturnedBytes: writeReturned,
            operation: 'onSubmittedWorkDone', phase: 'before-call' });
          const waitStart = performance.now();
          await device.queue.onSubmittedWorkDone();
          const gpuWaitMs = performance.now() - waitStart;
          if (lost) throw new Error(`Device lost: ${JSON.stringify(lost)}`);
          uploaded += chunkBytes;
          await checkpoint({ stage: 'resident-range-complete', ...range, gpuWeightUploaded: uploaded,
            gpuWriteReturnedBytes: writeReturned, gpuWriteMs, gpuWaitMs });
        }
        const encoder = device.createCommandEncoder();
        const pass = encoder.beginComputePass();
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [
          { binding: 0, resource: { buffer } }, { binding: 1, resource: { buffer: output } },
        ] }));
        pass.dispatchWorkgroups(1); pass.end();
        encoder.copyBufferToBuffer(output, 0, readback, 0, 256);
        await checkpoint({ stage: 'resident-compute', initializerName: init.name, gpuWeightAllocated: allocated,
          gpuWeightUploaded: uploaded, gpuWriteReturnedBytes: writeReturned, loadedInitializerCount: index });
        device.queue.submit([encoder.finish()]);
        await readback.mapAsync(GPUMapMode.READ);
        const sums = new Uint32Array(readback.getMappedRange());
        for (let lane = 0; lane < 64; lane++) {
          if (sums[lane] !== expected[lane]) throw new Error(`Resident buffer readback mismatch: ${init.name}, lane ${lane}`);
        }
        readback.unmap();
        await checkpoint({ stage: 'resident-complete', initializerName: init.name, loadedInitializerCount: index + 1,
          gpuWeightAllocated: allocated, gpuWeightUploaded: uploaded, gpuWriteReturnedBytes: writeReturned });
      }
      while (scopes) {
        scopes--; const error = await device.popErrorScope();
        if (error) { await checkpoint({ stage: 'gpu-error', operation: 'resident-error-scope', ...errorDetails(error) }); throw new Error(error.message); }
      }
      if (lost) throw new Error(`Device lost: ${JSON.stringify(lost)}`);
      await tracked.flush();
      if (tracked.ledger.lastError) throw new Error(`WebGPU: ${tracked.ledger.lastError}`);
      const count = manifest.initializers.filter(t => t.location).length;
      return { gpuWeightAllocated: allocated, gpuWeightUploaded: uploaded, gpuWriteReturnedBytes: writeReturned,
        loadedInitializerCount: count, expectedInitializerCount: count, gpuValidatedInitializerCount: count,
        gpuLedger: tracked.ledger, expectedBytes: manifest.totalExternalTensorBytes, allBytesUsed: true,
        inputSource: store ? 'opfs-cache' : 'synthetic', allocationOrder: 'manifest', verification: 'u32-fnv1a-64-lanes-v1',
        cpuStagingBytes: scratch.byteLength + expected.byteLength };
    } finally {
      while (scopes) { scopes--; await device.popErrorScope(); }
      await lossSaved; closed = true;
      for (const buffer of buffers) buffer.destroy();
      device.destroy();
    }
  } finally { await tracked.flush(); tracked.restore(); }
}

async function prepareFixture(store, checkpoint) {
  for (const file of FIXTURE.manifest.files) {
    await store.prepare(file, { checkpoint, openResponse: async () => {
      const bytes = new Uint8Array(file.bytes);
      bytes.set(new TextEncoder().encode('external-test'));
      for (const init of FIXTURE.manifest.initializers.filter(t => t.location === file.location)) {
        const values = new Float32Array(init.bytes / 4).fill(init.name === 'bias' ? 0.5 : 1);
        bytes.set(new Uint8Array(values.buffer), init.offset);
      }
      return new Response(bytes);
    } });
  }
}

async function opfsResidentProbe(manifest, checkpoint, stagingMiB, fixture) {
  const store = await OpfsWeightStore.open(manifest);
  let result;
  try {
    if (fixture) await prepareFixture(store, checkpoint);
    else {
      // Match a warm production load. Never mix download/migration with this
      // comparison or silently generate substitute weights for a real model.
      for (const file of manifest.files) {
        if (!await store.complete(file)) throw new Error(`저장된 가중치가 없습니다: ${file.location}. 전체 모델 로드에서 파일 준비를 먼저 완료해 주세요.`);
        await store.prepare(file, { checkpoint, openResponse: () => { throw new Error('OPFS comparison requires verified cache'); } });
      }
    }
    store.finishPreparation();
    await checkpoint({ stage: 'weights-prepared', storage: { ...store.metrics } });
    result = await residentProbe(manifest, checkpoint, stagingMiB, store);
  } finally { store.close(); }
  return { ...result, storage: { ...store.metrics } };
}

export async function runtimeProbe(mode, checkpoint, idleSeconds = 120, stagingMiB = 8) {
  if (!['asyncify', 'jspi'].includes(mode)) throw new Error('Invalid runtime mode');
  if (mode === 'jspi' && !(WebAssembly.Suspending && WebAssembly.promising)) throw new Error('이 브라우저는 JSPI를 지원하지 않습니다.');
  const { ort } = await loadOrt(mode);
  const manifest = FIXTURE.manifest;
  const store = await OpfsWeightStore.open(manifest);
  let tracked, loader, session;
  try {
    await prepareFixture(store, checkpoint);
    store.finishPreparation();
    await checkpoint({ stage: 'weights-prepared', storage: { ...store.metrics } });
    tracked = await installGpuTracking(manifest.largestInitializerBytes, record => {
      if (['gpu-error', 'gpu-uncaptured-error', 'device-lost'].includes(record.stage)) return checkpoint(record);
    }, { context: () => ({ initializerName: loader?.last?.initializerName }) });
    loader = new SessionRangeLoader({ manifest, stagingMiB, checkpoint, gpuTracker: tracked, storage: () => store.metrics });
    globalThis.__ortExternalTensorLoader = loader;
    await checkpoint({ stage: 'runtime-create', expectedInitializerCount: manifest.initializers.filter(t => t.location).length });
    const graph = Uint8Array.from(atob(FIXTURE.graph), c => c.charCodeAt(0));
    session = await ort.InferenceSession.create(graph, { executionProviders: ['webgpu'], graphOptimizationLevel: 'disabled',
      enableCpuMemArena: false, enableMemPattern: false,
      externalData: manifest.files.map(file => ({ path: file.location, data: store.descriptor(file) })) });
    await tracked.flush();
    if (tracked.ledger.lastError) throw new Error(`WebGPU: ${tracked.ledger.lastError}`);
    loader.close(true); store.close();
    const reads = loader.metrics.rangeReadCount;
    for (let i = 0; i < 2; i++) {
      const input = new ort.Tensor('float32', new Float32Array(1024).fill(1), [1, 1024]);
      const outputs = await session.run({ X: input });
      try {
        if (outputs.Y.data.length !== 2560 || outputs.Y.data.some(v => v !== 1024.5)) throw new Error('Wrong FP32 fixture output');
      } finally { input.dispose(); outputs.Y.dispose(); }
    }
    if (loader.metrics.rangeReadCount !== reads) throw new Error('Weights read during inference');
    await checkpoint({ stage: 'runtime-inference-complete', metrics: loader.sampleMetrics(), storage: { ...store.metrics } });
    await checkpoint({ stage: 'runtime-idle-start', idleSeconds });
    const idleStart = performance.now();
    for (let elapsed = 0; elapsed < idleSeconds; elapsed += 5) {
      await checkpoint({ stage: 'runtime-idle', elapsedSeconds: elapsed, idleElapsedMs: performance.now() - idleStart,
        metrics: loader.sampleMetrics(), gpuLedger: { ...tracked.ledger }, storage: { ...store.metrics } });
      await new Promise(resolve => setTimeout(resolve, Math.min(5, idleSeconds - elapsed) * 1000));
      if (loader.metrics.deviceLost) throw new Error('Device lost during idle');
    }
    const idleElapsedMs = performance.now() - idleStart;
    await checkpoint({ stage: 'runtime-idle-complete', idleSeconds, idleElapsedMs });
    await tracked.flush();
    if (tracked.ledger.lastError) throw new Error(`WebGPU: ${tracked.ledger.lastError}`);
    return { inferenceVerified: true, metrics: { ...loader.metrics }, storage: { ...store.metrics }, gpuLedger: { ...tracked.ledger },
      idleSeconds, idleElapsedMs, idleAcceptanceCompleted: idleSeconds >= 120 && idleElapsedMs >= 120000 };
  } finally {
    store.close();
    if (loader && !loader.closed) loader.close(false);
    await loader?.lossSaved;
    await session?.release();
    await tracked?.flush();
    for (const gpuDevice of tracked?.devices || []) gpuDevice.destroy();
    tracked?.restore();
  }
}

self.onmessage = async ({ data }) => {
  const stagingMiB = data.stagingMiB ?? 8, mode = data.mode || 'asyncify';
  const settings = executionSettings(data.kind, data), { idleSeconds } = settings;
  const journal = new RunDiagnostics(data.runId || newRunId(), { ...data.environment, userAgent: navigator.userAgent,
    experiment: data.kind, ...settings, stagingMiB,
    fixture: !!data.fixture, workerURL: self.location.href });
  const checkpoint = async record => {
    await journal.checkpoint(record);
    self.postMessage({ type: 'progress', record });
  };
  const start = performance.now();
  try {
    const { build } = await runtimeRelease();
    journal.state.environment.build = buildIdentity(build);
    await checkpoint({ stage: 'probe-start' });
    if (!STAGING_MIB.includes(stagingMiB) || !Number.isInteger(idleSeconds) || idleSeconds < 0 || idleSeconds > 600) throw new Error('Invalid probe settings');
    let result;
    if (isResident(data.kind)) {
      const manifest = data.fixture ? FIXTURE.manifest : await fetch(new URL('../../../model/initializers.json', import.meta.url)).then(r => r.json());
      if (data.kind === 'resident') result = await residentProbe(manifest, checkpoint, stagingMiB);
      else {
        const execute = async lock => {
          if (!lock) throw new Error('다른 탭에서 모델을 준비 중입니다. 해당 작업이 끝난 후 다시 시도해 주세요.');
          return opfsResidentProbe(manifest, checkpoint, stagingMiB, !!data.fixture);
        };
        result = navigator.locks ? await navigator.locks.request('didimdol-model-load', { ifAvailable: true }, execute) : await execute(true);
      }
    } else if (data.kind === 'runtime') result = await runtimeProbe(mode, checkpoint, idleSeconds, stagingMiB);
    else throw new Error(`Unknown experiment: ${data.kind}`);
    result = { ...result, success: !journal.state.fault, durationMs: performance.now() - start,
      releaseId: build.releaseId, stagingMiB, environment: journal.state.environment };
    await journal.finish('complete', result);
    self.postMessage({ type: 'result', result });
  } catch (error) {
    const result = { success: false, error: String(error.stack || error), durationMs: performance.now() - start };
    await journal.finish('failed', result);
    self.postMessage({ type: 'result', result });
  }
};
