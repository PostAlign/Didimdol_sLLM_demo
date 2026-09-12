import { errorDetails, gpuOperationContext } from '../diagnostics.js';
import { SessionRangeLoader, STAGING_MIB } from '../range-loader.js';
import { OpfsWeightStore } from '../opfs-store.js';
import { installGpuTracking } from '../gpu-device.js';
import { FIXTURE } from './fixture.js';
import { loadOrt } from '../ort-runtime.js';
import { disposeResources } from '../cleanup.js';

/** Use every uploaded element while retaining all weight buffers until the end. No ORT import. */
export async function residentProbe(manifest, checkpoint, stagingMiB = 8, store = null, borrowed = null) {
  if (!STAGING_MIB.includes(stagingMiB)) throw new Error('Invalid staging size');
  const persist = checkpoint;
  let current = {}, tracked, primaryError;
  tracked = borrowed?.tracker || await installGpuTracking(manifest.largestInitializerBytes, record => {
    if (['gpu-error', 'gpu-uncaptured-error', 'device-lost'].includes(record.stage)) {
      return persist({ ...record, gpuLedger: tracked?.ledger });
    }
  }, { context: () => gpuOperationContext(current) });
  checkpoint = async record => {
    current = record;
    borrowed?.signal?.throwIfAborted();
    await tracked.flush();
    await persist({ ...record, ...borrowed?.details?.(), gpuLedger: tracked.ledger, ...(store ? { storage: { ...store.metrics } } : {}) });
    if (tracked.ledger.lastError) throw new Error(`WebGPU: ${tracked.ledger.lastError}`);
  };
  try {
    let device = borrowed?.device;
    if (!device) {
      const adapter = await navigator.gpu.requestAdapter();
      if (!adapter) throw new Error('WebGPU adapter unavailable');
      const largest = manifest.largestInitializerBytes;
      device = await adapter.requestDevice({ requiredLimits: {
        maxBufferSize: Math.max(268435456, largest), maxStorageBufferBindingSize: Math.max(134217728, largest),
      } });
    }
    if (manifest.largestInitializerBytes > Math.min(device.limits.maxBufferSize, device.limits.maxStorageBufferBindingSize)) {
      throw new Error('Resident weights exceed the retained ORT device limits');
    }
    const buffers = [];
    let allocated = 0, uploaded = 0, writeReturned = 0, lost, closed = false, lossSaved = Promise.resolve(), probeError;
    device.lost.then(info => {
      if (closed) return;
      lost = { reason: info.reason, message: info.message };
      lossSaved = checkpoint({ stage: 'device-lost', info: lost, gpuWeightAllocated: allocated, gpuWeightUploaded: uploaded });
    });
    const onError = event => { void checkpoint({ stage: 'gpu-uncaptured-error', ...errorDetails(event.error) }); };
    device.addEventListener('uncapturederror', onError);
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
      tracked.setBufferRole(output, 'verification'); tracked.setBufferRole(readback, 'verification');
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
        cpuStagingBytes: scratch.byteLength + expected.byteLength, residentDeviceId: tracked.ledger.tracking.activeDeviceId };
    } catch (error) { probeError = error; throw error; }
    finally {
      closed = true;
      device.removeEventListener('uncapturederror', onError);
      // The outer owner always destroys its device, including allocation/limit
      // failures before the inner scope was entered. Borrowers release only buffers.
      const cleanup = await disposeResources({ timeoutMs: 1000, releaseSession: async () => {
        while (scopes) { scopes--; await device.popErrorScope(); }
        await lossSaved;
      } });
      for (const buffer of buffers) buffer.destroy();
      if (!cleanup.success && !probeError) throw new Error(`Resident cleanup failed: ${JSON.stringify(cleanup.errors)}`);
    }
  } catch (error) { primaryError = error; throw error; }
  finally {
    if (!borrowed) {
      const cleanup = await disposeResources({ tracker: tracked });
      if (!cleanup.success && !primaryError) throw new Error(`Resident device cleanup failed: ${JSON.stringify(cleanup.errors)}`);
    }
  }
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

export async function opfsResidentProbe(manifest, checkpoint, stagingMiB, fixture, borrowed = null) {
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
    result = await residentProbe(manifest, checkpoint, stagingMiB, store, borrowed);
  } finally { store.close(); }
  return { ...result, storage: { ...store.metrics } };
}

export async function runtimeProbe(mode, checkpoint, idleSeconds = 120, stagingMiB = 8, options = {}) {
  if (!['asyncify', 'jspi'].includes(mode)) throw new Error('Invalid runtime mode');
  if (mode === 'jspi' && !(WebAssembly.Suspending && WebAssembly.promising)) throw new Error('이 브라우저는 JSPI를 지원하지 않습니다.');
  const persist = checkpoint;
  let current = {}, primaryError;
  checkpoint = async record => {
    current = record;
    options.signal?.throwIfAborted();
    return persist(record);
  };
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
    }, { context: () => gpuOperationContext(current) });
    options.onTracker?.(tracked);
    loader = new SessionRangeLoader({ manifest, stagingMiB, checkpoint, gpuTracker: tracked,
      weightRole: options.residentManifest ? 'runtime-weight' : 'weight', signal: options.signal, storage: () => store.metrics });
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
      let outputs;
      try {
        outputs = await session.run({ X: input });
        if (outputs.Y.data.length !== 2560 || outputs.Y.data.some(v => v !== 1024.5)) throw new Error('Wrong FP32 fixture output');
      } finally { input.dispose(); outputs?.Y.dispose(); }
    }
    if (loader.metrics.rangeReadCount !== reads) throw new Error('Weights read during inference');
    await checkpoint({ stage: 'runtime-inference-complete', metrics: loader.sampleMetrics(), storage: { ...store.metrics } });
    if (options.residentManifest) {
      const runtimeDeviceId = tracked.ledger.tracking.activeDeviceId;
      const runtimeGpuBeforeResidency = tracked.ledger;
      const details = () => ({ runtimeMetrics: loader.sampleMetrics(), runtimeDeviceId, smallSessionRetained: true });
      await checkpoint({ stage: 'runtime-resident-start', ...details(), runtimeGpuBeforeResidency });
      const result = await opfsResidentProbe(options.residentManifest, checkpoint, stagingMiB, !!options.fixture, {
        tracker: tracked, device: tracked.device, details, signal: options.signal,
      });
      if (result.residentDeviceId !== runtimeDeviceId || tracked.ledger.tracking.deviceCount !== 1) {
        throw new Error('Combined comparison must use the same ORT GPUDevice');
      }
      // The small session remains reachable through verification of the final weight.
      if (!session) throw new Error('Small runtime session was released during residency');
      return { ...result, ...details(), runtimeGpuBeforeResidency, runtimeStorage: { ...store.metrics },
        sameDevice: true, ortWasmInstantiated: true, inferenceVerified: true,
        modelSessionCreated: false, tokenizerPrepared: false };
    }
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
  } catch (error) { primaryError = error; throw error; }
  finally {
    const cleanup = await disposeResources({ store, loader, tracker: tracked, releaseSession: () => session?.release() });
    await options.onCleanup?.(cleanup);
    if (!cleanup.success && !primaryError) throw new Error(`Runtime cleanup failed: ${JSON.stringify(cleanup.errors)}`);
  }
}

