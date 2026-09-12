import { BlobTensorSource, checkedRange } from './external-source.js';
import { errorDetails } from './diagnostics.js';

export const STAGING_MIB = [2, 4, 8, 16, 32, 64];

/** Called inside OrtCreateSession by the patched EM_ASYNC_JS bridge. */
export class SessionRangeLoader {
  constructor({ manifest, stagingMiB = 8, checkpoint = async () => {}, emit = () => {}, maxCpuTensorBytes = 65536, signal,
    gpuLedger = () => null, gpuTracker = null, storage = () => null, clock = () => performance.now() }) {
    if (!STAGING_MIB.includes(stagingMiB)) throw new RangeError('stagingMiB must be 2, 4, 8, 16, 32 or 64');
    this.stagingBytes = stagingMiB * 2 ** 20;
    this.checkpoint = checkpoint;
    this.emit = emit;
    this.signal = signal;
    this.gpuTracker = gpuTracker;
    this.gpuLedger = gpuTracker ? () => gpuTracker.ledger : gpuLedger;
    this.clock = clock;
    this.storage = storage;
    this.lossSaved = Promise.resolve();
    this.maxCpuTensorBytes = maxCpuTensorBytes;
    this.byRange = new Map(manifest.initializers.filter(t => t.location).map(t => [`${t.location}:${t.offset}:${t.bytes}`, t]));
    this.byName = new Map(manifest.initializers.map(t => [t.name, t]));
    this.completed = new Set();
    this.gpuBuffers = new WeakSet();
    this.devices = new WeakSet();
    this.closed = false;
    this.busy = false;
    this.scratch = null;
    this.metrics = {
      cpuStagingCurrent: 0, cpuStagingPeak: 0, wasmTempCurrent: 0, wasmTempPeak: 0,
      wasmHeapBytes: 0, wasmHeapPeak: 0, cpuInitializerBytes: 0,
      gpuWeightAllocated: 0, gpuWeightPeak: 0,
      gpuWeightUploaded: 0, gpuWeightBufferCount: 0, gpuWriteCalls: 0,
      gpuWriteReturnedBytes: 0, gpuQueueCompletedBytes: 0, gpuValidatedInitializerCount: 0,
      gpuWriteMs: 0, gpuWaitMs: 0, gpuWaitPeakMs: 0,
      totalExternalTensorBytes: manifest.totalExternalTensorBytes,
      loadedInitializerCount: 0, rangeReadCount: 0, deviceLost: null,
    };
  }
  event(stage, details = {}) {
    this.emit({ stage, ...details, metrics: { ...this.metrics }, timestamp: Date.now() });
  }
  async record(stage, details = {}) {
    const storage = this.storage();
    await this.checkpoint({ ...this.last, ...details, stage, metrics: this.sampleMetrics(), gpuLedger: this.gpuLedger(),
      ...(storage ? { storage: { ...storage } } : {}) });
  }
  check() {
    this.signal?.throwIfAborted();
    if (this.metrics.deviceLost) throw new Error('WebGPU device lost');
    if (this.gpuTracker?.ledger.lastError) throw new Error(`WebGPU: ${this.gpuTracker.ledger.lastError}`);
  }
  sampleMetrics() {
    if (this.getHeap) {
      this.metrics.wasmHeapBytes = this.getHeap().byteLength;
      this.metrics.wasmHeapPeak = Math.max(this.metrics.wasmHeapPeak, this.metrics.wasmHeapBytes);
    }
    return { ...this.metrics };
  }
  staging(delta) {
    this.metrics.cpuStagingCurrent += delta;
    this.metrics.cpuStagingPeak = Math.max(this.metrics.cpuStagingPeak, this.metrics.cpuStagingCurrent);
  }
  observeDevice(device) {
    if (this.devices.has(device)) return;
    this.devices.add(device);
    device.lost.then(info => {
      if (this.closed && info.reason === 'destroyed') return;
      this.metrics.deviceLost = { reason: info.reason, message: info.message };
      // The main thread may terminate this worker as soon as it sees the event.
      this.lossSaved = this.record('device-lost').then(() => this.event('device-lost'));
    });
  }
  async beforeAllocate(name, isCpu) {
    this.check();
    if (this.closed) throw new Error('Initializer allocation after session creation');
    const init = this.byName.get(name);
    if (!init) throw new Error(`Unknown initializer: ${name}`);
    if (isCpu && init.bytes > this.maxCpuTensorBytes) throw new Error(`Large initializer assigned to CPU: ${name}`);
    this.last = { initializerName: name, shape: init.shape, index: init.index,
      offset: init.offset, length: init.bytes, location: init.location };
    await this.gpuTracker?.flush();
    this.check();
    await this.record('allocate-initializer', { placement: isCpu ? 'cpu' : 'gpu' });
    this.event('initializer-before-allocate', this.last);
  }
  async load({ location, file, offset, length, target, loadType, gpu, getHeap }) {
    this.check();
    this.getHeap = getHeap;
    if (this.closed) throw new Error('External weights cannot be read after session creation');
    if (this.busy) throw new Error('Concurrent initializer loads would exceed the staging budget');
    const ranged = file?.ortRangeSource === 2;
    checkedRange(offset, length, ranged || file instanceof Blob ? file.size : file.byteLength);
    const init = this.byRange.get(`${location}:${offset}:${length}`);
    if (!init) throw new Error(`Initializer missing from verified manifest: ${location}:${offset}:${length}`);
    if (loadType !== 0 && loadType !== 1) throw new Error(`Unsupported load type ${loadType}`);
    if (loadType === 0 && length > this.maxCpuTensorBytes) {
      throw new Error(`Large initializer assigned to CPU: ${init.name} (${length} bytes). Refusing full WASM allocation path.`);
    }
    this.busy = true;
    let scopes = 0;
    let operation = 'upload-initializer', primaryError;
    const popScope = async () => {
      scopes--;
      const error = await gpu.device.popErrorScope();
      if (error) {
        await this.record('gpu-error', { ...errorDetails(error), operation: 'upload-error-scope' });
        throw new Error(`WebGPU upload: ${error.message}`, { cause: error });
      }
    };
    try {
      this.last = { initializerName: init.name, shape: init.shape, index: init.index, offset, length, location };
      this.event('initializer-start', this.last);
      // Await persistence BEFORE reading/uploading this initializer. Worker postMessage
      // alone cannot guarantee that the main thread saved the crash position in time.
      if (gpu) {
        this.observeDevice(gpu.device);
        this.gpuTracker?.observeBuffer(gpu.device, gpu.buffer);
        // Allocation scopes are opened by createBuffer interception, before the
        // native buffer exists. Persist their result before uploading any bytes.
        await this.gpuTracker?.flush();
        this.check();
        if (length % 4 || gpu.buffer.size < length || length > gpu.device.limits.maxBufferSize ||
            length > gpu.device.limits.maxStorageBufferBindingSize) throw new Error(`GPU size/alignment: ${init.name}`);
        if (!this.gpuBuffers.has(gpu.buffer)) {
          this.gpuBuffers.add(gpu.buffer);
          this.metrics.gpuWeightAllocated += gpu.buffer.size;
          this.metrics.gpuWeightBufferCount++;
          this.metrics.gpuWeightPeak = Math.max(this.metrics.gpuWeightPeak, this.metrics.gpuWeightAllocated);
          this.event('gpu-weight-buffer', { ...this.last, bytes: gpu.buffer.size });
        }
        gpu.device.pushErrorScope('out-of-memory'); scopes++;
        gpu.device.pushErrorScope('validation'); scopes++;
        this.event('wasm-staging-bypassed', { bytes: 0, ...this.last });
      } else {
        this.metrics.cpuInitializerBytes += length;
        this.event('wasm-cpu-initializer', { bytes: length, ...this.last });
      }
      await this.record('upload-initializer');
      if (!this.scratch) {
        this.scratch = new Uint8Array(this.stagingBytes);
        this.staging(this.scratch.byteLength);
        this.event('cpu-staging-allocate', { bytes: this.scratch.byteLength });
      }
      const source = file instanceof Blob
        ? new BlobTensorSource(new Map([[location, file]]), delta => this.staging(delta)) : null;
      for (let position = 0; position < length; position += this.stagingBytes) {
        const size = Math.min(this.stagingBytes, length - position);
        const data = this.scratch.subarray(0, size);
        this.check();
        const range = { fileOffset: offset + position, destinationOffset: position, chunkBytes: size };
        operation = 'range-read';
        await this.record('range-read', range);
        this.event('range-read-start', { offset: offset + position, bytes: size });
        if (ranged) await file.readRangeInto(offset + position, size, data);
        else if (source) await source.readRangeInto(location, offset + position, size, data);
        else data.set(file.subarray(offset + position, offset + position + size));
        this.metrics.rangeReadCount++;
        this.event('range-read-complete', { offset: offset + position, bytes: size });
        if (gpu) {
          this.check();
          await this.record('gpu-write', { ...range, operation: 'writeBuffer', phase: 'before-call' });
          // FP32 lengths, offsets and staging sizes are multiples of four. No
          // full-tensor MAP_WRITE buffer and no CPU-to-WASM copy on this path.
          operation = 'writeBuffer';
          const writeStart = this.clock();
          gpu.device.queue.writeBuffer(gpu.buffer, position, data.buffer, data.byteOffset, size);
          range.gpuWriteMs = this.clock() - writeStart;
          this.metrics.gpuWriteMs += range.gpuWriteMs;
          this.metrics.gpuWriteCalls++;
          this.metrics.gpuWriteReturnedBytes += size;
          this.event('gpu-write-buffer', { offset: position, bytes: size });
          // Bound driver upload backlog too, not just JS references.
          await this.record('gpu-wait', { ...range, operation: 'onSubmittedWorkDone', phase: 'before-call' });
          operation = 'onSubmittedWorkDone';
          const waitStart = this.clock();
          await gpu.device.queue.onSubmittedWorkDone();
          range.gpuWaitMs = this.clock() - waitStart;
          this.metrics.gpuWaitMs += range.gpuWaitMs;
          this.metrics.gpuWaitPeakMs = Math.max(this.metrics.gpuWaitPeakMs, range.gpuWaitMs);
          if (this.metrics.deviceLost) throw new Error('WebGPU device lost during upload');
          // Queue completion observed, not a physical-memory or error-scope measurement.
          this.metrics.gpuWeightUploaded += size;
          this.metrics.gpuQueueCompletedBytes += size;
        } else {
          const heap = getHeap();
          checkedRange(target + position, size, heap.byteLength);
          heap.set(data, target + position);
        }
        this.sampleMetrics();
        await this.record('range-complete', range);
      }
      operation = 'popErrorScope';
      while (scopes) await popScope();
      if (gpu) this.metrics.gpuValidatedInitializerCount++;
      this.completed.add(init.name);
      this.metrics.loadedInitializerCount = this.completed.size;
      await this.record('initializer-complete');
      this.event('initializer-complete', this.last);
    } catch (error) {
      primaryError = error;
      await this.record('loader-error', { ...errorDetails(error), operation });
      throw error;
    } finally {
      try {
        while (scopes) {
          try { await popScope(); }
          catch (error) { if (!primaryError) { primaryError = error; throw error; } }
        }
      } finally { this.busy = false; }
    }
  }
  close(success) {
    if (this.busy) throw new Error('Cannot close a running range loader');
    this.closed = true;
    if (this.scratch) {
      const bytes = this.scratch.byteLength;
      this.scratch = null;
      this.staging(-bytes);
      this.event('cpu-staging-release', { bytes });
    }
    this.event(success ? 'session-create-complete' : 'session-create-failed', this.last);
    return { ...this.metrics };
  }
}
