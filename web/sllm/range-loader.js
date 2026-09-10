import { BlobTensorSource, checkedRange } from './external-source.js';

export const STAGING_MIB = [8, 16, 32, 64];

/** Called inside OrtCreateSession by the patched EM_ASYNC_JS bridge. */
export class SessionRangeLoader {
  constructor({ manifest, stagingMiB = 16, checkpoint = async () => {}, emit = () => {}, maxCpuTensorBytes = 65536 }) {
    if (!STAGING_MIB.includes(stagingMiB)) throw new RangeError('stagingMiB must be 8, 16, 32 or 64');
    this.stagingBytes = stagingMiB * 2 ** 20;
    this.checkpoint = checkpoint;
    this.emit = emit;
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
      totalExternalTensorBytes: manifest.totalExternalTensorBytes,
      loadedInitializerCount: 0, rangeReadCount: 0, deviceLost: null,
    };
  }
  event(stage, details = {}) {
    this.emit({ stage, ...details, metrics: { ...this.metrics }, timestamp: Date.now() });
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
      this.event('device-lost');
      void this.checkpoint({ stage: 'device-lost', ...this.last, metrics: { ...this.metrics } });
    });
  }
  async beforeAllocate(name, isCpu) {
    if (this.closed) throw new Error('Initializer allocation after session creation');
    const init = this.byName.get(name);
    if (!init) throw new Error(`Unknown initializer: ${name}`);
    if (isCpu && init.bytes > this.maxCpuTensorBytes) throw new Error(`Large initializer assigned to CPU: ${name}`);
    this.last = { initializerName: name, shape: init.shape, index: init.index,
      offset: init.offset, length: init.bytes, location: init.location };
    await this.checkpoint({ ...this.last, stage: 'allocate-initializer', metrics: { ...this.metrics } });
    this.event('initializer-before-allocate', this.last);
  }
  async load({ location, file, offset, length, target, loadType, gpu, getHeap }) {
    if (this.closed) throw new Error('External weights cannot be read after session creation');
    if (this.busy) throw new Error('Concurrent initializer loads would exceed the staging budget');
    checkedRange(offset, length, file instanceof Blob ? file.size : file.byteLength);
    const init = this.byRange.get(`${location}:${offset}:${length}`);
    if (!init) throw new Error(`Initializer missing from verified manifest: ${location}:${offset}:${length}`);
    if (loadType !== 0 && loadType !== 1) throw new Error(`Unsupported load type ${loadType}`);
    if (loadType === 0 && length > this.maxCpuTensorBytes) {
      throw new Error(`Large initializer assigned to CPU: ${init.name} (${length} bytes). Refusing full WASM allocation path.`);
    }
    this.busy = true;
    let scopes = 0;
    try {
      this.last = { initializerName: init.name, shape: init.shape, index: init.index, offset, length, location };
      this.event('initializer-start', this.last);
      // Await persistence BEFORE reading/uploading this initializer. Worker postMessage
      // alone cannot guarantee that the main thread saved the crash position in time.
      await this.checkpoint({ ...this.last, stage: 'upload-initializer', metrics: { ...this.metrics } });
      if (gpu) {
        this.observeDevice(gpu.device);
        if (length % 4 || gpu.buffer.size < length || length > gpu.device.limits.maxBufferSize ||
            length > gpu.device.limits.maxStorageBufferBindingSize) throw new Error(`GPU size/alignment: ${init.name}`);
        if (!this.gpuBuffers.has(gpu.buffer)) {
          this.gpuBuffers.add(gpu.buffer);
          this.metrics.gpuWeightAllocated += gpu.buffer.size;
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
        if (this.metrics.deviceLost) throw new Error('WebGPU device lost');
        this.event('range-read-start', { offset: offset + position, bytes: size });
        if (source) await source.readRangeInto(location, offset + position, size, data);
        else data.set(file.subarray(offset + position, offset + position + size));
        this.metrics.rangeReadCount++;
        this.event('range-read-complete', { offset: offset + position, bytes: size });
        if (gpu) {
          // FP32 lengths, offsets and staging sizes are multiples of four. No
          // full-tensor MAP_WRITE buffer and no CPU-to-WASM copy on this path.
          gpu.device.queue.writeBuffer(gpu.buffer, position, data.buffer, data.byteOffset, size);
          this.event('gpu-write-buffer', { offset: position, bytes: size });
          // Bound driver upload backlog too, not just JS references.
          await gpu.device.queue.onSubmittedWorkDone();
          if (this.metrics.deviceLost) throw new Error('WebGPU device lost during upload');
        } else {
          const heap = getHeap();
          checkedRange(target + position, size, heap.byteLength);
          heap.set(data, target + position);
        }
        const heapBytes = getHeap().byteLength;
        this.metrics.wasmHeapBytes = heapBytes;
        this.metrics.wasmHeapPeak = Math.max(this.metrics.wasmHeapPeak, heapBytes);
      }
      while (scopes) {
        scopes--;
        const error = await gpu.device.popErrorScope();
        if (error) throw new Error(`WebGPU upload: ${error.message}`);
      }
      this.completed.add(init.name);
      this.metrics.loadedInitializerCount = this.completed.size;
      this.event('initializer-complete', this.last);
    } finally {
      while (scopes) { scopes--; await gpu.device.popErrorScope(); }
      this.busy = false;
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
