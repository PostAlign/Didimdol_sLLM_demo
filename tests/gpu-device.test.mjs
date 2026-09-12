import test from 'node:test';
import assert from 'node:assert/strict';
import { installGpuTracking } from '../web/sllm/gpu-device.js';

function fixture() {
  class Buffer {
    constructor(descriptor) { this.size = descriptor.size; this.destroyCalls = 0; }
    destroy() { this.destroyCalls++; }
  }
  class Device extends EventTarget {
    constructor() {
      super(); this.scopes = []; this.lost = new Promise(resolve => { this.lose = resolve; });
    }
    pushErrorScope(filter) { this.scopes.push({ filter, error: null }); }
    popErrorScope() { return Promise.resolve(this.scopes.pop().error); }
    destroy() { this.lose({ reason: 'destroyed', message: 'explicit destruction' }); }
    createBuffer(descriptor) {
      if (descriptor.size < 0) throw new RangeError('invalid buffer size');
      if (this.allocationError) this.scopes.findLast(s => s.filter === 'out-of-memory').error = this.allocationError;
      return new Buffer(descriptor);
    }
    createShaderModule() { return { shader: true }; }
    createComputePipeline() { return { pipeline: true }; }
    createComputePipelineAsync() { return this.pipelinePromise ||= Promise.resolve({ pipeline: true }); }
  }
  class Adapter {
    limits = { maxBufferSize: 2 ** 30, maxStorageBufferBindingSize: 2 ** 28 };
    async requestDevice() { return new Device(); }
  }
  const newGPUWrapper = () => ({ async requestAdapter() { return new Adapter(); } });
  const options = { gpu: newGPUWrapper(), adapterPrototype: Adapter.prototype,
    devicePrototype: Device.prototype, bufferPrototype: Buffer.prototype };
  return { options, Buffer, Device, Adapter, newGPUWrapper };
}

test('another GPU wrapper and multiple native devices are tracked without double-counting', async () => {
  const f = fixture(), original = f.Device.prototype.createBuffer;
  const tracked = await installGpuTracking(40, () => {}, f.options);
  try {
    assert.equal(tracked.ledger.tracking.status, 'unbound');
    assert.equal(tracked.ledger.requestedPeak, null);
    const first = await (await f.newGPUWrapper().requestAdapter()).requestDevice();
    const second = await (await f.newGPUWrapper().requestAdapter()).requestDevice();
    const a = first.createBuffer({ size: 40 });
    const b = second.createBuffer({ size: 80, mappedAtCreation: true });
    assert.ok(a instanceof f.Buffer, 'createBuffer must remain synchronous');
    tracked.observeBuffer(second, b); tracked.observeBuffer(second, b);
    await tracked.flush();
    assert.equal(tracked.device, second);
    assert.equal(tracked.ledger.tracking.status, 'complete');
    assert.equal(tracked.ledger.tracking.deviceCount, 2);
    assert.equal(tracked.ledger.requestedCurrent, 120);
    assert.equal(tracked.ledger.requestedPeak, 120);
    assert.equal(tracked.ledger.bufferCount, 2);
    assert.equal(tracked.ledger.mappedUploadRequested, 80);
    a.destroy(); a.destroy(); b.destroy();
    assert.equal(tracked.ledger.requestedCurrent, 0);
    assert.equal(tracked.ledger.liveBufferCount, 0);
    assert.equal(tracked.ledger.requestedPeak, 120);
  } finally { tracked.restore(); }
  assert.equal(f.Device.prototype.createBuffer, original);
});

test('buffer categories and compilation counters preserve synchronous APIs and original promises', async () => {
  const f = fixture();
  const tracked = await installGpuTracking(40, () => {}, { ...f.options, context: () => ({ observedDuring: 'ort-initializers-start' }) });
  try {
    const device = await (await f.options.gpu.requestAdapter()).requestDevice();
    const weight = device.createBuffer({ size: 40, usage: 136 });
    const tiny = device.createBuffer({ size: 8, usage: 72 });
    tracked.observeBuffer(device, weight);
    tracked.observeBuffer(device, tiny, 'runtime-weight');
    assert.equal(device.createShaderModule().shader, true);
    assert.equal(device.createComputePipeline().pipeline, true);
    const promise = device.createComputePipelineAsync();
    assert.equal(promise, device.pipelinePromise);
    await promise; await tracked.flush();
    assert.equal(tracked.ledger.programs.shaderModules, 1);
    assert.equal(tracked.ledger.programs.asyncPipelinesCompleted, 1);
    assert.equal(tracked.ledger.programs.coverage, 'complete');
    const groups = Object.fromEntries(tracked.ledger.categories.map(entry => [entry.role, entry]));
    assert.equal(groups.weight.requestedCurrent, 40);
    assert.equal(groups['runtime-weight'].requestedCurrent, 8);
    assert.equal(groups.other.requestedCurrent, 0);
    assert.equal(tracked.ledger.recentAllocations[0].usage, 136);
    assert.equal(tracked.ledger.recentAllocations[0].stage, 'ort-initializers-start');
    weight.destroy(); device.destroy();
    assert.ok(tracked.ledger.categories.every(entry => entry.requestedCurrent === 0 && entry.liveCount === 0));
  } finally { tracked.restore(); }
});

test('late binding repairs observed counts without inventing historical peaks or mapped allocations', async () => {
  const f = fixture();
  const nativeRequest = f.Adapter.prototype.requestDevice, nativeCreate = f.Device.prototype.createBuffer;
  const tracked = await installGpuTracking(40, () => {}, f.options);
  try {
    const device = await nativeRequest.call(new f.Adapter());
    const buffer = nativeCreate.call(device, { size: 40, mappedAtCreation: true });
    tracked.observeBuffer(device, buffer); tracked.observeBuffer(device, buffer);
    assert.equal(tracked.device, device);
    assert.equal(tracked.ledger.requestedCurrent, 40);
    assert.equal(tracked.ledger.tracking.status, 'partial');
    assert.equal(tracked.ledger.requestedPeak, null);
    assert.equal(tracked.ledger.mappedUploadRequested, null);
    assert.equal(tracked.ledger.lateBufferCount, 1);
    const next = device.createBuffer({ size: 80 });
    await tracked.flush();
    assert.equal(tracked.ledger.observedPeak, 120);
    buffer.destroy(); next.destroy();
    assert.equal(tracked.ledger.requestedCurrent, 0);
    assert.equal(tracked.ledger.tracking.status, 'partial');
  } finally { tracked.restore(); }
});

test('allocation scopes capture the original initializer and flush awaits durable fault writes', async () => {
  const f = fixture(), records = [];
  let current = 'W', release, saved = false;
  const gate = new Promise(resolve => { release = resolve; });
  const tracked = await installGpuTracking(40, async record => {
    if (record.stage === 'gpu-error') { records.push(record); await gate; saved = true; }
  }, { ...f.options, context: () => ({ initializerName: current, destinationOffset: 24 }) });
  try {
    const device = await (await f.options.gpu.requestAdapter()).requestDevice();
    device.allocationError = { name: 'GPUOutOfMemoryError', message: 'allocation failed' };
    const buffer = device.createBuffer({ size: 40 });
    assert.ok(buffer instanceof f.Buffer);
    assert.equal(device.scopes.length, 0, 'allocation scopes must be popped before the next operation');
    current = 'later initializer';
    let finished = false;
    const flush = tracked.flush().then(() => { finished = true; });
    await Promise.resolve(); await Promise.resolve();
    assert.equal(finished, false);
    release(); await flush;
    assert.equal(saved, true);
    assert.equal(records[0].initializerName, 'W');
    assert.equal(records[0].destinationOffset, 24);
    assert.equal(records[0].operation, 'createBuffer');
    assert.equal(records[0].bytes, 40);
    assert.throws(() => device.createBuffer({ size: -1 }), /invalid buffer size/);
    await tracked.flush();
    assert.equal(tracked.ledger.firstError.message, 'allocation failed');
    assert.equal(tracked.ledger.lastError, 'invalid buffer size');
  } finally { release(); tracked.restore(); }
});

test('unhookable allocation methods are reported as partial while bridge-observed buffers remain countable', async () => {
  const f = fixture();
  Object.defineProperty(f.Device.prototype, 'createBuffer', { writable: false, configurable: false });
  const tracked = await installGpuTracking(40, () => {}, f.options);
  try {
    const device = await (await f.options.gpu.requestAdapter()).requestDevice();
    // Block the per-device fallback too.
    Object.defineProperty(device, 'createBuffer', { value: f.Device.prototype.createBuffer, configurable: false });
    const buffer = device.createBuffer({ size: 40 });
    tracked.observeBuffer(device, buffer);
    assert.equal(tracked.ledger.tracking.status, 'partial');
    assert.equal(tracked.ledger.requestedCurrent, 40);
    assert.equal(tracked.ledger.requestedPeak, null);
  } finally { tracked.restore(); }
});

test('explicit device destruction releases observed live counts and is not a GPU fault', async () => {
  const f = fixture(), records = [];
  const tracked = await installGpuTracking(40, record => { records.push(record); }, f.options);
  try {
    const device = await (await f.options.gpu.requestAdapter()).requestDevice();
    const buffer = device.createBuffer({ size: 40 });
    await tracked.flush();
    device.destroy(); device.destroy(); buffer.destroy();
    await Promise.resolve(); await tracked.flush();
    assert.equal(tracked.ledger.requestedCurrent, 0);
    assert.equal(tracked.ledger.liveBufferCount, 0);
    assert.equal(tracked.ledger.requestedPeak, 40);
    assert.equal(tracked.ledger.deviceLost, null);
    assert.equal(records.some(record => record.stage === 'device-lost'), false);
  } finally { tracked.restore(); }
});
