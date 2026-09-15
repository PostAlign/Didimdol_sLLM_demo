import test from 'node:test';
import assert from 'node:assert/strict';
import { BlobTensorSource, HttpTensorSource, exactInteger } from '../web/sllm/external-source.js';
import { SessionRangeLoader } from '../web/sllm/range-loader.js';

test('Blob ranges preserve offsets and never request the whole blob arrayBuffer', async () => {
  const blob = new Blob([new Uint8Array([10, 20, 30, 40, 50])]);
  blob.arrayBuffer = () => { throw new Error('whole-file materialization'); };
  const source = new BlobTensorSource(new Map([['weights', blob]]));
  const dest = new Uint8Array(3);
  await source.readRangeInto('weights', 1n, 3, dest);
  assert.deepEqual([...dest], [20, 30, 40]);
  await assert.rejects(source.readRangeInto('weights', 4, 3, dest), /exceeds/);
  assert.throws(() => exactInteger(9007199254740993n), /Invalid/);
});

test('HTTP rejects ignored Range and detects short bodies', async () => {
  const files = new Map([['w', { url: 'https://example.test/w', size: 16 }]]);
  const bad = new HttpTensorSource(files, { fetchImpl: async () => new Response(new Uint8Array(16)) });
  await assert.rejects(bad.readRangeInto('w', 4, 4, new Uint8Array(4)), /206/);
  const short = new HttpTensorSource(files, { fetchImpl: async () => new Response(new Uint8Array(3), {
    status: 206, headers: { 'content-range': 'bytes 4-7/16' },
  }) });
  await assert.rejects(short.readRangeInto('w', 4, 4, new Uint8Array(4)), /Short/);
});

function fixture(size = 10 * 2**20, stagingMiB = 8) {
  const bytes = new Uint8Array(size + 13);
  for (let i = 0; i < size; i++) bytes[i + 13] = i % 251;
  const file = new Blob([bytes]);
  const init = { name: 'W', shape: [size / 4], index: 0, bytes: size, offset: 13, location: 'w' };
  const manifest = { totalExternalTensorBytes: size, initializers: [init] };
  const uploaded = new Uint8Array(size);
  const writes = [];
  let pending = false;
  let checkpointDone = false;
  const device = {
    limits: { maxBufferSize: 2**30, maxStorageBufferBindingSize: 2**27 },
    lost: new Promise(() => {}), pushErrorScope() {}, async popErrorScope() { return null; },
    queue: {
      writeBuffer(buffer, offset, data, dataOffset, length) {
        assert.equal(checkpointDone, true);
        assert.equal(pending, false, 'each queue submission must finish before the next write');
        pending = true;
        writes.push({ offset, length });
        uploaded.set(new Uint8Array(data, dataOffset, length), offset);
      },
      async onSubmittedWorkDone() { await Promise.resolve(); pending = false; },
    },
  };
  const loader = new SessionRangeLoader({ manifest, stagingMiB,
    checkpoint: async () => { await Promise.resolve(); checkpointDone = true; } });
  return { loader, bytes, uploaded, writes,
    request: { location: 'w', file, offset: 13, length: size, target: 123, loadType: 1,
      gpu: { device, buffer: { size } }, getHeap: () => new Uint8Array(65536) } };
}

test('chunk upload has exact bytes, queue backpressure, zero WASM staging and no decode reads', async () => {
  const f = fixture();
  await f.loader.load(f.request);
  assert.deepEqual(f.writes, [{ offset: 0, length: 8 * 2**20 }, { offset: 8 * 2**20, length: 2 * 2**20 }]);
  assert.deepEqual(f.uploaded, f.bytes.subarray(13));
  const m = f.loader.close(true);
  assert.equal(m.cpuStagingCurrent, 0);
  assert.ok(m.cpuStagingPeak <= 16 * 2**20); // reusable scratch + one bounded stream chunk
  assert.equal(m.wasmTempPeak, 0);
  assert.equal(m.loadedInitializerCount, 1);
  await assert.rejects(f.loader.load(f.request), /after session/);
});

test('CPU fallback guard and GPU errors fail explicitly and release staging', async () => {
  const f = fixture();
  await assert.rejects(f.loader.load({ ...f.request, loadType: 0, gpu: null }), /assigned to CPU/);
  let first = true;
  f.request.gpu.device.popErrorScope = async () => first ? (first = false, { message: 'validation failure' }) : null;
  await assert.rejects(f.loader.load(f.request), /validation failure/);
  assert.equal(f.loader.metrics.gpuQueueCompletedBytes, 10 * 2**20);
  assert.equal(f.loader.metrics.gpuValidatedInitializerCount, 0, 'a completed queue does not imply valid upload scopes');
  assert.equal(f.loader.metrics.loadedInitializerCount, 0);
  assert.equal(f.loader.close(false).cpuStagingCurrent, 0);
});

test('heap is reacquired after async reads instead of retaining a detached view', async () => {
  const f = fixture(16);
  const heap = new Uint8Array(65536);
  await f.loader.load({ ...f.request, loadType: 0, gpu: null, getHeap: () => heap });
  assert.deepEqual(heap.subarray(123, 139), f.bytes.subarray(13));
  assert.equal(f.loader.close(true).cpuInitializerBytes, 16);
});

test('range descriptors preserve bytes and checkpoint GPU writes with current allocation counts', async () => {
  const f = fixture();
  const records = [];
  f.loader.checkpoint = async record => { records.push(structuredClone(record)); };
  let readBytes = 0;
  f.loader.storage = () => ({ rangeReadBytes: readBytes });
  // Remove the fixture's separate checkpoint flag; this test inspects the actual sequence.
  f.request.gpu.device.queue.writeBuffer = (buffer, offset, data, dataOffset, length) => {
    assert.equal(records.at(-1).stage, 'gpu-write');
    assert.equal(records.at(-1).destinationOffset, offset);
    assert.equal(records.at(-1).metrics.gpuWeightAllocated, f.uploaded.byteLength);
    f.uploaded.set(new Uint8Array(data, dataOffset, length), offset);
  };
  const descriptor = { ortRangeSource: 2, size: f.bytes.byteLength,
    async readRangeInto(offset, length, destination) {
      assert.equal(records.at(-1).stage, 'range-read');
      destination.set(f.bytes.subarray(offset, offset + length));
      readBytes += length;
    } };
  await f.loader.load({ ...f.request, file: descriptor });
  assert.deepEqual(f.uploaded, f.bytes.subarray(13));
  assert.equal(records.at(-1).stage, 'initializer-complete');
  assert.equal(f.loader.metrics.cpuStagingPeak, 8 * 2**20);
  assert.equal(records.at(-1).storage.rangeReadBytes, 10 * 2**20);
  assert.equal(records.find(r => r.stage === 'upload-initializer').metrics.wasmHeapBytes, 65536);
  // Pre-call position records keep the crash position and progress counters
  // only; the allocation and completion records carry the full snapshots.
  const positions = records.filter(r => SessionRangeLoader.POSITION_STAGES.has(r.stage));
  assert.deepEqual(positions.map(r => r.stage), ['upload-initializer', 'range-read', 'gpu-write', 'gpu-wait', 'range-read', 'gpu-write', 'gpu-wait']);
  for (const record of positions) {
    assert.equal(record.storage, undefined, `${record.stage} omits storage`);
    assert.deepEqual(Object.keys(record.metrics), SessionRangeLoader.POSITION_METRICS);
    assert.equal(record.initializerName, 'W');
  }
  assert.equal(positions[3].destinationOffset, 0);
  assert.equal(positions[6].destinationOffset, 8 * 2**20);
  assert.equal(positions[6].metrics.gpuQueueCompletedBytes, 8 * 2**20, 'progress counters stay on position records');
  assert.equal(JSON.stringify(positions[6]).length < JSON.stringify(records.at(-1)).length, true);
  // `allocate-initializer` is recorded by the createBuffer hook, which this direct load() call bypasses.
  const full = records.filter(r => !SessionRangeLoader.POSITION_STAGES.has(r.stage));
  assert.deepEqual(full.map(r => r.stage), ['initializer-complete']);
  assert.ok(full[0].storage && full[0].metrics.gpuWriteMs != null, 'the completion record keeps the full snapshot');
});

test('position records carry a slim ledger with tracking status and omit recent allocations', async () => {
  const f = fixture(16), records = [];
  const checkpoint = f.loader.checkpoint;
  f.loader.checkpoint = async record => { records.push(structuredClone(record)); await checkpoint(record); };
  f.loader.gpuLedger = () => ({ requestedCurrent: 16, observedPeak: 16, liveBufferCount: 1, bufferCount: 1, deviceLost: null, lastError: null,
    tracking: { status: 'complete', activeDeviceId: 1, deviceCount: 1, devices: [{ id: 1, fullHistory: true }] },
    categories: [{ role: 'weight' }], programs: { shaderModules: 0 }, recentAllocations: [{ id: 1 }, { id: 2 }] });
  await f.loader.load(f.request);
  const position = records.find(r => r.stage === 'gpu-wait');
  assert.deepEqual(position.gpuLedger, { requestedCurrent: 16, observedPeak: 16, liveBufferCount: 1, bufferCount: 1, deviceLost: null, lastError: null,
    tracking: { status: 'complete', activeDeviceId: 1, deviceCount: 1 }, recentAllocationsOmitted: 2, positionRecord: true });
  const complete = records.find(r => r.stage === 'initializer-complete').gpuLedger;
  assert.equal(complete.recentAllocations, undefined, 'ring records drop the allocation list');
  assert.equal(complete.recentAllocationsOmitted, 2);
  assert.deepEqual(complete.categories, [{ role: 'weight' }], 'ring records keep the category totals');
  assert.deepEqual(complete.tracking.devices, [{ id: 1, fullHistory: true }]);
  assert.equal(SessionRangeLoader.ringLedger(null), null);
  assert.deepEqual(SessionRangeLoader.ringLedger({ requestedCurrent: 4 }), { requestedCurrent: 4 });
  assert.equal(SessionRangeLoader.positionLedger(null), null);
  assert.equal(SessionRangeLoader.positionLedger({ requestedCurrent: 4 }).recentAllocationsOmitted, 0);
});

test('device-lost is emitted only after durable persistence finishes', async () => {
  const f = fixture(16);
  let lose, release, emitted = false;
  f.request.gpu.device.lost = new Promise(resolve => { lose = resolve; });
  const saved = new Promise(resolve => { release = resolve; });
  f.loader.checkpoint = async record => { if (record.stage === 'device-lost') await saved; };
  f.loader.emit = record => { if (record.stage === 'device-lost') emitted = true; };
  f.loader.observeDevice(f.request.gpu.device);
  lose({ reason: 'unknown', message: 'test loss' });
  await Promise.resolve();
  assert.equal(emitted, false);
  release(); await f.loader.lossSaved;
  assert.equal(emitted, true);
  await assert.rejects(f.loader.load(f.request), /device lost/);
});

test('2/4/8 MiB transfers preserve every byte and keep the same destination allocation', async () => {
  for (const stagingMiB of [2, 4, 8]) {
    const f = fixture(10 * 2**20, stagingMiB);
    await f.loader.load(f.request);
    assert.deepEqual(f.uploaded, f.bytes.subarray(13));
    assert.equal(f.loader.metrics.gpuWeightAllocated, 10 * 2**20);
    assert.equal(f.loader.metrics.gpuWeightUploaded, 10 * 2**20);
    assert.equal(f.loader.metrics.gpuWeightBufferCount, 1);
    assert.ok(f.writes.every(write => write.length <= stagingMiB * 2**20));
    assert.equal(f.loader.metrics.gpuWriteCalls, f.writes.length);
  }
});

test('upload completion excludes pending writes and GPU timings exclude checkpoint latency', async () => {
  const f = fixture();
  let clock = 0, release, entered;
  const firstWait = new Promise(resolve => { entered = resolve; });
  const gate = new Promise(resolve => { release = resolve; });
  const checkpoint = f.loader.checkpoint;
  f.loader.clock = () => clock;
  f.loader.checkpoint = async record => { await checkpoint(record); clock += 100; };
  const write = f.request.gpu.device.queue.writeBuffer;
  f.request.gpu.device.queue.writeBuffer = (...args) => { write(...args); clock += 3; };
  const wait = f.request.gpu.device.queue.onSubmittedWorkDone;
  let waits = 0;
  f.request.gpu.device.queue.onSubmittedWorkDone = async () => {
    if (++waits === 1) { entered(); await gate; }
    clock += 7; await wait();
  };
  const loading = f.loader.load(f.request);
  await firstWait;
  assert.equal(f.loader.metrics.gpuWeightAllocated, 10 * 2**20);
  assert.equal(f.loader.metrics.gpuWeightUploaded, 0);
  assert.equal(f.loader.metrics.gpuWriteReturnedBytes, 8 * 2**20);
  assert.equal(f.loader.metrics.gpuQueueCompletedBytes, 0);
  assert.equal(f.loader.metrics.gpuValidatedInitializerCount, 0);
  assert.equal(f.writes.length, 1);
  release(); await loading;
  assert.equal(f.loader.metrics.gpuWeightUploaded, 10 * 2**20);
  assert.equal(f.loader.metrics.gpuWriteReturnedBytes, 10 * 2**20);
  assert.equal(f.loader.metrics.gpuQueueCompletedBytes, 10 * 2**20);
  assert.equal(f.loader.metrics.gpuValidatedInitializerCount, 1);
  assert.equal(f.loader.metrics.gpuWriteMs, 6);
  assert.equal(f.loader.metrics.gpuWaitMs, 14);
  assert.equal(f.loader.metrics.gpuWaitPeakMs, 7);
});

test('queue errors are persisted with their operation before cleanup can fail', async () => {
  const f = fixture(16), records = [];
  const checkpoint = f.loader.checkpoint;
  f.loader.checkpoint = async record => { records.push(structuredClone(record)); await checkpoint(record); };
  f.request.gpu.device.queue.onSubmittedWorkDone = async () => { throw new Error('queue rejected'); };
  f.request.gpu.device.popErrorScope = async () => { throw new Error('cleanup rejected'); };
  await assert.rejects(f.loader.load(f.request), /queue rejected/);
  const fault = records.find(record => record.stage === 'loader-error');
  assert.equal(fault.operation, 'onSubmittedWorkDone');
  assert.equal(fault.message, 'queue rejected');
  assert.equal(fault.metrics.gpuWeightUploaded, 0);
  assert.equal(fault.metrics.gpuWriteReturnedBytes, 16);
  assert.equal(fault.metrics.gpuQueueCompletedBytes, 0);
  assert.equal(f.loader.busy, false);
  f.loader.close(false);
});

test('sealed loaders record auxiliary ORT sessions and still refuse weight reads', async () => {
  const f = fixture(), records = [];
  const checkpoint = f.loader.checkpoint;
  f.loader.checkpoint = async record => { records.push(structuredClone(record)); await checkpoint(record); };
  await f.loader.phase('ort-session-start', { sessionDiagnosticsVersion: 1 });
  await f.loader.load(f.request);
  await f.loader.phase('ort-session-complete');
  f.loader.close(true);
  // transformers.js creates the top_k session on the first sampled token, after the model session is sealed.
  await f.loader.phase('ort-session-start', { sessionDiagnosticsVersion: 1 });
  await f.loader.phase('ort-plan-start');
  await f.loader.phase('ort-session-complete');
  await f.loader.phase('ort-session-start', { sessionDiagnosticsVersion: 1 });
  const auxiliary = records.filter(record => record.stage.startsWith('auxiliary-'));
  assert.deepEqual(auxiliary.map(record => [record.stage, record.ortPhase, record.auxiliarySession]), [
    ['auxiliary-session-start', 'ort-session-start', 1], ['auxiliary-session-phase', 'ort-plan-start', 1],
    ['auxiliary-session-complete', 'ort-session-complete', 1], ['auxiliary-session-start', 'ort-session-start', 2]]);
  assert.equal(auxiliary[0].sessionDiagnosticsVersion, 1);
  assert.equal(auxiliary[0].metrics.loadedInitializerCount, 1, 'auxiliary records carry the sealed model metrics');
  assert.deepEqual(records.filter(record => record.stage.startsWith('ort-')).map(record => record.stage),
    ['ort-session-start', 'ort-session-complete'], 'the model session milestones are never overwritten');
  assert.equal(f.loader.auxiliarySessionCount, 2);
  await assert.rejects(f.loader.phase('session-start'), /Invalid ORT lifecycle phase/);
  await assert.rejects(f.loader.load(f.request), /after session creation/);
  await assert.rejects(f.loader.beforeAllocate('W', false), /after session creation/);
});

test('a loader created after WASM initialization samples the inherited heap before its first weight read', async () => {
  const manifest = { initializers: [], totalExternalTensorBytes: 0 };
  const records = [];
  const heap = new Uint8Array(24 * 2 ** 20);
  const loader = new SessionRangeLoader({ manifest, stagingMiB: 2, getHeap: () => heap, checkpoint: async record => { records.push(record); } });
  await loader.phase('ort-session-start');
  await loader.phase('ort-plan-start');
  assert.deepEqual(records.map(record => [record.stage, record.metrics.wasmHeapBytes]),
    [['ort-session-start', heap.byteLength], ['ort-plan-start', heap.byteLength]]);
  const late = new SessionRangeLoader({ manifest, stagingMiB: 2, checkpoint: async record => { records.push(record); } });
  await late.phase('ort-session-start');
  assert.equal(records.at(-1).metrics.wasmHeapBytes, 0, 'without an accessor the heap stays unmeasured rather than invented');
});
