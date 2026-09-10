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

function fixture(size = 10 * 2**20) {
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
  const loader = new SessionRangeLoader({ manifest, stagingMiB: 8,
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
  assert.equal(f.loader.close(false).cpuStagingCurrent, 0);
});

test('heap is reacquired after async reads instead of retaining a detached view', async () => {
  const f = fixture(16);
  const heap = new Uint8Array(65536);
  await f.loader.load({ ...f.request, loadType: 0, gpu: null, getHeap: () => heap });
  assert.deepEqual(heap.subarray(123, 139), f.bytes.subarray(13));
  assert.equal(f.loader.close(true).cpuInitializerBytes, 16);
});
