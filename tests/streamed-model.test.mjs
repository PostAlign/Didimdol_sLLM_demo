import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { bodyManifest, StreamedSession, TIMELINE_PROJECTIONS } from '../web/sllm/streamed-model.js';
import { StreamedWeights, DEFAULT_OUTPUT_BUFFERS } from '../web/sllm/streamed-weights.js';
import { acquireModelLease } from '../web/sllm/opfs-store.js';
import { executionEvidence } from '../web/sllm/experiments/results.js';
import { diagnosticSummary } from '../web/sllm/diagnostics.js';

const source = JSON.parse(await readFile(new URL('../model/initializers.json', import.meta.url)));
const descriptor = JSON.parse(await readFile(new URL('../model/streamed/manifest.json', import.meta.url)));
test('streamed artifacts exclude the entire tied embedding and preserve source ranges', async () => {
  const body = bodyManifest(source, descriptor);
  assert.equal(body.totalExternalTensorBytes, 401304064);
  assert.equal(body.initializers.filter(x => x.location).length, 235);
  for (const name of ['body', 'head']) {
    const bytes = await readFile(new URL(`../model/streamed/${name}.onnx`, import.meta.url));
    assert.equal(createHash('sha256').update(bytes).digest('hex'), descriptor.graphs[name].sha256);
  }
  const wrong = structuredClone(descriptor); wrong.chunks[0].offset += 4;
  assert.throws(() => bodyManifest(source, wrong), /embedding\/source mismatch/);
  const retained = structuredClone(descriptor); retained.bodyInitializerNames.push('embed_tokens.chunk0');
  assert.throws(() => bodyManifest(source, retained), /body manifest/);
});

test('embedding lookup reads exact rows across chunk boundaries and honors cancellation', async () => {
  const reads = [], controller = new AbortController();
  // Exercise the reader without a GPU: constructor allocations are covered by the browser test.
  const weights = Object.assign(Object.create(StreamedWeights.prototype), {
    ort: { Tensor: class { constructor(type, data, dims) { Object.assign(this, { type, data, dims }); } } },
    tracker: { ledger: {} }, descriptor, files: new Map(source.files.map(x => [x.location, x])), signal: controller.signal,
    metrics: { embeddingReadBytes: 0, embeddingMs: 0 },
    store: { async readRangeInto(file, offset, size, destination) {
      reads.push({ file: file.location, offset, size }); destination.fill(reads.length);
    } },
  });
  const result = await weights.embeddings({ dims: [1, 4], data: BigInt64Array.from([0n, 16383n, 16384n, 262143n]) });
  assert.deepEqual(result.dims, [1, 4, 640]);
  assert.deepEqual(reads.map(x => x.offset), [descriptor.chunks[0].offset,
    descriptor.chunks[0].offset + 16383 * 2560, descriptor.chunks[1].offset, descriptor.chunks[15].offset + 16383 * 2560]);
  assert.ok(reads.every(x => x.size === 2560));
  await assert.rejects(weights.embeddings({ dims: [1, 1], data: [262144n] }), /Invalid token/);
  controller.abort();
  await assert.rejects(weights.embeddings({ dims: [1, 1], data: [0n] }), /abort/i);
  assert.equal(reads.length, 4);
});

test('the streaming lease survives preparation and releases on disposal', async () => {
  let held = false;
  const locks = { async request(name, options, callback) {
    if (held) return callback(null);
    held = true;
    try { return await callback({ name }); } finally { held = false; }
  } };
  const release = await acquireModelLease(locks);
  assert.equal(held, true);
  await assert.rejects(acquireModelLease(locks), /다른 탭/);
  await release();
  assert.equal(held, false);
  await assert.rejects(acquireModelLease(null), /잠금/);
});

test('completed exports retain measured inference streaming totals rather than preparation totals', () => {
  const streaming = { projections: 4, outputReadBytes: 4 * 671088640 };
  const run = { status: 'complete', last: { stage: 'complete' },
    environment: { modelExecution: 'streamed' }, records: [{ stage: 'streamed-step-complete', streaming }],
    summary: { streaming: { projections: 0 } } };
  assert.deepEqual(diagnosticSummary(run).streaming, streaming);
  assert.deepEqual(executionEvidence({ kind: 'probe', success: true }, run).streaming, streaming);
});

test('two output buffers alternate, wait once per chunk unless told not to, and stop after cancelled reads', async () => {
  const originalUsage = globalThis.GPUBufferUsage;
  globalThis.GPUBufferUsage = { STORAGE: 128, COPY_DST: 8 };
  let created = 0, destroyed = 0, writes = 0, pending = 0, reads = 0, waits = 0, maxPending = 0, scopes = 0, maxScopes = 0;
  const controller = new AbortController();
  const buffers = [], written = [];
  const device = {
    createBuffer() { created++; const buffer = { id: created, destroy() { destroyed++; } }; buffers.push(buffer); return buffer; },
    pushErrorScope() { scopes++; maxScopes = Math.max(maxScopes, scopes); },
    async popErrorScope() { scopes--; return null; },
    queue: { writeBuffer(buffer, offset, data) {
      assert.equal(scopes, 2, 'both scopes bracket the write itself');
      assert.ok(data.length <= 8 * 2**20); assert.equal(offset, (writes % 5) * 8 * 2**20); writes++; pending++;
      written.push(buffer.id);
      maxPending = Math.max(maxPending, pending);
    }, async onSubmittedWorkDone() { assert.equal(scopes, 0, 'no scope is held across the queue wait'); waits++; pending = 0; } },
  };
  const ort = { Tensor: { fromGpuBuffer(buffer) { return { buffer, dispose() {} }; } } };
  const store = { async readRangeInto(file, offset, size) { assert.equal(scopes, 0, 'no scope is held across a disk read'); assert.ok(size <= 8 * 2**20); reads++; } };
  let weights;
  try {
    weights = new StreamedWeights({ ort, device, tracker: { ledger: {}, observeBuffer() {} }, store,
      manifest: source, descriptor, signal: controller.signal, outputBuffers: 2 });
    assert.equal(weights.metrics.cpuScratchBytes, 8 * 2**20, 'the default scratch is the 8 MiB staging size');
    assert.equal(created, 2, 'two output buffers when requested');
    assert.equal(weights.metrics.gpuBufferBytes, 80 * 2**20); assert.equal(weights.metrics.gpuBufferCount, 2);
    assert.deepEqual([weights.slot(0), weights.slot(1), weights.slot(2), weights.slot(15)], [0, 1, 0, 1]);
    assert.equal(weights.tensors[1].buffer, buffers[1]);
    const timing = await weights.upload(0); await weights.upload(1); await weights.upload(2);
    assert.ok(timing.start <= timing.end && timing.readMs >= 0 && timing.waitMs >= 0 && timing.readMs + timing.waitMs <= timing.end - timing.start + 1);
    assert.equal(reads, 15); assert.equal(writes, 15);
    assert.deepEqual(written, [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1], 'chunks alternate between the two buffers');
    assert.equal(waits, 3, 'one queue wait per chunk, not per staging piece');
    assert.equal(maxPending, 5, 'a whole chunk of writes is queued before the wait');
    assert.equal(maxScopes, 2, 'scopes never nest beyond one write');
    assert.equal(weights.metrics.uploadedBytes, 120 * 2**20);
    assert.equal(weights.metrics.readCalls, 15); assert.equal(weights.metrics.writeCalls, 15); assert.equal(weights.metrics.queueWaits, 3);
    assert.ok(weights.metrics.outputReadMs <= weights.metrics.uploadMs);
    // The overlapped loop skips the wait: the readback that follows bounds the backlog.
    const unwaited = await weights.upload(3, { queueWait: false });
    assert.equal(waits, 3, 'queueWait: false submits the chunk without waiting');
    assert.equal(unwaited.waitMs, 0); assert.equal(weights.metrics.queueWaits, 3); assert.equal(weights.metrics.chunks, 4);
    assert.equal(weights.metrics.uploadedBytes, 160 * 2**20);
    store.readRangeInto = async () => controller.abort();
    await assert.rejects(weights.upload(4), /abort/i);
    assert.equal(writes, 20, 'cancel during disk read must not enqueue another upload');
    assert.equal(weights.metrics.uploadedBytes, 160 * 2**20, 'an interrupted chunk is not counted as uploaded');
    weights.dispose(); weights.dispose();
    assert.equal(destroyed, 2);
  } finally { weights?.dispose(); globalThis.GPUBufferUsage = originalUsage; }
});

test('one output buffer is the default serial path and other counts are refused', () => {
  const originalUsage = globalThis.GPUBufferUsage;
  globalThis.GPUBufferUsage = { STORAGE: 128, COPY_DST: 8 };
  let created = 0;
  const device = { createBuffer() { created++; return { destroy() {} }; } };
  const ort = { Tensor: { fromGpuBuffer() { return { dispose() {} }; } } };
  const tracker = { ledger: {}, observeBuffer() {} };
  try {
    assert.equal(DEFAULT_OUTPUT_BUFFERS, 1, 'the September 14 phone paid 30-36% per token for the overlap');
    const single = new StreamedWeights({ ort, device, tracker, store: {}, manifest: source, descriptor });
    assert.equal(created, 1, 'one output buffer by default');
    assert.equal(single.metrics.gpuBufferBytes, 40 * 2**20); assert.equal(single.metrics.gpuBufferCount, 1);
    assert.deepEqual([single.slot(0), single.slot(1), single.slot(15)], [0, 0, 0]);
    single.dispose();
    for (const outputBuffers of [0, 3, '2', null]) {
      assert.throws(() => new StreamedWeights({ ort, device, tracker, store: {}, manifest: source, descriptor, outputBuffers }), RangeError, String(outputBuffers));
    }
  } finally { globalThis.GPUBufferUsage = originalUsage; }
});

test('the streamed scratch follows the staging size and refuses sizes a chunk cannot hold', async () => {
  const originalUsage = globalThis.GPUBufferUsage;
  globalThis.GPUBufferUsage = { STORAGE: 128, COPY_DST: 8 };
  let reads = 0, waits = 0;
  const device = { createBuffer() { return { destroy() {} }; }, pushErrorScope() {}, async popErrorScope() { return null; },
    queue: { writeBuffer(buffer, offset, data) { assert.ok(data.length <= 2 * 2**20); }, async onSubmittedWorkDone() { waits++; } } };
  const ort = { Tensor: { fromGpuBuffer() { return { dispose() {} }; } } };
  const store = { async readRangeInto() { reads++; } };
  const tracker = { ledger: {}, observeBuffer() {} };
  try {
    const small = new StreamedWeights({ ort, device, tracker, store, manifest: source, descriptor, scratchBytes: 2 * 2**20, outputBuffers: 1 });
    await small.upload(0);
    assert.equal(reads, 20); assert.equal(waits, 1);
    assert.equal(small.metrics.cpuScratchBytes, 2 * 2**20);
    small.dispose();
    for (const scratchBytes of [0, 6, 41 * 2**20, 2**20 + 1]) {
      assert.throws(() => new StreamedWeights({ ort, device, tracker, store, manifest: source, descriptor, scratchBytes }), RangeError, String(scratchBytes));
    }
  } finally { globalThis.GPUBufferUsage = originalUsage; }
});

test('a disk error remains the primary failure while both GPU error scopes are drained', async () => {
  const diskError = new Error('disk read failed');
  let popped = 0, reads = 0;
  const weights = Object.assign(Object.create(StreamedWeights.prototype), {
    check() {}, descriptor, scratch: new Uint8Array(16), files: new Map(source.files.map(x => [x.location, x])),
    buffers: [{}], metrics: { outputReadBytes: 0, outputReadMs: 0, readCalls: 0, writeCalls: 0 },
    store: { async readRangeInto() { if (reads++) throw diskError; } },
    device: { pushErrorScope() {}, async popErrorScope() { popped++; throw new Error('device lost'); }, queue: { writeBuffer() {} } },
  });
  await assert.rejects(weights.upload(0), error => error === diskError);
  assert.equal(popped, 2, 'the scopes of the one completed write are drained');
});

/** A session over fake ORT sessions and a fake queue that logs the order of GPU work and disk reads. */
function fakeSession({ outputBuffers, log, failReadAt = null }) {
  const originalUsage = globalThis.GPUBufferUsage;
  globalThis.GPUBufferUsage = { STORAGE: 128, COPY_DST: 8 };
  let buffers = 0, waits = 0, downloads = 0;
  const device = {
    createBuffer() { return { id: ++buffers, destroy() {} }; },
    pushErrorScope() {}, async popErrorScope() { return null; },
    queue: { writeBuffer(buffer, offset) { if (offset === 0) log.push(`write ${buffer.id}`); }, async onSubmittedWorkDone() { waits++; log.push('wait'); } },
  };
  const ort = { Tensor: class {
    constructor(type, data, dims) { Object.assign(this, { type, data, dims }); }
    static fromGpuBuffer(buffer) { return { buffer, dispose() {} }; }
    dispose() {}
  } };
  let reads = 0;
  const store = { async readRangeInto(file, offset, size, destination) {
    if (failReadAt != null && ++reads === failReadAt) throw new Error('disk read failed');
    if (size === 2560) destination.fill(1);
  } };
  const hidden = { dispose() { log.push('dispose hidden'); } };
  const body = { inputNames: ['input_ids', 'streamed_embeddings', 'attention_mask'], inputMetadata: [], outputNames: ['hidden_states', 'present.0.key'],
    outputMetadata: [], async run(feeds) { assert.equal(feeds.streamed_embeddings.dims[2], 640); return { hidden_states: hidden, 'present.0.key': { dispose() {} } }; } };
  const head = { async run({ weight }) {
    const chunk = weight.buffer.id, index = log.filter(x => x.startsWith('run')).length;
    log.push(`run ${index} on ${chunk}`);
    await new Promise(resolve => setTimeout(resolve, 1));
    const values = new Float32Array(16384).fill(index);
    return { chunk_logits: { getData() { log.push(`readback ${index}`); downloads++; return new Promise(resolve => setTimeout(() => resolve(values), 1)); }, dispose() {} } };
  } };
  globalThis.GPUBufferUsage = originalUsage;
  const checkpoints = [];
  globalThis.GPUBufferUsage = { STORAGE: 128, COPY_DST: 8 };
  try {
    const session = new StreamedSession({ ort, body, head, descriptor: { ...descriptor, hiddenOutput: 'hidden_states' }, store, manifest: source,
      tracker: { device, ledger: {}, observeBuffer() {} }, checkpoint: async record => { checkpoints.push(structuredClone(record)); }, outputBuffers });
    return { session, checkpoints, counters: { get waits() { return waits; }, get downloads() { return downloads; } } };
  } finally { globalThis.GPUBufferUsage = originalUsage; }
}
const feeds = { input_ids: { dims: [1, 2], data: BigInt64Array.from([5n, 7n]) }, attention_mask: { dims: [1, 2], data: BigInt64Array.from([1n, 1n]) } };

test('the overlapped projection submits each readback before the next chunk is written and skips the upload wait', async () => {
  const log = [];
  const { session, checkpoints, counters } = fakeSession({ outputBuffers: 2, log });
  const outputs = await session.run(feeds);
  assert.equal(outputs.logits.dims[2], 262144);
  for (let i = 0; i < 16; i++) assert.equal(outputs.logits.data[i * 16384 + 7], i, `chunk ${i} lands at its rows`);
  assert.equal(counters.downloads, 16);
  assert.equal(counters.waits, 1, 'only the first chunk, uploaded before the loop, waits for the queue');
  const find = (prefix, from = 0) => log.findIndex((entry, index) => index >= from && entry.startsWith(prefix));
  for (let i = 0; i < 15; i++) {
    const readback = find(`readback ${i}`), nextWrite = find(`write ${i % 2 === 0 ? 2 : 1}`, find(`run ${i} on`));
    assert.ok(readback >= 0 && nextWrite > readback, `chunk ${i + 1} is written after readback ${i} is submitted: ${log.slice(readback - 1, nextWrite + 1)}`);
    assert.ok(find(`run ${i + 1} on`) > nextWrite, 'the next projection runs after its chunk is written');
  }
  assert.equal(checkpoints.length, 1);
  const [step] = checkpoints;
  assert.equal(step.stage, 'streamed-step-complete');
  assert.equal(step.streaming.projections, 1); assert.equal(step.streaming.queueWaits, 1); assert.equal(step.streaming.chunks, 16);
  assert.equal(step.chunkTimelines.length, 1);
  const [timeline] = step.chunkTimelines;
  assert.equal(timeline.projection, 1); assert.equal(timeline.outputBuffers, 2); assert.equal(timeline.chunks.length, 16);
  for (const [i, entry] of timeline.chunks.entries()) {
    assert.equal(entry.chunk, i); assert.equal(entry.slot, i % 2);
    for (const key of ['uploadStart', 'uploadEnd', 'runSubmit', 'runResolved', 'readbackSubmit', 'readbackResolved', 'readMs', 'waitMs']) {
      assert.equal(typeof entry[key], 'number', `${key} of chunk ${i}`);
    }
    assert.ok(entry.uploadEnd >= entry.uploadStart && entry.runResolved >= entry.runSubmit && entry.readbackResolved >= entry.readbackSubmit);
    if (i > 0) assert.ok(entry.uploadStart >= timeline.chunks[i - 1].readbackSubmit, `chunk ${i} upload starts after readback ${i - 1} is submitted`);
  }
  assert.equal(timeline.chunks[0].waitMs >= 0 && timeline.chunks[1].waitMs, 0);
  assert.ok(JSON.stringify(timeline).length < 4096, 'a timeline stays small enough for the journal');
  // Later projections carry totals only; a checkpoint every N projections still carries the first timelines.
  session.stepCheckpointEvery = 3;
  for (let i = 0; i < 3; i++) await session.run(feeds);
  assert.equal(checkpoints.length, 2);
  assert.equal(checkpoints[1].streaming.projections, 3);
  assert.deepEqual(checkpoints[1].chunkTimelines.map(t => t.projection), [2], `projections beyond ${TIMELINE_PROJECTIONS} are not timed`);
  assert.equal(session.timelines.length, 0);
});

test('the serial projection waits per chunk, uploads after each readback and reports a disk failure with its position', async () => {
  const log = [];
  const { session, checkpoints, counters } = fakeSession({ outputBuffers: 1, log });
  const outputs = await session.run(feeds);
  for (let i = 0; i < 16; i++) assert.equal(outputs.logits.data[i * 16384], i);
  assert.equal(counters.waits, 16, 'one queue wait per chunk, inside the upload');
  const find = (prefix, from = 0) => log.findIndex((entry, index) => index >= from && entry.startsWith(prefix));
  for (let i = 0; i < 15; i++) {
    const readback = find(`readback ${i}`), write = find('write 1', readback);
    assert.ok(write > readback && find(`run ${i + 1} on`) > write, 'the single buffer is rewritten only after the readback');
  }
  assert.equal(checkpoints[0].chunkTimelines[0].chunks.every(entry => entry.slot === 0 && entry.waitMs >= 0), true);
  // Two embedding reads, five pieces each for chunks 0 and 1, then the third piece of chunk 2.
  const failing = fakeSession({ outputBuffers: 2, log: [], failReadAt: 2 + 5 + 5 + 3 });
  await assert.rejects(failing.session.run(feeds), /disk read failed/);
  const error = failing.checkpoints.find(record => record.stage === 'streamed-error');
  assert.equal(error.position.operation, 'output-read');
  assert.equal(error.position.chunk, 2, 'the failing read belongs to the chunk being uploaded during projection 1');
  assert.deepEqual(error.computing, { chunk: 1, slot: 1 });
  assert.equal(failing.session.busy, false);
});
