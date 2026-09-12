import test from 'node:test';
import assert from 'node:assert/strict';
import { RunDiagnostics, readRun, recoveryEvidence, diagnosticSummary } from '../web/sllm/diagnostics.js';

const snapshotRun = (id, environment, persist) => new RunDiagnostics(id, { ...environment, diagnosticsMode: 'snapshot' }, persist);

function journalStore() {
  const values = new Map(), writes = [];
  return {
    values, writes, failNext: false,
    async persistBatch(entries) {
      writes.push(structuredClone(entries));
      if (this.failNext) { this.failNext = false; return false; }
      for (const [key, value] of entries) values.set(key, structuredClone(value));
      return true;
    },
    read: async key => values.get(key),
    readEntries: async id => [...values].filter(([key]) => key.startsWith(`journal:${id}:`)).map(([, value]) => value),
  };
}

test('incremental journals recover an interrupted pre-call checkpoint and retry atomic metadata after failure', async () => {
  const store = journalStore();
  const run = new RunDiagnostics('compact', { build: { releaseId: 'pinned' } }, undefined,
    { persistBatch: entries => store.persistBatch(entries) });
  store.failNext = true;
  assert.equal(await run.checkpoint({ stage: 'graph-verified', expectedInitializerCount: 251 }), false);
  await run.checkpoint({ stage: 'ort-initializers-start' });
  for (let i = 0; i < 70; i++) await run.checkpoint({ stage: 'allocate-initializer', initializerName: `W${i}`, length: 40 });
  await run.checkpoint({ stage: 'gpu-wait', phase: 'before-call', destinationOffset: 8,
    metrics: { gpuWriteReturnedBytes: 16, gpuQueueCompletedBytes: 8 } });
  const recovered = await readRun('compact', store);
  assert.equal(recovered.status, 'running');
  assert.equal(recovered.last.stage, 'gpu-wait');
  assert.equal(recovered.last.metrics.gpuQueueCompletedBytes, 8);
  assert.equal(recovered.milestones['graph-verified'].expectedInitializerCount, 251);
  assert.ok(recovered.milestones['ort-initializers-start']);
  assert.equal(recovered.initializerOrder.length, 70);
  assert.equal(recovered.records.length, 64);
  assert.equal(recovered.persistence.failures, 1);
  assert.equal(recovered.environment.build.releaseId, 'pinned');
  assert.equal(store.writes.slice(2).some(entries => entries.some(([key]) => key === 'run:compact')), false, 'header is saved once after successful retry');
  assert.equal(store.values.size, 1 + 1 + 64 + 70 + 2, 'history cannot grow beyond the ring');
  await run.checkpoint({ stage: 'gpu-error', message: 'original allocation failure' });
  await run.finish('failed', { error: 'outer failure' });
  store.values.set('cleanup:compact', { success: false, errors: ['cleanup error'] });
  const failed = await readRun('compact', store);
  assert.equal(failed.firstFault.message, 'original allocation failure');
  assert.equal(failed.summary.error, 'outer failure');
  assert.equal(failed.cleanup.success, false);
  assert.equal(failed.last.stage, 'failed');
});

test('incremental writes reduce serialized payload while preserving legacy and snapshot exports', async () => {
  const store = journalStore();
  assert.equal(await readRun(undefined, store), null);
  assert.equal(await readRun(null, store), null);
  const environment = { build: { releaseId: 'release', inventory: 'x'.repeat(12000) } };
  const run = new RunDiagnostics('compact-bytes', environment, undefined, { persistBatch: entries => store.persistBatch(entries) });
  let snapshotBytes = 0;
  const snapshot = snapshotRun('snapshot-bytes', environment, async state => { snapshotBytes += JSON.stringify(state).length; return true; });
  for (let i = 0; i < 200; i++) {
    const record = { stage: 'gpu-wait', destinationOffset: i * 8, phase: 'before-call',
      metrics: { gpuWeightAllocated: 40, gpuQueueCompletedBytes: i * 8 } };
    await run.checkpoint(record); await snapshot.checkpoint(record);
  }
  const compactBytes = store.writes.reduce((sum, entries) => sum + JSON.stringify(entries).length, 0);
  assert.ok(compactBytes < snapshotBytes / 5, `${compactBytes} vs ${snapshotBytes}`);
  for (const schemaVersion of [2, 3]) {
    const old = { schemaVersion, runId: `legacy-${schemaVersion}`, status: 'running', last: { stage: 'gpu-wait' } };
    store.values.set(`run:${old.runId}`, old);
    assert.deepEqual(await readRun(old.runId, store), old);
  }
});

test('oversized recent records are bounded without losing the durable latest position or original fault', async () => {
  const store = journalStore();
  const run = new RunDiagnostics('bounded', {}, undefined, { persistBatch: entries => store.persistBatch(entries) });
  const message = 'failure detail '.repeat(4000);
  await run.checkpoint({ stage: 'gpu-error', message, destinationOffset: 24 });
  const interrupted = await readRun('bounded', store);
  assert.equal(interrupted.last.message, message);
  assert.equal(interrupted.firstFault.message, message);
  assert.equal(interrupted.records[0].historyTruncated, true);
  assert.ok(JSON.stringify(interrupted.records[0]).length < 16384);
  await run.finish('failed');
  const finished = await readRun('bounded', store);
  assert.equal(finished.last.stage, 'failed');
  assert.equal(finished.firstFault.message, message);
});

test('evicting recent events preserves preparation, allocation order and the original fault', async () => {
  let saved;
  const run = snapshotRun('retained', {}, async value => { saved = structuredClone(value); return true; });
  await run.checkpoint({ stage: 'graph-verified', graphSha256: 'graph', expectedInitializerCount: 251 });
  await run.checkpoint({ stage: 'weight-ready', location: 'weights', source: 'opfs-cache' });
  await run.checkpoint({ stage: 'session-create', storage: { cacheHits: 9 } });
  for (let i = 0; i < 70; i++) await run.checkpoint({ stage: 'allocate-initializer', initializerName: `w${i}`, length: 40 });
  await run.checkpoint({ stage: 'gpu-error', errorType: 'GPUOutOfMemoryError', message: 'allocation failed' });
  await run.checkpoint({ stage: 'device-lost', message: 'later device loss' });
  await run.finish('failed', { error: 'cleanup error' });
  assert.equal(saved.schemaVersion, 4);
  assert.equal(saved.records.length, 64);
  assert.equal(saved.milestones['graph-verified'].expectedInitializerCount, 251);
  assert.equal(saved.milestones['session-create'].storage.cacheHits, 9);
  assert.equal(saved.files.weights.source, 'opfs-cache');
  assert.deepEqual(saved.initializerOrder.map(value => value.name), Array.from({ length: 70 }, (_, i) => `w${i}`));
  assert.equal(saved.firstFault.errorType, 'GPUOutOfMemoryError');
  assert.equal(saved.fault.stage, 'device-lost');
  assert.equal(saved.status, 'device-lost');
  assert.equal(saved.persistence.completed, saved.recordCount - 1);
});

test('recovery associates lifecycle with the run interval and UUID without rewriting the worker', () => {
  const run = { runId: 'current', startedAt: 100, status: 'running', last: { stage: 'gpu-wait', timestamp: 120 } };
  const lifecycle = [
    { event: 'pagehide', timestamp: 96 },
    { event: 'visibilitychange', timestamp: 105, runId: 'another-run' },
    { event: 'pagehide', timestamp: 122, runId: 'current' },
    { event: 'pageshow', timestamp: 131, runId: 'current' },
  ];
  const recovery = recoveryEvidence(run, { lifecycle }, 130);
  assert.deepEqual(recovery.lifecycle, [lifecycle[2]]);
  assert.deepEqual(recovery.lifecycleHistory, lifecycle);
  assert.equal(recovery.cause, 'unknown');
  assert.equal(run.status, 'running');
  assert.equal(diagnosticSummary({ ...run, recovery }).effectiveStatus, 'interrupted');
  assert.deepEqual(recoveryEvidence({ ...run, startedAt: undefined }, { lifecycle }, 130).lifecycle, []);
});

test('old zero GPU ledgers are partial and missing measurements remain unknown', () => {
  const run = { status: 'running', recovery: { classification: 'interrupted' }, last: {
    stage: 'gpu-wait', initializerName: 'embed_tokens.chunk0', destinationOffset: 24 * 2**20,
    metrics: { gpuWeightAllocated: 922290176, gpuWeightUploaded: 905512960, loadedInitializerCount: 214 },
    gpuLedger: { requestedCurrent: 0, requestedPeak: 0, bufferCount: 0 },
  } };
  const summary = diagnosticSummary(run);
  assert.equal(summary.trackingStatus, 'partial');
  assert.equal(summary.gpuQueueCompletedBytes, 905512960);
  assert.equal(summary.gpuWriteReturnedBytes, null);
  assert.equal(summary.gpuValidatedInitializerCount, null);
  assert.equal(summary.destinationOffset, 24 * 2**20);
  assert.equal(diagnosticSummary({ schemaVersion: 2, last: {} }).trackingStatus, 'unknown');
  assert.equal(diagnosticSummary({ schemaVersion: 2, last: {} }).gpuWeightAllocated, null);
});

test('fault/stop summaries retain progress and cache facts after a terminal record', () => {
  const run = { status: 'failed', fault: { stage: 'gpu-error', initializerName: 'failed-weight' },
    last: { stage: 'failed' }, summary: { error: 'error' },
    milestones: { 'weights-prepared': { storage: { cacheHits: 9, totalFiles: 9 } } },
    records: [{ stage: 'gpu-wait', initializerName: 'failed-weight', destinationOffset: 8,
      metrics: { gpuWeightAllocated: 40, loadedInitializerCount: 1 }, gpuLedger: { tracking: { status: 'complete' } } }] };
  const summary = diagnosticSummary(run);
  assert.equal(summary.gpuWeightAllocated, 40);
  assert.equal(summary.loadedInitializerCount, 1);
  assert.equal(summary.initializerName, 'failed-weight');
  assert.equal(summary.trackingStatus, 'complete');
  assert.equal(summary.storage.cacheHits, 9);
});

test('failed persistence does not break later checkpoints, and queued records are snapshots', async () => {
  let saved, calls = 0;
  const run = snapshotRun('persistence', {}, async value => {
    if (++calls === 1) throw new Error('storage unavailable');
    saved = structuredClone(value); return true;
  });
  const record = { stage: 'gpu-error', message: 'original', metrics: { gpuWeightUploaded: 0 } };
  const pending = run.checkpoint(record);
  record.message = 'changed'; record.metrics.gpuWeightUploaded = 40;
  await pending;
  await run.finish('complete');
  assert.equal(saved.firstFault.message, 'original');
  assert.equal(saved.firstFault.metrics.gpuWeightUploaded, 0);
  assert.equal(saved.persistence.failures, 1);
  assert.equal(saved.status, 'failed');
});

test('storage progress survives interruption, while a finished session reports closed handles', () => {
  const run = { status: 'running', last: { stage: 'gpu-wait', storage: { rangeReadBytes: 24, openHandles: 1 },
    metrics: { gpuWeightAllocated: 40, wasmHeapBytes: 64 },
    gpuLedger: { requestedCurrent: 56, tracking: { status: 'complete' } } },
    milestones: { 'weights-prepared': { storage: { rangeReadBytes: 0, openHandles: 0 } } } };
  assert.equal(diagnosticSummary(run).storage.rangeReadBytes, 24);
  assert.equal(diagnosticSummary(run).gpuRequestedCurrent, 56);
  assert.equal(diagnosticSummary(run).wasmHeapBytes, 64);
  const finished = { ...run, status: 'ready', last: { stage: 'ready' }, records: [run.last],
    summary: { storage: { rangeReadBytes: 40, openHandles: 0 }, timings: { modelLoadCallMs: 10, tokenizerPreparationMs: 5 } } };
  assert.equal(diagnosticSummary(finished).storage.openHandles, 0);
  assert.equal(diagnosticSummary(finished, { timings: { modelLoadCallMs: 10 } }).timings.tokenizerPreparationMs, 5);
  assert.equal(diagnosticSummary({ ...run, last: { ...run.last, gpuLedger: { requestedCurrent: 0 } } }).gpuRequestedCurrent, null);
});

test('run journals serialize persistence, isolate IDs and retain a GPU fault after cleanup', async () => {
  const saved = new Map();
  let pending = false;
  const persist = async (state, key) => {
    assert.equal(pending, false); pending = true;
    await Promise.resolve(); saved.set(key, structuredClone(state)); pending = false;
  };
  const run = snapshotRun('one', { browser: 'test' }, persist);
  await Promise.all(Array.from({ length: 70 }, (_, i) => run.checkpoint({ stage: 'range-read', destinationOffset: i })));
  await run.checkpoint({ stage: 'device-lost', metrics: { deviceLost: { reason: 'unknown' } } });
  await run.checkpoint({ stage: 'gpu-uncaptured-error', message: 'later cleanup error' });
  await run.finish('failed', { stage: 'session-create-failed' });
  const record = saved.get('run:one');
  assert.equal(record.records.length, 64);
  assert.equal(record.fault.stage, 'device-lost');
  assert.equal(record.status, 'device-lost');
  assert.equal(record.last.stage, 'device-lost');
  await snapshotRun('two', {}, persist).checkpoint({ stage: 'load-start' });
  assert.equal(saved.get('run:one').runId, 'one');
  assert.equal(saved.get('run:two').runId, 'two');
});
