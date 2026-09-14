import test from 'node:test';
import assert from 'node:assert/strict';
import { RunDiagnostics, readRun, recoveryEvidence, diagnosticSummary, errorText, localTimestamp, deviceClock, ortPlanEvidence, stopLabel } from '../web/sllm/diagnostics.js';

test('reentry distinguishes completion, observation interruption and unfinished cleanup', () => {
  const session = { startedAt: 10, status: 'complete', summary: { modelSessionCreated: true }, last: { timestamp: 20 } };
  assert.equal(recoveryEvidence(session).classification, 'completed-run-reentry');
  const pending = { ...session, cleanup: { stage: 'cleanup-start', startedAt: 21 } };
  const recovery = recoveryEvidence(pending, { phase: 'cleanup' });
  assert.equal(recovery.classification, 'cleanup-reentry');
  assert.equal(diagnosticSummary({ ...pending, recovery }).effectiveStatus, 'interrupted');
  assert.equal(recoveryEvidence({ ...session, cleanup: { success: true } }, { phase: 'cleanup' }).classification, 'completed-run-reentry');
  const observing = { ...session, status: 'running', summary: null, milestones: { 'session-create-complete': {} } };
  assert.equal(recoveryEvidence(observing).interruptedPhase, 'after-session-create');
  assert.equal(recoveryEvidence(observing).cause, 'unknown');
});

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

test('position records update the durable head only and stay out of the event ring', async () => {
  const store = journalStore();
  const run = new RunDiagnostics('positions', {}, undefined, { persistBatch: entries => store.persistBatch(entries) });
  await run.checkpoint({ stage: 'graph-verified', expectedInitializerCount: 2 });
  // A load of two initializers: the six records of each are one allocation, four
  // pre-call positions and one completion. Only the first and last go into the ring.
  for (const name of ['W0', 'W1']) {
    await run.checkpoint({ stage: 'allocate-initializer', initializerName: name, length: 40, metrics: { gpuWeightAllocated: 40 }, storage: { rangeReadBytes: 0 } });
    for (const stage of ['upload-initializer', 'range-read', 'gpu-write', 'gpu-wait']) {
      await run.checkpoint({ stage, initializerName: name, destinationOffset: 0, metrics: { gpuWeightAllocated: 40, loadedInitializerCount: 0 } });
    }
    await run.checkpoint({ stage: 'initializer-complete', initializerName: name, metrics: { gpuWeightAllocated: 40, loadedInitializerCount: 1 }, storage: { rangeReadBytes: 40 } });
  }
  assert.equal(run.state.recordCount, 13);
  assert.equal(run.state.eventCount, 5);
  assert.deepEqual(run.state.records.map(record => record.stage),
    ['graph-verified', 'allocate-initializer', 'initializer-complete', 'allocate-initializer', 'initializer-complete']);
  const entries = [...store.values.keys()];
  assert.equal(entries.filter(key => key.includes(':event:')).length, 5, 'position records write no ring slot');
  const heads = store.writes.map(batch => batch.find(([key]) => key.endsWith(':head'))?.[1]).filter(Boolean);
  assert.equal(heads.filter(head => head.last).length, 8, 'every position record is the head\'s last');
  assert.equal(heads.at(-1).last, undefined, 'a ring record is found by the head\'s event count instead');
  assert.equal(heads.at(-1).persistence.note, undefined, 'the fixed note is not repeated with every head');
  let recovered = await readRun('positions', store);
  assert.equal(recovered.last.stage, 'initializer-complete');
  assert.equal(recovered.records.length, 5);
  assert.equal(recovered.eventCount, 5);
  assert.ok(recovered.persistence.note.includes('serializedBytes'), 'readRun re-attaches the note');
  assert.equal(diagnosticSummary(recovered).storage.rangeReadBytes, 40);
  // Interrupted before a queue wait: the position is durable and the summary reads
  // the position's counters, while storage comes from the ring's last full record.
  await run.checkpoint({ stage: 'allocate-initializer', initializerName: 'W2', length: 40, metrics: { gpuWeightAllocated: 80 }, storage: { rangeReadBytes: 80 } });
  await run.checkpoint({ stage: 'gpu-wait', initializerName: 'W2', destinationOffset: 8, phase: 'before-call',
    metrics: { gpuWeightAllocated: 80, gpuQueueCompletedBytes: 8, loadedInitializerCount: 2 }, gpuLedger: { requestedCurrent: 80, tracking: { status: 'complete' } } });
  recovered = await readRun('positions', store);
  assert.equal(recovered.last.stage, 'gpu-wait');
  assert.equal(recovered.records.at(-1).stage, 'allocate-initializer');
  const summary = diagnosticSummary({ ...recovered, recovery: { classification: 'interrupted', interruptedPhase: 'execution' } });
  assert.equal(summary.stage, 'gpu-wait');
  assert.equal(summary.initializerName, 'W2');
  assert.equal(summary.destinationOffset, 8);
  assert.equal(summary.loadedInitializerCount, 2);
  assert.equal(summary.gpuRequestedCurrent, 80);
  assert.equal(summary.storage.rangeReadBytes, 80);
  // Journals written before this change numbered ring slots by recordCount; they still read.
  const legacy = journalStore();
  legacy.values.set('run:old', { schemaVersion: 4, storageFormat: 'incremental-v1', runId: 'old', startedAt: 1, environment: {} });
  legacy.values.set('journal:old:head', { kind: 'head', status: 'running', recordCount: 2, persistence: { mode: 'compact' } });
  legacy.values.set('journal:old:event:1', { kind: 'event', sequence: 1, value: { stage: 'load-start' } });
  legacy.values.set('journal:old:event:2', { kind: 'event', sequence: 2, value: { stage: 'gpu-wait' } });
  const old = await readRun('old', legacy);
  assert.equal(old.last.stage, 'gpu-wait');
  assert.equal(old.records.length, 2);
  assert.ok(old.persistence.note);
});

test('stop labels name the tester\'s stop and the rows an evaluation finished', async () => {
  const store = journalStore();
  const run = new RunDiagnostics('stopped', {}, undefined, { persistBatch: entries => store.persistBatch(entries) });
  await run.checkpoint({ stage: 'row-complete', row: 1, nTok: 3, result: { totalMs: 9 } });
  await run.finish('cancelled', { error: 'AbortError: The operation was aborted.', stopSource: 'user-stop', completedRows: 1 });
  const recovered = await readRun('stopped', store);
  const summary = diagnosticSummary(recovered);
  assert.equal(summary.stopSource, 'user-stop');
  assert.equal(summary.completedRows, 1);
  assert.equal(stopLabel(summary), '사용자 정지 · 1행 완료');
  assert.equal(stopLabel(diagnosticSummary({ status: 'cancelled', last: { stage: 'user-cancelled' } })), '사용자 정지');
  // Older cancelled journals carry no stop source; their rows still count.
  const older = diagnosticSummary({ status: 'cancelled', last: { stage: 'cancelled' }, milestones: { 'evaluation-rows': { rows: 100 } },
    rows: [{ row: 1 }, { row: 2 }], summary: { completedRows: undefined } });
  assert.equal(older.stopSource, null);
  assert.equal(stopLabel(older), '중단됨 (정지 출처 미기록) · 2행 완료');
  // A stopped load or probe has no evaluation rows to count.
  assert.equal(diagnosticSummary({ status: 'cancelled', last: { stage: 'user-cancelled' }, rows: [] }).completedRows, null);
  assert.equal(stopLabel(null), '중단됨 (정지 출처 미기록)');
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

test('recovery records unload evidence and the re-entry gap without upgrading the cause', () => {
  const run = { runId: 'current', startedAt: 100, status: 'running', last: { stage: 'gpu-wait', timestamp: 120 } };
  const silent = recoveryEvidence(run, { lifecycle: [], navigationType: 'reload' }, 963);
  assert.equal(silent.unloadEvidence, 'no-unload-event');
  assert.equal(silent.unloadObserved, false);
  assert.equal(silent.reentryGapMs, 843);
  assert.equal(silent.navigationType, 'reload');
  assert.equal(silent.cause, 'unknown', 'a silent re-entry is consistent with a process kill but never proves one');
  const summary = diagnosticSummary({ ...run, recovery: silent });
  assert.equal(summary.unloadEvidence, 'no-unload-event');
  assert.equal(summary.reentryGapMs, 843);
  assert.equal(summary.navigationType, 'reload');
  const hidden = recoveryEvidence(run, { lifecycle: [{ event: 'visibilitychange', visibility: 'hidden', timestamp: 121, runId: 'current' }] }, 130);
  assert.equal(hidden.unloadEvidence, 'unload-observed');
  assert.equal(hidden.unloadObserved, true);
  const earlier = recoveryEvidence(run, { lifecycle: [{ event: 'pagehide', timestamp: 90 }] }, 130);
  assert.equal(earlier.unloadEvidence, 'no-unload-event', 'events before the run interval do not explain this run');
  assert.equal(recoveryEvidence({ ...run, status: 'complete' }, {}, 130).unloadEvidence, null);
  assert.equal(recoveryEvidence({ ...run, fault: { stage: 'device-lost' } }, {}, 130).unloadEvidence, 'recorded-fault');
  assert.equal(recoveryEvidence({ ...run, last: null }, {}, 130).reentryGapMs, null);
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
  await Promise.all(Array.from({ length: 70 }, (_, i) => run.checkpoint({ stage: 'initializer-complete', destinationOffset: i })));
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

test('error text keeps the message when a JavaScriptCore stack omits it', () => {
  const jsc = Object.assign(new Error('Session phase after loader close'),
    { stack: 'phase@https://example.test/range-loader.js:77:37\n@https://example.test/ort.asyncify.mjs:2:8897' });
  assert.equal(errorText(jsc), `Session phase after loader close\n${jsc.stack}`);
  const v8 = Object.assign(new Error('boom'), { stack: 'Error: boom\n    at f (x.js:1:1)' });
  assert.equal(errorText(v8), v8.stack, 'a stack that already carries the message is stored once');
  assert.equal(errorText(Object.assign(new Error('no stack'), { stack: undefined })), 'no stack');
  assert.equal(errorText('plain'), 'plain');
});

test('local timestamps match device log clocks and summaries expose the last record time', () => {
  // 2026-09-13T07:31:04.759Z, the last 2c record of the September 13 export, on a KST phone.
  assert.equal(localTimestamp(1789284664759, -540), '2026-09-13 16:31:04.759 +09:00');
  assert.equal(localTimestamp(1789284664759, 0), '2026-09-13 07:31:04.759 +00:00');
  assert.equal(localTimestamp(1789284664759, 300), '2026-09-13 02:31:04.759 -05:00');
  assert.equal(localTimestamp(null), null);
  const clock = deviceClock(new Date(1789284664759));
  assert.equal(typeof clock.timezoneOffsetMinutes, 'number');
  assert.equal(clock.exportedAtLocal, localTimestamp(1789284664759, clock.timezoneOffsetMinutes));
  assert.equal(diagnosticSummary({ last: { stage: 'gpu-wait', timestamp: 1789284664759 } }).lastRecordAt, 1789284664759);
  assert.equal(diagnosticSummary({ last: {} }).lastRecordAt, null);
});

test('an ort-wasm-error checkpoint is the run fault and stays visible after the failed summary', async () => {
  const saved = new Map();
  const run = snapshotRun('wasm', { modelExecution: 'streamed' }, async (value, key) => { saved.set(key, structuredClone(value)); return true; });
  await run.checkpoint({ stage: 'session-create', modelExecution: 'streamed' });
  await run.checkpoint({ stage: 'streamed-head-create', componentPhase: 'ort-wasm-start' });
  await run.checkpoint({ stage: 'ort-wasm-error', errorType: 'RuntimeError', message: 'Aborted(NetworkError)', wasmURL: 'x.wasm', status: 200 });
  await run.finish('failed', { error: 'no available backend found', ortWasmInstantiated: false });
  const state = saved.get('run:wasm');
  assert.equal(state.status, 'failed');
  assert.equal(state.fault.stage, 'ort-wasm-error');
  assert.equal(state.fault.status, 200);
  assert.equal(state.milestones['ort-wasm-error'].message, 'Aborted(NetworkError)');
  const summary = diagnosticSummary(state);
  assert.equal(summary.effectiveStatus, 'failed');
  assert.equal(summary.faultStage, 'ort-wasm-error');
  assert.deepEqual(summary.ortPhases.map(phase => phase.stage), ['ort-wasm-error']);
});

test('completed evaluation rows survive the event ring and a cancelled run keeps them', async () => {
  const store = journalStore();
  const run = new RunDiagnostics('rows', { modelExecution: 'streamed' }, undefined, { persistBatch: entries => store.persistBatch(entries) });
  await run.checkpoint({ stage: 'evaluation-rows', rows: 3, totalRows: 100, rowLimit: 3 });
  for (let row = 1; row <= 3; row++) {
    await run.checkpoint({ stage: 'row-start', row });
    for (let step = 0; step < 30; step++) await run.checkpoint({ stage: 'streamed-step-complete', durationMs: 665 });
    await run.checkpoint({ stage: 'row-complete', row, promptLen: 50 + row, nTok: 137,
      inference: { durationMs: 91638 }, result: { ttft: 2186, totalMs: 91638, tps: 1.5, eos: true, rouge: { p: 0.1, r: 0.2, f1: 0.13 } } });
  }
  await run.finish('cancelled', { completedRows: 3, rows: 3, totalRows: 100, rowLimit: 3 });
  const recovered = await readRun('rows', store);
  assert.equal(recovered.records.length, 64, 'the ring no longer holds the first rows');
  assert.deepEqual(recovered.rows.map(row => [row.row, row.promptLen, row.nTok, row.durationMs, row.tps, row.rouge.f1]),
    [[1, 51, 137, 91638, 1.5, 0.13], [2, 52, 137, 91638, 1.5, 0.13], [3, 53, 137, 91638, 1.5, 0.13]]);
  assert.equal(recovered.milestones['evaluation-rows'].rowLimit, 3);
  assert.equal(recovered.summary.rowLimit, 3);
  assert.equal(recovered.persistence.serializedBytes > 0, true);
  assert.equal(diagnosticSummary(recovered).rows, 3);
  assert.equal(diagnosticSummary({ last: {} }).rows, null);
  const snapshot = snapshotRun('rows-snapshot', {}, async () => true);
  await snapshot.checkpoint({ stage: 'row-complete', row: 1, nTok: 2 });
  await snapshot.checkpoint({ stage: 'row-complete', row: 'x', nTok: 2 });
  assert.equal(snapshot.state.rows.length, 1, 'rows without an integer index are not summarized');
  assert.equal(snapshot.state.rows[0].durationMs, null);
});

test('graph planning evidence compares repeats that end before any weight is read', () => {
  const milestones = { 'ort-wasm-complete': { elapsedMs: 900 }, 'ort-session-start': { elapsedMs: 1646, metrics: { wasmHeapBytes: 24248320 } },
    'ort-plan-start': { elapsedMs: 4379, metrics: { wasmHeapBytes: 24248320 }, gpuLedger: { requestedCurrent: 0 } } };
  assert.deepEqual(ortPlanEvidence(milestones), { elapsedMs: 4379, sinceSessionStartMs: 2733, sinceWasmCompleteMs: 3479,
    wasmHeapBytes: 24248320, gpuRequestedCurrent: 0 });
  assert.equal(ortPlanEvidence({ 'ort-session-start': { elapsedMs: 1 } }), null);
  assert.equal(ortPlanEvidence(undefined), null);
  assert.equal(ortPlanEvidence({ 'ort-plan-start': {} }).sinceSessionStartMs, null);
  const summary = diagnosticSummary({ status: 'running', last: { stage: 'ort-plan-start' }, milestones });
  assert.equal(summary.ortPlan.sinceSessionStartMs, 2733);
  assert.equal(diagnosticSummary({ last: {} }).ortPlan, null);
});
