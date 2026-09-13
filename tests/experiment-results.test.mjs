import test from 'node:test';
import assert from 'node:assert/strict';
import { executionSettings, executionEvidence, seriesSummary, isSimpleProbe, applicationOperation, canRepeat, describeDevice, runContext, parseDeviceLogNote } from '../web/sllm/experiments/results.js';

test('the device description carries OS and browser versions from the user agent', () => {
  const iphone = 'Mozilla/5.0 (iPhone; CPU iPhone OS 26_6_1 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) CriOS/153.0.8010.24 Mobile/15E148 Safari/604.1';
  assert.equal(describeDevice(iphone), 'iPhone / iOS 26.6.1 / Chrome 153.0.8010.24');
  const safari = 'Mozilla/5.0 (iPhone; CPU iPhone OS 26_4 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/26.4 Mobile/15E148 Safari/604.1';
  assert.equal(describeDevice(safari), 'iPhone / iOS 26.4 / Safari 26.4');
  assert.equal(describeDevice('Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/141.0.0.0 Safari/537.36'), 'Linux / Chrome 141.0.0.0');
  assert.equal(describeDevice(''), '');
});

test('interrupted loading recovers tokenizer facts from milestones without inventing session success', () => {
  const run = { environment: { tokenizerFormat: 'prepared' }, milestones: {
    'tokenizer-ready': { tokenizerFormat: 'prepared' }, 'session-create': { tokenizerPrepared: true },
  } };
  const evidence = executionEvidence({ kind: 'load', success: false }, run);
  assert.equal(evidence.tokenizerPrepared, true);
  assert.equal(evidence.tokenizerFormat, 'prepared');
  assert.equal(evidence.modelSessionCreated, null);
  assert.equal(evidence.completedScope, null);
  assert.equal(executionSettings('session-only', { idleSeconds: 120 }).idleSeconds, 120);
});

test('combined residency requires real WASM, retained small session, and the same device', () => {
  const result = { kind: 'runtime-resident', success: true, mode: 'asyncify', allBytesUsed: true,
    inferenceVerified: true, ortWasmInstantiated: true, smallSessionRetained: true, sameDevice: true, modelSessionCreated: false };
  assert.equal(executionEvidence(result).completedScope, 'runtime-and-gpu-residency');
  assert.equal(executionEvidence(result).runtimeMode, 'asyncify');
  assert.equal(executionEvidence(result).idleRequestedSeconds, 0);
  assert.equal(executionEvidence(result, { summary: result, cleanup: { success: false } }).completedScope, null);
  for (const key of ['allBytesUsed', 'inferenceVerified', 'ortWasmInstantiated', 'smallSessionRetained', 'sameDevice']) {
    assert.equal(executionEvidence({ ...result, [key]: false }).completedScope, null, key);
  }
  assert.equal(applicationOperation(result.kind), result.kind);
  assert.equal(isSimpleProbe(result.kind), false);
  assert.equal(canRepeat(result.kind), true);
});

test('comparison scopes require both components and cannot claim application readiness', () => {
  const resident = { kind: 'resident-opfs-tokenizer', success: true };
  assert.equal(executionEvidence(resident).completedScope, null);
  assert.equal(executionEvidence({ ...resident, allBytesUsed: true }).completedScope, null);
  const summary = { allBytesUsed: true, tokenizerPrepared: true, modelSessionCreated: false, ortWasmInstantiated: false };
  const evidence = executionEvidence(resident, { summary, environment: { ortJavaScriptLoaded: true, ortJavaScriptMode: 'asyncify' } });
  assert.equal(evidence.completedScope, 'tokenizer-and-gpu-residency');
  assert.equal(evidence.runtimeMode, null);
  assert.equal(evidence.ortWasmInstantiated, false);
  assert.equal(evidence.ortJavaScriptLoaded, true);
  const session = { kind: 'session-only', success: true, modelSessionCreated: true, tokenizerPrepared: false };
  assert.equal(executionEvidence(session).completedScope, 'model-session');
  assert.equal(executionEvidence({ ...session, modelSessionCreated: false }).completedScope, null);
  for (const kind of ['resident-opfs', 'resident-opfs-tokenizer', 'session-only']) {
    assert.equal(isSimpleProbe(kind), false, 'comparison uses the application worker');
    assert.equal(applicationOperation(kind), kind);
    assert.equal(canRepeat(kind), true);
  }
});

test('tokenizer completion requires preparation evidence and cannot claim model loading or inference', () => {
  const result = { kind: 'tokenizer', success: true, mode: 'asyncify' };
  assert.equal(executionEvidence(result).completedScope, null);
  const run = { summary: { tokenizerPrepared: true, modelSessionCreated: false,
    tokenizer: { loadOrder: 'tokenizer-only', tokenizerBuild: { implementation: 'incremental-bpe-v1' } } } };
  const evidence = executionEvidence(result, run);
  assert.equal(evidence.completedScope, 'tokenizer-preparation');
  assert.equal(evidence.loadOrder, 'tokenizer-only');
  assert.equal(evidence.tokenizerBuild.implementation, 'incremental-bpe-v1');
  assert.equal(executionEvidence({ ...result, success: false }, run).completedScope, null);
});

test('old resident exports cannot claim an ORT run, idle acceptance, or three completed attempts', () => {
  const result = { kind: 'resident', mode: 'asyncify', success: true, allBytesUsed: true,
    idleSeconds: 120, repeats: 3, runId: 'resident-one' };
  const evidence = executionEvidence(result);
  assert.equal(evidence.runtimeMode, null);
  assert.equal(evidence.idleRequestedSeconds, 0);
  assert.equal(evidence.idleElapsedMs, 0);
  assert.equal(evidence.idleAcceptanceCompleted, false);
  assert.equal(evidence.completedScope, 'gpu-residency');
  assert.equal(seriesSummary([result])[0].startedRuns, 1);
  assert.equal(seriesSummary([result])[0].requestedRuns, null);
  assert.equal(executionSettings('resident-opfs', { mode: 'jspi', idleSeconds: 120 }).runtimeMode, null);
});

test('runtime acceptance requires measured full idle, with partial observations preserved on cancellation', () => {
  const result = { kind: 'runtime', success: true, inferenceVerified: true, idleSeconds: 120 };
  assert.equal(executionEvidence(result).completedScope, 'small-runtime-inference');
  assert.equal(executionEvidence({ ...result, idleElapsedMs: 1000, idleAcceptanceCompleted: true }).idleAcceptanceCompleted, false);
  assert.equal(executionEvidence({ ...result, idleElapsedMs: 120001, idleAcceptanceCompleted: true }).completedScope, 'small-runtime-and-idle');
  const stopped = { milestones: { 'runtime-idle-start': { idleSeconds: 120 } },
    records: [{ stage: 'runtime-idle', idleElapsedMs: 15000 }] };
  const evidence = executionEvidence({ ...result, success: false }, stopped);
  assert.equal(evidence.idleElapsedMs, 15000);
  assert.equal(evidence.idleAcceptanceCompleted, false);
  assert.equal(evidence.completedScope, null);
});

test('repeat counts track experiments rather than load, probe and evaluation subruns', () => {
  const first = { seriesId: 'series', requestedRuns: 3, kind: 'load', runId: 'a', runIds: ['a'], success: true };
  const interrupted = { ...first, runId: 'b', runIds: ['b'], success: false };
  const active = { ...first, runId: 'c', runIds: ['c', 'probe', 'eval'], success: false };
  const [series] = seriesSummary([first, interrupted], active);
  assert.equal(series.requestedRuns, 3);
  assert.equal(series.startedRuns, 3);
  assert.equal(series.successfulRuns, 1);
  assert.deepEqual(series.runIds, ['a', 'b', 'c']);
});

test('run context counts results since the last interruption and measures the gap from the previous run', () => {
  assert.deepEqual(runContext([], 1000), { resultsSinceInterruption: 0, gapSincePreviousMs: null, previous: null });
  const results = [
    { kind: 'load', modelExecution: 'resident', success: false, interrupted: true, startedAt: 0, durationMs: null },
    { kind: 'session-only', modelExecution: 'streamed', success: true, interrupted: false, startedAt: 100, durationMs: 50,
      comparison: { gpuRequestedCurrent: 7, gpuWeightAllocated: 5 } },
    { kind: 'warm-load', modelExecution: 'resident', success: true, interrupted: false, startedAt: 200, endedAt: 260 },
  ];
  const context = runContext(results, 1000);
  assert.equal(context.resultsSinceInterruption, 2);
  assert.equal(context.gapSincePreviousMs, 740);
  assert.deepEqual(context.previous, { kind: 'warm-load', modelExecution: 'resident', success: true, interrupted: false, endedAt: 260,
    gpuRequestedCurrent: null, gpuWeightAllocated: null });
  const older = runContext(results.slice(0, 2), 1000);
  assert.equal(older.gapSincePreviousMs, 850, 'old rows without endedAt fall back to start plus duration');
  assert.equal(older.previous.gpuRequestedCurrent, 7);
  assert.equal(runContext(results.slice(0, 1), 1000).resultsSinceInterruption, 0);
  assert.equal(runContext(results.slice(0, 1), 1000).gapSincePreviousMs, null, 'an interrupted row has no known end');
});

test('device log notes keep the text and extract the file, reason and footprint when present', () => {
  assert.equal(parseDeviceLogNote(''), null);
  assert.equal(parseDeviceLogNote('   '), null);
  const note = parseDeviceLogNote('JetsamEvent-2026-09-13-163104.ips · per-process-limit · 1,024 MiB');
  assert.deepEqual(note, { note: 'JetsamEvent-2026-09-13-163104.ips · per-process-limit · 1,024 MiB',
    file: 'JetsamEvent-2026-09-13-163104.ips', reason: 'per-process-limit', footprintMiB: 1024 });
  assert.equal(parseDeviceLogNote('com.apple.WebKit.WebContent 1.5 GB vm-pageshortage').footprintMiB, 1430.5);
  assert.equal(parseDeviceLogNote('rpages 65,536 highwater').footprintMiB, 1024);
  assert.equal(parseDeviceLogNote('"rpages": 32768').footprintMiB, 512);
  assert.equal(parseDeviceLogNote('65536 pages').footprintMiB, 1024);
  assert.deepEqual(parseDeviceLogNote('no file found in Analytics Data'), { note: 'no file found in Analytics Data', file: null, reason: null, footprintMiB: null });
});
