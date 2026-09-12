import test from 'node:test';
import assert from 'node:assert/strict';
import { executionSettings, executionEvidence, seriesSummary, isSimpleProbe, applicationOperation, canRepeat } from '../web/sllm/experiments/results.js';

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
