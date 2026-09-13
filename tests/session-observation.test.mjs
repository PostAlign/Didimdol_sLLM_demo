import test from 'node:test';
import assert from 'node:assert/strict';
import { observeSession } from '../web/sllm/session-observation.js';
import { executionEvidence } from '../web/sllm/experiments/results.js';

test('session observation requires elapsed time, checks the live owner, and records only coarse progress', async () => {
  let time = 0, samples = 0;
  const records = [], summary = await observeSession({ seconds: 120, clock: () => time,
    wait: async ms => { assert.ok(ms <= 5000); time += ms; },
    sample: () => { samples++; return { modelSessionCreated: true }; }, checkpoint: async record => records.push(record) });
  assert.equal(summary.idleElapsedMs, 120000);
  assert.equal(summary.idleAcceptanceCompleted, true);
  assert.equal(samples, 26);
  assert.equal(records.at(-1).stage, 'session-idle-complete');
  const result = { kind: 'session-only', success: true, modelSessionCreated: true, tokenizerPrepared: false, ...summary };
  assert.equal(executionEvidence(result).completedScope, 'model-session-and-idle');
  assert.equal(executionEvidence({ ...result, idleElapsedMs: 10 }).completedScope, 'model-session');
});

test('observation cancellation preserves partial progress and never claims completion', async () => {
  const controller = new AbortController(), records = [];
  let time = 0;
  await assert.rejects(observeSession({ seconds: 120, signal: controller.signal, clock: () => time,
    wait: async ms => { time += ms; if (time === 10000) controller.abort(); }, checkpoint: async record => records.push(record) }), { name: 'AbortError' });
  assert.equal(records.at(-1).idleElapsedMs, 5000);
  assert.equal(records.some(record => record.stage === 'session-idle-complete'), false);
  const run = { milestones: { 'session-idle-start': records[0], 'session-create-complete': { modelSessionCreated: true } }, records };
  const evidence = executionEvidence({ kind: 'session-only', success: false }, run);
  assert.equal(evidence.modelSessionCreated, true);
  assert.equal(evidence.idleElapsedMs, 5000);
  assert.equal(evidence.idleAcceptanceCompleted, false);
  assert.equal(evidence.completedScope, null);
  const realWait = new AbortController();
  const pending = observeSession({ seconds: 120, signal: realWait.signal, checkpoint: async () => {} });
  setTimeout(() => realWait.abort(), 5);
  await assert.rejects(pending, { name: 'AbortError' });
});
