import test from 'node:test';
import assert from 'node:assert/strict';
import { observeSession, observeInference, inferenceDigest } from '../web/sllm/session-observation.js';
import { executionEvidence } from '../web/sllm/experiments/results.js';
import { recoveryEvidence, diagnosticSummary } from '../web/sllm/diagnostics.js';

// Drive the sampler's wait from the test: each tick advances the clock by the requested interval.
function manualClock() {
  let time = 0, release = null;
  const wait = (ms, signal) => new Promise((resolve, reject) => {
    release = () => { release = null; time += ms; resolve(); };
    signal.addEventListener('abort', () => reject(signal.reason), { once: true });
  });
  const settle = () => new Promise(resolve => setImmediate(resolve));
  return { clock: () => time, wait, settle, tick: async () => { await settle(); release?.(); await settle(); } };
}

test('inference sampling records ledger changes, heartbeats and the first token without waiting on the GPU', async () => {
  const { clock, wait, tick, settle } = manualClock();
  const ledger = { requestedCurrent: 1000, observedPeak: 1000, liveBufferCount: 3, bufferCount: 3, deviceLost: null, lastError: null,
    programs: { shaderModules: 0, computePipelines: 0, asyncPipelinesStarted: 0, asyncPipelinesCompleted: 0, asyncPipelinesFailed: 0 } };
  const records = [];
  const observation = observeInference({ clock, wait, context: { phase: 'warmup' }, checkpoint: async record => records.push(record),
    sample: () => ({ gpuLedger: structuredClone(ledger), metrics: { wasmHeapBytes: 24 } }) });
  await tick();
  assert.equal(records.length, 1);
  assert.equal(records[0].stage, 'inference-sample');
  assert.equal(records[0].reason, 'change');
  assert.equal(records[0].phase, 'warmup');
  assert.equal(records[0].inferenceElapsedMs, 500);
  assert.equal(records[0].sampleIndex, 1);
  await tick();
  assert.equal(records.length, 1, 'an unchanged ledger is not rewritten every interval');
  ledger.requestedCurrent = 900; ledger.liveBufferCount = 5; ledger.bufferCount = 7;
  await tick();
  assert.equal(records.length, 1, 'per-step buffer churn without peak growth is not a change');
  ledger.programs.computePipelines = 3; ledger.requestedCurrent = 1100; ledger.observedPeak = 1100;
  await tick();
  assert.equal(records.length, 2);
  assert.equal(records[1].gpuLedger.programs.computePipelines, 3);
  for (let i = 0; i < 10; i++) await tick();
  assert.equal(records.length, 3, 'a heartbeat keeps a long unchanged inference visible');
  assert.equal(records[2].reason, 'heartbeat');
  assert.equal(records[2].inferenceElapsedMs, 7000);
  await observation.mark('first-token');
  assert.equal(records.at(-1).reason, 'first-token');
  const summary = await observation.stop();
  await settle();
  assert.equal(summary.samples, 4);
  assert.equal(summary.changes, 2);
  assert.equal(summary.marks, 1);
  assert.equal(summary.phase, 'warmup');
  assert.equal(summary.durationMs, 7000);
  assert.equal(summary.lastSample.gpuRequestedCurrent, 1100);
  assert.equal(summary.lastSample.programs.computePipelines, 3);
  assert.equal(summary.lastSample.wasmHeapBytes, 24);
  assert.deepEqual(inferenceDigest(records[1]).programs, ledger.programs);
  assert.throws(() => observeInference({ intervalMs: 0, checkpoint: async () => {}, sample: () => ({}) }));
  assert.throws(() => observeInference({ intervalMs: 500, heartbeatMs: 100, checkpoint: async () => {}, sample: () => ({}) }));
});

test('stopping the sampler does not wait out the interval, and sampler failures are recorded rather than thrown', async () => {
  const records = [];
  const observation = observeInference({ intervalMs: 500, checkpoint: async record => records.push(record), sample: () => { throw new Error('ledger gone'); } });
  const started = Date.now();
  await observation.mark('first-token');
  assert.equal(records[0].sampleError.message, 'ledger gone');
  const summary = await observation.stop();
  assert.ok(Date.now() - started < 400, 'stop must abort the pending wait');
  assert.equal(summary.samples, 1);
  assert.equal(summary.lastSample.gpuRequestedCurrent, null);
});

test('an interruption inside the first inference is classified and summarized from the last sample', () => {
  const run = { runId: 'run', startedAt: 10, status: 'running', milestones: { 'run-start': {} }, last: { stage: 'warmup-start', timestamp: 20 } };
  const recovery = recoveryEvidence(run);
  assert.equal(recovery.interruptedPhase, 'inference');
  assert.deepEqual(recovery.inference, { phase: 'warmup', row: null, reason: null, inferenceElapsedMs: 0 });
  const sample = { stage: 'inference-sample', phase: 'warmup', reason: 'change', inferenceElapsedMs: 1500, sampleIndex: 2, timestamp: 22,
    gpuLedger: { requestedCurrent: 1100, observedPeak: 1100, liveBufferCount: 9, bufferCount: 9, tracking: { status: 'complete' },
      programs: { shaderModules: 4, computePipelines: 3 } }, metrics: { wasmHeapBytes: 64 } };
  const sampled = { ...run, last: sample, records: [run.last, sample] };
  const summary = diagnosticSummary({ ...sampled, recovery: recoveryEvidence(sampled) });
  assert.equal(summary.effectiveStatus, 'interrupted');
  assert.equal(summary.interruptedPhase, 'inference');
  assert.equal(summary.interruptedInference.inferenceElapsedMs, 1500);
  assert.equal(summary.inference.gpuRequestedCurrent, 1100);
  assert.equal(summary.inference.programs.computePipelines, 3);
  assert.equal(summary.inference.wasmHeapBytes, 64);
  assert.equal(summary.gpuRequestedCurrent, 1100, 'the table reads the sampled ledger');
  const finished = { ...run, status: 'complete', last: { stage: 'row-complete', row: 1, inference: { phase: 'evaluation', row: 1, samples: 3 } } };
  assert.equal(diagnosticSummary(finished).lastInference.samples, 3);
  assert.equal(diagnosticSummary(finished).inference, null);
  assert.equal(recoveryEvidence({ ...run, last: { stage: 'row-complete', row: 1, timestamp: 20 } }).interruptedPhase, 'after-session-create');
  assert.equal(recoveryEvidence({ ...run, last: { stage: 'row-start', row: 7, timestamp: 20 } }).inference.row, 7);
  assert.equal(recoveryEvidence({ ...run, milestones: {}, last: { stage: 'gpu-wait', timestamp: 20 } }).interruptedPhase, 'execution');
  assert.equal(recoveryEvidence({ ...run, status: 'complete' }).inference, null);
});

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
