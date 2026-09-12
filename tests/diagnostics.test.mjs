import test from 'node:test';
import assert from 'node:assert/strict';
import { RunDiagnostics } from '../web/sllm/diagnostics.js';

test('evicting recent events preserves preparation, allocation order and the original fault', async () => {
  let saved;
  const run = new RunDiagnostics('retained', {}, async value => { saved = structuredClone(value); return true; });
  await run.checkpoint({ stage: 'graph-verified', graphSha256: 'graph', expectedInitializerCount: 251 });
  await run.checkpoint({ stage: 'weight-ready', location: 'weights', source: 'opfs-cache' });
  await run.checkpoint({ stage: 'session-create', storage: { cacheHits: 9 } });
  for (let i = 0; i < 70; i++) await run.checkpoint({ stage: 'allocate-initializer', initializerName: `w${i}`, length: 40 });
  await run.checkpoint({ stage: 'gpu-error', errorType: 'GPUOutOfMemoryError', message: 'allocation failed' });
  await run.checkpoint({ stage: 'device-lost', message: 'later device loss' });
  await run.finish('failed', { error: 'cleanup error' });
  assert.equal(saved.schemaVersion, 3);
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

test('failed persistence does not break later checkpoints, and queued records are snapshots', async () => {
  let saved, calls = 0;
  const run = new RunDiagnostics('persistence', {}, async value => {
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

test('run journals serialize persistence, isolate IDs and retain a GPU fault after cleanup', async () => {
  const saved = new Map();
  let pending = false;
  const persist = async (state, key) => {
    assert.equal(pending, false); pending = true;
    await Promise.resolve(); saved.set(key, structuredClone(state)); pending = false;
  };
  const run = new RunDiagnostics('one', { browser: 'test' }, persist);
  await Promise.all(Array.from({ length: 70 }, (_, i) => run.checkpoint({ stage: 'range-read', destinationOffset: i })));
  await run.checkpoint({ stage: 'device-lost', metrics: { deviceLost: { reason: 'unknown' } } });
  await run.checkpoint({ stage: 'gpu-uncaptured-error', message: 'later cleanup error' });
  await run.finish('failed', { stage: 'session-create-failed' });
  const record = saved.get('run:one');
  assert.equal(record.records.length, 64);
  assert.equal(record.fault.stage, 'device-lost');
  assert.equal(record.status, 'device-lost');
  assert.equal(record.last.stage, 'device-lost');
  await new RunDiagnostics('two', {}, persist).checkpoint({ stage: 'load-start' });
  assert.equal(saved.get('run:one').runId, 'one');
  assert.equal(saved.get('run:two').runId, 'two');
});
