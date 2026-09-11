import test from 'node:test';
import assert from 'node:assert/strict';
import { RunDiagnostics } from '../web/sllm/diagnostics.js';

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
