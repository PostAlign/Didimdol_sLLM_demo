import test from 'node:test';
import assert from 'node:assert/strict';
import { disposeResources } from '../web/sllm/cleanup.js';

test('session release failure still closes storage, destroys devices and restores tracking', async () => {
  const calls = [];
  const tracker = { devices: [{ queue: { onSubmittedWorkDone: async () => calls.push('queue') }, destroy: () => calls.push('destroy') }],
    flush: async () => calls.push('flush'), restore: () => calls.push('restore'), ledger: {} };
  const result = await disposeResources({ tracker, store: { close: () => calls.push('store') },
    releaseSession: async () => { throw new Error('release failed'); } });
  assert.deepEqual(calls, ['store', 'queue', 'flush', 'destroy', 'restore']);
  assert.equal(result.success, false);
  assert.equal(result.errors[0].message, 'release failed');
});

test('a stalled session has a bounded cleanup deadline and cannot prevent device destruction', async () => {
  let destroyed = false, restored = false;
  const result = await disposeResources({ timeoutMs: 10, releaseSession: () => new Promise(() => {}),
    tracker: { devices: [{ destroy: () => { destroyed = true; } }], restore: () => { restored = true; } } });
  assert.equal(result.timedOut, true);
  assert.equal(destroyed, true);
  assert.equal(restored, true);
});
