import test from 'node:test';
import assert from 'node:assert/strict';
import { loadExperimentState, saveExperimentState, DURABLE_KEY, TAB_KEY, MAX_RESULTS } from '../web/sllm/experiments/state-store.js';

const store = () => { const map = new Map(); return { getItem: key => map.has(key) ? map.get(key) : null, setItem: (key, value) => map.set(key, String(value)) }; };

test('results persist across tabs while live progress stays tab-scoped, migrating the old single-tab state once', () => {
  const local = store(), session = store();
  session.setItem(TAB_KEY, JSON.stringify({ results: [{ runId: 'a' }], device: 'iPhone 14 Pro Max', mode: 'jspi', active: { runId: 'b' } }));
  const state = loadExperimentState({ local, session });
  assert.deepEqual(state.results, [{ runId: 'a' }]);
  assert.equal(state.device, 'iPhone 14 Pro Max');
  assert.equal(state.mode, 'jspi');
  assert.equal(state.active.runId, 'b');
  assert.equal(state.continue, null);
  assert.deepEqual(saveExperimentState(state, { local, session }), []);
  assert.deepEqual(JSON.parse(session.getItem(TAB_KEY)), { active: { runId: 'b' }, continue: null });
  assert.equal(JSON.parse(local.getItem(DURABLE_KEY)).results.length, 1);
  assert.equal(JSON.parse(local.getItem(DURABLE_KEY)).active, undefined);
  const otherTab = loadExperimentState({ local, session: store() });
  assert.deepEqual(otherTab.results, [{ runId: 'a' }]);
  assert.equal(otherTab.device, 'iPhone 14 Pro Max');
  assert.equal(otherTab.active, null, 'another tab must never recover this tab\'s running experiment');
  session.setItem(TAB_KEY, JSON.stringify({ results: [{ runId: 'stale' }], active: null }));
  assert.deepEqual(loadExperimentState({ local, session }).results, [{ runId: 'a' }], 'durable results win over a stale legacy tab');
});

test('missing, corrupt or unavailable storage yields defaults and falls back to the tab', () => {
  assert.equal(loadExperimentState({}).mode, 'asyncify');
  assert.deepEqual(loadExperimentState({}).results, []);
  assert.deepEqual(loadExperimentState({ local: { getItem: () => '{bad' }, session: { getItem: () => '[]' } }).results, []);
  const session = store(), local = { getItem: () => null, setItem: () => { throw new Error('quota'); } };
  const state = { results: [{ runId: 'a' }], device: 'x', mode: 'asyncify', active: { runId: 'b' }, continue: null };
  assert.deepEqual(saveExperimentState(state, { local, session }), ['local']);
  const reloaded = loadExperimentState({ local, session });
  assert.deepEqual(reloaded.results, [{ runId: 'a' }]);
  assert.equal(reloaded.active.runId, 'b');
  const durable = store();
  durable.setItem(DURABLE_KEY, JSON.stringify({ results: Array.from({ length: MAX_RESULTS + 10 }, (_, i) => ({ runId: String(i) })) }));
  const capped = loadExperimentState({ local: durable, session: store() });
  assert.equal(capped.results.length, MAX_RESULTS);
  assert.equal(capped.results.at(-1).runId, String(MAX_RESULTS + 9), 'the newest results are kept');
});
