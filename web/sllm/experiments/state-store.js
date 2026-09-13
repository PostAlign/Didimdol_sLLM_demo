// Experiment results outlive the tab: a run interrupted by the OS is usually
// followed by the tester closing the browser, which used to discard the results
// list before it could be exported. Live progress (`active`, `continue`) stays
// tab-scoped so a second tab can never recover another tab's running experiment.
export const DURABLE_KEY = 'didimdol.device-experiments.results.v3';
export const TAB_KEY = 'didimdol.device-experiments.v2';
export const MAX_RESULTS = 30;

const parse = (storage, key) => {
  try { const value = JSON.parse(storage?.getItem(key)); return value && typeof value === 'object' ? value : null; }
  catch { return null; }
};

export function loadExperimentState({ local, session } = {}) {
  const tab = parse(session, TAB_KEY) || {};
  const { active = null, continue: next = null, ...legacy } = tab;
  // An older single-tab state carried its results; adopt them once when no durable results exist.
  const base = parse(local, DURABLE_KEY) || legacy;
  return { device: '', mode: 'asyncify', ...base,
    results: Array.isArray(base.results) ? base.results.slice(-MAX_RESULTS) : [], active, continue: next };
}

/** Returns the storages that refused the write; results fall back to the tab when durable storage is unavailable. */
export function saveExperimentState(state, { local, session } = {}) {
  const { active = null, continue: next = null, ...durable } = state;
  const failures = [];
  try { local.setItem(DURABLE_KEY, JSON.stringify(durable)); } catch { failures.push('local'); }
  const tab = failures.length ? { ...durable, active, continue: next } : { active, continue: next };
  try { session.setItem(TAB_KEY, JSON.stringify(tab)); } catch { failures.push('session'); }
  return failures;
}
