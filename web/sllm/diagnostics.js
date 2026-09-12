const DB = 'didimdol-runtime-diagnostics';
const STORE = 'runs';
let connection;
let warned = false;
async function database() {
  if (!connection) connection = new Promise((resolve, reject) => {
    const request = indexedDB.open(DB, 1);
    request.onupgradeneeded = () => request.result.createObjectStore(STORE);
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
  return connection;
}
export async function saveCheckpoint(value, key = 'last') {
  try {
    const db = await database();
    await new Promise((resolve, reject) => {
      const tx = db.transaction(STORE, 'readwrite');
      tx.objectStore(STORE).put({ ...value, timestamp: Date.now() }, key);
      tx.oncomplete = resolve;
      tx.onerror = tx.onabort = () => reject(tx.error);
    });
    return true;
  } catch (error) {
    if (!warned) console.warn('Diagnostic persistence unavailable:', error);
    warned = true;
    return false;
  }
}
export async function readCheckpoint(key = 'last') {
  try {
    const db = await database();
    return await new Promise((resolve, reject) => {
      const request = db.transaction(STORE).objectStore(STORE).get(key);
      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error);
    });
  } catch { return null; }
}

export const newRunId = () => crypto.randomUUID();
export const runKey = runId => `run:${runId}`;
const MILESTONES = new Set(['load-start', 'run-start', 'probe-start', 'graph-verified', 'weights-prepared',
  'session-create', 'session-create-complete', 'session-create-failed', 'tokenizer-load', 'runtime-create',
  'runtime-inference-complete', 'runtime-idle-start', 'runtime-idle-complete', 'ready', 'complete', 'failed', 'cancelled']);
const FAULTS = new Set(['device-lost', 'worker-error', 'gpu-uncaptured-error', 'gpu-error', 'loader-error']);

// Keep binary hashes in diagnostics without copying the entire site inventory at every checkpoint.
export const buildIdentity = build => ({ releaseId: build.releaseId, provenance: build.provenance,
  ortVersion: build.ortVersion, transformersVersion: build.transformersVersion, rangeLoaderVersion: build.rangeLoaderVersion,
  modes: build.modes, builds: build.builds,
  runtimeAssets: Object.fromEntries(Object.entries(build.assets || {}).filter(([name]) => name.startsWith('web/vendor/'))) });

export const errorDetails = error => ({ errorType: error?.constructor?.name || error?.name || 'Error',
  message: String(error?.message ?? error) });

/** Recovery is evidence from a later page, never a rewrite of the interrupted worker's last/fault. */
export async function recordRecovery(run, context = {}) {
  if (!run?.runId || (run.status !== 'running' && !run.fault)) return null;
  const recovery = { observedAt: Date.now(), priorStatus: run.status, lastTimestamp: run.last?.timestamp ?? null,
    classification: run.fault ? 'known-fault' : 'interrupted', cause: run.fault ? 'recorded-fault' : 'unknown', ...context };
  await saveCheckpoint(recovery, `recovery:${run.runId}`);
  return recovery;
}

/** Serialize durable checkpoints; keep a bounded history without tensors or prompts. */
export class RunDiagnostics {
  constructor(runId = newRunId(), environment = {}, persist = saveCheckpoint) {
    this.persist = persist;
    this.state = { schemaVersion: 3, runId, startedAt: Date.now(), status: 'running', environment,
      records: [], last: null, fault: null, firstFault: null, summary: null,
      milestones: {}, files: {}, initializerOrder: [], droppedInitializers: 0, recordCount: 0,
      persistence: { completed: 0, failures: 0, totalMs: 0, peakMs: 0,
        note: 'Timings cover completed writes before this snapshot; they are excluded from GPU operation timings.' },
      memoryNote: 'Allocation counters and WASM capacity are not process RSS or driver memory.' };
    this.pending = Promise.resolve();
  }
  checkpoint = record => {
    const snapshot = structuredClone(record);
    const occurredAt = Date.now();
    this.pending = this.pending.then(async () => {
      const value = { ...snapshot, runId: this.state.runId, timestamp: occurredAt,
        elapsedMs: occurredAt - this.state.startedAt };
      if (FAULTS.has(value.stage) && !this.state.firstFault) this.state.firstFault = value;
      if (value.stage === 'device-lost' || (!this.state.fault && ['worker-error', 'gpu-uncaptured-error'].includes(value.stage))) this.state.fault = value;
      if (!this.state.fault && FAULTS.has(value.stage)) this.state.fault = value;
      if (MILESTONES.has(value.stage)) this.state.milestones[value.stage] = value;
      if (value.stage.startsWith('weight-') && value.location &&
          (this.state.files[value.location] || Object.keys(this.state.files).length < 64)) this.state.files[value.location] = value;
      if (value.stage === 'allocate-initializer' || value.stage === 'resident-allocate') {
        if (this.state.initializerOrder.length < 2048) this.state.initializerOrder.push({ name: value.initializerName,
          length: value.length, placement: value.placement || 'gpu', timestamp: value.timestamp,
          gpuWeightAllocated: value.metrics?.gpuWeightAllocated ?? value.gpuWeightAllocated });
        else this.state.droppedInitializers++;
      }
      this.state.recordCount++;
      this.state.last = value;
      this.state.records.push(value);
      if (this.state.records.length > 64) this.state.records.shift();
      this.state.updatedAt = value.timestamp;
      const start = performance.now();
      let saved;
      try { saved = await this.persist(this.state, runKey(this.state.runId)); }
      catch { saved = false; }
      const ms = performance.now() - start;
      this.state.persistence.completed++;
      if (saved === false) this.state.persistence.failures++;
      this.state.persistence.totalMs += ms;
      this.state.persistence.peakMs = Math.max(this.state.persistence.peakMs, ms);
      return saved;
    });
    return this.pending;
  };
  async finish(status, summary = {}) {
    await this.pending;
    this.state.status = this.state.fault?.stage === 'device-lost' ? 'device-lost'
      : this.state.fault && ['ready', 'complete'].includes(status) ? 'failed' : status;
    this.state.summary = summary;
    return this.checkpoint({ ...summary, stage: this.state.status });
  }
}

export async function readRun(runId) {
  const value = runId ? await readCheckpoint(runKey(runId)) : null;
  if (value?.runId !== runId) return null;
  // Schema 2 remains readable; missing counters are unknown, never fabricated zeros.
  const recovery = await readCheckpoint(`recovery:${runId}`);
  return { ...value, ...(recovery ? { recovery } : {}) };
}
