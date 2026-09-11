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

/** Serialize durable checkpoints; keep a bounded history without tensors or prompts. */
export class RunDiagnostics {
  constructor(runId = newRunId(), environment = {}, persist = saveCheckpoint) {
    this.persist = persist;
    this.state = { schemaVersion: 2, runId, startedAt: Date.now(), status: 'running', environment,
      records: [], last: null, fault: null, summary: null,
      memoryNote: 'Allocation counters and WASM capacity are not process RSS or driver memory.' };
    this.pending = Promise.resolve();
  }
  checkpoint = record => {
    this.pending = this.pending.then(async () => {
      const value = { ...record, runId: this.state.runId, timestamp: Date.now() };
      if (value.stage === 'device-lost' || (!this.state.fault && ['worker-error', 'gpu-uncaptured-error'].includes(value.stage))) this.state.fault = value;
      this.state.last = value;
      this.state.records.push(value);
      if (this.state.records.length > 64) this.state.records.shift();
      this.state.updatedAt = value.timestamp;
      return this.persist(this.state, runKey(this.state.runId));
    });
    return this.pending;
  };
  async finish(status, summary = {}) {
    await this.pending;
    this.state.status = this.state.fault?.stage === 'device-lost' ? 'device-lost' : status;
    this.state.summary = summary;
    return this.checkpoint({ ...summary, stage: this.state.status });
  }
}

export async function readRun(runId) {
  const value = runId ? await readCheckpoint(runKey(runId)) : null;
  return value?.runId === runId ? value : null;
}
