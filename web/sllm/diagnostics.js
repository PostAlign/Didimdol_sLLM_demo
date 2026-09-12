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
  'resident-start',
  'session-create', 'session-create-complete', 'session-create-failed', 'tokenizer-load', 'runtime-create',
  'runtime-inference-complete', 'runtime-idle-start', 'runtime-idle-complete', 'ready', 'complete', 'failed', 'cancelled']);
const FAULTS = new Set(['device-lost', 'worker-error', 'gpu-uncaptured-error', 'gpu-error', 'loader-error',
  'tokenizer-error', 'template-error', 'evaluation-data-error']);
const isPreparation = stage => /^(tokenizer|template|evaluation-data)-/.test(stage);

// Keep binary hashes in diagnostics without copying the entire site inventory at every checkpoint.
export const buildIdentity = build => ({ releaseId: build.releaseId, provenance: build.provenance,
  tokenizer: build.tokenizer ?? null,
  ortVersion: build.ortVersion, transformersVersion: build.transformersVersion, rangeLoaderVersion: build.rangeLoaderVersion,
  modes: build.modes, builds: build.builds,
  runtimeAssets: Object.fromEntries(Object.entries(build.assets || {}).filter(([name]) => name.startsWith('web/vendor/'))) });

export const errorDetails = error => ({ errorType: error?.constructor?.name || error?.name || 'Error',
  message: String(error?.message ?? error) });

export const gpuOperationContext = record => ({ observedDuring: record?.stage,
  ...Object.fromEntries(['initializerName', 'shape', 'index', 'offset', 'length', 'location',
    'fileOffset', 'destinationOffset', 'chunkBytes'].filter(key => record?.[key] !== undefined).map(key => [key, record[key]])) });

export function recoveryEvidence(run, context = {}, observedAt = Date.now()) {
  const { lifecycle = [], ...details } = context;
  const startedAt = run.startedAt ?? run.milestones?.['load-start']?.timestamp;
  return { ...details, observedAt, priorStatus: run.status, lastTimestamp: run.last?.timestamp ?? null,
    classification: run.fault ? 'known-fault' : 'interrupted', cause: run.fault ? 'recorded-fault' : 'unknown',
    // Retain historical hints, but only associate events inside this run's time
    // interval and, for new records, with its explicit UUID.
    lifecycleHistory: lifecycle,
    lifecycle: lifecycle.filter(event => Number.isFinite(startedAt) && event.timestamp >= startedAt &&
      event.timestamp <= observedAt && (!event.runId || event.runId === run.runId)) };
}

export function diagnosticSummary(run, sessionFallback = null) {
  if (!run) return null;
  const last = run.last || {}, fault = run.fault;
  const session = run.summary?.sessionMetrics || sessionFallback || run.summary;
  const progress = [last, ...(run.records || []).slice().reverse()].find(r => r.metrics || r.gpuWeightAllocated != null) || last;
  const metrics = progress.metrics || session?.metrics || {};
  const ledger = last.gpuLedger || progress.gpuLedger || run.summary?.gpuLedger || session?.gpuLedger;
  const allocated = metrics.gpuWeightAllocated ?? progress.gpuWeightAllocated ?? run.summary?.gpuWeightAllocated ?? null;
  let trackingStatus = ledger?.tracking?.status ?? 'unknown';
  // Old schema-3 exports can contain apparently valid zero ledgers. Do not
  // upgrade those to complete, or invent measurements for schema-2 exports.
  if (!ledger?.tracking && allocated > 0 && ledger && (ledger.bufferCount === 0 || ledger.requestedCurrent === 0)) trackingStatus = 'partial';
  const interrupted = run.status === 'running' && !!run.recovery;
  const storage = last.storage ?? session?.storage
    ?? [...(run.records || [])].reverse().find(record => record.storage)?.storage
    ?? run.milestones?.['weights-prepared']?.storage ?? null;
  return { effectiveStatus: fault ? 'failed' : interrupted ? 'interrupted' : run.status,
    file: fault?.file ?? last.file ?? null, observedDuring: fault?.observedDuring ?? last.observedDuring ?? null,
    loadOrder: run.environment?.loadOrder ?? session?.loadOrder ?? null,
    tokenizer: run.summary?.tokenizer ?? session?.tokenizer ?? run.milestones?.['tokenizer-ready'] ?? null,
    jsMemory: last.jsMemory ?? null,
    stage: last.stage ?? null, initializerName: fault?.initializerName ?? progress.initializerName ?? session?.lastInitializer?.initializerName ?? null,
    destinationOffset: fault?.destinationOffset ?? progress.destinationOffset ?? null, faultStage: fault?.stage ?? null,
    loadedInitializerCount: metrics.loadedInitializerCount ?? progress.loadedInitializerCount ?? run.summary?.loadedInitializerCount ?? null,
    expectedInitializerCount: run.milestones?.['graph-verified']?.expectedInitializerCount ??
      run.milestones?.['resident-start']?.expectedInitializerCount ?? run.milestones?.['runtime-create']?.expectedInitializerCount ??
      run.summary?.expectedInitializerCount ?? null,
    gpuWeightAllocated: allocated,
    gpuWriteReturnedBytes: metrics.gpuWriteReturnedBytes ?? progress.gpuWriteReturnedBytes ?? run.summary?.gpuWriteReturnedBytes ?? null,
    gpuQueueCompletedBytes: metrics.gpuQueueCompletedBytes ?? metrics.gpuWeightUploaded ?? progress.gpuWeightUploaded ?? run.summary?.gpuWeightUploaded ?? null,
    gpuValidatedInitializerCount: metrics.gpuValidatedInitializerCount ?? run.summary?.gpuValidatedInitializerCount ?? null,
    trackingStatus, storage,
    gpuRequestedCurrent: trackingStatus === 'complete' ? ledger.requestedCurrent ?? null : null,
    wasmHeapBytes: metrics.wasmHeapBytes ?? null,
    timings: run.summary?.sessionMetrics?.timings ?? run.summary?.timings
      ?? session?.timings ?? run.milestones?.['session-create']?.timings ?? null,
    releaseId: run.environment?.build?.releaseId ?? null };
}

export const trackingLabel = status => ({ complete: '정상', partial: '부분 관측', unbound: '연결 미확인', unknown: '미기록' }[status] || '미기록');

/** Recovery is evidence from a later page, never a rewrite of the interrupted worker's last/fault. */
export async function recordRecovery(run, context = {}) {
  if (!run?.runId || (run.status !== 'running' && !run.fault)) return null;
  const recovery = recoveryEvidence(run, context);
  await saveCheckpoint(recovery, `recovery:${run.runId}`);
  return recovery;
}

/** Serialize durable checkpoints; keep a bounded history without tensors or prompts. */
export class RunDiagnostics {
  constructor(runId = newRunId(), environment = {}, persist = saveCheckpoint) {
    this.persist = persist;
    this.state = { schemaVersion: 3, runId, startedAt: Date.now(), status: 'running', environment,
      records: [], last: null, fault: null, firstFault: null, summary: null,
      milestones: {}, files: {}, preparation: {}, initializerOrder: [], droppedInitializers: 0, recordCount: 0,
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
      if (isPreparation(value.stage)) {
        this.state.milestones[value.stage] = value;
        const file = value.file || 'shared';
        this.state.preparation[file] ||= {};
        this.state.preparation[file][value.stage] = value;
      }
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
