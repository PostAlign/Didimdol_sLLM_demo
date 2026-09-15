import { inferenceDigest } from './session-observation.js';

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

// One transaction commits the latest event and its sequence together. A page
// interruption cannot expose a new head with an older event in the ring.
export async function saveJournalEntries(entries) {
  try {
    const db = await database();
    await new Promise((resolve, reject) => {
      const tx = db.transaction(STORE, 'readwrite');
      tx.oncomplete = resolve;
      tx.onerror = tx.onabort = () => reject(tx.error);
      try { for (const [key, value] of entries) tx.objectStore(STORE).put(value, key); }
      catch (error) { tx.abort(); reject(error); }
    });
    return true;
  } catch {
    if (!warned) console.warn('Diagnostic persistence unavailable: journal transaction failed');
    warned = true;
    return false;
  }
}
const journalPrefix = runId => `journal:${runId}:`;
async function readJournalEntries(runId) {
  try {
    const db = await database(), prefix = journalPrefix(runId);
    return await new Promise((resolve, reject) => {
      const request = db.transaction(STORE).objectStore(STORE).getAll(IDBKeyRange.bound(prefix, `${prefix}\uffff`));
      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error);
    });
  } catch { return null; }
}

export const newRunId = () => crypto.randomUUID();
export const runKey = runId => `run:${runId}`;
const MILESTONES = new Set(['load-start', 'run-start', 'probe-start', 'graph-verified', 'weights-prepared',
  'resident-start', 'evaluation-rows',
  'session-create', 'session-create-complete', 'session-create-failed', 'tokenizer-load', 'runtime-create',
  'runtime-inference-complete', 'runtime-idle-start', 'runtime-idle-complete', 'session-idle-start', 'session-idle-complete',
  'auxiliary-session-start', 'auxiliary-session-complete',
  'ready', 'complete', 'failed', 'cancelled']);
const FAULTS = new Set(['device-lost', 'worker-error', 'gpu-uncaptured-error', 'gpu-error', 'loader-error', 'streamed-error',
  'ort-wasm-error', 'tokenizer-error', 'template-error', 'evaluation-data-error']);
const isPreparation = stage => /^(tokenizer|template|evaluation-data)-/.test(stage);
// Pre-call position records: written before a guarded read, write or wait so an
// interruption leaves its position, then superseded by the next record. They
// update the durable head only and stay out of the 64-slot event ring, which
// keeps the allocation/completion pace of every initializer instead of the
// three or four pieces of the last few. A full load on the September 14 phone
// wrote 3,255 entries (6.3 MB) with them in the ring.
const POSITION_ONLY = new Set(['upload-initializer', 'range-read', 'gpu-write', 'gpu-wait',
  'resident-read', 'resident-write', 'resident-wait']);
const PERSISTENCE_NOTE = 'Timings cover completed writes before this snapshot; they are excluded from GPU operation timings. '
  + 'serializedBytes is the JSON length of committed compact entries, not the bytes IndexedDB wrote to disk.';

// Keep binary hashes in diagnostics without copying the entire site inventory at every checkpoint.
export const buildIdentity = build => ({ releaseId: build.releaseId, provenance: build.provenance,
  tokenizer: build.tokenizer ?? null,
  ortVersion: build.ortVersion, transformersVersion: build.transformersVersion, rangeLoaderVersion: build.rangeLoaderVersion,
  modes: build.modes, builds: build.builds,
  runtimeAssets: Object.fromEntries(Object.entries(build.assets || {}).filter(([name]) => name.startsWith('web/vendor/'))) });

export const errorDetails = error => ({ errorType: error?.constructor?.name || error?.name || 'Error',
  message: String(error?.message ?? error) });

/** Message plus stack for exports. JavaScriptCore stacks omit the message, so it is never dropped here. */
export const errorText = error => {
  const { message } = errorDetails(error);
  const stack = error?.stack == null ? '' : String(error.stack);
  return !stack ? message : stack.includes(message) ? stack : `${message}\n${stack}`;
};

export const gpuOperationContext = record => ({ observedDuring: record?.stage,
  ...Object.fromEntries(['initializerName', 'shape', 'index', 'offset', 'length', 'location',
    'fileOffset', 'destinationOffset', 'chunkBytes'].filter(key => record?.[key] !== undefined).map(key => [key, record[key]])) });

// Records that place an interruption inside a generate() call. `-complete`
// records are deliberately absent: after them the worker is between inferences.
const INFERENCE_START = { 'warmup-start': 'warmup', 'row-start': 'evaluation', 'probe-inference-start': 'probe' };
export function inferenceContext(record) {
  if (!record?.stage) return null;
  if (record.stage === 'inference-sample') return { phase: record.phase ?? null, row: record.row ?? null,
    reason: record.reason ?? null, inferenceElapsedMs: record.inferenceElapsedMs ?? null };
  const phase = INFERENCE_START[record.stage];
  return phase ? { phase, row: record.row ?? null, reason: null, inferenceElapsedMs: 0 } : null;
}

/** Local wall-clock text for an epoch-millisecond timestamp: `YYYY-MM-DD HH:mm:ss.SSS +HH:MM`. */
export function localTimestamp(ms, offsetMinutes) {
  if (!Number.isFinite(ms)) return null;
  const offset = Number.isFinite(offsetMinutes) ? offsetMinutes : new Date(ms).getTimezoneOffset();
  const shifted = new Date(ms - offset * 60000).toISOString();
  const sign = offset <= 0 ? '+' : '-', abs = Math.abs(offset);
  return `${shifted.slice(0, 10)} ${shifted.slice(11, 23)} ${sign}${String(Math.floor(abs / 60)).padStart(2, '0')}:${String(abs % 60).padStart(2, '0')}`;
}

/** Device clock facts for matching exports against JetsamEvent/crash files, which are named and stamped in device local time. */
export function deviceClock(now = new Date()) {
  let timeZone = null;
  try { timeZone = Intl.DateTimeFormat().resolvedOptions().timeZone ?? null; } catch { /* unavailable */ }
  return { timeZone, timezoneOffsetMinutes: now.getTimezoneOffset(),
    exportedAtLocal: localTimestamp(now.getTime(), now.getTimezoneOffset()),
    note: 'Run timestamps are epoch milliseconds (UTC). JetsamEvent-*.ips names and bodies use the device local clock; compare them with the *Local fields.' };
}

/** Navigation type of the page observing a recovery (reload, navigate, back_forward) when the browser exposes it. */
export function pageNavigationType() {
  try { return globalThis.performance?.getEntriesByType?.('navigation')?.[0]?.type ?? null; } catch { return null; }
}

export function recoveryEvidence(run, context = {}, observedAt = Date.now()) {
  const { lifecycle = [], navigationType = null, ...details } = context;
  const startedAt = run.startedAt ?? run.milestones?.['load-start']?.timestamp;
  const cleanupPending = run.cleanup?.stage === 'cleanup-start' || (context.phase === 'cleanup' && !run.cleanup);
  // Evaluation/probe runs are separate journals that only start with a created session.
  const sessionCreated = !!run.milestones?.['session-create-complete'] || run.summary?.modelSessionCreated === true ||
    !!run.milestones?.['run-start'] || !!run.milestones?.['probe-start'];
  const inference = run.status === 'running' && !cleanupPending ? inferenceContext(run.last) : null;
  const completed = ['complete', 'ready'].includes(run.status);
  const terminalReentry = run.status === 'cancelled' ? 'cancelled-run-reentry' : run.status === 'failed' ? 'failed-run-reentry' : null;
  const classification = run.fault ? 'known-fault' : cleanupPending ? 'cleanup-reentry' : completed ? 'completed-run-reentry' : terminalReentry || 'interrupted';
  // Retain historical hints, but only associate events inside this run's time
  // interval and, for new records, with its explicit UUID.
  const events = lifecycle.filter(event => Number.isFinite(startedAt) && event.timestamp >= startedAt &&
    event.timestamp <= observedAt && (!event.runId || event.runId === run.runId));
  const lastTimestamp = run.last?.timestamp ?? null;
  // A normal navigation or reload leaves pagehide/hidden events before the page goes away. Their absence, with the
  // recovering page appearing shortly after the last checkpoint, matches a process termination but does not prove an
  // OS memory kill: only device logs can. The original cause therefore stays 'unknown'.
  const unloadObserved = events.some(event => event.event === 'pagehide' || (event.event === 'visibilitychange' && event.visibility === 'hidden'));
  // A `pagehide` with `persisted: true` means the page was kept (back/forward cache or, on iOS, the app leaving the
  // foreground) rather than torn down by a navigation or reload; the re-entry then followed a background discard or
  // a later navigation. It distinguishes the September 14 evening streamed probe from the silent resident re-entries.
  const unloadKind = !unloadObserved ? null
    : events.some(event => event.event === 'pagehide' && event.persisted === true) ? 'persisted-hide' : 'unload';
  const reentryGapMs = Number.isFinite(lastTimestamp) && Number.isFinite(observedAt) ? Math.max(0, observedAt - lastTimestamp) : null;
  const unloadEvidence = run.fault ? 'recorded-fault' : !['interrupted', 'cleanup-reentry'].includes(classification) ? null
    : unloadObserved ? 'unload-observed' : 'no-unload-event';
  return { ...details, observedAt, priorStatus: run.status, lastTimestamp,
    classification,
    interruptedPhase: cleanupPending ? 'cleanup' : run.status === 'running' ? (inference ? 'inference' : sessionCreated ? 'after-session-create' : 'execution') : null,
    inference,
    cause: run.fault ? 'recorded-fault' : 'unknown',
    unloadEvidence, unloadObserved, unloadKind, reentryGapMs, navigationType,
    lifecycleHistory: lifecycle,
    lifecycle: events };
}

/** Human-readable unload evidence for an interrupted run; empty when there is nothing to add. */
export function unloadLabel(recovery) {
  if (!recovery?.unloadEvidence || recovery.unloadEvidence === 'recorded-fault') return '';
  const gap = recovery.reentryGapMs == null ? '' : ` ${(recovery.reentryGapMs / 1000).toFixed(1)}초 뒤 다시 열림`;
  const navigation = recovery.navigationType ? ` · 탐색 유형 ${recovery.navigationType}` : '';
  if (recovery.unloadEvidence === 'unload-observed') {
    return recovery.unloadKind === 'persisted-hide' ? `페이지 가려짐(pagehide persisted·hidden) 기록됨${gap}${navigation}`
      : `페이지 이동·새로고침 이벤트 기록됨${gap}${navigation}`;
  }
  return `언로드 이벤트 없이${gap}${navigation}`;
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
  const interrupted = !!run.recovery && (run.status === 'running' || run.recovery.classification === 'cleanup-reentry');
  const storage = last.storage ?? session?.storage
    ?? [...(run.records || [])].reverse().find(record => record.storage)?.storage
    ?? run.milestones?.['weights-prepared']?.storage ?? null;
  const recent = [last, ...(run.records || []).slice().reverse()];
  return { effectiveStatus: fault || run.cleanup?.success === false || run.cleanupError ? 'failed' : interrupted ? 'interrupted' : run.status,
    interruptedPhase: run.recovery?.interruptedPhase ?? null, interruptedInference: run.recovery?.inference ?? null,
    // Last sample taken inside a generate() call, and the summary of the last generate() that finished.
    inference: inferenceDigest(recent.find(record => record.stage === 'inference-sample') ?? null),
    lastInference: recent.find(record => record.inference)?.inference ?? run.summary?.inference ?? null,
    modelExecution: run.environment?.modelExecution ?? session?.modelExecution ?? null,
    streaming: [last, ...(run.records || []).slice().reverse()].find(record => record.streaming)?.streaming ?? session?.streaming ?? null,
    file: fault?.file ?? last.file ?? null, observedDuring: fault?.observedDuring ?? last.observedDuring ?? null,
    loadOrder: run.environment?.loadOrder ?? session?.loadOrder ?? null,
    tokenizer: run.summary?.tokenizer ?? session?.tokenizer ?? run.milestones?.['tokenizer-ready'] ?? null,
    tokenizerPrepared: run.summary?.tokenizerPrepared ?? (run.milestones?.['tokenizer-ready'] ? true :
      run.milestones?.['session-create']?.tokenizerPrepared ?? null),
    tokenizerFormat: run.summary?.tokenizer?.tokenizerFormat ?? run.milestones?.['tokenizer-ready']?.tokenizerFormat ?? null,
    recoveryClassification: run.recovery?.classification ?? null,
    // The worker writes `stopSource` when the page's stop message ended a run;
    // the pages write a `user-cancelled` position when they end a worker themselves.
    stopSource: run.summary?.stopSource ?? (last.stage === 'user-cancelled' ? 'user-stop' : null),
    completedRows: run.summary?.completedRows
      ?? (run.status === 'cancelled' && run.milestones?.['evaluation-rows'] ? (Array.isArray(run.rows) ? run.rows.length : 0) : null),
    unloadEvidence: run.recovery?.unloadEvidence ?? null, unloadKind: run.recovery?.unloadKind ?? null,
    reentryGapMs: run.recovery?.reentryGapMs ?? null, navigationType: run.recovery?.navigationType ?? null,
    jsMemory: last.jsMemory ?? null,
    lastRecordAt: last.timestamp ?? null,
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
    wasmHeapBytes: metrics.wasmHeapBytes || last.runtimeMetrics?.wasmHeapBytes || run.summary?.runtimeMetrics?.wasmHeapBytes || null,
    gpuCategories: ledger?.categories ?? null, gpuPrograms: ledger?.programs ?? null,
    cleanup: run.cleanup ?? null, persistence: run.persistence ?? null,
    ortPhases: Object.values(run.milestones || {}).filter(record => record.stage?.startsWith('ort-'))
      .map(({ stage, elapsedMs, sessionDiagnosticsVersion }) => ({ stage, elapsedMs, sessionDiagnosticsVersion })),
    ortPlan: ortPlanEvidence(run.milestones),
    rows: Array.isArray(run.rows) ? run.rows.length : null,
    timings: run.summary?.sessionMetrics?.timings ?? run.summary?.timings
      ?? session?.timings ?? run.milestones?.['session-create']?.timings ?? null,
    releaseId: run.environment?.build?.releaseId ?? null };
}

/**
 * Where a run stood when ORT began planning the graph: the gap from
 * `ort-session-start` and the WASM heap at that moment. Two September 13
 * sessions ended a third rapid cached load at `ort-plan-start` with no weights,
 * so repeats are compared on this point rather than on upload counters.
 */
export function ortPlanEvidence(milestones) {
  const plan = milestones?.['ort-plan-start'];
  if (!plan) return null;
  const sessionStart = milestones['ort-session-start'];
  const wasmComplete = milestones['ort-wasm-complete'];
  const since = reference => Number.isFinite(plan.elapsedMs) && Number.isFinite(reference?.elapsedMs) ? plan.elapsedMs - reference.elapsedMs : null;
  return { elapsedMs: plan.elapsedMs ?? null, sinceSessionStartMs: since(sessionStart), sinceWasmCompleteMs: since(wasmComplete),
    wasmHeapBytes: plan.metrics?.wasmHeapBytes ?? null, gpuRequestedCurrent: plan.gpuLedger?.requestedCurrent ?? null };
}

export const trackingLabel = status => ({ complete: '정상', partial: '부분 관측', unbound: '연결 미확인', unknown: '미기록' }[status] || '미기록');

/** Label for a cancelled run: who stopped it and how many evaluation rows had finished. */
export function stopLabel(summary) {
  const rows = Number.isInteger(summary?.completedRows) ? ` · ${summary.completedRows}행 완료` : '';
  return summary?.stopSource === 'user-stop' ? `사용자 정지${rows}` : `중단됨 (정지 출처 미기록)${rows}`;
}

/** Recovery is evidence from a later page, never a rewrite of the interrupted worker's last/fault. */
export async function recordRecovery(run, context = {}) {
  if (!run?.runId) return null;
  const recovery = recoveryEvidence(run, context);
  await saveCheckpoint(recovery, `recovery:${run.runId}`);
  return recovery;
}

const MAX_HISTORY_CHARACTERS = 16384;
const MAX_ROWS = 128;
function historyRecord(value) {
  // Bound both the number and size of recent records. The full latest record,
  // faults, milestones and terminal summary are persisted separately if large.
  if (JSON.stringify(value).length <= MAX_HISTORY_CHARACTERS) return value;
  return { stage: String(value.stage).slice(0, 160), runId: value.runId, timestamp: value.timestamp,
    elapsedMs: value.elapsedMs, initializerName: String(value.initializerName || '').slice(0, 1024),
    phase: value.phase, destinationOffset: value.destinationOffset, historyTruncated: true };
}

/** Serialize durable checkpoints; keep a bounded history without tensors or prompts. */
export class RunDiagnostics {
  constructor(runId = newRunId(), environment = {}, persist = saveCheckpoint, { persistBatch = saveJournalEntries } = {}) {
    this.persist = persist;
    this.persistBatch = persistBatch;
    this.mode = environment.diagnosticsMode || 'compact';
    if (!['compact', 'snapshot'].includes(this.mode)) throw new Error('Invalid diagnostics mode');
    this.headerSaved = false;
    this.dirty = new Map();
    this.state = { schemaVersion: 4, runId, startedAt: Date.now(), status: 'running', environment,
      records: [], last: null, fault: null, firstFault: null, summary: null,
      milestones: {}, files: {}, preparation: {}, initializerOrder: [], rows: [], droppedInitializers: 0, recordCount: 0, eventCount: 0,
      persistence: { mode: this.mode, completed: 0, failures: 0, totalMs: 0, peakMs: 0, entriesWritten: 0, serializedBytes: 0,
        note: PERSISTENCE_NOTE },
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
      if (MILESTONES.has(value.stage) || value.stage.startsWith('ort-')) this.state.milestones[value.stage] = value;
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
      // Row summaries outlive the 64-event ring: a cancelled or interrupted
      // evaluation keeps every completed row's timing without prompt text.
      if (value.stage === 'row-complete' && Number.isInteger(value.row) && this.state.rows.length < MAX_ROWS) {
        this.state.rows.push({ row: value.row, promptLen: value.promptLen ?? null, nTok: value.nTok ?? null,
          durationMs: value.inference?.durationMs ?? value.result?.totalMs ?? null, ...(value.result || {}), timestamp: value.timestamp });
      }
      this.state.recordCount++;
      this.state.last = value;
      const history = historyRecord(value);
      const positionOnly = POSITION_ONLY.has(value.stage);
      if (!positionOnly) {
        this.state.eventCount++;
        this.state.records.push(history);
        if (this.state.records.length > 64) this.state.records.shift();
      }
      this.state.updatedAt = value.timestamp;
      // Metadata deltas survive a failed transaction and are retried with the
      // next event. The history is a 64-slot ring; initializer metadata is capped.
      const delta = (key, kind, data) => this.dirty.set(key, { kind, ...data });
      if (this.state.milestones[value.stage] === value) delta(`milestone:${value.stage}`, 'milestone', { name: value.stage, value });
      if (isPreparation(value.stage)) delta(`preparation:${value.file || 'shared'}:${value.stage}`, 'preparation', { file: value.file || 'shared', name: value.stage, value });
      if (value.location && this.state.files[value.location] === value) delta(`file:${value.location}`, 'file', { name: value.location, value });
      const initIndex = this.state.initializerOrder.length - 1;
      const initializer = this.state.initializerOrder[initIndex];
      if (initializer?.timestamp === value.timestamp && ['allocate-initializer', 'resident-allocate'].includes(value.stage)) {
        delta(`initializer:${initIndex}`, 'initializer', { index: initIndex, value: initializer });
      }
      const row = this.state.rows.at(-1);
      if (value.stage === 'row-complete' && row?.timestamp === value.timestamp) delta(`row:${row.row}`, 'row', { value: row });
      if (this.state.fault === value || this.state.firstFault === value) delta('faults', 'faults', { fault: this.state.fault, firstFault: this.state.firstFault });
      // A position record or a truncated one is the head's `last`; otherwise the
      // ring slot with the head's event count is. The persistence note is fixed
      // text, re-attached by readRun rather than written with every head.
      const { note, ...persistence } = this.state.persistence;
      delta('head', 'head', { ...(positionOnly || history !== value ? { last: value } : {}), status: this.state.status, updatedAt: value.timestamp,
        recordCount: this.state.recordCount, eventCount: this.state.eventCount,
        droppedInitializers: this.state.droppedInitializers, persistence });
      if (!positionOnly) delta(`event:${this.state.eventCount % 64}`, 'event', { sequence: this.state.eventCount, value: history });
      const prefix = journalPrefix(this.state.runId);
      const entries = [...this.dirty].map(([key, entry]) => [prefix + key, entry]);
      if (!this.headerSaved) entries.unshift([runKey(this.state.runId), {
        schemaVersion: 4, storageFormat: 'incremental-v1', runId: this.state.runId, startedAt: this.state.startedAt,
        environment: this.state.environment, memoryNote: this.state.memoryNote,
      }]);
      const start = performance.now();
      let saved;
      try { saved = this.mode === 'snapshot' ? await this.persist(this.state, runKey(this.state.runId)) : await this.persistBatch(entries); }
      catch { saved = false; }
      if (saved !== false) {
        this.headerSaved = true;
        this.state.persistence.entriesWritten += this.mode === 'snapshot' ? 1 : entries.length;
        if (this.mode !== 'snapshot') this.state.persistence.serializedBytes += JSON.stringify(entries).length;
        this.dirty.clear();
      }
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
    this.dirty.set('summary', { kind: 'summary', value: summary });
    return this.checkpoint({ stage: this.state.status });
  }
}

/** Reconstruct schema 4 exports; schema 2/3 and snapshot journals stay readable. */
export async function readRun(runId, options = {}) {
  const { read = readCheckpoint, readEntries = readJournalEntries } = options && typeof options === 'object' ? options : {};
  let value = runId ? await read(runKey(runId)) : null;
  if (!value || value.runId !== runId) return null;
  if (value.storageFormat === 'incremental-v1') {
    const entries = await readEntries(runId);
    if (!entries) return null;
    const head = entries.find(entry => entry.kind === 'head');
    if (!head) return null;
    const { storageFormat, ...header } = value;
    const { kind, ...state } = head;
    value = { ...header, ...state, records: [], last: null, summary: null, fault: null, firstFault: null,
      milestones: {}, files: {}, preparation: {}, initializerOrder: [], rows: [] };
    const events = [];
    // Journals before position-only records numbered ring slots by recordCount.
    const eventCount = head.eventCount ?? head.recordCount;
    if (value.persistence && !value.persistence.note) value.persistence = { ...value.persistence, note: PERSISTENCE_NOTE };
    for (const entry of entries) {
      if (entry.kind === 'event' && entry.sequence <= eventCount && entry.sequence > eventCount - 64) events.push(entry);
      if (entry.kind === 'milestone') value.milestones[entry.name] = entry.value;
      if (entry.kind === 'file') value.files[entry.name] = entry.value;
      if (entry.kind === 'preparation') {
        value.preparation[entry.file] ||= {};
        value.preparation[entry.file][entry.name] = entry.value;
      }
      if (entry.kind === 'initializer') value.initializerOrder[entry.index] = entry.value;
      if (entry.kind === 'row') value.rows.push(entry.value);
      if (entry.kind === 'summary') value.summary = entry.value;
      if (entry.kind === 'faults') { value.fault = entry.fault; value.firstFault = entry.firstFault; }
    }
    events.sort((a, b) => a.sequence - b.sequence);
    value.rows.sort((a, b) => a.row - b.row);
    value.records = events.map(entry => entry.value);
    value.last = head.last ?? events.find(entry => entry.sequence === eventCount)?.value ?? null;
    value.initializerOrder = value.initializerOrder.filter(Boolean);
  }
  const recovery = await read(`recovery:${runId}`);
  const cleanup = await read(`cleanup:${runId}`);
  return { ...value, ...(recovery ? { recovery } : {}), ...(cleanup ? { cleanup } : {}) };
}
