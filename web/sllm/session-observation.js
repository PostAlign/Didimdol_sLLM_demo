export function pause(ms, signal) {
  return new Promise((resolve, reject) => {
    const finish = error => {
      clearTimeout(timer); signal?.removeEventListener('abort', abort);
      if (error) reject(error); else resolve();
    };
    const abort = () => finish(signal.reason);
    const timer = setTimeout(() => finish(), ms);
    signal?.addEventListener('abort', abort, { once: true });
    if (signal?.aborted) abort();
  });
}

/** The caller owns and retains the session until observation and cleanup finish. */
export async function observeSession({ seconds, signal, checkpoint, sample = () => ({}),
  clock = () => performance.now(), wait = pause }) {
  if (!Number.isFinite(seconds) || seconds < 0 || seconds > 120) throw new Error('Session observation must be 0–120 seconds');
  if (!seconds) return { idleSeconds: 0, idleElapsedMs: 0, idleAcceptanceCompleted: false };
  signal?.throwIfAborted();
  const started = clock();
  await checkpoint({ stage: 'session-idle-start', idleSeconds: seconds, idleElapsedMs: 0, ...sample() });
  while (clock() - started < seconds * 1000) {
    await wait(Math.min(5000, seconds * 1000 - (clock() - started)), signal);
    signal?.throwIfAborted();
    await checkpoint({ stage: 'session-idle', idleSeconds: seconds, idleElapsedMs: clock() - started, ...sample() });
  }
  signal?.throwIfAborted();
  const summary = { idleSeconds: seconds, idleElapsedMs: clock() - started,
    idleAcceptanceCompleted: seconds >= 120 && clock() - started >= 120000 };
  await checkpoint({ stage: 'session-idle-complete', ...summary, ...sample() });
  signal?.throwIfAborted();
  return summary;
}

// ── 추론 중 샘플링 ──────────────────────────────────────────────────────────
// A generate() call leaves no checkpoint between its start and end records. When
// the page is terminated during the first inference, nothing shows how much the
// GPU or WASM heap had grown. The sampler reads the ledger between event-loop
// turns; it never waits on the GPU queue and never touches ORT state.

const programKeys = ['shaderModules', 'computePipelines', 'asyncPipelinesStarted', 'asyncPipelinesCompleted', 'asyncPipelinesFailed'];

/** Compact view of one sample; used for change detection and for summaries. */
export function inferenceDigest(value) {
  if (!value) return null;
  const ledger = value.gpuLedger || {}, programs = ledger.programs || {}, metrics = value.metrics || {};
  return { phase: value.phase ?? null, row: value.row ?? null, reason: value.reason ?? null,
    inferenceElapsedMs: value.inferenceElapsedMs ?? null, sampleIndex: value.sampleIndex ?? null,
    gpuRequestedCurrent: ledger.requestedCurrent ?? null, gpuObservedPeak: ledger.observedPeak ?? null,
    liveBufferCount: ledger.liveBufferCount ?? null, bufferCount: ledger.bufferCount ?? null,
    deviceLost: ledger.deviceLost ?? null, lastError: ledger.lastError ?? null,
    programs: Object.fromEntries(programKeys.map(key => [key, programs[key] ?? null])),
    wasmHeapBytes: metrics.wasmHeapBytes ?? null,
    streamingUploadedBytes: value.streaming?.uploadedBytes ?? null };
}

// Growth signals only. Current bytes and buffer counts move every decode step,
// which would turn "record on change" into a write every interval.
export const inferenceSignature = value => {
  const digest = inferenceDigest(value);
  if (!digest) return 'none';
  return [digest.gpuObservedPeak, digest.deviceLost, digest.lastError, ...programKeys.map(key => digest.programs[key]),
    digest.wasmHeapBytes].join('|');
};

/** Record `inference-sample` checkpoints while a generate call runs. Stop before disposing the outputs. */
export function observeInference({ checkpoint, sample, context = {}, intervalMs = 500, heartbeatMs = 5000,
  signature = inferenceSignature, clock = () => performance.now(), wait = pause }) {
  if (!(intervalMs > 0) || !(heartbeatMs >= intervalMs)) throw new Error('Invalid inference sampling interval');
  const controller = new AbortController();
  const started = clock();
  const summary = { ...context, samples: 0, changes: 0, marks: 0, durationMs: 0, lastSample: null };
  let lastSignature = null, lastAt = started;
  const record = async (trigger, details = {}) => {
    let value;
    try { value = sample(); }
    catch (error) { value = { sampleError: { errorType: error?.constructor?.name || 'Error', message: String(error?.message ?? error) } }; }
    const now = clock();
    const current = signature(value);
    const changed = current !== lastSignature;
    const interval = trigger === 'interval';
    if (interval && !changed && now - lastAt < heartbeatMs) return false;
    lastSignature = current; lastAt = now;
    summary.samples++;
    if (changed) summary.changes++;
    if (!interval) summary.marks++;
    const record = { stage: 'inference-sample', reason: interval ? (changed ? 'change' : 'heartbeat') : trigger,
      inferenceElapsedMs: now - started, sampleIndex: summary.samples, ...context, ...details, ...value };
    summary.lastSample = inferenceDigest(record);
    await checkpoint(record);
    return true;
  };
  const loop = (async () => {
    while (!controller.signal.aborted) {
      try { await wait(intervalMs, controller.signal); } catch { break; }
      if (controller.signal.aborted) break;
      await record('interval');
    }
  })();
  return {
    mark: (reason, details) => record(reason, details),
    async stop() {
      controller.abort();
      await loop;
      summary.durationMs = clock() - started;
      return { ...summary };
    },
  };
}
