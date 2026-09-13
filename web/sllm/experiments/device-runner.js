import { newRunId, readRun, saveCheckpoint, runKey, recordRecovery, buildIdentity, diagnosticSummary, trackingLabel, pageNavigationType, unloadLabel, localTimestamp, deviceClock } from '../diagnostics.js';
import { runtimeRelease } from '../ort-runtime.js';
import { isResident, isSimpleProbe, canRepeat, applicationOperation, executionSettings, executionEvidence, scopeLabel, seriesSummary, describeDevice, runContext, parseDeviceLogNote } from './results.js';
import { loadExperimentState, saveExperimentState, MAX_RESULTS } from './state-store.js';
import { parseDeviceReport, matchDeviceReports, coverageLabel } from './device-log.js';

const $ = id => document.getElementById(id);
const storages = { local: localStorage, session: sessionStorage };
const state = loadExperimentState(storages);
const save = () => saveExperimentState(state, storages);
let worker, sessionResult, evaluationCount = 0, finishing = false, repeatWait = null;
// iOS defaults follow the September 13 phone results: resident inference ended
// in every session, streamed loads and probes completed, and a third rapid
// cached load twice ended at `ort-plan-start`. Stored choices still win.
const isIOS = /iPad|iPhone|iPod/.test(navigator.userAgent);
const IOS_DEFAULTS = { modelExecution: 'streamed', repeatDelay: 30 };
const controlIds = ['device', 'mode', 'staging', 'repeats', 'repeatDelay', 'inspector', 'kind', 'diagnosticsMode', 'tokenizerFormat', 'sessionIdle', 'modelExecution'];
// Exports without OS/browser versions cannot be compared; the model name is still typed by hand.
$('device').value = state.device || describeDevice(navigator.userAgent); $('mode').value = state.mode;
$('staging').value = String(state.stagingMiB || 8);
$('diagnosticsMode').value = state.diagnosticsMode || 'compact';
$('tokenizerFormat').value = state.tokenizerFormat || 'json';
$('modelExecution').value = state.modelExecution || (isIOS ? IOS_DEFAULTS.modelExecution : 'resident');
$('sessionIdle').value = String(state.sessionIdle ?? 0);
$('inspector').value = state.inspector || 'unknown';
$('repeats').value = String(state.repeats || 1);
$('repeatDelay').value = String(state.repeatDelay ?? (isIOS ? IOS_DEFAULTS.repeatDelay : 0));
$('kind').value = state.kind || 'resident';
function updateControls() {
  const evaluationURL = new URL('../../../index.html', location.href);
  for (const [key, id] of [['ortMode', 'mode'], ['stagingMiB', 'staging'], ['diagnosticsMode', 'diagnosticsMode'], ['tokenizerFormat', 'tokenizerFormat'], ['modelExecution', 'modelExecution']]) {
    evaluationURL.searchParams.set(key, $(id).value);
  }
  $('evaluationLink').href = evaluationURL.href;
  $('start').disabled = !!state.active;
  $('export').disabled = false;
  for (const id of controlIds) $(id).disabled = !!state.active;
  if (!state.active) {
    $('mode').disabled = isResident($('kind').value);
    $('repeats').disabled = !canRepeat($('kind').value);
    $('repeatDelay').disabled = !(canRepeat($('kind').value) || $('kind').value === 'warm-load');
    $('staging').disabled = $('kind').value === 'tokenizer';
    $('tokenizerFormat').disabled = ['resident', 'resident-opfs', 'runtime', 'runtime-resident', 'session-only'].includes($('kind').value);
    $('sessionIdle').disabled = $('kind').value !== 'session-only';
    // The staging size applies to streamed loading and its per-token output
    // scratch as well; 2 MiB is no longer forced for the streamed path.
    $('modelExecution').disabled = !['session-only', 'load', 'warm-load', 'probe', 'evaluation'].includes($('kind').value);
  }
}
$('kind').addEventListener('change', updateControls);
for (const id of ['mode', 'staging', 'diagnosticsMode', 'tokenizerFormat', 'modelExecution']) $(id).addEventListener('change', updateControls);
// Session start to graph planning, with the WASM heap at that point: the
// comparison for repeats that end at `ort-plan-start` before any weight.
const ortPlanLabel = plan => !plan ? '—' : [plan.sinceSessionStartMs == null ? null : `세션 시작 후 ${(plan.sinceSessionStartMs / 1000).toFixed(1)}초`,
  plan.wasmHeapBytes == null ? null : `heap ${(plan.wasmHeapBytes / 2**20).toFixed(1)} MiB`].filter(Boolean).join(' · ') || '기록됨';
const inferenceLabel = phase => ({ warmup: '워밍업(첫 추론)', evaluation: '평가 추론', probe: '짧은 추론' }[phase] || '추론');
function render() {
  const series = seriesSummary(state.results, state.active);
  // Newest first: the row that needs an export is the one just recovered.
  $('rows').replaceChildren(...state.results.slice().reverse().map(result => {
    const tr = document.createElement('tr');
    if (result.interrupted) tr.className = 'interrupted';
    const comparison = result.comparison || {}, storage = comparison.storage;
    const execution = result.execution || executionEvidence(result);
    const group = series.find(group => group.seriesId === (result.seriesId || result.runIds?.[0] || result.runId));
    const mib = value => value == null ? '—' : (value / 2**20).toFixed(2);
    const cache = storage ? `${storage.cacheHits}/${storage.totalFiles} · 이전 ${storage.migratedFiles} · 다운로드 ${storage.downloadedFiles}` : '—';
    const location = [comparison.observedDuring || comparison.faultStage || comparison.stage, comparison.file, comparison.initializerName,
      comparison.destinationOffset == null ? null : `${mib(comparison.destinationOffset)} MiB 위치`].filter(Boolean).join(' · ') || '—';
    const idle = execution.idleRequestedSeconds === 0 ? '대기 없음'
      : `${execution.idleElapsedMs == null ? '미기록' : (execution.idleElapsedMs / 1000).toFixed(1)} / ${execution.idleRequestedSeconds ?? '?'}초`;
    const setting = (result.kind === 'tokenizer' ? '모델 세션 없음' : `${result.stagingMiB ?? '?'} MiB`) +
      (execution.modelExecution === 'streamed' ? ' · FP32 순차 로딩' : '') +
      (execution.tokenizerFormat ? ` · 토크나이저 ${execution.tokenizerFormat}` : '');
    const recovered = comparison.recoveryClassification;
    const unload = unloadLabel(comparison);
    const unconfirmed = unload ? `원인 미확인 · ${unload}` : '원인 미확인';
    for (const value of [`${result.kind} · ${setting} · ${isResident(result.kind) ? 'ORT 세션 없음' : execution.runtimeMode || '미기록'} · ${{ compact: '변경 항목 저장', snapshot: '전체 상태 저장' }[execution.diagnosticsMode] || '저장 방식 미기록'}`,
      recovered === 'cleanup-reentry' ? '자원 정리 중 재진입 · 정리 완료 미확인' :
      result.interrupted ? (comparison.interruptedPhase === 'inference' ? `${inferenceLabel(comparison.interruptedInference?.phase)} 중 중단 (${unconfirmed})`
        : execution.modelSessionCreated ? `세션 생성 후 중단 (${unconfirmed})` : `실행 중 중단 (${unconfirmed})`) :
      result.success ? scopeLabel(execution.completedScope) + (recovered === 'completed-run-reentry' ? ' · 완료 후 재진입' : '') : result.cancelled ? '사용자 중단' : '실패',
      `${group.startedRuns}회 시작 · ${group.successfulRuns}회 성공 / 요청 ${group.requestedRuns ?? '?'}회`, idle,
      `${result.reportedDevice || '기기 미기록'} · 검사기 ${{ attached: '연결', detached: '미연결' }[result.inspector] || '미기록'}`,
      (comparison.releaseId || result.releaseId)?.slice(0, 12) || '—', cache,
      `${comparison.loadedInitializerCount ?? '—'} / ${comparison.expectedInitializerCount ?? '?'}`,
      mib(comparison.gpuWeightAllocated), mib(comparison.gpuRequestedCurrent), `${mib(comparison.gpuWriteReturnedBytes)} / ${mib(comparison.gpuQueueCompletedBytes)}`,
      location, trackingLabel(comparison.trackingStatus), ortPlanLabel(comparison.ortPlan),
      result.durationMs == null ? '—' : `${(result.durationMs / 1000).toFixed(1)}초`]) {
      const td = document.createElement('td'); td.textContent = value; tr.append(td);
    }
    tr.append(deviceLogCell(result));
    return tr;
  }));
}
// The tester matches each interrupted row with the JetsamEvent/WebContent file
// from the phone's Analytics Data list. Those files are stamped in device local
// time, so the row shows its last record in the same clock.
function deviceLogCell(result) {
  const td = document.createElement('td');
  const at = result.lastRecordAt ?? result.endedAt ?? result.startedAt;
  const when = localTimestamp(at);
  const log = result.deviceLog;
  // The OS footprint is what Jetsam acted on; the page's GPU request is what the
  // worker counted. On the September 13 phone the first was 1.87× and 2.20× the second.
  const requested = result.comparison?.gpuRequestedCurrent;
  const ratio = log?.footprintMiB != null && requested > 0 ? (log.footprintMiB / (requested / 2 ** 20)).toFixed(2) : null;
  const summary = log ? [log.file, log.reason, log.footprintMiB == null ? null : `${log.footprintMiB} MiB`,
    log.lifetimeMaxMiB == null ? null : `최대 ${log.lifetimeMaxMiB} MiB`, ratio == null ? null : `페이지 GPU 요청의 ${ratio}배`,
    log.freeMiB == null ? null : `시스템 여유 ${log.freeMiB} MiB`, log.matchedBy === 'report' ? '파일에서 대조' : null]
    .filter(Boolean).join(' · ') || log.note : '기록 없음';
  const text = document.createElement('div'); text.textContent = `${when ? `${when.slice(11, 19)} 기준 · ` : ''}${summary}`;
  const button = document.createElement('button'); button.type = 'button'; button.textContent = log ? '기기 로그 수정' : '기기 로그 메모';
  button.onclick = () => {
    const note = prompt(`이 실행${when ? ` (${when})` : ''}과 같은 시각의 기기 로그를 적어 주세요.\n예: JetsamEvent-2026-09-13-163104.ips · per-process-limit · 1,024 MiB`, log?.note || '');
    if (note == null) return;
    const parsed = parseDeviceLogNote(note);
    result.deviceLog = parsed ? { ...parsed, recordedAt: Date.now() } : null;
    save(); render();
  };
  td.append(text, button);
  return td;
}
if (state.active) {
  const diagnostic = await readRun(state.active.runId);
  const recovery = await recordRecovery(diagnostic, { visibility: document.visibilityState, experiment: state.active.kind,
    phase: state.active.phase, navigation: state.active.navigation ?? null, lifecycle: state.active.lifecycle || [],
    navigationType: pageNavigationType() });
  const kind = state.active.kind;
  const storage = diagnostic?.summary?.storage;
  const success = recovery?.classification !== 'cleanup-reentry' && !diagnostic?.fault && diagnostic?.cleanup?.success !== false && !diagnostic?.cleanupError && ((kind === 'load' && diagnostic?.status === 'ready') ||
    (kind === 'tokenizer' && diagnostic?.status === 'complete' && diagnostic.summary?.tokenizerPrepared && diagnostic.summary?.modelSessionCreated === false) ||
    (kind === 'session-only' && diagnostic?.status === 'complete' && diagnostic.summary?.modelSessionCreated && diagnostic.summary?.tokenizerPrepared === false) ||
    (['resident-opfs', 'resident-opfs-tokenizer', 'runtime-resident'].includes(kind) && diagnostic?.status === 'complete' &&
      !!executionEvidence({ ...state.active, success: true }, diagnostic).completedScope) ||
    (kind === 'warm-load' && diagnostic?.status === 'ready' && storage?.totalFiles > 0 && storage.cacheHits === storage.totalFiles) ||
    ((isSimpleProbe(kind) || kind === 'probe') && diagnostic?.status === 'complete') ||
    (kind === 'evaluation' && state.active.evaluations?.length === 2 && state.active.evaluations.every(value => value.failed === 0)));
  const cancelled = diagnostic?.status === 'cancelled';
  const knownFailure = ['failed', 'device-lost'].includes(diagnostic?.status) || !!diagnostic?.fault ||
    diagnostic?.cleanup?.success === false || !!diagnostic?.cleanupError ||
    (kind === 'warm-load' && diagnostic?.status === 'ready' && !success);
  state.results.push({ ...state.active, success, cancelled, interrupted: !success && !cancelled && !knownFailure,
    endedAt: recovery?.observedAt ?? Date.now(), lastRecordAt: diagnostic?.last?.timestamp ?? null,
    execution: executionEvidence({ ...state.active, success }, diagnostic),
    comparison: diagnosticSummary(diagnostic && { ...diagnostic, recovery }, state.active.sessionResult) });
  if (state.results.length > MAX_RESULTS) state.results.shift();
  $('last').textContent = JSON.stringify(diagnostic, null, 2);
  $('status').textContent = success ? '이전 실험 완료 기록을 복구했습니다.' : '이전 실험 기록을 복구했습니다. 진단 JSON을 저장해 주세요.';
  state.active = null; state.continue = null; save();
}
render();
updateControls();
for (const eventName of ['pagehide', 'pageshow', 'visibilitychange']) {
  addEventListener(eventName, event => {
    if (!state.active) return;
    state.active.lifecycle ||= [];
    state.active.lifecycle.push({ event: eventName, visibility: document.visibilityState, persisted: event.persisted,
      runId: state.active.runId, timestamp: Date.now() });
    state.active.lifecycle = state.active.lifecycle.slice(-32); save();
  });
}
function reloadFor(config, reason = 'experiment-start', previousRunId = null) {
  config.runId = newRunId();
  config.navigation = { id: newRunId(), reason, previousRunId, runId: config.runId, timestamp: Date.now() };
  state.continue = config; save();
  const url = new URL(location.href); url.searchParams.set('next', config.navigation.id); location.replace(url);
}
async function closeWorker(active, result) {
  if (!worker) return;
  const current = worker;
  if (!isSimpleProbe(active.kind)) {
    const cleanup = await new Promise(resolve => {
      const listener = ({ data }) => {
        if (data.type !== 'disposed') return;
        clearTimeout(timer); current.removeEventListener('message', listener); resolve(data.error || null);
      };
      const timer = setTimeout(() => { current.removeEventListener('message', listener); resolve('GPU cleanup timed out'); }, 5000);
      current.addEventListener('message', listener); current.postMessage({ type: 'dispose' });
    });
    if (cleanup) { result.success = false; result.cleanupError = cleanup; result.error ||= cleanup; }
  }
  current.terminate(); worker = null;
}
async function finish(result) {
  const active = state.active;
  if (!active || finishing) return;
  finishing = true;
  active.phase = 'cleanup'; save();
  $('stop').disabled = true;
  await closeWorker(active, result);
  const diagnostic = await readRun(active.runId);
  if (diagnostic?.fault && !result.cancelled) { result.success = false; result.error ||= diagnostic.fault.message || diagnostic.fault.stage; }
  state.results.push({ ...active, ...result, releaseId: diagnostic?.environment?.build?.releaseId || active.releaseId,
    execution: executionEvidence({ ...active, ...result }, diagnostic),
    durationMs: Date.now() - active.startedAt, endedAt: Date.now(), lastRecordAt: diagnostic?.last?.timestamp ?? null,
    sessionResult, comparison: diagnosticSummary(diagnostic, sessionResult) });
  if (state.results.length > MAX_RESULTS) state.results.shift();
  state.active = null; finishing = false;
  save(); render();
  $('start').disabled = false; $('stop').disabled = true;
  updateControls();
  $('status').textContent = result.success ? scopeLabel(state.results.at(-1).execution.completedScope) : result.cancelled ? '사용자가 중단했습니다.' : '실험 실패 · 진단 JSON을 저장해 주세요.';
  $('last').textContent = JSON.stringify({ ...state.results.at(-1), diagnostic }, null, 2);
  if (result.success && active.remaining > 1) {
    const next = { kind: active.kind, remaining: active.remaining - 1,
      seriesId: active.seriesId, requestedRuns: active.requestedRuns, attemptNumber: active.attemptNumber + 1,
      diagnosticsMode: active.diagnosticsMode, mode: active.mode, stagingMiB: active.stagingMiB, inspector: active.inspector,
      tokenizerFormat: active.tokenizerFormat, modelExecution: active.modelExecution, idleSeconds: active.idleSeconds,
      repeatDelaySeconds: active.repeatDelaySeconds ?? 0, reportedDevice: active.reportedDevice, releaseId: active.releaseId };
    const delayMs = (active.repeatDelaySeconds || 0) * 1000;
    if (!delayMs) return reloadFor(next, 'repeat', active.runId);
    // The wait is observable and cancellable; nothing is persisted until the
    // reload is actually scheduled, so closing the tab simply ends the series.
    const due = Date.now() + delayMs;
    const tick = () => { $('status').textContent = `다음 반복까지 ${Math.max(0, Math.ceil((due - Date.now()) / 1000))}초 대기 중 (중단 버튼으로 취소)`; };
    tick();
    $('stop').disabled = false; $('start').disabled = true;
    repeatWait = { interval: setInterval(tick, 500),
      timeout: setTimeout(() => { cancelRepeatWait(); if (!state.active) reloadFor(next, 'repeat', active.runId); }, delayMs) };
  }
}
function cancelRepeatWait() {
  if (!repeatWait) return false;
  clearInterval(repeatWait.interval); clearTimeout(repeatWait.timeout); repeatWait = null;
  $('stop').disabled = true; $('start').disabled = !!state.active;
  return true;
}
async function begin(config) {
  if (state.active) return;
  config = { diagnosticsMode: state.diagnosticsMode || 'compact', tokenizerFormat: state.tokenizerFormat || 'json', mode: state.mode, stagingMiB: 8, inspector: 'unknown', reportedDevice: state.device, ...config };
  config.seriesId ||= newRunId();
  config.requestedRuns ??= config.remaining || 1;
  config.attemptNumber ??= 1;
  if (isResident(config.kind)) config.mode = null;
  if (config.kind === 'tokenizer') config.stagingMiB = null;
  if (!['session-only', 'load', 'warm-load', 'probe', 'evaluation'].includes(config.kind)) config.modelExecution = 'resident';
  state.kind = config.kind; $('kind').value = config.kind;
  const runId = config.runId || newRunId();
  state.active = { ...config, runId, runIds: [runId], phase: 'execution', startedAt: Date.now() }; save();
  // A stop becomes available after the initial journal write and worker creation,
  // so startup persistence cannot overwrite a just-recorded cancellation.
  $('start').disabled = true; $('stop').disabled = true;
  for (const id of controlIds) $(id).disabled = true;
  $('status').textContent = '실험 준비 중…';
  sessionResult = null; evaluationCount = 0;
  let release;
  try {
    release = await runtimeRelease();
    if (config.releaseId && config.releaseId !== release.build.releaseId) throw new Error('반복 실행 중 빌드가 변경됐습니다. 같은 빌드로 실험을 다시 시작해 주세요.');
  } catch (error) {
    if (!state.active) return;
    const fault = { stage: 'build-error', message: error.message, timestamp: Date.now() };
    await saveCheckpoint({ schemaVersion: 3, runId, status: 'failed', fault, last: fault }, runKey(runId));
    await finish({ success: false, error: error.message }); return;
  }
  if (!state.active) return;
  state.active.releaseId = release.build.releaseId; save();
  // Previous-run context is captured before this run writes anything, so an
  // interruption still leaves what preceded it in the run header.
  const context = runContext(state.results, Date.now());
  state.active.runContext = context; save();
  const environment = { reportedDevice: config.reportedDevice, userAgent: navigator.userAgent, experiment: config.kind,
    inspector: config.inspector, diagnosticsMode: config.diagnosticsMode, stagingMiB: config.stagingMiB, ...executionSettings(config.kind, config),
    tokenizerFormat: config.tokenizerFormat, modelExecution: config.modelExecution || 'resident', navigation: config.navigation ?? null,
    seriesId: config.seriesId, requestedRuns: config.requestedRuns, attemptNumber: config.attemptNumber,
    repeatDelaySeconds: config.repeatDelaySeconds ?? 0, runContext: context };
  await saveCheckpoint({ schemaVersion: 3, runId, status: 'running', environment: { ...environment, build: buildIdentity(release.build) },
    last: { stage: 'worker-start', timestamp: Date.now() } }, runKey(runId));
  if (!state.active) return;
  const simple = isSimpleProbe(config.kind);
  const url = new URL(simple ? './device-probes.js' : '../worker.js', import.meta.url);
  if (config.mode) url.searchParams.set('ortMode', config.mode);
  worker = new Worker(url, { type: 'module' });
  worker.onerror = async event => {
    const run = await readRun(state.active?.runId);
    if (state.active) await saveCheckpoint({ ...run, runId: state.active.runId, status: 'failed',
      fault: { stage: 'worker-error', message: event.message } }, runKey(state.active.runId));
    await finish({ success: false, error: event.message });
  };
  const nextRun = type => {
    const id = newRunId(); state.active.runId = id; state.active.runIds.push(id); save();
    worker.postMessage({ type, runId: id, environment });
  };
  worker.onmessage = async ({ data }) => {
    if (!state.active || finishing) return;
    if (data.type === 'worker-ready') worker.postMessage({ type: applicationOperation(config.kind),
      device: 'webgpu', stagingMiB: config.stagingMiB, runId, environment });
    if (data.type === 'preparation') $('last').textContent = JSON.stringify(data.record, null, 2);
    if (data.type === 'phase') $('status').textContent = data.text;
    if (data.type === 'progress') { $('status').textContent = data.record.stage; $('last').textContent = JSON.stringify(data.record, null, 2); }
    if (data.type === 'dl') $('status').textContent = `${data.which === 'tok' ? '토크나이저' : '모델'} 준비 ${Math.round((data.loaded || 0) / 2**20)} MiB`;
    if (data.type === 'diagnostic') {
      $('last').textContent = JSON.stringify(data.record, null, 2);
      if (data.record.stage === 'device-lost') await finish({ success: false, error: 'GPU device lost' });
    }
    if (data.type === 'session-result') { sessionResult = data.result; state.active.sessionResult = sessionResult; save(); }
    if (data.type === 'ready') {
      if (config.kind === 'probe') nextRun('probe');
      else if (config.kind === 'evaluation') nextRun('run');
      else if (config.kind === 'warm-load' && sessionResult?.storage.cacheHits !== sessionResult?.storage.totalFiles) {
        await finish({ success: false, error: '이번 실행에서 파일을 새로 준비했습니다. 파일 저장이 끝났으므로 저장된 모델 로드 5회를 다시 시작해 주세요.' });
      }
      else await finish({ success: true });
    }
    if (data.type === 'done') {
      evaluationCount++;
      state.active.evaluations ||= []; state.active.evaluations.push(data); save();
      if (evaluationCount < 2) nextRun('run');
      else await finish({ success: state.active.evaluations.every(value => value.failed === 0) });
    }
    if (data.type === 'result' || data.type === 'probe-result') await finish(data.result);
    if (data.type === 'fatal') await finish({ success: false, error: data.error, cancelled: data.cancelled });
    if (data.type === 'aborted') await finish({ success: false, cancelled: true });
  };
  if (simple) worker.postMessage({ kind: config.kind, runId, environment, mode: config.mode, stagingMiB: config.stagingMiB });
  $('stop').disabled = false;
}
$('start').onclick = () => {
  cancelRepeatWait();
  state.device = $('device').value; state.mode = $('mode').value;
  state.diagnosticsMode = $('diagnosticsMode').value;
  state.modelExecution = $('modelExecution').value;
  state.tokenizerFormat = $('tokenizerFormat').value; state.sessionIdle = Number($('sessionIdle').value);
  state.stagingMiB = Number($('staging').value); state.inspector = $('inspector').value; state.repeats = Number($('repeats').value);
  state.repeatDelay = Number($('repeatDelay').value) || 0;
  const kind = $('kind').value;
  state.kind = kind;
  reloadFor({ kind, remaining: kind === 'warm-load' ? 5 : canRepeat(kind) ? state.repeats : 1,
    tokenizerFormat: state.tokenizerFormat, modelExecution: state.modelExecution, ...(kind === 'session-only' ? { idleSeconds: state.sessionIdle } : {}),
    repeatDelaySeconds: state.repeatDelay, diagnosticsMode: state.diagnosticsMode, mode: state.mode, stagingMiB: state.stagingMiB,
    inspector: state.inspector, reportedDevice: state.device });
};
$('stop').onclick = async () => {
  if (cancelRepeatWait()) { $('status').textContent = '남은 반복을 취소했습니다.'; updateControls(); return; }
  const active = state.active;
  if (!active || finishing) return;
  $('stop').disabled = true;
  if (worker) {
    worker.onmessage = null; worker.onerror = null;
    worker.postMessage({ type: 'stop' });
  }
  // Give the application worker a bounded opportunity to abort and dispose.
  // Persist the main-thread cancellation only after it can no longer write.
  const result = { success: false, cancelled: true };
  await closeWorker(active, result);
  const run = await readRun(active.runId);
  await saveCheckpoint({ ...run, runId: active.runId, status: 'cancelled',
    last: { stage: 'user-cancelled', timestamp: Date.now() }, cleanupError: result.cleanupError }, runKey(active.runId));
  await finish(result);
};
// Reports are matched to interrupted rows by time and stored in the same shape
// as a hand-written note. A note the tester typed is never replaced by a file.
$('deviceLogFiles').addEventListener('change', async event => {
  const files = [...(event.target.files || [])];
  if (!files.length) return;
  const clock = deviceClock();
  const reports = [];
  for (const file of files) {
    let text = '';
    try { text = await file.text(); } catch { reports.push({ file: file.name, kind: 'unknown', kills: [], webContent: [], error: 'unreadable' }); continue; }
    reports.push(parseDeviceReport(text, file.name, { timezoneOffsetMinutes: clock.timezoneOffsetMinutes }));
  }
  const { matched, unmatched, context, ignored } = matchDeviceReports(state.results, reports);
  const kept = [];
  let applied = 0;
  for (const { index, deviceLog } of matched) {
    const current = state.results[index]?.deviceLog;
    if (current && current.matchedBy !== 'report') { kept.push(deviceLog.file); continue; }
    state.results[index].deviceLog = deviceLog; applied++;
  }
  // Every report the tester dropped in is kept with the results, matched or
  // not, so the export says which files were examined and what they showed.
  const clockedAt = Date.now();
  const summaries = [
    ...context.map(report => ({ ...report, matchedRows: [] })),
    ...ignored.map(report => ({ ...report, coverage: null, matchedRows: [] })),
    ...[...new Set(matched.map(entry => entry.file))].map(file => {
      const report = reports.find(value => value.file === file);
      return { file, kind: report?.kind ?? 'jetsam', reportedAt: report?.reportedAt ?? null, reportedAtText: report?.reportedAtText ?? null,
        coverage: 'within-runs', freeMiB: report?.freeMiB ?? null, compressorMiB: report?.compressorMiB ?? null,
        suspendedMiB: report?.suspendedMiB ?? null, suspendedTop: report?.suspendedTop ?? [], frontmost: report?.frontmost ?? [],
        kills: (report?.kills ?? []).map(kill => `${kill.name} ${kill.reason}`),
        matchedRows: matched.filter(entry => entry.file === file).map(entry => entry.runId) };
    }),
  ].map(report => ({ ...report, recordedAt: clockedAt }));
  state.deviceReports = [...(state.deviceReports || []).filter(report => !summaries.some(value => value.file === report.file)), ...summaries].slice(-32);
  save(); render();
  const parts = [`${applied}개 행에 기기 로그를 기록했습니다.`];
  if (kept.length) parts.push(`손으로 적은 메모가 있어 건너뜀 ${kept.length}건`);
  if (unmatched.length) parts.push(`짝지을 중단 행이 없는 WebContent 종료 ${unmatched.length}건 (${unmatched.map(kill => `${kill.file} pid ${kill.pid} ${kill.reason} ${kill.footprintMiB ?? '?'} MiB`).join(', ')})`);
  for (const report of context) {
    const web = report.webContent.map(process => `WebContent pid ${process.pid} ${process.footprintMiB ?? '?'} MiB (최대 ${process.lifetimeMaxMiB ?? '?'})`).join(', ');
    const pressure = [report.freeMiB == null ? null : `시스템 여유 ${report.freeMiB} MiB`, report.compressorMiB == null ? null : `압축기 ${report.compressorMiB} MiB`,
      report.suspendedMiB == null ? null : `일시정지 앱 ${report.suspendedMiB} MiB`, report.frontmost?.length ? `전면 ${report.frontmost.join(', ')}` : null].filter(Boolean).join(' · ');
    parts.push(`${report.file}: ${coverageLabel(report)} · WebContent 종료 없음${report.kills.length ? ` · 종료 ${report.kills.join(', ')}` : ''}${pressure ? ` · ${pressure}` : ''}${web ? ` · ${web}` : ' · 브라우저 WebContent 프로세스 없음'}`);
  }
  for (const report of ignored) parts.push(`${report.file}: ${report.kind === 'resource' ? `리소스 리포트 (${report.event ?? '종류 미상'}), 종료 아님` : report.error || '읽을 수 없는 형식'}`);
  $('status').textContent = parts.join(' · ');
  event.target.value = '';
});
$('export').onclick = async () => {
  const ids = new Set(state.results.flatMap(result => result.runIds || [result.runId]));
  for (const id of state.active?.runIds || []) ids.add(id);
  const runs = await Promise.all([...ids].map(readRun));
  const byId = new Map(runs.filter(Boolean).map(run => [run.runId, run]));
  const clock = deviceClock();
  const results = state.results.map(result => {
    const lastRecordAt = result.lastRecordAt ?? byId.get(result.runId)?.last?.timestamp ?? null;
    return { ...result, execution: executionEvidence(result, byId.get(result.runId)),
      startedAtLocal: localTimestamp(result.startedAt, clock.timezoneOffsetMinutes),
      lastRecordAt, lastRecordAtLocal: localTimestamp(lastRecordAt, clock.timezoneOffsetMinutes),
      deviceLog: result.deviceLog ?? null };
  });
  const blob = new Blob([JSON.stringify({ schemaVersion: 4, exportedAt: new Date().toISOString(), deviceClock: clock, userAgent: navigator.userAgent,
    screenSettings: { device: state.device, diagnosticsMode: state.diagnosticsMode, tokenizerFormat: state.tokenizerFormat, modelExecution: state.modelExecution,
      sessionIdle: state.sessionIdle, mode: state.mode, stagingMiB: state.stagingMiB, inspector: state.inspector, repeats: state.repeats,
      repeatDelay: state.repeatDelay ?? 0 },
    active: state.active, results, series: seriesSummary(results, state.active),
    deviceReports: state.deviceReports || [],
    memoryNote: 'Logical allocation counters are not process RSS. Device logs are needed to confirm termination causes.',
    runs: runs.filter(Boolean) }, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a'); a.href = url; a.download = 'iphone-fp32-experiments.json'; a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
};
if (state.continue && state.continue.navigation?.id === new URLSearchParams(location.search).get('next')) {
  const config = state.continue; state.continue = null; save();
  await begin(config);
} else if (state.continue) {
  state.continue = null; save();
  $('status').textContent = '예정된 페이지 이동을 확인하지 못해 자동 실행을 중단했습니다. 실험 시작 버튼으로 다시 실행하세요.';
}
