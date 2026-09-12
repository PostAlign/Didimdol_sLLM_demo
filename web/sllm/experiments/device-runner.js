import { newRunId, readRun, saveCheckpoint, runKey, recordRecovery, buildIdentity, diagnosticSummary, trackingLabel } from '../diagnostics.js';
import { runtimeRelease } from '../ort-runtime.js';
import { isResident, isSimpleProbe, executionSettings, executionEvidence, scopeLabel, seriesSummary } from './results.js';

const $ = id => document.getElementById(id);
const key = 'didimdol.device-experiments.v2';
let state;
try { state = JSON.parse(sessionStorage.getItem(key)); } catch {}
state ||= { results: [], active: null, device: '', mode: 'asyncify' };
const save = () => sessionStorage.setItem(key, JSON.stringify(state));
let worker, sessionResult, evaluationCount = 0;
$('device').value = state.device; $('mode').value = state.mode;
$('staging').value = String(state.stagingMiB || 8);
$('inspector').value = state.inspector || 'unknown';
$('repeats').value = String(state.repeats || 1);
$('kind').value = state.kind || 'resident';
function updateControls() {
  for (const id of ['device', 'mode', 'staging', 'repeats', 'inspector', 'kind']) $(id).disabled = !!state.active;
  if (!state.active) {
    $('mode').disabled = isResident($('kind').value);
    $('repeats').disabled = $('kind').value !== 'load';
  }
}
$('kind').addEventListener('change', updateControls);
function render() {
  const series = seriesSummary(state.results, state.active);
  $('rows').replaceChildren(...state.results.map(result => {
    const tr = document.createElement('tr');
    const comparison = result.comparison || {}, storage = comparison.storage;
    const execution = result.execution || executionEvidence(result);
    const group = series.find(group => group.seriesId === (result.seriesId || result.runIds?.[0] || result.runId));
    const mib = value => value == null ? '—' : (value / 2**20).toFixed(2);
    const cache = storage ? `${storage.cacheHits}/${storage.totalFiles} · 이전 ${storage.migratedFiles} · 다운로드 ${storage.downloadedFiles}` : '—';
    const location = [comparison.faultStage || comparison.stage, comparison.initializerName,
      comparison.destinationOffset == null ? null : `${mib(comparison.destinationOffset)} MiB 위치`].filter(Boolean).join(' · ') || '—';
    const idle = execution.idleRequestedSeconds === 0 ? '대기 없음'
      : `${execution.idleElapsedMs == null ? '미기록' : (execution.idleElapsedMs / 1000).toFixed(1)} / ${execution.idleRequestedSeconds ?? '?'}초`;
    for (const value of [`${result.kind} · ${result.stagingMiB ?? '?'} MiB · ${isResident(result.kind) ? 'ORT 사용 안 함' : execution.runtimeMode || '미기록'}`,
      result.interrupted ? '중단 (원인 미확인)' : result.success ? scopeLabel(execution.completedScope) : result.cancelled ? '사용자 중단' : '실패',
      `${group.startedRuns}회 시작 · ${group.successfulRuns}회 성공 / 요청 ${group.requestedRuns ?? '?'}회`, idle,
      `${result.reportedDevice || '기기 미기록'} · 검사기 ${{ attached: '연결', detached: '미연결' }[result.inspector] || '미기록'}`,
      (comparison.releaseId || result.releaseId)?.slice(0, 12) || '—', cache,
      `${comparison.loadedInitializerCount ?? '—'} / ${comparison.expectedInitializerCount ?? '?'}`,
      mib(comparison.gpuWeightAllocated), `${mib(comparison.gpuWriteReturnedBytes)} / ${mib(comparison.gpuQueueCompletedBytes)}`,
      location, trackingLabel(comparison.trackingStatus),
      result.durationMs == null ? '—' : `${(result.durationMs / 1000).toFixed(1)}초`]) {
      const td = document.createElement('td'); td.textContent = value; tr.append(td);
    }
    return tr;
  }));
}
if (state.active) {
  const diagnostic = await readRun(state.active.runId);
  const recovery = await recordRecovery(diagnostic, { visibility: document.visibilityState, experiment: state.active.kind,
    lifecycle: state.active.lifecycle || [] });
  const kind = state.active.kind;
  const storage = diagnostic?.summary?.storage;
  const success = !diagnostic?.fault && ((kind === 'load' && diagnostic?.status === 'ready') ||
    (kind === 'warm-load' && diagnostic?.status === 'ready' && storage?.totalFiles > 0 && storage.cacheHits === storage.totalFiles) ||
    ((isSimpleProbe(kind) || kind === 'probe') && diagnostic?.status === 'complete') ||
    (kind === 'evaluation' && state.active.evaluations?.length === 2 && state.active.evaluations.every(value => value.failed === 0)));
  const cancelled = diagnostic?.status === 'cancelled';
  const knownFailure = ['failed', 'device-lost'].includes(diagnostic?.status) || !!diagnostic?.fault ||
    (kind === 'warm-load' && diagnostic?.status === 'ready' && !success);
  state.results.push({ ...state.active, success, cancelled, interrupted: !success && !cancelled && !knownFailure,
    execution: executionEvidence({ ...state.active, success }, diagnostic),
    comparison: diagnosticSummary(diagnostic && { ...diagnostic, recovery }, state.active.sessionResult) });
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
function reloadFor(config) {
  state.continue = config; save();
  const url = new URL(location.href); url.searchParams.set('next', '1'); location.replace(url);
}
async function closeWorker(active, result) {
  if (!worker) return;
  const current = worker;
  if (result.success && !isSimpleProbe(active.kind)) {
    const cleanup = await new Promise(resolve => {
      const listener = ({ data }) => {
        if (data.type !== 'disposed') return;
        clearTimeout(timer); current.removeEventListener('message', listener); resolve(data.error || null);
      };
      const timer = setTimeout(() => { current.removeEventListener('message', listener); resolve('GPU cleanup timed out'); }, 5000);
      current.addEventListener('message', listener); current.postMessage({ type: 'dispose' });
    });
    if (cleanup) { result.success = false; result.error = cleanup; }
  }
  current.terminate(); worker = null;
}
async function finish(result) {
  const active = state.active;
  if (!active) return;
  state.active = null;
  await closeWorker(active, result);
  const diagnostic = await readRun(active.runId);
  if (diagnostic?.fault && !result.cancelled) { result.success = false; result.error ||= diagnostic.fault.message || diagnostic.fault.stage; }
  state.results.push({ ...active, ...result, releaseId: diagnostic?.environment?.build?.releaseId || active.releaseId,
    execution: executionEvidence({ ...active, ...result }, diagnostic),
    durationMs: Date.now() - active.startedAt, sessionResult, comparison: diagnosticSummary(diagnostic, sessionResult) });
  if (state.results.length > 30) state.results.shift();
  save(); render();
  $('start').disabled = false; $('stop').disabled = true;
  updateControls();
  $('status').textContent = result.success ? scopeLabel(state.results.at(-1).execution.completedScope) : result.cancelled ? '사용자가 중단했습니다.' : '실험 실패 · 진단 JSON을 저장해 주세요.';
  $('last').textContent = JSON.stringify({ ...state.results.at(-1), diagnostic }, null, 2);
  if (result.success && active.remaining > 1) reloadFor({ kind: active.kind, remaining: active.remaining - 1,
    seriesId: active.seriesId, requestedRuns: active.requestedRuns, attemptNumber: active.attemptNumber + 1,
    mode: active.mode, stagingMiB: active.stagingMiB, inspector: active.inspector,
    reportedDevice: active.reportedDevice, releaseId: active.releaseId });
}
async function begin(config) {
  if (state.active) return;
  config = { mode: state.mode, stagingMiB: 8, inspector: 'unknown', reportedDevice: state.device, ...config };
  config.seriesId ||= newRunId();
  config.requestedRuns ??= config.remaining || 1;
  config.attemptNumber ??= 1;
  if (isResident(config.kind)) config.mode = null;
  state.kind = config.kind; $('kind').value = config.kind;
  const runId = newRunId();
  state.active = { ...config, runId, runIds: [runId], startedAt: Date.now() }; save();
  // A stop becomes available after the initial journal write and worker creation,
  // so startup persistence cannot overwrite a just-recorded cancellation.
  $('start').disabled = true; $('stop').disabled = true;
  for (const id of ['device', 'mode', 'staging', 'repeats', 'inspector', 'kind']) $(id).disabled = true;
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
  const environment = { reportedDevice: config.reportedDevice, userAgent: navigator.userAgent, experiment: config.kind,
    inspector: config.inspector, stagingMiB: config.stagingMiB, ...executionSettings(config.kind, config),
    seriesId: config.seriesId, requestedRuns: config.requestedRuns, attemptNumber: config.attemptNumber };
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
    if (!state.active) return;
    if (data.type === 'worker-ready') worker.postMessage({ type: 'load', device: 'webgpu', stagingMiB: config.stagingMiB, runId, environment });
    if (data.type === 'phase') $('status').textContent = data.text;
    if (data.type === 'progress') { $('status').textContent = data.record.stage; $('last').textContent = JSON.stringify(data.record, null, 2); }
    if (data.type === 'dl') $('status').textContent = `모델 준비 ${Math.round((data.loaded || 0) / 2**20)} MiB`;
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
  state.device = $('device').value; state.mode = $('mode').value;
  state.stagingMiB = Number($('staging').value); state.inspector = $('inspector').value; state.repeats = Number($('repeats').value);
  const kind = $('kind').value;
  state.kind = kind;
  reloadFor({ kind, remaining: kind === 'warm-load' ? 5 : kind === 'load' ? state.repeats : 1,
    mode: state.mode, stagingMiB: state.stagingMiB, inspector: state.inspector, reportedDevice: state.device });
};
$('stop').onclick = async () => {
  const active = state.active;
  if (!active) return;
  if (worker) { worker.onmessage = null; worker.onerror = null; worker.terminate(); }
  const run = await readRun(active.runId);
  await saveCheckpoint({ ...run, runId: active.runId, status: 'cancelled', last: { stage: 'user-cancelled', timestamp: Date.now() } }, runKey(active.runId));
  await finish({ success: false, cancelled: true });
};
$('export').onclick = async () => {
  const ids = new Set(state.results.flatMap(result => result.runIds || [result.runId]));
  for (const id of state.active?.runIds || []) ids.add(id);
  const runs = await Promise.all([...ids].map(readRun));
  const byId = new Map(runs.filter(Boolean).map(run => [run.runId, run]));
  const results = state.results.map(result => ({ ...result,
    execution: executionEvidence(result, byId.get(result.runId)) }));
  const blob = new Blob([JSON.stringify({ schemaVersion: 3, exportedAt: new Date().toISOString(), userAgent: navigator.userAgent,
    screenSettings: { device: state.device, mode: state.mode, stagingMiB: state.stagingMiB, inspector: state.inspector, repeats: state.repeats },
    active: state.active, results, series: seriesSummary(results, state.active),
    memoryNote: 'Logical allocation counters are not process RSS. Device logs are needed to confirm termination causes.',
    runs: runs.filter(Boolean) }, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a'); a.href = url; a.download = 'iphone-fp32-experiments.json'; a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
};
if (state.continue && new URLSearchParams(location.search).has('next')) {
  const config = state.continue; state.continue = null; save();
  await begin(config);
}
