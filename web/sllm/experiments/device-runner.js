import { newRunId, readRun, saveCheckpoint, runKey } from '../diagnostics.js';

const $ = id => document.getElementById(id);
const key = 'didimdol.device-experiments.v2';
let state;
try { state = JSON.parse(sessionStorage.getItem(key)); } catch {}
state ||= { results: [], active: null, device: '', mode: 'asyncify' };
const save = () => sessionStorage.setItem(key, JSON.stringify(state));
let worker, sessionResult, evaluationCount = 0;
$('device').value = state.device; $('mode').value = state.mode;
function render() {
  $('rows').replaceChildren(...state.results.map(result => {
    const tr = document.createElement('tr');
    for (const value of [result.kind, result.interrupted ? '중단 (원인 미확인)' : result.success ? '성공' : result.cancelled ? '사용자 중단' : '실패',
      result.durationMs == null ? '—' : `${(result.durationMs / 1000).toFixed(1)}초`]) {
      const td = document.createElement('td'); td.textContent = value; tr.append(td);
    }
    return tr;
  }));
}
if (state.active) {
  const diagnostic = await readRun(state.active.runId);
  const kind = state.active.kind;
  const storage = diagnostic?.summary?.storage;
  const success = !diagnostic?.fault && ((kind === 'load' && diagnostic?.status === 'ready') ||
    (kind === 'warm-load' && diagnostic?.status === 'ready' && storage?.totalFiles > 0 && storage.cacheHits === storage.totalFiles) ||
    (['resident', 'runtime', 'probe'].includes(kind) && diagnostic?.status === 'complete') ||
    (kind === 'evaluation' && state.active.evaluations?.length === 2 && state.active.evaluations.every(value => value.failed === 0)));
  const cancelled = diagnostic?.status === 'cancelled';
  const knownFailure = ['failed', 'device-lost'].includes(diagnostic?.status) || !!diagnostic?.fault ||
    (kind === 'warm-load' && diagnostic?.status === 'ready' && !success);
  state.results.push({ ...state.active, success, cancelled, interrupted: !success && !cancelled && !knownFailure });
  $('last').textContent = JSON.stringify(diagnostic, null, 2);
  $('status').textContent = success ? '이전 실험 완료 기록을 복구했습니다.' : '이전 실험 기록을 복구했습니다. 진단 JSON을 저장해 주세요.';
  state.active = null; state.continue = null; save();
}
render();
function reloadFor(config) {
  state.continue = config; save();
  const url = new URL(location.href); url.searchParams.set('next', '1'); location.replace(url);
}
async function closeWorker(active, result) {
  if (!worker) return;
  const current = worker;
  if (result.success && !['resident', 'runtime'].includes(active.kind)) {
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
  state.results.push({ ...active, ...result, durationMs: Date.now() - active.startedAt, sessionResult });
  if (state.results.length > 30) state.results.shift();
  save(); render();
  $('start').disabled = false; $('stop').disabled = true;
  $('status').textContent = result.success ? '실험 완료' : result.cancelled ? '사용자가 중단했습니다.' : '실험 실패 · 진단 JSON을 저장해 주세요.';
  $('last').textContent = JSON.stringify({ ...state.results.at(-1), diagnostic }, null, 2);
  if (result.success && active.remaining > 1) reloadFor({ kind: active.kind, remaining: active.remaining - 1 });
}
async function begin(config) {
  if (state.active) return;
  const runId = newRunId();
  state.active = { ...config, runId, runIds: [runId], startedAt: Date.now() }; save();
  $('start').disabled = true; $('stop').disabled = false;
  $('status').textContent = '실험 준비 중…';
  sessionResult = null; evaluationCount = 0;
  const environment = { reportedDevice: state.device, userAgent: navigator.userAgent, experiment: config.kind };
  const simple = ['resident', 'runtime'].includes(config.kind);
  const url = new URL(simple ? './device-probes.js' : '../worker.js', import.meta.url);
  url.searchParams.set('ortMode', state.mode);
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
    if (data.type === 'worker-ready') worker.postMessage({ type: 'load', device: 'webgpu', stagingMiB: 8, runId, environment });
    if (data.type === 'phase') $('status').textContent = data.text;
    if (data.type === 'progress') { $('status').textContent = data.record.stage; $('last').textContent = JSON.stringify(data.record, null, 2); }
    if (data.type === 'dl') $('status').textContent = `모델 준비 ${Math.round((data.loaded || 0) / 2**20)} MiB`;
    if (data.type === 'diagnostic') {
      $('last').textContent = JSON.stringify(data.record, null, 2);
      if (data.record.stage === 'device-lost') await finish({ success: false, error: 'GPU device lost' });
    }
    if (data.type === 'session-result') sessionResult = data.result;
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
  if (simple) worker.postMessage({ kind: config.kind, runId, environment, mode: state.mode });
}
$('start').onclick = () => {
  state.device = $('device').value; state.mode = $('mode').value;
  const kind = $('kind').value;
  reloadFor({ kind, remaining: kind === 'warm-load' ? 5 : 1 });
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
  const blob = new Blob([JSON.stringify({ ...state, userAgent: navigator.userAgent,
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
