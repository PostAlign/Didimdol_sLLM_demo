import { readCheckpoint } from '../diagnostics.js';
export const MATRIX = [
  { id: 'A', kind: 'stock', fileMiB: 128 }, { id: 'B', kind: 'stock', fileMiB: 64 }, { id: 'C', kind: 'stock', fileMiB: 32 },
  ...[64, 32, 16, 8].map((stagingMiB, i) => ({ id: 'DEFG'[i], kind: 'range', fileMiB: 128, stagingMiB })),
];
const key = 'didimdol.memory-experiments.v1';
let state = JSON.parse(localStorage.getItem(key) || '{"results":[],"active":null}');
const $ = id => document.getElementById(id);
const save = () => localStorage.setItem(key, JSON.stringify(state));
const mib = bytes => bytes == null ? '미측정' : `${(bytes / 2**20).toFixed(2)} MiB`;
function render() {
  $('rows').replaceChildren(...state.results.map(result => {
    const tr = document.createElement('tr');
    for (const value of [result.id, result.pageReloadObserved ? '중단/페이지 재시작' : result.success ? '성공' : '실패',
      result.sessionCreateMs == null ? '미측정' : `${result.sessionCreateMs.toFixed(0)} ms`,
      mib(result.metrics?.cpuStagingPeak), mib(result.metrics?.wasmHeapPeak), mib(result.metrics?.gpuWeightAllocated)]) {
      const td = document.createElement('td'); td.textContent = value; tr.append(td);
    }
    return tr;
  }));
}
if (state.source) $('source').value = state.source;
if (state.active) {
  const last = await readCheckpoint(`experiment-${state.active.id}`);
  // A remaining marker indicates interruption; browser OOM is one possible cause.
  state.results.push({ ...state.active, success: false, pageReloadObserved: true, lastCheckpoint: last });
  $('last').textContent = 'LAST CRASH POSITION\n' + JSON.stringify(last, null, 2);
  state.active = null; save();
}
render();
$('start').onclick = () => {
  const experiment = MATRIX.find(entry => !state.results.some(result => result.id === entry.id));
  if (!experiment) { $('status').textContent = '완료'; return; }
  state.source = new URL($('source').value, location.href).href;
  state.active = experiment; save();
  $('start').disabled = true; $('reset').disabled = true;
  $('status').textContent = `${experiment.id} 실행 중…`;
  const worker = new Worker(new URL('./worker.js', import.meta.url), { type: 'module' });
  worker.onmessage = ({ data }) => {
    if (data.type === 'progress') { $('last').textContent = JSON.stringify(data.result, null, 2); return; }
    if (data.type !== 'result') return;
    state.results.push(data.result); state.active = null; save(); worker.terminate();
    const url = new URL(location.href); url.searchParams.set('continue', '1'); location.replace(url);
  };
  worker.onerror = error => {
    state.results.push({ ...experiment, success: false, error: error.message }); state.active = null; save();
    worker.terminate(); location.reload();
  };
  worker.postMessage({ source: state.source, experiment });
};
$('export').onclick = () => {
  const url = URL.createObjectURL(new Blob([JSON.stringify({ userAgent: navigator.userAgent, ...state }, null, 2)], { type: 'application/json' }));
  const a = document.createElement('a'); a.href = url; a.download = 'session-memory-experiments.json'; a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
};
$('reset').onclick = () => { localStorage.removeItem(key); location.href = location.pathname; };
if (new URLSearchParams(location.search).has('continue') && state.results.length < MATRIX.length) $('start').click();
