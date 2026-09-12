/** FP32 WebGPU evaluation worker. Range loading is only permitted during
 * OrtCreateSession. See docs/session-create-memory.md for the source audit. */

const runtimeMode = new URL(self.location.href).searchParams.get('ortMode') || 'asyncify';
const { AutoModelForCausalLM, AutoTokenizer, BaseStreamer,
  InterruptableStoppingCriteria, env, random, build: runtimeBuild } = await import(`./runtime.js?mode=${runtimeMode}`);
import { SessionRangeLoader } from './range-loader.js';
import { RunDiagnostics, newRunId, buildIdentity, gpuOperationContext } from './diagnostics.js';
import { OpfsWeightStore } from './opfs-store.js';
import { installGpuTracking } from './gpu-device.js';

import { makeRouge1 } from './rouge.js';

const REPO = 'PostAlign/Didimdol_sLLM';
// main 이 아니라 커밋 SHA 로 고정한다. transformers.js 의 브라우저 캐시 키는 원격 URL 이고
// ETag 검증을 하지 않으므로, main 에 새 가중치를 올리면 재방문자가 옛 모델을
// 무기한 캐시에서 계속 쓰게 된다. SHA 를 갈면 URL 이 바뀌어 캐시가 자연히 갈린다.
// 갈린 옛 SHA 항목(1 GB)은 저절로 사라지지 않으므로 load() 의 pruneStaleCache 가 지운다.
// upload_to_hf.py 로 새 산출물을 올린 뒤에는 반드시 그 커밋 SHA 로 바꾼다.
const REVISION = '8c50d7686bb1b205c02d42bb4b64505483c41a2b';

const GEN = {
  do_sample: true,
  temperature: 0.3,
  max_new_tokens: 512,
  // top_k 64 / top_p 0.95 는 generation_config.json 에서 상속한다(덮어쓰지 않음).
};
const SEED = 42;
const EOS = new Set([1, 106]); // <eos>, <end_of_turn>

// ── 정적 사이트에서 서빙할 파일들 ───────────────────────────────────────────
const LOCAL = Object.fromEntries(Object.entries({
  'config.json':            '../../model/config.json',
  'generation_config.json': '../../model/generation_config.json',
  'tokenizer.json':         '../../tokenizer/tokenizer.json',
  'tokenizer_config.json':  '../../tokenizer/tokenizer_config.json',
  'special_tokens_map.json':'../../tokenizer/special_tokens_map.json',
  'added_tokens.json':      '../../tokenizer/added_tokens.json',
}).map(([k, v]) => [k, new URL(v, import.meta.url).href]));

const post = (m) => self.postMessage(m);

const GRAPH = 'model.onnx';
const resolveUrl = (name) => `https://huggingface.co/${REPO}/resolve/${REVISION}/${name}`;

let verifiedGraph = null;
const baseFetch = env.fetch ?? fetch;
env.fetch = (input, init) => {
  const url = typeof input === 'string' ? input : (input?.url ?? String(input));
  if (url.includes('huggingface.co')) {
    const name = url.split('?')[0].split('/').pop();
    if (name === GRAPH && verifiedGraph) return Promise.resolve(new Response(verifiedGraph));
    if (LOCAL[name]) return fetch(LOCAL[name], init);          // → 정적 사이트
  }
  return baseFetch(input, init);                               // → HF
};

// Files are verified on disk before ORT receives small range-source descriptors.
let weightStore, journal, loadController, trackedGpu, sessionMetrics;
let operation = null, queuedRun = null;
const trace = new URL(self.location.href).searchParams.get('trace') === '1';
async function mountWeights(manifest) {
  weightStore = await OpfsWeightStore.open(manifest);
  let cache;
  try { cache = await caches.open(env.cacheKey); } catch {}
  const total = manifest.files.reduce((n, file) => n + file.bytes, 0);
  let before = 0;
  const out = [];
  for (const [i, file] of manifest.files.entries()) {
    post({ type: 'phase', text: `가중치 ${i + 1}/${manifest.files.length} 저장·검증 중…` });
    const url = resolveUrl(file.location);
    await weightStore.prepare(file, {
      cache, url, signal: loadController.signal,
      openResponse: () => baseFetch(url, { signal: loadController.signal }),
      checkpoint: journal.checkpoint,
      progress: (loaded, stage) => post({ type: 'dl', which: 'model', status: 'progress', loaded: before + loaded, total, stage }),
    });
    before += file.bytes;
    out.push({ path: file.location, data: weightStore.descriptor(file) });
  }
  weightStore.finishPreparation();
  await journal.checkpoint({ stage: 'weights-prepared', storage: { ...weightStore.metrics } });
  post({ type: 'dl', which: 'model', status: 'done' });
  return out;
}

// ── TTFT 계측 전용 스트리머 ─────────────────────────────────────────────────
// TextStreamer 는 토큰마다 decode 를 돌린다. 그 비용이 TTFT·총시간에 섞이면
// 측정값이 오염되므로, 타임스탬프만 찍는 최소 구현을 쓴다.
class TimingStreamer extends BaseStreamer {
  constructor() { super(); this.reset(); }
  reset() { this.t0 = 0; this.ttft = null; this.isPrompt = true; }
  start() { this.reset(); this.t0 = performance.now(); }
  put(value) {
    if (this.isPrompt) { this.isPrompt = false; return; }  // 프롬프트 통째로 1회
    if (this.ttft === null) this.ttft = performance.now() - this.t0;
  }
  end() {}
}

let tokenizer = null, model = null, rouge1 = null;
let chatTemplate = null, rows = null;
let device = 'wasm', aborted = false;
const stopper = new InterruptableStoppingCriteria();

// ── 옛 REVISION 캐시 정리 ───────────────────────────────────────────────────
// 이전 로더가 Cache Storage에 저장한 옛 revision을 정리한다.
// 현재 revision의 파일은 mountWeights에서 OPFS로 검증·이전한 뒤 삭제한다.
// 이 레포 항목만 건드리고 다른 레포·앱 항목은 두지 않는다.
async function pruneStaleCache() {
  if (typeof caches === 'undefined') return;
  try {
    const cache = await caches.open(env.cacheKey);
    const stale = (await cache.keys()).filter(({ url }) =>
      url.includes(`/${REPO}/`) && !url.includes(`/${REVISION}/`));
    await Promise.all(stale.map((req) => cache.delete(req)));
    if (stale.length) console.info(`옛 REVISION 캐시 ${stale.length}개 삭제`);
  } catch (e) {
    // 시크릿 모드·iframe 등에서 open 이 거부될 수 있다. 캐시 정리는 부가 기능이므로 넘어간다.
    console.warn('캐시 정리 건너뜀:', e);
  }
}

// ── 로드 ────────────────────────────────────────────────────────────────────
// A failed WebGPU session is not retried in a heap that may retain allocations.

async function load({ device: preferred, stagingMiB = 8 }) {
  const loadStarted = performance.now(), timings = {};
  if (preferred !== 'webgpu') throw new Error('이 FP32 메모리 실험은 WebGPU가 필요합니다.');
  env.useBrowserCache = true;   // 그래프(1 MB)·설정·토크나이저만 transformers.js 가 캐시한다. 가중치는 여기서.
  await pruneStaleCache();

  // 모바일 빌드 자체가 단일 스레드이며 런타임 설정도 일치시킨다.
  try {
    env.backends.onnx.wasm.numThreads = 1;
  } catch { /* ORT 백엔드가 아직 준비되지 않았으면 기본값을 쓴다 */ }

  const progress_callback = (p) => {
    const f = String(p.file ?? '');
    const which = /(^|\/)model\.onnx$/.test(f) ? 'graph' : f.endsWith('tokenizer.json') ? 'tok' : null;
    if (!which) return;
    // 그래프 파일(1 MB)이 오고 나면 콜백 없는 구간(세션 생성)만 남는다. 문구로 단계를 드러낸다.
    if (which === 'graph' && p.status === 'done') {
      post({ type: 'phase', text: 'ORT 세션 생성 중… (진행률 없음 · 메모리 최대 구간)' });
    }
    if (which !== 'tok' || (p.status !== 'progress' && p.status !== 'done')) return;
    post({ type: 'dl', which, status: p.status, loaded: p.loaded, total: p.total });
  };

  post({ type: 'phase', text: '그래프와 initializer 메타데이터 확인 중…' });
  const manifestResponse = await fetch(new URL('../../model/initializers.json', import.meta.url));
  if (!manifestResponse.ok) throw new Error('initializer manifest missing; run tools/inspect_initializers.py');
  const manifest = await manifestResponse.json();
  if (manifest.revision !== REVISION) throw new Error('Manifest/model revision mismatch');
  const graphResponse = await baseFetch(resolveUrl(GRAPH));
  if (!graphResponse.ok) throw new Error(`Graph HTTP ${graphResponse.status}`);
  verifiedGraph = new Uint8Array(await graphResponse.arrayBuffer()); // graph only (~0.96 MB)
  const hash = Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256', verifiedGraph)),
    b => b.toString(16).padStart(2, '0')).join('');
  if (hash !== manifest.graphSha256) throw new Error('Graph/initializer manifest SHA-256 mismatch');
  await journal.checkpoint({ stage: 'graph-verified', graphSha256: hash, revision: REVISION, graphBytes: verifiedGraph.byteLength,
    expectedInitializerCount: manifest.initializers.filter(t => t.location).length,
    expectedGpuResidentBytes: manifest.totalExternalTensorBytes });
  console.info('[MODEL] graph loaded', { bytes: verifiedGraph.byteLength, sha256: hash });
  timings.graphPreparationMs = performance.now() - loadStarted;
  // Ensure from_pretrained consumes the verified graph via env.fetch rather
  // than bypassing it with a separately cached graph response.
  env.useBrowserCache = false;

  const preparationStarted = performance.now();
  const externalData = await mountWeights(manifest);
  timings.weightPreparationMs = performance.now() - preparationStarted;
  const tracked = trackedGpu = await installGpuTracking(manifest.largestInitializerBytes, record => {
    if (trace) console.info('[GPU]', record);
    if (['gpu-uncaptured-error', 'gpu-error', 'device-lost'].includes(record.stage)) return journal.checkpoint({
      ...record,
      gpuLedger: trackedGpu ? { ...trackedGpu.ledger } : null });
  }, { context: () => gpuOperationContext(journal.state.last) });
  const started = performance.now();
  const loader = new SessionRangeLoader({
    manifest, stagingMiB, signal: loadController.signal,
    gpuTracker: tracked,
    storage: () => weightStore.metrics,
    checkpoint: record => journal.checkpoint(record),
    emit: record => {
      if (trace) console.info(`[${record.stage.startsWith('gpu') ? 'GPU' : record.stage.startsWith('wasm') ? 'WASM' :
        record.stage.startsWith('cpu') ? 'CPU' : record.stage.startsWith('range') ? 'EXT' :
        record.stage.startsWith('initializer') ? 'INIT' : 'SESSION'}]`, record);
      if (record.stage === 'initializer-start' || record.stage === 'initializer-complete' || record.stage === 'device-lost') {
        post({ type: 'diagnostic', record });
      }
    },
  });
  globalThis.__ortExternalTensorLoader = loader;
  await journal.checkpoint({ stage: 'session-create', runtimeMode, stagingMiB, timings: { ...timings }, storage: { ...weightStore.metrics } });
  let success = false;
  const modelCallStarted = performance.now();
  try {
    model = await AutoModelForCausalLM.from_pretrained(REPO, {
      revision: REVISION, subfolder: '', dtype: 'fp32', device: 'webgpu',
      use_external_data_format: false, progress_callback,
      session_options: { externalData, graphOptimizationLevel: 'disabled',
        executionProviders: ['webgpu'],
        enableCpuMemArena: false, enableMemPattern: false },
    });
    timings.modelLoadCallMs = performance.now() - modelCallStarted;
    await tracked.flush();
    if (tracked.ledger.lastError) throw new Error(`WebGPU: ${tracked.ledger.lastError}`);
    if (loader.metrics.loadedInitializerCount !== manifest.initializers.filter(t => t.location).length) throw new Error('Incomplete external initializer load');
    loadController.signal.throwIfAborted();
    success = true;
  } finally {
    timings.modelLoadCallMs ??= performance.now() - modelCallStarted;
    await tracked.flush();
    if (!success) tracked.restore();
    await loader.lossSaved;
    const metrics = loader.close(success);
    weightStore.close();
    const result = { stage: success ? 'session-create-complete' : 'session-create-failed',
      runtimeMode, stagingMiB, durationMs: performance.now() - started, metrics,
      gpuLedger: { ...tracked.ledger }, storage: { ...weightStore.metrics }, lastInitializer: loader.last };
    result.timings = { ...timings };
    sessionMetrics = result;
    await journal.checkpoint(result);
    post({ type: 'session-result', result });
    // The sealed loader stays installed: a later weight read is an error.
    externalData.length = 0;
    verifiedGraph = null;
  }
  env.useBrowserCache = true;
  const tokenizerStarted = performance.now();
  await journal.checkpoint({ stage: 'tokenizer-load', metrics: loader.sampleMetrics(), gpuLedger: { ...tracked.ledger } });
  post({ type: 'phase', text: '토크나이저 내려받는 중…' });
  [tokenizer, chatTemplate, rows] = await Promise.all([
    AutoTokenizer.from_pretrained(REPO, { revision: REVISION, progress_callback }),
    // chat_template 은 tokenizer_config.json 에 없고, AutoTokenizer 는 .jinja 를
    // 받아오지 않는다(그 경로는 Processor 전용). 직접 읽어서 명시적으로 넘긴다.
    fetch(new URL('../../tokenizer/chat_template.jinja', import.meta.url)).then(r => r.text()),
    fetch(new URL('./data.jsonl', import.meta.url)).then(r => r.text()).then(t =>
      t.split('\n').filter(Boolean).map(JSON.parse)),
  ]);
  rouge1 = makeRouge1(tokenizer);
  timings.tokenizerPreparationMs = performance.now() - tokenizerStarted;
  timings.totalLoadMs = performance.now() - loadStarted;
  sessionMetrics.timings = { ...timings };

  loadController.signal.throwIfAborted();
  await tracked.flush();
  if (tracked.ledger.lastError) throw new Error(`WebGPU: ${tracked.ledger.lastError}`);
  device = 'webgpu';
  await journal.finish('ready', sessionMetrics);
  post({ type: 'ready', device, dtype: 'fp32', rows: rows.length });
}

// ── 한 행 실행 ──────────────────────────────────────────────────────────────
const streamer = new TimingStreamer();

async function generateOwned(inputs, options) {
  try { return await model.generate({ ...inputs, ...options }); }
  finally { for (const value of new Set(Object.values(inputs))) value?.dispose?.(); }
}

async function recordMemory(stage, details = {}) {
  const loader = globalThis.__ortExternalTensorLoader;
  await trackedGpu.device.queue.onSubmittedWorkDone();
  await trackedGpu.flush();
  if (trackedGpu.ledger.lastError) throw new Error(`WebGPU: ${trackedGpu.ledger.lastError}`);
  if (loader.metrics.rangeReadCount !== sessionMetrics.metrics.rangeReadCount) throw new Error('Weights reloaded during inference');
  await journal.checkpoint({ stage, ...details, metrics: loader.sampleMetrics(), gpuLedger: { ...trackedGpu.ledger } });
}

async function runRow(row) {
  const msgs = row.messages;
  const ctx = msgs.slice(0, -1);          // 마지막 assistant 직전까지가 문맥
  const ref = msgs.at(-1).content;        // 마지막 assistant 가 정답

  const inputs = tokenizer.apply_chat_template(ctx, {
    chat_template: chatTemplate,
    add_generation_prompt: true,
    return_dict: true,
  });
  const promptLen = inputs.input_ids.dims.at(-1);

  stopper.reset();
  streamer.start();
  const t0 = performance.now();
  const out = await generateOwned(inputs, {
    ...GEN,
    streamer,
    stopping_criteria: stopper,
  });
  const total = performance.now() - t0;

  let all;
  try { all = out.tolist()[0].map(Number); }
  finally { out.dispose(); }
  const gen = all.slice(promptLen);
  const eos = gen.length > 0 && EOS.has(gen.at(-1));
  const text = tokenizer.decode(gen, { skip_special_tokens: true }).trim();

  // EOS 토큰 자체는 산출물이 아니므로 속도 계산에서 뺀다.
  const nTok = eos ? gen.length - 1 : gen.length;
  const ttft = streamer.ttft ?? total;
  const tps = nTok > 1 && total > ttft ? ((nTok - 1) / (total - ttft)) * 1000 : 0;

  return {
    turns: msgs.length / 2, promptLen, nTok, eos, ttft, total, tps,
    rouge: rouge1(text, ref), pred: text, ref,
  };
}

// ── 전체 평가 ───────────────────────────────────────────────────────────────
async function runAll() {
  if (aborted) { await journal.finish('cancelled', { completedRows: 0 }); post({ type: 'aborted', at: 0 }); return; }
  random.seed(SEED);   // 동일 시드 → 동일 결과. do_sample 이라 이게 없으면 매번 달라진다.

  // 워밍업: 첫 추론에는 ORT 커널 컴파일이 섞인다. 1행 TTFT 가 혼자 튀지 않게 버린다.
  post({ type: 'phase', text: '워밍업 중…' });
  stopper.reset();
  await recordMemory('warmup-start');
  const warmup = await generateOwned(tokenizer.apply_chat_template(rows[0].messages.slice(0, -1), {
      chat_template: chatTemplate, add_generation_prompt: true, return_dict: true,
    }),
    { do_sample: false, max_new_tokens: 4, stopping_criteria: stopper,
  });
  warmup.dispose();
  await recordMemory('warmup-complete');
  random.seed(SEED);   // 워밍업이 소비한 난수를 되돌린다.

  const wall = performance.now();
  const done = [];
  for (let i = 0; i < rows.length; ++i) {
    if (aborted) { await journal.finish('cancelled', { completedRows: i }); post({ type: 'aborted', at: i }); return; }
    post({ type: 'phase', text: `평가 중… ${i + 1}/${rows.length}` });
    await recordMemory('row-start', { row: i + 1 });
    const t0 = performance.now();
    try {
      const r = await runRow(rows[i]);
      if (aborted) { await journal.finish('cancelled', { completedRows: i }); post({ type: 'aborted', at: i }); return; }
      done.push(r);
      await recordMemory('row-complete', { row: i + 1, promptLen: r.promptLen, nTok: r.nTok });
      post({ type: 'row', i, r, progress: (i + 1) / rows.length });
    } catch (e) {
      // 실패 위치를 표시하고 종료한다. 불완전한 실행의 평균은 표시하지 않는다.
      const elapsed = performance.now() - t0;
      const error = String(e?.stack ?? e?.message ?? e);
      done.push({
        turns: rows[i].messages.length / 2, promptLen: 0, nTok: 0, eos: false,
        ttft: elapsed, total: elapsed, tps: 0,
        rouge: { p: 0, r: 0, f1: 0 }, pred: '', ref: rows[i].messages.at(-1).content,
        failed: true, error,
      });
      post({ type: 'row', i, error, progress: (i + 1) / rows.length });
      // generate() does not expose every intermediate GPU tensor on failure.
      // End this worker instead of continuing in an uncertain native allocator.
      throw e;
    }
  }
  if (aborted) { await journal.finish('cancelled', { completedRows: rows.length }); post({ type: 'aborted', at: rows.length }); return; }

  // done.length === rows.length. 512 상한에 걸린 행도 포함한 전체 평균.
  const avg = (f) => done.reduce((s, r) => s + f(r), 0) / done.length;
  const result = {
    type: 'done',
    n: done.length, total: rows.length,
    failed: done.filter(r => r.failed).length,
    ttft: avg(r => r.ttft), totalMs: avg(r => r.total), tps: avg(r => r.tps),
    p: avg(r => r.rouge.p), r: avg(r => r.rouge.r), f1: avg(r => r.rouge.f1),
    eos: done.filter(r => r.eos).length,
    wall: performance.now() - wall,
  };
  await journal.finish('complete', { result, gpuLedger: { ...trackedGpu.ledger },
    rows: done.map(({ pred, ref, ...metrics }) => metrics) });
  post(result);
}

async function runProbe(maxNewTokens = 32) {
  if (!Number.isInteger(maxNewTokens) || maxNewTokens < 1 || maxNewTokens > 32) throw new Error('Probe token count must be 1–32');
  const lengths = rows.map((row, index) => {
    const inputs = tokenizer.apply_chat_template(row.messages.slice(0, -1), {
      chat_template: chatTemplate, add_generation_prompt: true, return_dict: true });
    const length = inputs.input_ids.dims.at(-1);
    for (const tensor of Object.values(inputs)) tensor.dispose();
    return { index, length };
  }).sort((a, b) => a.length - b.length);
  const outputs = [];
  for (const { index, length } of [lengths[0], lengths.at(-1)]) {
    if (aborted) break;
    stopper.reset();
    await recordMemory('probe-inference-start', { row: index + 1, promptLen: length });
    post({ type: 'phase', text: `짧은 추론 확인 중… 입력 ${length}토큰` });
    const inputs = tokenizer.apply_chat_template(rows[index].messages.slice(0, -1), {
      chat_template: chatTemplate, add_generation_prompt: true, return_dict: true });
    const start = performance.now();
    const result = await generateOwned(inputs, { do_sample: false, max_new_tokens: maxNewTokens, stopping_criteria: stopper });
    try { outputs.push({ row: index + 1, promptLen: length, durationMs: performance.now() - start,
      tokens: result.tolist()[0].slice(length).map(Number) }); }
    finally { result.dispose(); }
    await recordMemory('probe-inference-complete', { row: index + 1 });
  }
  const result = { success: !aborted, outputs, gpuLedger: { ...trackedGpu.ledger }, sessionMetrics };
  await journal.finish(aborted ? 'cancelled' : 'complete', result);
  post({ type: 'probe-result', result });
}

self.onmessage = async ({ data }) => {
  if (data.type === 'stop') { aborted = true; loadController?.abort(); stopper.interrupt(); return; }
  if (data.type === 'dispose') {
    if (operation) { queuedRun = data; return; }
    operation = 'dispose';
    try {
      await model?.dispose(); model = null;
      for (const gpuDevice of trackedGpu?.devices || []) await gpuDevice.queue.onSubmittedWorkDone();
      await trackedGpu?.flush();
      for (const gpuDevice of trackedGpu?.devices || []) gpuDevice.destroy();
      trackedGpu?.restore();
      post({ type: 'disposed' });
    } catch (error) { post({ type: 'disposed', error: String(error) }); }
    finally { trackedGpu?.restore(); operation = null; }
    return;
  }
  if (!['load', 'run', 'probe'].includes(data.type)) return;
  if (operation) { if (data.type !== 'load') queuedRun = data; return; }
  operation = data.type;
  aborted = false;
  if (operation === 'load') loadController = new AbortController();
  journal = new RunDiagnostics(data.runId || newRunId(), { ...data.environment, userAgent: navigator.userAgent, runtimeMode,
    stagingMiB: operation === 'load' ? data.stagingMiB ?? 8 : sessionMetrics?.stagingMiB ?? 8, workerURL: self.location.href });
  try {
    journal.state.environment.build = buildIdentity(runtimeBuild);
    await journal.checkpoint({ stage: `${operation}-start` });
    if (data.type === 'load') {
      if (model) throw new Error('A model is already loaded in this worker');
      if (runtimeMode === 'stock') throw new Error('Production loading requires the OPFS range runtime. Use experiments for stock comparisons.');
      if (runtimeBuild.rangeLoaderVersion !== 2) throw new Error('런타임을 새로 빌드해야 합니다: range-loader ABI 2 필요');
      const execute = async lock => {
        if (!lock) throw new Error('다른 탭에서 모델을 준비 중입니다. 해당 작업이 끝난 후 다시 시도해 주세요.');
        await load(data);
      };
      if (navigator.locks) await navigator.locks.request('didimdol-model-load', { ifAvailable: true }, execute);
      else await execute(true);
    } else {
      if (!model) throw new Error('Model is not loaded');
      if (data.type === 'probe') await runProbe(data.maxNewTokens);
      else await runAll();
    }
  } catch (err) {
    const cancelled = data.type === 'load' && loadController?.signal.aborted;
    await trackedGpu?.flush();
    trackedGpu?.restore();
    await journal.finish(cancelled ? 'cancelled' : 'failed', { error: String(err?.stack ?? err), sessionMetrics });
    // A failed create/generate may retain native allocations. Use a fresh worker/page.
    post({ type: 'fatal', cancelled, error: cancelled ? '모델 준비를 중단했습니다. 페이지를 새로 열어 다시 시작해 주세요.' : String(err?.stack ?? err) });
  } finally {
    weightStore?.close();
    operation = null;
    if (queuedRun) { const next = queuedRun; queuedRun = null; queueMicrotask(() => self.onmessage({ data: next })); }
  }
};
post({ type: 'worker-ready' });
