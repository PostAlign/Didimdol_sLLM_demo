/** FP32 WebGPU evaluation worker. Range loading is only permitted during
 * OrtCreateSession. See docs/session-create-memory.md for the source audit. */

const runtimeMode = new URL(self.location.href).searchParams.get('ortMode') || 'asyncify';
const { AutoModelForCausalLM, AutoTokenizer, BaseStreamer,
  InterruptableStoppingCriteria, env, random } = await import(`./runtime.js?mode=${runtimeMode}`);
import { SessionRangeLoader } from './range-loader.js';
import { saveCheckpoint } from './diagnostics.js';
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
const isWeights = (name) => /(^|\/)model\.onnx_data(_\d+)?$/.test(name.split('?')[0]);
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

// ── 가중치 파일 목록 ────────────────────────────────────────────────────────
// 파일 수는 build_web_models.py 의 FILE_CAP 에 따라 달라지므로 코드에 박지 않고 레포에서 읽는다.
// HF tree API 한 번이면 이름과 크기(진행률 분모)가 나온다. API 가 막히면 HEAD 로 하나씩 더듬는다.
async function listWeightFiles() {
  const at = `${REPO}@${REVISION.slice(0, 7)}`;
  let entries = null;
  try {
    const r = await baseFetch(`https://huggingface.co/api/models/${REPO}/tree/${REVISION}`);
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    entries = await r.json();
  } catch (e) {
    console.warn('tree API 실패, HEAD 로 파일을 더듬는다:', e);
  }
  let files;
  if (entries) {
    if (!entries.some((f) => f.type === 'file' && f.path === GRAPH)) {
      throw new Error(`${GRAPH} 이(가) 레포 ${at} 에 없습니다`);
    }
    files = entries.filter((f) => f.type === 'file' && isWeights(f.path))
                   .map((f) => ({ name: f.path, size: f.lfs?.size ?? f.size ?? 0 }));
  } else {
    files = [];
    for (let i = 0; i < 64; ++i) {
      const name = i === 0 ? 'model.onnx_data' : `model.onnx_data_${i}`;
      const r = await baseFetch(resolveUrl(name), { method: 'HEAD' });
      if (!r.ok) break;
      files.push({ name, size: Number(r.headers.get('content-length')) || 0 });
    }
  }
  if (!files.length) throw new Error(`model.onnx_data 파일이 레포 ${at} 에 없습니다`);
  return sortWeights(files);
}
const chunkNo = (name) => Number(name.match(/_data(?:_(\d+))?$/)?.[1] ?? 0);
const sortWeights = (files) => files.sort((a, b) => chunkNo(a.name) - chunkNo(b.name));

// ── 가중치 파일 하나 → Blob ───────────────────────────────────────────────────
// 진행률은 transformers.js 를 거치지 않으므로 청크를 세어 직접 보낸다. onBytes(loaded) 는 이 파일 안의
// 누적 바이트다. 캐시 저장이 중간에 실패하면 다시 받으므로 0 부터 다시 센다.
function counted(res, onBytes) {
  const reader = res.body.getReader();
  let loaded = 0;
  const body = new ReadableStream({
    async pull(ctrl) {
      const { done, value } = await reader.read();
      if (done) { ctrl.close(); return; }
      loaded += value.byteLength;
      onBytes(loaded);
      ctrl.enqueue(value);
    },
    cancel(reason) { reader.cancel(reason).catch(() => {}); },
  });
  return new Response(body, { status: 200, headers: res.headers });
}

async function weightBlob(file, onBytes) {
  const url = resolveUrl(file.name);
  let cache = null;
  try { cache = await caches.open(env.cacheKey); } catch { /* 시크릿 모드 등: 캐시 없이 받는다 */ }
  if (cache) {
    const hit = await cache.match(url);
    if (hit) return { blob: await hit.blob(), stage: 'cache' };
  }
  const fetchOk = async () => {
    const r = await baseFetch(url);
    if (!r.ok || !r.body) throw new Error(`${file.name}: HTTP ${r.status}`);
    return r;
  };
  let res = await fetchOk();
  if (cache) {
    try {
      await cache.put(url, counted(res, onBytes));
      const hit = await cache.match(url);
      if (hit) return { blob: await hit.blob(), stage: 'net' };
      // The response was consumed by cache.put; fetch a new stream if evicted immediately.
      res = await fetchOk();
    } catch (e) {
      // 쿼터 초과 등. 스트림은 이미 소비됐으므로 새로 받는다. 이번 방문에서만 쓰고 저장하지 않는다.
      console.warn(`${file.name} 캐시 저장 실패, 저장 없이 다시 받는다:`, e);
      onBytes(0);
      res = await fetchOk();
    }
  }
  return { blob: await counted(res, onBytes).blob(), stage: 'net' };
}

// Stock comparison preserves the old BlobFile path without eager full-file
// materialization. Production passes real Blob objects to the patched API.
class StockBlobFile extends Uint8Array {
  constructor(blob) { super(0); this.blob = blob; this.reader = new FileReaderSync(); }
  get byteLength() { return this.blob.size; }
  get length() { return this.blob.size; }
  subarray(begin = 0, end = this.blob.size) {
    return new Uint8Array(this.reader.readAsArrayBuffer(this.blob.slice(begin, end)));
  }
}
let weightFiles = null;
let weightBlobs = null;
async function mountWeights(manifest) {
  weightFiles ??= await listWeightFiles();
  weightBlobs ??= {};
  const total = weightFiles.reduce((sum, file) => sum + file.size, 0);
  let before = 0;
  const out = [];
  for (const [i, file] of weightFiles.entries()) {
    post({ type: 'phase', text: `가중치 ${i + 1}/${weightFiles.length} 준비 중…` });
    const got = await weightBlob(file, loaded => post({ type: 'dl', which: 'model', status: 'progress',
      loaded: before + loaded, total, stage: 'net' }));
    const blob = got.blob;
    const expected = manifest.files.find(f => f.location === file.name);
    if (!expected || blob.size < expected.minimumBytes || (file.size && blob.size !== file.size)) {
      throw new Error(`가중치 크기 검증 실패: ${file.name}`);
    }
    weightBlobs[file.name] = blob;
    before += blob.size;
    out.push({ path: file.name, data: runtimeMode === 'stock' ? new StockBlobFile(blob) : blob });
    post({ type: 'dl', which: 'model', status: 'progress', loaded: before, total, stage: got.stage });
  }
  if (manifest.files.some(f => !weightBlobs[f.location])) throw new Error('외부 데이터 파일 누락');
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
// 받은 파일은 Cache API 의 env.cacheKey 저장소에 원격 URL 을 키로 들어간다 (transformers.js 와
// 위 weightBlob 모두). REVISION 을 갈면 새 URL 로 다시 받지만 옛 SHA 의 가중치는
// 그대로 남는다. 새 파일을 받기 *전에* 지워야 한다 — Safari 처럼 용량이 빡빡한 곳에서 옛 1 GB 가
// 남아 있으면 새 1 GB 의 cache.put 이 QuotaExceeded 로 실패하고, 그러면 다음 방문에
// 또 재다운로드하게 된다. 이 레포 항목만 건드리고 다른 레포·앱 항목은 두지 않는다.
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

async function load({ device: preferred, stagingMiB = 16 }) {
  if (preferred !== 'webgpu') throw new Error('이 FP32 메모리 실험은 WebGPU가 필요합니다.');
  env.useBrowserCache = true;   // 그래프(1 MB)·설정·토크나이저만 transformers.js 가 캐시한다. 가중치는 여기서.
  await pruneStaleCache();

  // COOP/COEP 가 없으면 SharedArrayBuffer 를 못 써서 ORT 가 어차피 싱글스레드로 떨어진다.
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
  console.info('[MODEL] graph loaded', { bytes: verifiedGraph.byteLength, sha256: hash });
  // Ensure from_pretrained consumes the verified graph via env.fetch rather
  // than bypassing it with a separately cached graph response.
  env.useBrowserCache = false;

  const externalData = await mountWeights(manifest);
  const tracked = await installGpuTracking(manifest.largestInitializerBytes, record => console.info('[GPU]', record));
  const started = performance.now();
  const loader = runtimeMode === 'stock' ? null : new SessionRangeLoader({
    manifest, stagingMiB,
    checkpoint: saveCheckpoint,
    emit: record => {
      console.info(`[${record.stage.startsWith('gpu') ? 'GPU' : record.stage.startsWith('wasm') ? 'WASM' :
        record.stage.startsWith('cpu') ? 'CPU' : record.stage.startsWith('range') ? 'EXT' :
        record.stage.startsWith('initializer') ? 'INIT' : 'SESSION'}]`, record);
      if (record.stage === 'initializer-start' || record.stage === 'initializer-complete' || record.stage === 'device-lost') {
        post({ type: 'diagnostic', record });
      }
    },
  });
  if (loader) globalThis.__ortExternalTensorLoader = loader;
  await saveCheckpoint({ stage: 'session-create', runtimeMode, stagingMiB });
  let success = false;
  try {
    model = await AutoModelForCausalLM.from_pretrained(REPO, {
      revision: REVISION, subfolder: '', dtype: 'fp32', device: 'webgpu',
      use_external_data_format: false, progress_callback,
      session_options: { externalData, graphOptimizationLevel: 'disabled',
        executionProviders: ['webgpu'],
        enableCpuMemArena: false, enableMemPattern: false },
    });
    success = true;
  } finally {
    tracked.restore();
    const metrics = loader?.close(success) ?? null;
    const result = { stage: success ? 'session-create-complete' : 'session-create-failed',
      runtimeMode, stagingMiB, durationMs: performance.now() - started, metrics,
      gpuLedger: tracked.ledger, lastInitializer: loader?.last };
    await saveCheckpoint(result);
    post({ type: 'session-result', result });
    // The sealed loader stays installed: a later weight read is an error.
    externalData.length = 0;
    weightBlobs = null;
    verifiedGraph = null;
  }
  device = 'webgpu';
  post({ type: 'ready', device, dtype: 'fp32', rows: rows.length });
}

// ── 한 행 실행 ──────────────────────────────────────────────────────────────
const streamer = new TimingStreamer();

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
  const out = await model.generate({
    input_ids: inputs.input_ids,
    attention_mask: inputs.attention_mask,
    ...GEN,
    streamer,
    stopping_criteria: stopper,
  });
  const total = performance.now() - t0;

  const all = out.tolist()[0].map(Number);
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
  aborted = false;
  random.seed(SEED);   // 동일 시드 → 동일 결과. do_sample 이라 이게 없으면 매번 달라진다.

  // 워밍업: 첫 추론에는 ORT 커널 컴파일이 섞인다. 1행 TTFT 가 혼자 튀지 않게 버린다.
  post({ type: 'phase', text: '워밍업 중…' });
  await model.generate({
    ...tokenizer.apply_chat_template(rows[0].messages.slice(0, -1), {
      chat_template: chatTemplate, add_generation_prompt: true, return_dict: true,
    }),
    do_sample: false, max_new_tokens: 4,
  });
  random.seed(SEED);   // 워밍업이 소비한 난수를 되돌린다.

  const wall = performance.now();
  const done = [];
  for (let i = 0; i < rows.length; ++i) {
    if (aborted) { post({ type: 'aborted', at: i }); return; }
    post({ type: 'phase', text: `평가 중… ${i + 1}/${rows.length}` });
    const t0 = performance.now();
    try {
      const r = await runRow(rows[i]);
      done.push(r);
      post({ type: 'row', i, r, progress: (i + 1) / rows.length });
    } catch (e) {
      // 실패한 행도 분모에 넣는다. 평균은 항상 전체 행 기준이어야 하므로
      // 산출물 없음(ROUGE 0, EOS 아님, 속도 0)으로 기록하고 시간은 실패까지 실제 걸린 만큼 잡는다.
      const elapsed = performance.now() - t0;
      const error = String(e?.stack ?? e?.message ?? e);
      done.push({
        turns: rows[i].messages.length / 2, promptLen: 0, nTok: 0, eos: false,
        ttft: elapsed, total: elapsed, tps: 0,
        rouge: { p: 0, r: 0, f1: 0 }, pred: '', ref: rows[i].messages.at(-1).content,
        failed: true, error,
      });
      post({ type: 'row', i, error, progress: (i + 1) / rows.length });
    }
  }
  if (aborted) { post({ type: 'aborted', at: rows.length }); return; }

  // done.length === rows.length. 512 상한에 걸린 행, 예외로 실패한 행 모두 포함한 전체 평균.
  const avg = (f) => done.reduce((s, r) => s + f(r), 0) / done.length;
  post({
    type: 'done',
    n: done.length, total: rows.length,
    failed: done.filter(r => r.failed).length,
    ttft: avg(r => r.ttft), totalMs: avg(r => r.total), tps: avg(r => r.tps),
    p: avg(r => r.rouge.p), r: avg(r => r.rouge.r), f1: avg(r => r.rouge.f1),
    eos: done.filter(r => r.eos).length,
    wall: performance.now() - wall,
  });
}

self.onmessage = async (e) => {
  const { type } = e.data;
  try {
    if (type === 'load') await load(e.data);
    else if (type === 'run') await runAll();
    else if (type === 'stop') { aborted = true; stopper.interrupt(); }
  } catch (err) {
    post({ type: 'fatal', error: String(err?.stack ?? err) });
  }
};
post({ type: 'worker-ready' });
