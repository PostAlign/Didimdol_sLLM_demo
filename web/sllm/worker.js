/**
 * 평가 워커 — 모델 로드 · 생성 · 채점을 전부 여기서 한다.
 *
 * 메인 스레드에서 돌리면 100행 × 최대 512토큰 디코드가 UI 를 얼려서
 * "스크롤하며 결과 보기" 자체가 성립하지 않는다.
 *
 * 파일 출처가 둘로 갈린다.
 *   model.onnx / model.onnx_data* → Hugging Face (커밋 SHA 고정)
 *   그 외 config/토크나이저       → 이 정적 사이트
 * transformers.js 는 한 곳에서만 받아오므로 env.fetch 를 후킹해 갈라 보낸다.
 *
 * 정밀도는 fp32 하나다. fp16 은 품질이 평가 기준이 못 되어 폐기했다.
 *
 * ── 가중치는 transformers.js 에 맡기지 않고 여기서 직접 다룬다 ──────────────────────
 * transformers.js 의 로더는 가중치 파일을 통째로 JS 버퍼(1.07 GB)에 읽은 뒤 ORT 에 넘긴다.
 * WASM 이면 ORT 가 그걸 텐서마다 wasm 힙으로 복사하므로 같은 바이트가 두 벌(2.1 GB)이 되고,
 * WebGPU 라도 세션이 만들어질 때까지 1 GB 버퍼가 살아 있다. 스마트폰은 여기서 죽는다.
 *
 * 대신 이렇게 한다 (build_web_models.py 가 가중치를 128 MiB 이하 파일 여러 개로 나눠 둔다).
 *   ① 파일마다: Cache API 에 있으면 그대로, 없으면 네트워크 스트림을 cache.put 에 흘려 넣는다.
 *      JS 힙에 버퍼를 만들지 않는다. 캐시가 안 되면 그냥 스트림으로 Blob 을 만든다.
 *   ② 파일마다 Response.blob() 으로 Blob 을 얻는다. Blob 은 브라우저가 디스크/브라우저 프로세스에
 *      두므로 워커 힙에는 없다.
 *   ③ ORT 에는 Blob 을 감싼 BlobFile 을 외부 데이터로 마운트한다. ORT 웹의 외부 데이터 로더는
 *      마운트된 파일에서 byteLength 와 subarray(offset, offset+len) 만 쓰므로, subarray 를
 *      FileReaderSync 로 그 구간만 동기적으로 읽어 돌려주면 텐서 하나 분량만 메모리를 지난다.
 *      WebGPU 는 그 조각을 GPU 버퍼에 바로 올리고, WASM 은 wasm 힙에 복사한다.
 * 그래서 로드 중 워커 힙의 피크는 "가장 큰 텐서(41.9 MB) 한 개" 수준이다. 가중치 총량은 GPU(또는
 * wasm 힙)에 한 벌만 있다.
 *
 * ORT 로더가 mountedFile.byteLength / .subarray 만 쓴다는 건 transformers.js 4.2.0 이 묶은
 * onnxruntime-web 1.26.0-dev.20260416 의 asyncify 빌드 기준이다. transformers.js 를 올리면 다시 확인한다.
 * 만약 그 경로가 깨지면(외부 데이터 오류) 옛 방식(파일을 통째로 버퍼에 읽기)으로 한 번 더 시도한다.
 */

import {
  AutoModelForCausalLM, AutoTokenizer, BaseStreamer,
  InterruptableStoppingCriteria, env, random,
} from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.2.0';

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

const baseFetch = env.fetch ?? fetch;
env.fetch = (input, init) => {
  const url = typeof input === 'string' ? input : (input?.url ?? String(input));
  if (url.includes('huggingface.co')) {
    const name = url.split('?')[0].split('/').pop();
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
    } catch (e) {
      // 쿼터 초과 등. 스트림은 이미 소비됐으므로 새로 받는다. 이번 방문에서만 쓰고 저장하지 않는다.
      console.warn(`${file.name} 캐시 저장 실패, 저장 없이 다시 받는다:`, e);
      onBytes(0);
      res = await fetchOk();
    }
  }
  return { blob: await counted(res, onBytes).blob(), stage: 'net' };
}

// ── ORT 외부 데이터로 마운트할 Blob 래퍼 ───────────────────────────────────────
// ORT 웹 로더는 마운트된 파일에서 byteLength 와 subarray 만 쓴다 (파일 머리말 참고).
// Uint8Array 를 상속해야 onnxruntime-web 의 loadFile 이 통과시킨다 (아니면 new Uint8Array(obj) 로
// 통째로 복사하려 든다). 본체는 길이 0 이고, 실제 바이트는 subarray 가 Blob 에서 그 구간만 읽는다.
class BlobFile extends Uint8Array {
  constructor(blob) { super(0); this.blob = blob; this.reader = new FileReaderSync(); }
  get byteLength() { return this.blob.size; }
  get length() { return this.blob.size; }
  subarray(begin = 0, end = this.blob.size) {
    return new Uint8Array(this.reader.readAsArrayBuffer(this.blob.slice(begin, end)));
  }
}

// 가중치 전부를 [{ path, data }] 로. buffered 면 옛 방식(파일을 통째로 버퍼에) — BlobFile 경로가
// 깨졌을 때의 마지막 수단이다. FileReaderSync 가 없는 환경(사실상 없음)도 여기로 온다.
let weightFiles = null;   // listWeightFiles() 결과. 시도마다 다시 조회하지 않는다.
let weightBlobs = null;   // { name → Blob }. WASM 폴백 때 다시 받지 않는다.

async function mountWeights({ buffered }) {
  weightFiles ??= await listWeightFiles();
  weightBlobs ??= {};
  const total = weightFiles.reduce((s, f) => s + f.size, 0);
  let before = 0;
  const out = [];
  for (const [i, file] of weightFiles.entries()) {
    let blob = weightBlobs[file.name];
    if (!blob) {
      let stage = 'net';
      const onBytes = (loaded) => post({ type: 'dl', which: 'model', status: 'progress',
                                         loaded: before + loaded, total, stage });
      post({ type: 'phase', text: `가중치 ${i + 1}/${weightFiles.length} 준비 중…` });
      const got = await weightBlob(file, onBytes);
      blob = weightBlobs[file.name] = got.blob;
      stage = got.stage;
      post({ type: 'dl', which: 'model', status: 'progress', loaded: before + blob.size, total, stage });
    }
    before += blob.size;
    const data = !buffered && typeof FileReaderSync === 'function'
      ? new BlobFile(blob)
      : new Uint8Array(await blob.arrayBuffer());
    out.push({ path: file.name, data });
  }
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
// 시도 순서: 선호 device → WASM. 앞 것이 예외로 실패하면 다음으로 넘어간다.
// 메모리 초과로 프로세스가 죽는 경우는 예외가 아니라 페이지 재시작이라 여기서 못 잡는다.
// 그건 index.js 가 localStorage 표식으로 감지해 안내한다.
const plan = (preferred) => preferred === 'wasm' ? ['wasm'] : [preferred, 'wasm'];

async function load({ device: preferred }) {
  env.useBrowserCache = true;   // 그래프(1 MB)·설정·토크나이저만 transformers.js 가 캐시한다. 가중치는 여기서.
  await pruneStaleCache();

  // COOP/COEP 가 없으면 SharedArrayBuffer 를 못 써서 ORT 가 어차피 싱글스레드로 떨어진다.
  try {
    env.backends.onnx.wasm.numThreads =
      self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 4) : 1;
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

  post({ type: 'phase', text: '가중치 목록 조회 중…' });
  let externalData = await mountWeights({ buffered: false });

  const opts = {
    revision: REVISION,
    subfolder: '',                   // 레포가 평면 구조다. 기본값 'onnx/' 로 찾으면 404.
    dtype: 'fp32',
    use_external_data_format: false, // 가중치는 위에서 직접 마운트한다. transformers.js 가 받게 두지 않는다.
    progress_callback,
  };
  let lastErr = null;
  for (const [i, dev] of plan(preferred).entries()) {
    const label = dev === 'webgpu' ? 'WebGPU · fp32' : 'WASM · fp32';
    if (i > 0) post({ type: 'fallback', why: String(lastErr?.message ?? lastErr), next: label, device: dev });
    post({ type: 'phase', text: `그래프 내려받고 ORT 세션 생성 중… (${label})` });
    try {
      model = await AutoModelForCausalLM.from_pretrained(REPO, {
        ...opts, device: dev, session_options: { externalData },
      });
    } catch (e) {
      console.warn(`${label} 실패:`, e);
      lastErr = e;
      // BlobFile 경로 자체가 깨진 경우(ORT 가 외부 데이터를 못 읽음)만 옛 방식으로 한 번 더 시도한다.
      // 그 밖의 실패(메모리·커널 미지원 등)는 다음 device 로 넘긴다.
      if (/external data/i.test(String(e?.message ?? e)) && externalData.some((x) => x.data instanceof BlobFile)) {
        post({ type: 'fallback', why: `Blob 마운트 실패 → 파일을 통째로 읽어 다시 시도: ${e?.message ?? e}`,
               next: label, device: dev });
        try {
          externalData = await mountWeights({ buffered: true });
          model = await AutoModelForCausalLM.from_pretrained(REPO, {
            ...opts, device: dev, session_options: { externalData },
          });
        } catch (e2) { console.warn(`${label} (버퍼) 실패:`, e2); lastErr = e2; continue; }
      } else {
        continue;
      }
    }
    device = dev;
    post({ type: 'ready', device, dtype: 'fp32', rows: rows.length });
    return;
  }
  throw lastErr;
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
