/**
 * 평가 워커 — 모델 로드 · 생성 · 채점을 전부 여기서 한다.
 *
 * 메인 스레드에서 돌리면 100행 × 최대 512토큰 디코드가 UI 를 얼려서
 * "스크롤하며 결과 보기" 자체가 성립하지 않는다.
 *
 * 파일 출처가 둘로 갈린다.
 *   model*.onnx / model*.onnx_data → Hugging Face (커밋 SHA 고정)
 *   그 외 config/토크나이저        → 이 정적 사이트
 * transformers.js 는 한 곳에서만 받아오므로 env.fetch 를 후킹해 갈라 보낸다.
 *
 * 모델은 외부 데이터 포맷이다 (그래프 model.onnx 수 MB + 가중치 model.onnx_data).
 * 단일 파일이면 ORT 가 JS 버퍼를 wasm 힙에 통째로 복사해 같은 바이트가 두 벌이 되고,
 * iPhone 은 그 순간 WebContent 한계(약 2 GB)를 넘어 오류 없이 죽는다. 외부 데이터는
 * 텐서 단위로 잘라 WebGPU 버퍼에 바로 올리므로 사본이 없다 (build_web_models.py 참고).
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
const REVISION = '2ec0c707259429aa93fe4130b40f9f04d9ff4b7d';

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
  'config.json':            '../model/config.json',
  'generation_config.json': '../model/generation_config.json',
  'tokenizer.json':         '../tokenizer/tokenizer.json',
  'tokenizer_config.json':  '../tokenizer/tokenizer_config.json',
  'special_tokens_map.json':'../tokenizer/special_tokens_map.json',
  'added_tokens.json':      '../tokenizer/added_tokens.json',
}).map(([k, v]) => [k, new URL(v, import.meta.url).href]));

const post = (m) => self.postMessage(m);

const isWeights = (url) => /\.onnx_data(_\d+)?(\?|$)/.test(url);
let streamCache = false;      // iOS: 가중치를 스트리밍으로 캐시에 먼저 넣고, 캐시에서 읽는다
let weightsStage = 'net';     // 진행률 라벨용: 'net' | 'cache'

const baseFetch = env.fetch ?? fetch;
env.fetch = (input, init) => {
  const url = typeof input === 'string' ? input : (input?.url ?? String(input));
  if (url.includes('huggingface.co')) {
    const name = url.split('?')[0].split('/').pop();
    if (LOCAL[name]) return fetch(LOCAL[name], init);          // → 정적 사이트
    if (streamCache && isWeights(url)) return streamCachedFetch(url, init);
  }
  return baseFetch(input, init);                               // → HF
};

// ── iOS 전용 가중치 캐시 ────────────────────────────────────────────────────
// transformers.js 의 캐시는 "버퍼 전부 받기 → Response 로 감싸 put" 이라 그 순간 가중치가
// 메모리에 두 벌이다 (fp32 면 2.1 GB → iPhone 즉사). 여기서는 순서를 바꾼다.
//   ① 네트워크 스트림을 그대로 cache.put 에 흘려 넣는다 (JS 힙에 버퍼를 만들지 않는다)
//   ② 끝나면 cache.match 로 다시 열어 transformers.js 에 넘긴다 → 이때 1 벌만 생긴다
// 두 단계가 순차라 피크가 한 벌 크기를 넘지 않는다. put 이 거부되면(쿼터 등) 그냥 다시 받는다.
// ①에서는 transformers.js 가 진행률을 낼 수 없으므로 청크를 세어 직접 보낸다.
async function streamCachedFetch(url, init) {
  let cache = null;
  try { cache = await caches.open(env.cacheKey); } catch { /* 캐시 불가 → 직접 받기 */ }
  if (cache) {
    const hit = await cache.match(url);
    if (hit) { weightsStage = 'cache'; return hit; }
  }
  weightsStage = 'net';
  const res = await baseFetch(url, init);
  if (!res.ok || !cache || !res.body) return res;

  const total = Number(res.headers.get('content-length')) || 0;
  const reader = res.body.getReader();
  let loaded = 0;
  const counted = new ReadableStream({
    async pull(ctrl) {
      const { done, value } = await reader.read();
      if (done) { ctrl.close(); return; }
      loaded += value.byteLength;
      post({ type: 'dl', which: 'model', status: 'progress', loaded, total, stage: 'net' });
      ctrl.enqueue(value);
    },
    cancel(reason) { reader.cancel(reason).catch(() => {}); },
  });
  try {
    post({ type: 'phase', text: '가중치를 캐시에 스트리밍 저장 중…' });
    await cache.put(url, new Response(counted, { status: 200, headers: res.headers }));
    const hit = await cache.match(url);
    if (hit) { weightsStage = 'cache'; return hit; }
  } catch (e) {
    console.warn('가중치 캐시 저장 실패, 직접 받는다:', e);
    try { await reader.cancel(); } catch { /* 이미 닫힘 */ }
  }
  weightsStage = 'net';
  return baseFetch(url, init);
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
let device = 'wasm', dtype = 'fp32', aborted = false;
const stopper = new InterruptableStoppingCriteria();

// ── 옛 REVISION 캐시 정리 ───────────────────────────────────────────────────
// 받은 파일은 Cache API 의 env.cacheKey 저장소에 원격 URL 을 키로 들어간다 (transformers.js 와
// 위 streamCachedFetch 모두). REVISION 을 갈면 새 URL 로 다시 받지만 옛 SHA 의 가중치는
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
// 시도 순서. 앞 것이 예외로 실패하면 다음으로 넘어간다.
//   (선호 device, 고른 dtype) → [webgpu 면] (webgpu, fp16) → (wasm, fp32)
// fp16 은 WebGPU 에서만 시도한다. wasm CPU 는 fp16 커널이 드물어 느리거나 실패한다.
// 메모리 초과로 프로세스가 죽는 경우는 예외가 아니라 페이지 재시작이라 여기서 못 잡는다.
// 그건 app.js 가 localStorage 표식으로 감지해 다음 시도 dtype 을 fp16 으로 내린다.
function plan(preferred, wanted) {
  const p = [{ device: preferred, dtype: wanted }];
  if (preferred === 'webgpu' && wanted === 'fp32') p.push({ device: 'webgpu', dtype: 'fp16' });
  if (!(preferred === 'wasm' && wanted === 'fp32')) p.push({ device: 'wasm', dtype: 'fp32' });
  return p;
}

// 고정 REVISION 에 그 dtype 의 파일이 실제로 있는지 본다. 레포에 fp32 만 올라가 있는 상태에서
// fp16 을 "있다고 가정하고" 받으러 가면 404 를 중간에 맞고 다음 단계로 밀린다. 그러면 화면에는
// 원인이 뭉개진 실패만 남으므로, 없으면 그 시도를 건너뛰고 이유를 그대로 보여 준다.
const FILE = { fp32: 'model.onnx', fp16: 'model_fp16.onnx' };
async function available(dtype) {
  const url = `https://huggingface.co/${REPO}/resolve/${REVISION}/${FILE[dtype]}`;
  try {
    const r = await baseFetch(url, { method: 'HEAD' });
    return r.ok;
  } catch {
    return true;   // 네트워크 오류는 존재 여부와 무관하다. 본 시도에서 다시 겪게 둔다.
  }
}

async function load({ device: preferred, dtype: wanted, ios }) {
  streamCache = !!ios;
  env.useBrowserCache = !ios;   // iOS 는 가중치를 streamCachedFetch 가 직접 캐시한다
  await pruneStaleCache();

  // COOP/COEP 가 없으면 SharedArrayBuffer 를 못 써서 ORT 가 어차피 싱글스레드로 떨어진다.
  try {
    env.backends.onnx.wasm.numThreads =
      self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 4) : 1;
  } catch { /* ORT 백엔드가 아직 준비되지 않았으면 기본값을 쓴다 */ }

  // 가중치(.onnx_data)가 100% 에 닿은 뒤에도 보이지 않는 단계가 남는다.
  //   데스크톱: transformers.js 의 cache.put (사본 +1) → 'done' → 세션 생성 (콜백 없음)
  //   iOS     : 캐시에서 읽기 → 'done' → 세션 생성
  // 이 구간에 문구가 없으면 "다운로드에서 멈춘 것"처럼 보이므로 phase 로 단계를 드러낸다.
  let weightsFull = false;
  const progress_callback = (p) => {
    const f = String(p.file ?? '');
    const which = isWeights(f) ? 'model'
                : /(^|\/)model(_fp16)?\.onnx$/.test(f) ? 'graph'
                : f.endsWith('tokenizer.json') ? 'tok' : null;
    if (!which) return;
    if (which === 'model') {
      if (p.status === 'progress' && p.total && p.loaded >= p.total && !weightsFull) {
        weightsFull = true;
        post({ type: 'phase', text: streamCache
          ? '가중치 읽기 완료 · ORT 세션 생성 중… (진행률 없음)'
          : '가중치 수신 완료 · 브라우저 캐시에 저장 중… (사본 생성)' });
      } else if (p.status === 'done') {
        post({ type: 'phase', text: '가중치 준비 완료 · ORT 세션 생성 중… (진행률 없음 · 메모리 최대 구간)' });
      }
    }
    if (p.status !== 'progress' && p.status !== 'done') return;
    post({ type: 'dl', which, status: p.status, loaded: p.loaded, total: p.total,
           stage: which === 'model' ? weightsStage : undefined });
  };

  post({ type: 'phase', text: '토크나이저 내려받는 중…' });
  [tokenizer, chatTemplate, rows] = await Promise.all([
    AutoTokenizer.from_pretrained(REPO, { revision: REVISION, progress_callback }),
    // chat_template 은 tokenizer_config.json 에 없고, AutoTokenizer 는 .jinja 를
    // 받아오지 않는다(그 경로는 Processor 전용). 직접 읽어서 명시적으로 넘긴다.
    fetch(new URL('../tokenizer/chat_template.jinja', import.meta.url)).then(r => r.text()),
    fetch(new URL('../data.jsonl', import.meta.url)).then(r => r.text()).then(t =>
      t.split('\n').filter(Boolean).map(JSON.parse)),
  ]);
  rouge1 = makeRouge1(tokenizer);

  const opts = {
    revision: REVISION,
    subfolder: '',                  // 레포가 평면 구조다. 기본값 'onnx/' 로 찾으면 404.
    use_external_data_format: true, // model*.onnx (그래프) + model*.onnx_data (가중치)
    progress_callback,
  };
  let lastErr = null;
  for (const [i, attempt] of plan(preferred, wanted).entries()) {
    const label = `${attempt.device === 'webgpu' ? 'WebGPU' : 'WASM'} · ${attempt.dtype}`;
    if (!(await available(attempt.dtype))) {
      lastErr = new Error(`${FILE[attempt.dtype]} 이(가) 레포 ${REPO}@${REVISION.slice(0, 7)} 에 없어 ${attempt.dtype} 은 건너뜁니다`);
      post({ type: 'skip', why: lastErr.message, ...attempt });
      continue;
    }
    if (i > 0) post({ type: 'fallback', why: String(lastErr?.message ?? lastErr), next: label, ...attempt });
    post({ type: 'phase', text: `모델 내려받는 중… (${label})` });
    weightsFull = false;
    post({ type: 'dl', which: 'model', status: 'reset' });
    try {
      model = await AutoModelForCausalLM.from_pretrained(REPO, { ...opts, ...attempt });
      device = attempt.device; dtype = attempt.dtype;
      post({ type: 'ready', device, dtype, rows: rows.length });
      return;
    } catch (e) {
      console.warn(`${label} 실패:`, e);
      lastErr = e;
    }
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
