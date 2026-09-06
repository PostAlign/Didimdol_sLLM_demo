/** UI — 기기 판정, 진행바, 결과 카드, 평균 패널. 추론은 전부 워커에서 돈다. */

const $ = (s) => document.querySelector(s);
const els = {
  blocker: $('#blocker'), blockWhy: $('#blockWhy'), badges: $('#badges'),
  start: $('#start'), stop: $('#stop'), phase: $('#phase'),
  avg: $('#avg'), prep: $('#prep'), runbar: $('#runbar'), rows: $('#rows'),
};
const bars = {
  model: $('#dlModel'), tok: $('#dlTok'), run: $('#runbar .dl'),
};

// ── 표시 헬퍼 ───────────────────────────────────────────────────────────────
const ms   = (v) => v >= 1000 ? `${(v / 1000).toFixed(2)}<span class="u">s</span>`
                              : `${Math.round(v)}<span class="u">ms</span>`;
const mb   = (b) => `${(b / 1e6).toFixed(1)} MB`;
const pct  = (x) => `${(x * 100).toFixed(1)}%`;
const f3   = (x) => x.toFixed(3);
const badge = (text, cls = '') => {
  const s = document.createElement('span');
  s.className = `badge ${cls}`; s.textContent = text; els.badges.append(s);
};

function setBar(el, fraction, label) {
  el.querySelector('.fill').style.width = `${Math.max(0, Math.min(1, fraction)) * 100}%`;
  el.querySelector('.s').textContent = label;
}

// ── iOS 게이트 ──────────────────────────────────────────────────────────────
// WebGPU 없이 1.02 GB fp32 를 iOS WASM 으로 돌리면 탭 메모리 한계에서 죽는다.
// iOS 의 WebGPU 는 Safari 26 부터라, 그 아래는 시도하지 않고 안내 후 차단한다.
function iosBlock() {
  const ua = navigator.userAgent;
  const isIOS = /iPad|iPhone|iPod/.test(ua)
             || (/Macintosh/.test(ua) && navigator.maxTouchPoints > 1);
  if (!isIOS) return null;
  // 실제로 쓰는 건 navigator.gpu 다. 버전 문자열은 안내문에만 쓴다 — Chrome iOS·카카오톡 등
  // 인앱 브라우저는 UA 에 "Version/N" 이 없어서 버전 기준으로 판정하면 WebGPU 가 있어도 막힌다.
  if (navigator.gpu) return null;
  const major = Number(ua.match(/Version\/(\d+)/)?.[1] ?? 0);
  return major
    ? `감지된 Safari 버전: ${major} · WebGPU 사용 불가`
    : 'WebGPU 를 사용할 수 없는 iOS 브라우저입니다.';
}

// ── 백엔드 판정 ─────────────────────────────────────────────────────────────
// tied embedding 이 [262144, 640] fp32 = 671 MB 짜리 단일 텐서다. WebGPU 에서는
// 이게 스토리지 버퍼 하나로 바인딩되므로, 어댑터 한계가 이보다 작으면 1 GB 를
// 다 받은 뒤 세션 생성에서 터진다. 받기 전에 미리 확인한다.
const NEED = 262144 * 640 * 4;

async function pickDevice() {
  if (!navigator.gpu) return { device: 'wasm', why: 'WebGPU 미지원 브라우저' };
  let adapter;
  try { adapter = await navigator.gpu.requestAdapter(); }
  catch { return { device: 'wasm', why: 'GPU 어댑터 요청 실패' }; }
  if (!adapter) return { device: 'wasm', why: 'GPU 어댑터 없음' };

  const { maxStorageBufferBindingSize: b, maxBufferSize: m } = adapter.limits;
  const limit = Math.min(b, m);
  if (limit < NEED) {
    return { device: 'wasm', limit, why: `GPU 버퍼 한계 부족 (${mb(limit)} < ${mb(NEED)})` };
  }
  return { device: 'webgpu', limit, why: adapter.info?.description || adapter.info?.vendor || 'GPU' };
}

// ── 진행바 (rAF 스로틀) ─────────────────────────────────────────────────────
// progress 이벤트는 매우 자주 온다. 그대로 DOM 에 반영하면 렌더가 밀린다.
const dl = {
  model: { seen: false, done: false, loaded: 0, total: 0, markT: 0, markL: 0, bps: 0 },
  tok:   { seen: false, done: false, loaded: 0, total: 0, markT: 0, markL: 0, bps: 0 },
};
let dirty = false;

function onDl({ which, status, loaded, total }) {
  const d = dl[which];
  if (status === 'done') { d.done = true; dirty = true; return; }
  d.seen = true; d.loaded = loaded; d.total = total;

  // 속도는 0.25초 이상 벌어진 표본으로만 갱신한다. progress 이벤트는 같은 밀리초에
  // 여러 번 오기도 해서, 매 이벤트로 나누면 Infinity 가 튀어나온다.
  const now = performance.now();
  const dt = (now - d.markT) / 1000;
  if (d.markT && dt >= 0.25) {
    const inst = (loaded - d.markL) / dt;
    if (Number.isFinite(inst)) d.bps = d.bps ? d.bps * 0.75 + inst * 0.25 : inst;
  }
  if (!d.markT || dt >= 0.25) { d.markT = now; d.markL = loaded; }
  dirty = true;
}

function paint() {
  if (dirty) {
    dirty = false;
    for (const which of ['model', 'tok']) {
      const d = dl[which], el = bars[which];
      if (d.done) { setBar(el, 1, d.seen ? '완료' : '캐시됨'); continue; }
      if (!d.seen || !d.total) continue;
      const frac = d.loaded / d.total;
      const speed = Number.isFinite(d.bps) && d.bps > 0 ? d.bps : 0;
      const eta = speed ? (d.total - d.loaded) / speed : 0;
      setBar(el, frac,
        `${mb(d.loaded)} / ${mb(d.total)} (${Math.floor(frac * 100)}%)`
        + (speed ? ` \u00b7 ${mb(speed)}/s` : '')
        + (eta > 1 ? ` \u00b7 약 ${eta > 90 ? `${Math.round(eta / 60)}분` : `${Math.round(eta)}초`} 남음` : ''));
    }
  }
  requestAnimationFrame(paint);
}
requestAnimationFrame(paint);

// ── 결과 카드 ───────────────────────────────────────────────────────────────
function addRow(i, r, error) {
  const d = document.createElement('details');
  d.className = 'r' + (error ? ' err' : '');
  if (error) {
    d.innerHTML = `<summary><span class="idx">${i < 0 ? '!' : `#${i + 1}`}</span>
      <span class="m" style="color:#b91c1c">${i < 0 ? '오류' : '실패'}</span></summary>
      <div class="body"><p>${esc(error)}</p></div>`;
  } else {
    d.innerHTML = `<summary>
      <span class="idx">#${i + 1}</span>
      <span class="turn">${r.turns}턴</span>
      <span class="m">TTFT <b>${ms(r.ttft)}</b></span>
      <span class="m">총 <b>${ms(r.total)}</b></span>
      <span class="m">${r.nTok}토큰 · ${r.tps.toFixed(1)} tok/s</span>
      ${r.eos ? '' : '<span class="warn">512 상한 도달</span>'}
      <span class="f1">R1 ${f3(r.rouge.f1)}</span>
    </summary>
    <div class="body">
      <div><h4>생성 (P ${f3(r.rouge.p)} / R ${f3(r.rouge.r)} · 프롬프트 ${r.promptLen}토큰)</h4>
        <p class="pred">${esc(r.pred)}</p></div>
      <div><h4>정답</h4><p class="ref">${esc(r.ref)}</p></div>
    </div>`;
  }
  els.rows.append(d);
  d.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
}
const esc = (s) => String(s).replace(/[&<>]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' }[c]));

// ── 워커 ────────────────────────────────────────────────────────────────────
const worker = new Worker(new URL('./worker.js', import.meta.url), { type: 'module' });
let loaded = false, chosen = null, nRows = 100;

worker.onmessage = ({ data: m }) => {
  switch (m.type) {
    case 'phase':    els.phase.textContent = m.text; break;
    case 'dl':       onDl(m); break;
    case 'fallback':
      badge('⚠ WebGPU 초기화 실패 → WASM', 'cpu');
      console.warn('WebGPU fallback:', m.why);
      break;
    case 'ready':
      loaded = true; nRows = m.rows;
      els.prep.hidden = true;
      els.runbar.hidden = false;
      setBar(bars.run, 0, `0 / ${nRows}`);
      worker.postMessage({ type: 'run' });
      break;
    case 'row':
      addRow(m.i, m.r, m.error);
      setBar(bars.run, m.progress, `${m.i + 1} / ${nRows}`);
      break;
    case 'done':
      finish(m);
      break;
    case 'aborted':
      els.phase.textContent = `중단됨 (${m.at}행까지 실행)`;
      els.start.disabled = false; els.stop.disabled = true;
      break;
    case 'fatal':
      els.phase.textContent = '오류';
      els.prep.hidden = true;
      addRow(-1, null, m.error);
      els.start.disabled = false; els.stop.disabled = true;
      break;
  }
};

function finish(m) {
  $('#aTtft').innerHTML  = ms(m.ttft);
  $('#aTotal').innerHTML = ms(m.totalMs);
  $('#aTps').innerHTML   = `${m.tps.toFixed(1)}<span class="u">tok/s</span>`;
  $('#aF1').textContent  = f3(m.f1);
  $('#aP').textContent   = f3(m.p);
  $('#aR').textContent   = f3(m.r);
  $('#aEos').innerHTML   = `${m.eos}<span class="u">/ ${m.n}</span>`;
  $('#aWall').innerHTML  = `${(m.wall / 60000).toFixed(1)}<span class="u">분</span>`;
  els.avg.hidden = false;                       // 100/100 완료 시에만 노출
  els.phase.textContent = `완료 · ${m.n}/${m.total}행`;
  els.start.disabled = false; els.stop.disabled = true;
  setBar(bars.run, 1, `${m.n} / ${m.total}`);
}

els.start.onclick = () => {
  els.avg.hidden = true;                        // 시작하면 평균은 즉시 다시 숨긴다
  els.rows.replaceChildren();
  els.start.disabled = true; els.stop.disabled = false;
  if (loaded) {
    els.runbar.hidden = false;
    setBar(bars.run, 0, `0 / ${nRows}`);
    worker.postMessage({ type: 'run' });
  } else {
    els.prep.hidden = false;
    worker.postMessage({ type: 'load', device: chosen.device });
  }
};
els.stop.onclick = () => {
  els.stop.disabled = true;
  els.phase.textContent = '중단하는 중…';
  worker.postMessage({ type: 'stop' });
};

// ── 부팅 ────────────────────────────────────────────────────────────────────
(async () => {
  const blocked = iosBlock();
  if (blocked) {
    els.blockWhy.textContent = blocked;
    els.blocker.hidden = false;
    els.phase.textContent = '지원되지 않는 기기';
    return;
  }
  chosen = await pickDevice();
  if (chosen.device === 'webgpu') badge(`⚡ WebGPU · ${chosen.why}`, 'gpu');
  else                            badge(`🐢 WASM(CPU) · ${chosen.why}`, 'cpu');
  if (chosen.limit) badge(`GPU 버퍼 한계 ${mb(chosen.limit)}`);
  badge(`fp32 · 임계 ${mb(NEED)}`);
  badge('temperature 0.3 · top_k 64 · top_p 0.95');
  badge('seed 42');
  badge('max_new_tokens 512');
  if (!self.crossOriginIsolated) badge('COOP/COEP 미적용 → WASM 싱글스레드');
  els.phase.textContent = '준비 완료';
  els.start.disabled = false;
})();
