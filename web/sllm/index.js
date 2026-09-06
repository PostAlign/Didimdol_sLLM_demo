/**
 * sLLM 성능 평가 탭 — 기기 판정, 진행바, 결과 카드, 평균 패널. 추론은 전부 워커에서 돈다.
 *
 * initSllm(root) 는 #tab-sllm 패널 루트를 받아 그 안에서만 DOM 을 찾는다. 셸(app.js)이나
 * 다른 탭의 요소는 건드리지 않는다. iOS 차단 안내(#blocker)도 이 패널 안의 카드다 —
 * WebGPU 가 필요한 건 이 탭뿐이라 페이지 전체를 막을 이유가 없다.
 */

import { ms, mb, f3, esc, setBar, makeBadge } from '../ui.js';

export function initSllm(root) {
  const $ = (s) => root.querySelector(s);
  const els = {
    body: $('#sllmBody'), blocker: $('#blocker'), blockWhy: $('#blockWhy'), badges: $('#badges'),
    start: $('#start'), stop: $('#stop'), phase: $('#phase'), homeHint: $('#homeHint'),
    dtype: $('#dtype'),
    avg: $('#avg'), prep: $('#prep'), runbar: $('#runbar'), rows: $('#rows'),
  };
  const bars = {
    model: $('#dlModel'), tok: $('#dlTok'), run: $('#runbar .dl'),
  };
  const badge = makeBadge(els.badges);

  // ── iOS 게이트 ──────────────────────────────────────────────────────────────
  const isIOS = /iPad|iPhone|iPod/.test(navigator.userAgent)
             || (/Macintosh/.test(navigator.userAgent) && navigator.maxTouchPoints > 1);
  // 디버그: ?ios=1 이면 데스크톱에서도 iOS 의 가중치 로드 경로(스트리밍 캐시)를 태운다. 차단 판정에는 영향 없다.
  const iosLoadPath = isIOS || new URLSearchParams(location.search).get('ios') === '1';
  // 홈 화면에서 실행 중인지. iOS 는 navigator.standalone, 그 외는 display-mode 로 판정한다.
  const isStandalone = navigator.standalone === true
                    || matchMedia('(display-mode: standalone)').matches;

  // iOS 에서 WASM 은 가중치를 wasm 힙에 한 벌 더 복사해 fp32 는 물론 fp16 도 탭 메모리 한계에
  // 걸린다. WebGPU 는 텐서를 GPU 프로세스로 바로 올려 그 사본이 없다.
  // iOS 의 WebGPU 는 Safari 26 부터라, 그 아래는 시도하지 않고 안내 후 차단한다.
  function iosBlock() {
    const ua = navigator.userAgent;
    if (!isIOS) return null;
    // 실제로 쓰는 건 navigator.gpu 다. 버전 문자열은 안내문에만 쓴다 — Chrome iOS·카카오톡 등
    // 인앱 브라우저는 UA 에 "Version/N" 이 없어서 버전 기준으로 판정하면 WebGPU 가 있어도 막힌다.
    if (navigator.gpu) return null;
    const major = Number(ua.match(/Version\/(\d+)/)?.[1] ?? 0);
    return major
      ? `감지된 Safari 버전: ${major} · WebGPU 사용 불가`
      : 'WebGPU 를 사용할 수 없는 iOS 브라우저입니다.';
  }

  // ── 저장소 영구화 ───────────────────────────────────────────────────────────
  // 내려받은 model.onnx (1 GB) 는 transformers.js 가 Cache API 에 넣어 재방문 때 다시 쓴다.
  // Cache API 는 기본이 best-effort 라 디스크가 부족하면 브라우저가 지울 수 있다.
  // persist() 가 승인되면 그 삭제 대상에서 빠진다. persist() 는 Window 전용이라
  // 워커가 아니라 여기서 부른다. 결과는 흐름에 영향 없고 콘솔에만 남긴다.
  //   Chrome  : 사이트 관여도 기준으로 조용히 승인/거절
  //   Firefox : 권한 프롬프트가 뜰 수 있음
  //   Safari  : 홈 화면 앱일 때만 사실상 승인. 일반 탭의 7일 미상호작용 삭제(ITP)는
  //             승인 여부와 무관하게 적용되므로, iOS 는 홈 화면 추가 안내로 보완한다.
  async function persistStorage() {
    try {
      const ok = await navigator.storage?.persist?.();
      if (ok !== undefined) console.info(`storage.persist(): ${ok ? '승인' : '거절'}`);
    } catch (e) {
      console.warn('storage.persist() 실패:', e);
    }
  }

  // ── 백엔드 판정 ─────────────────────────────────────────────────────────────
  // tied embedding 이 [262144, 640] 짜리 단일 텐서다 (fp32 671 MB · fp16 335 MB). WebGPU 에서는
  // 이게 스토리지 버퍼 하나로 바인딩되므로, 어댑터 한계가 이보다 작으면 가중치를
  // 다 받은 뒤 세션 생성에서 터진다. 받기 전에 미리 확인한다.
  const NEED = (dtype) => 262144 * 640 * (dtype === 'fp16' ? 2 : 4);

  async function pickDevice() {
    if (!navigator.gpu) return { device: 'wasm', why: 'WebGPU 미지원 브라우저' };
    let adapter;
    try { adapter = await navigator.gpu.requestAdapter(); }
    catch { return { device: 'wasm', why: 'GPU 어댑터 요청 실패' }; }
    if (!adapter) return { device: 'wasm', why: 'GPU 어댑터 없음' };

    const { maxStorageBufferBindingSize: b, maxBufferSize: m } = adapter.limits;
    const limit = Math.min(b, m);
    // transformers.js 는 fp16 세션을 만들기 직전에 adapter.features.has('shader-f16') 를 보고,
    // 없으면 "The device (webgpu) does not support fp16." 로 던진다. 버퍼 한계만 보고 fp16 을
    // 계획하면 이 예외를 맞고 워커의 다음 시도(WASM · fp32)로 밀려난다 — 사용자 눈에는
    // "fp16 을 골랐는데 fp32 로 받는다"로 보인다. 그래서 여기서 미리 확인한다.
    const f16 = adapter.features.has('shader-f16');
    return { device: 'webgpu', limit, f16,
             why: adapter.info?.description || adapter.info?.vendor || 'GPU' };
  }

  // fp16 을 실제로 돌릴 수 있는가. 셋 다 만족해야 한다. 못 쓰면 그 사유를 문자열로 돌려준다.
  //   WebGPU 어댑터가 있고 · shader-f16 을 갖고 있고 · 임베딩 텐서가 버퍼 한계 안에 들어온다
  // WASM 은 fp16 커널이 드물어 애초에 시도하지 않으므로 여기서 함께 막는다.
  function fp16Blocked(chosen) {
    if (chosen.device !== 'webgpu') return chosen.why;
    if (!chosen.f16) return 'GPU 어댑터가 shader-f16 미지원';
    if (chosen.limit < NEED('fp16')) {
      return `GPU 버퍼 한계 ${mb(chosen.limit)} < fp16 필요량 ${mb(NEED('fp16'))}`;
    }
    return null;
  }

  // 고른 dtype 이 안 되면 한 단계 내린다: fp32 → fp16 → WASM(fp32).
  // 내려간 이유는 반드시 note 로 남긴다. 선택이 조용히 버려지면 원인을 찾을 수 없다.
  function resolvePlan(chosen, dtype) {
    const noF16 = fp16Blocked(chosen);
    const toWasm = (why) => ({ device: 'wasm', dtype: 'fp32', note: why });

    if (dtype === 'fp16') {
      return noF16 ? toWasm(`fp16 사용 불가(${noF16}) → WASM · fp32`) : { device: 'webgpu', dtype: 'fp16' };
    }
    // fp32
    if (chosen.device !== 'webgpu') return toWasm(chosen.why);
    if (chosen.limit >= NEED('fp32')) return { device: 'webgpu', dtype: 'fp32' };
    const over = `GPU 버퍼 한계 ${mb(chosen.limit)} < fp32 필요량 ${mb(NEED('fp32'))}`;
    if (!noF16) return { device: 'webgpu', dtype: 'fp16', note: `${over} → fp16` };
    return toWasm(`${over} · fp16 도 사용 불가(${noF16}) → WASM`);
  }

  // dtype 선택은 새로고침 뒤에도 남긴다. 실패로 fp16 에 내려간 뒤에도 그 선택이 유지되게.
  const DTYPE_KEY = 'didimdol.dtype';
  const getDtype = () => els.dtype.value;
  function setDtype(v, persist = true) {
    els.dtype.value = v;
    if (persist) { try { localStorage.setItem(DTYPE_KEY, v); } catch {} }
  }

  // ── 진행바 (rAF 스로틀) ─────────────────────────────────────────────────────
  // progress 이벤트는 매우 자주 온다. 그대로 DOM 에 반영하면 렌더가 밀린다.
  const dl = {
    model: { seen: false, done: false, loaded: 0, total: 0, markT: 0, markL: 0, bps: 0 },
    tok:   { seen: false, done: false, loaded: 0, total: 0, markT: 0, markL: 0, bps: 0 },
  };
  let dirty = false;

  function onDl({ which, status, loaded, total, stage }) {
    const d = dl[which];
    if (!d) return;                                   // 그래프 파일(수 MB) 등은 표시하지 않는다
    if (status === 'reset') {                         // 다음 dtype 시도 → 바를 처음부터
      Object.assign(d, { seen: false, done: false, loaded: 0, total: 0, markT: 0, markL: 0, bps: 0, stage: undefined });
      dirty = true; return;
    }
    if (status === 'done') { d.done = true; dirty = true; return; }
    d.seen = true; d.loaded = loaded; d.total = total;
    if (stage && stage !== d.stage) { d.stage = stage; d.markT = 0; d.bps = 0; }  // 단계가 바뀌면 속도 표본 리셋

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
          (d.stage === 'cache' ? '캐시에서 읽는 중 · ' : d.stage === 'net' ? '내려받는 중 · ' : '')
          + `${mb(d.loaded)} / ${mb(d.total)} (${Math.floor(frac * 100)}%)`
          + (speed ? ` · ${mb(speed)}/s` : '')
          + (eta > 1 ? ` · 약 ${eta > 90 ? `${Math.round(eta / 60)}분` : `${Math.round(eta)}초`} 남음` : ''));
      }
    }
    requestAnimationFrame(paint);
  }
  requestAnimationFrame(paint);

  // ── 결과 카드 ───────────────────────────────────────────────────────────────
  function addRow(i, r, error, attempts = 1) {
    const d = document.createElement('div');
    d.className = 'r' + (error ? ' err' : '');
    if (error) {
      d.innerHTML = `<div class="head">
        <span class="idx">${i < 0 ? '!' : `#${i + 1}`}</span>
        <span class="m" style="color:#b91c1c">${i < 0 ? '오류' : attempts > 1 ? `${attempts}회 실패` : '실패'}</span>
        <span class="m">${esc(error)}</span>
      </div>`;
    } else {
      d.innerHTML = `<div class="head">
        <span class="idx">#${i + 1}</span>
        <span class="m">TTFT <b>${ms(r.ttft)}</b></span>
        <span class="m">총 <b>${ms(r.total)}</b></span>
        <span class="m">${r.nTok}토큰 · ${r.tps.toFixed(1)} tok/s</span>
        ${r.eos ? '' : '<span class="warn">512 상한 도달</span>'}
        ${r.attempts > 1 ? `<span class="warn">재시도 후 성공 (${r.attempts}회차)</span>` : ''}
        <span class="f1">R1 ${f3(r.rouge.f1)} <span class="pr">(P ${f3(r.rouge.p)} / R ${f3(r.rouge.r)})</span></span>
      </div>`;
    }
    els.rows.append(d);
    // 탭이 숨겨져 있으면 스크롤할 대상이 없다. 보이는 동안만 따라간다.
    if (!root.hidden) d.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
  }

  // ── 로드 도중 프로세스 종료 감지 ────────────────────────────────────────────
  // iOS 는 탭 메모리 한계를 넘으면 WebContent 프로세스를 죽이고 페이지를 조용히 다시 연다.
  // 콘솔도 오류도 남지 않아 "다운로드만 반복"처럼 보인다. 로드 시작 시 표식을 남기고
  // ready/fatal 에서 지우면, 표식이 남은 채 부팅된 경우 = 지난 시도가 도중에 끊긴 것이다.
  // 예외로 잡히는 실패는 워커가 다음 dtype 으로 넘어가지만, 프로세스 종료는 여기서만 잡을 수 있다.
  // fp32 로 죽었으면 다음 시도의 dtype 을 fp16 으로 내려 둔다 — "fp32 로 하고, 안 되면 fp16".
  const ATTEMPT_KEY = 'didimdol.loadAttempt';
  const markAttempt = (device, dtype) => {
    try { localStorage.setItem(ATTEMPT_KEY, JSON.stringify({ device, dtype, t: Date.now() })); } catch {}
  };
  const clearAttempt = () => { try { localStorage.removeItem(ATTEMPT_KEY); } catch {} };
  function reportBrokenAttempt(noF16) {
    let prev = null;
    try { prev = JSON.parse(localStorage.getItem(ATTEMPT_KEY) ?? 'null'); } catch {}
    if (!prev) return;
    clearAttempt();
    const when = new Date(prev.t).toLocaleTimeString('ko-KR');
    let msg = `지난 시도(${when} · ${prev.device} · ${prev.dtype})가 모델 로드 도중 끝났습니다. `
      + `iOS 에서는 탭 메모리 한계(약 2 GB)를 넘으면 오류 없이 페이지가 다시 열립니다.`;
    if (prev.dtype !== 'fp32') {
      msg += ' fp16 도 넘는다면 이 기기에서는 실행할 수 없습니다.';
    } else if (noF16) {
      // 여기서 fp16 으로 내려 봤자 워커가 다시 fp32 로 되돌린다. 그럴 바엔 사실대로 말한다.
      msg += ` 이 기기는 fp16 을 쓸 수 없어(${noF16}) 정밀도를 더 낮출 수 없습니다.`;
    } else {
      setDtype('fp16');
      msg += ' 다음 시도는 fp16 으로 바꿔 두었습니다. 원하면 위에서 fp32 로 되돌릴 수 있습니다.';
    }
    addRow(-1, null, msg);
  }

  // ── 워커 ────────────────────────────────────────────────────────────────────
  const worker = new Worker(new URL('./worker.js', import.meta.url), { type: 'module' });
  let loaded = false, chosen = null, nRows = 100;

  // 모듈 import 실패 등 워커 스크립트 자체의 오류는 onmessage 로 오지 않는다.
  worker.onerror = (e) => {
    els.phase.textContent = '오류';
    els.prep.hidden = true;
    addRow(-1, null, `워커 오류: ${e.message ?? e}`);
    els.start.disabled = false; els.stop.disabled = true;
  };

  worker.onmessage = ({ data: m }) => {
    switch (m.type) {
      case 'phase':    els.phase.textContent = m.text; break;
      case 'dl':       onDl(m); break;
      case 'skip':
        badge(`⚠ ${m.dtype} 파일 없음 → 건너뜀`, 'cpu');
        addRow(-1, null, m.why);
        break;
      case 'fallback':
        badge(`⚠ 실패 → ${m.next}`, 'cpu');
        addRow(-1, null, `이전 시도 실패, ${m.next} 로 다시 시도합니다: ${m.why}`);
        markAttempt(m.device, m.dtype);
        console.warn('fallback:', m.why);
        break;
      case 'ready':
        clearAttempt();
        badge(`${m.device === 'webgpu' ? '⚡ WebGPU' : '🐢 WASM(CPU)'} · ${m.dtype} 로 실행 중`,
              m.device === 'webgpu' ? 'gpu' : 'cpu');
        loaded = true; nRows = m.rows;
        els.prep.hidden = true;
        els.runbar.hidden = false;
        setBar(bars.run, 0, `0 / ${nRows}`);
        worker.postMessage({ type: 'run' });
        break;
      case 'row':
        addRow(m.i, m.r, m.error, m.attempts);
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
        clearAttempt();
        els.phase.textContent = '오류';
        els.prep.hidden = true;
        addRow(-1, null, m.error);
        els.start.disabled = false; els.stop.disabled = true;
        if (!loaded) els.dtype.disabled = false;   // 로드 자체가 실패했으면 정밀도를 바꿔 재시도할 수 있어야 한다
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
    const notes = [];
    if (m.retried) notes.push(`재시도 ${m.retried}행`);
    if (m.failed) notes.push(`실패 ${m.failed}행 포함, 평균은 전체 기준`);
    els.phase.textContent = `완료 · ${m.n}/${m.total}행` + (notes.length ? ` (${notes.join(' · ')})` : '');
    els.start.disabled = false; els.stop.disabled = true;
    setBar(bars.run, 1, `${m.n} / ${m.total}`);
  }

  els.start.onclick = () => {
    els.avg.hidden = true;                        // 시작하면 평균은 즉시 다시 숨긴다
    els.rows.replaceChildren();
    els.start.disabled = true; els.stop.disabled = false;
    els.dtype.disabled = true;                    // 로드된 모델은 바꿀 수 없다. 바꾸려면 새로고침.
    if (loaded) {
      els.runbar.hidden = false;
      setBar(bars.run, 0, `0 / ${nRows}`);
      worker.postMessage({ type: 'run' });
    } else {
      els.prep.hidden = false;
      const p = resolvePlan(chosen, getDtype());
      if (p.note) badge(`↓ ${p.note}`, 'cpu');
      markAttempt(p.device, p.dtype);
      // f16: 워커가 fp32 실패 뒤 fp16 으로 되물러설 수 있는지. 못 하면 그 시도를 큐에 넣지 않는다.
      worker.postMessage({ type: 'load', device: p.device, dtype: p.dtype,
                           ios: iosLoadPath, f16: !fp16Blocked(chosen) });
    }
  };
  els.dtype.onchange = () => setDtype(getDtype());
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
      els.body.hidden = true;                     // 이 탭의 컨트롤만 치운다. 다른 탭은 그대로.
      return;
    }
    chosen = await pickDevice();
    const noF16 = fp16Blocked(chosen);
    try { setDtype(localStorage.getItem(DTYPE_KEY) || 'fp32', false); } catch { setDtype('fp32', false); }
    if (noF16) {
      // 못 쓰는 선택지를 열어 두면 "골랐는데 무시된다"가 된다. 잠그고 사유를 라벨에 붙인다.
      // 저장된 선택(localStorage)은 건드리지 않는다 — fp16 이 되는 기기에서 다시 열면 살아나야 한다.
      const opt = els.dtype.querySelector('option[value="fp16"]');
      if (opt) { opt.disabled = true; opt.textContent = `fp16 (사용 불가 · ${noF16})`; }
      if (getDtype() === 'fp16') setDtype('fp32', false);
    }
    reportBrokenAttempt(noF16);                   // fp32 로 죽었으면 여기서 fp16 으로 내린다
    persistStorage();                             // 기다리지 않는다. 승인 여부가 로드를 막지 않는다.
    // iOS Safari 일반 탭은 7일간 상호작용이 없으면 캐시를 통째로 지운다. 홈 화면 앱은
    // 저장소가 분리되어 이 규칙에서 빠지므로, 홈 화면이 아닌 iOS 에서만 한 줄 안내한다.
    if (isIOS && !isStandalone) els.homeHint.hidden = false;
    if (chosen.device === 'webgpu') badge(`⚡ WebGPU · ${chosen.why}`, 'gpu');
    else                            badge(`🐢 WASM(CPU) · ${chosen.why}`, 'cpu');
    if (chosen.limit) badge(`GPU 버퍼 한계 ${mb(chosen.limit)} (fp32 ${mb(NEED('fp32'))} · fp16 ${mb(NEED('fp16'))} 필요)`);
    if (noF16) badge(`⚠ fp16 사용 불가 · ${noF16}`, 'cpu');   // 쓸 수 있으면 선택지가 열려 있는 것으로 충분하다
    els.phase.textContent = '준비 완료';
    els.start.disabled = false;
  })();
}
