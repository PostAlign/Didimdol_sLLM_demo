/**
 * sLLM 성능 평가 탭 — 기기 판정, 진행바, 결과 카드, 평균 패널. 추론은 전부 워커에서 돈다.
 *
 * initSllm(root) 는 #tab-sllm 패널 루트를 받아 그 안에서만 DOM 을 찾는다. 셸(app.js)이나
 * 다른 탭의 요소는 건드리지 않는다. iOS 차단 안내(#blocker)도 이 패널 안의 카드다 —
 * WebGPU 가 필요한 건 이 탭뿐이라 페이지 전체를 막을 이유가 없다.
 *
 * 정밀도는 fp32 하나다. fp16 은 품질이 평가 기준이 못 되어 폐기했다. 스마트폰에서도 fp32 가
 * 올라가도록 모델 쪽(임베딩 분할 · 마지막 토큰 로짓 · 파일 분할)과 워커 쪽(파일 단위 캐시 ·
 * Blob 마운트)을 손봤다. 자세한 건 build_web_models.py 와 worker.js 머리말.
 */

import { ms, mb, f3, esc, setBar, makeBadge } from '../ui.js';

export function initSllm(root) {
  const $ = (s) => root.querySelector(s);
  const els = {
    body: $('#sllmBody'), blocker: $('#blocker'), blockWhy: $('#blockWhy'), badges: $('#badges'),
    start: $('#start'), stop: $('#stop'), phase: $('#phase'), homeHint: $('#homeHint'),
    avg: $('#avg'), prep: $('#prep'), runbar: $('#runbar'), rows: $('#rows'),
  };
  const bars = {
    model: $('#dlModel'), tok: $('#dlTok'), run: $('#runbar .dl'),
  };
  const badge = makeBadge(els.badges);

  // ── iOS 게이트 ──────────────────────────────────────────────────────────────
  const isIOS = /iPad|iPhone|iPod/.test(navigator.userAgent)
             || (/Macintosh/.test(navigator.userAgent) && navigator.maxTouchPoints > 1);
  // 홈 화면에서 실행 중인지. iOS 는 navigator.standalone, 그 외는 display-mode 로 판정한다.
  const isStandalone = navigator.standalone === true
                    || matchMedia('(display-mode: standalone)').matches;

  // iOS 에서 WASM 은 가중치를 wasm 힙에 통째로 올려야 해서(1.07 GB) 탭 메모리 한계에 걸린다.
  // WebGPU 는 텐서를 GPU 프로세스로 바로 올려 그 사본이 없다.
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
  // 내려받은 가중치(1 GB, 파일 여러 개)는 워커가 Cache API 에 넣어 재방문 때 다시 쓴다.
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
  // WebGPU 는 텐서 하나가 스토리지 버퍼 하나라, 가장 큰 텐서가 어댑터 한계 안에 들어와야 한다.
  // 원본은 tied embedding [262144, 640] 한 덩어리(671 MB)라 스마트폰 GPU(한계 128 MiB)에서 막혔다.
  // build_web_models.py 가 이걸 CHUNK_ROWS=16384 행 청크로 나눠 두었으므로 가장 큰 텐서는
  // 16384 × 640 × 4 B = 41.9 MB 다. 그래도 넘는 어댑터는 사실상 없지만 받기 전에 확인한다.
  const NEED = 16384 * 640 * 4;

  async function pickDevice() {
    if (!navigator.gpu) return { device: 'wasm', why: 'WebGPU 미지원 브라우저' };
    let adapter;
    try { adapter = await navigator.gpu.requestAdapter(); }
    catch { return { device: 'wasm', why: 'GPU 어댑터 요청 실패' }; }
    if (!adapter) return { device: 'wasm', why: 'GPU 어댑터 없음' };

    const { maxStorageBufferBindingSize: b, maxBufferSize: m } = adapter.limits;
    return { device: 'webgpu', limit: Math.min(b, m),
             why: adapter.info?.description || adapter.info?.vendor || 'GPU' };
  }

  // WebGPU 가 안 되면 WASM 으로 내린다. 내려간 이유는 반드시 note 로 남긴다.
  // 선택이 조용히 버려지면 원인을 찾을 수 없다.
  function resolvePlan(chosen) {
    if (chosen.device !== 'webgpu') return { device: 'wasm', note: chosen.why };
    if (chosen.limit >= NEED) return { device: 'webgpu' };
    return { device: 'wasm', note: `GPU 버퍼 한계 ${mb(chosen.limit)} < 텐서 최대 ${mb(NEED)} → WASM` };
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
    if (!d) return;                                   // 그래프 파일(1 MB) 등은 표시하지 않는다
    if (status === 'reset') {                         // 다음 시도 → 바를 처음부터
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
  // 스마트폰은 탭 메모리 한계를 넘으면 렌더러/WebContent 프로세스를 죽이고 페이지를 조용히 다시 연다.
  // 콘솔도 오류도 남지 않아 "다운로드만 반복"처럼 보인다. 로드 시작 시 표식을 남기고
  // ready/fatal 에서 지우면, 표식이 남은 채 부팅된 경우 = 지난 시도가 도중에 끊긴 것이다.
  // 예외로 잡히는 실패는 워커가 WASM 으로 넘어가지만, 프로세스 종료는 여기서만 잡을 수 있다.
  const ATTEMPT_KEY = 'didimdol.loadAttempt';
  const markAttempt = (device) => {
    try { localStorage.setItem(ATTEMPT_KEY, JSON.stringify({ device, t: Date.now() })); } catch {}
  };
  const clearAttempt = () => { try { localStorage.removeItem(ATTEMPT_KEY); } catch {} };
  function reportBrokenAttempt() {
    let prev = null;
    try { prev = JSON.parse(localStorage.getItem(ATTEMPT_KEY) ?? 'null'); } catch {}
    if (!prev) return;
    clearAttempt();
    const when = new Date(prev.t).toLocaleTimeString('ko-KR');
    addRow(-1, null,
      `지난 시도(${when} · ${prev.device})가 모델 로드 도중 끝났습니다. 스마트폰은 탭 메모리 한계를 넘으면 `
      + `오류 없이 페이지가 다시 열립니다. 다른 탭을 닫고 다시 시도해 보세요. `
      + `가중치는 파일 단위로 캐시되므로 이미 받은 부분은 다시 받지 않습니다.`);
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
      case 'fallback':
        badge(`⚠ 실패 → ${m.next}`, 'cpu');
        addRow(-1, null, `이전 시도 실패, ${m.next} 로 다시 시도합니다: ${m.why}`);
        markAttempt(m.device);
        console.warn('fallback:', m.why);
        break;
      case 'ready':
        clearAttempt();
        badge(`${m.device === 'webgpu' ? '⚡ WebGPU' : '🐢 WASM(CPU)'} · fp32 로 실행 중`,
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
    if (loaded) {
      els.runbar.hidden = false;
      setBar(bars.run, 0, `0 / ${nRows}`);
      worker.postMessage({ type: 'run' });
    } else {
      els.prep.hidden = false;
      const p = resolvePlan(chosen);
      if (p.note) badge(`↓ ${p.note}`, 'cpu');
      markAttempt(p.device);
      worker.postMessage({ type: 'load', device: p.device });
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
      els.body.hidden = true;                     // 이 탭의 컨트롤만 치운다. 다른 탭은 그대로.
      return;
    }
    chosen = await pickDevice();
    reportBrokenAttempt();
    persistStorage();                             // 기다리지 않는다. 승인 여부가 로드를 막지 않는다.
    // iOS Safari 일반 탭은 7일간 상호작용이 없으면 캐시를 통째로 지운다. 홈 화면 앱은
    // 저장소가 분리되어 이 규칙에서 빠지므로, 홈 화면이 아닌 iOS 에서만 한 줄 안내한다.
    if (isIOS && !isStandalone) els.homeHint.hidden = false;
    if (chosen.device === 'webgpu') badge(`⚡ WebGPU · ${chosen.why}`, 'gpu');
    else                            badge(`🐢 WASM(CPU) · ${chosen.why}`, 'cpu');
    if (chosen.limit) badge(`GPU 버퍼 한계 ${mb(chosen.limit)} (텐서 최대 ${mb(NEED)} · 임베딩 16분할)`);
    els.phase.textContent = '준비 완료';
    els.start.disabled = false;
  })();
}
