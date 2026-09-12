/**
 * 셸 — 탭 전환만 한다. 각 탭의 동작은 자기 모듈이 갖는다.
 *
 *   sLLM 성능 평가         → ./sllm/index.js (기기 판정 · 워커 · 모델 로드 · 생성 · 채점)
 *   견인 모드 추천 정확도   → ./traction/index.js (순수 JS 추론 · 데이터셋 100건 TOP-1 · TOP-3)
 *
 * 현재 탭은 location.hash (#sllm / #traction) 로 기억한다. 새로고침이나 홈 화면 앱 재실행에서도
 * 보던 탭이 유지되고, 링크로 특정 탭을 바로 열 수 있다.
 */

import { initSllm } from './sllm/index.js';
import { initTraction } from './traction/index.js';

const TABS = ['sllm', 'traction'];
const panels  = Object.fromEntries(TABS.map((t) => [t, document.getElementById(`tab-${t}`)]));
const buttons = Object.fromEntries(TABS.map((t) => [t, document.querySelector(`.tabs [data-tab="${t}"]`)]));

const current = () => {
  const h = location.hash.slice(1);
  return TABS.includes(h) ? h : TABS[0];
};

function show(id) {
  for (const t of TABS) {
    panels[t].hidden = t !== id;
    buttons[t].setAttribute('aria-selected', String(t === id));
  }
}

// 탭이 숨겨져 있어도 sLLM 모듈은 바로 준비한다. 기기 판정과 지난 시도 복구는 탭과 무관하고,
// 평가가 도는 중에 탭을 옮겨도 워커는 계속 돌아야 한다. 탭 전환은 display 만 바꾼다.
// 견인 모듈도 DOM 만 잡아 둔다. 가중치는 그 탭의 '성능 평가 시작' 을 눌렀을 때 받는다.
initSllm(panels.sllm);
initTraction(panels.traction);

for (const t of TABS) {
  buttons[t].onclick = () => {
    // pushState 가 아니라 replaceState 다. 탭을 오가는 게 뒤로 가기 이력에 쌓이면 안 된다.
    history.replaceState(null, '', `#${t}`);
    show(t);
  };
}
addEventListener('hashchange', () => show(current()));
show(current());
