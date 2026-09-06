/**
 * 탭 공용 표시 헬퍼.
 *
 * 탭 모듈(sllm/, 앞으로 traction/)은 자기 패널 루트 안에서만 DOM 을 찾고, 화면에 값을
 * 찍는 잔손질은 여기 것을 쓴다. 셸(app.js)이나 다른 탭의 요소를 건드리지 않는다.
 */

export const ms  = (v) => v >= 1000 ? `${(v / 1000).toFixed(2)}<span class="u">s</span>`
                                    : `${Math.round(v)}<span class="u">ms</span>`;
export const mb  = (b) => `${(b / 1e6).toFixed(1)} MB`;
export const pct = (x) => `${(x * 100).toFixed(1)}%`;
export const f3  = (x) => x.toFixed(3);
export const esc = (s) => String(s).replace(/[&<>]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' }[c]));

/** .dl 진행바 (index.html 의 .bar > .fill 과 .row > .s 구조) 에 비율과 라벨을 찍는다. */
export function setBar(el, fraction, label) {
  el.querySelector('.fill').style.width = `${Math.max(0, Math.min(1, fraction)) * 100}%`;
  el.querySelector('.s').textContent = label;
}

/** 배지 컨테이너에 칩을 하나씩 붙이는 함수를 만든다. cls 는 'gpu' | 'cpu' | ''. */
export function makeBadge(container) {
  return (text, cls = '') => {
    const s = document.createElement('span');
    s.className = `badge ${cls}`;
    s.textContent = text;
    container.append(s);
  };
}
