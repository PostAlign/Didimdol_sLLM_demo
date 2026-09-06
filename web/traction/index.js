/**
 * 견인 모드 추천 정확도 탭 — 견인 모드 추천 데이터셋 100건을 브라우저에서 직접 추론해
 * TOP-1 · TOP-3 정확도와 행별 예측(정답 · TOP-1 · TOP-2 · TOP-3 · 권장 force/time)을 표로 보여 준다.
 * 개별 행에 확률은 적지 않는다.
 *
 * initTraction(root) 는 #tab-traction 패널 루트를 받아 그 안에서만 DOM 을 찾는다. 셸(app.js)이나
 * 다른 탭의 요소는 건드리지 않는다. 가중치(1.5 MB)는 '성능 평가 시작' 을 눌렀을 때 처음 받고,
 * 그 뒤로는 받아 둔 모델로 바로 다시 돌린다.
 *
 * 모델은 ./model.js 의 순수 JS 추론기(의존성 없음)라 100건을 한 번에 돌리면 수십 ms 만에 끝난다.
 * 그런데 그렇게 하면 결과가 통째로 튀어나와 미리 계산해 둔 것과 구별이 안 된다. 그래서 한 행씩
 * 추론하고 그 행을 바로 표에 붙인다 — 행마다 짧게 쉬어 진행바와 정확도가 실시간으로 올라가는
 * 것이 보이게 한다. 워커는 필요 없다. 한 행이 1 ms 안팎이라 메인 스레드가 막히지 않는다.
 *
 * 경로는 반드시 import.meta.url 기준이다. 페이지 기준 './model' 은 루트의 model/ 즉 sLLM ONNX
 * 디렉터리를 가리키게 된다.
 */

import { pct, esc, setBar } from '../ui.js';
import { createModel } from './model.js';

export function initTraction(root) {
  const $ = (s) => root.querySelector(s);
  const els = {
    start: $('#trStart'), status: $('#trStatus'), error: $('#trError'),
    n: $('#trN'), top1: $('#trTop1'), top3: $('#trTop3'),
    run: $('#trRun'), runbar: $('#trRun .dl'),
    stats: $('#trStats'), table: $('#trTableWrap'), wrap: $('#trTableWrap .tablewrap'), tbody: $('#trTable tbody'),
  };
  // 행 사이 간격. 100행이면 3초 남짓 — 돌아가는 게 보이면서 기다리기엔 짧다.
  const ROW_GAP_MS = 30;

  const getJson = async (rel) => {
    const r = await fetch(new URL(rel, import.meta.url));
    if (!r.ok) throw new Error(`${rel} 로드 실패: ${r.status}`);
    return r.json();
  };

  // 모델과 데이터셋은 첫 실행 때 한 번만 받아 둔다. 실패하면 비워 두어 다음 클릭에서 다시 받는다.
  let loaded = null;
  async function load() {
    if (loaded) return loaded;
    els.status.textContent = '모델과 데이터셋을 받는 중…';
    const [weights, pre, dataset] = await Promise.all([
      getJson('./model/weights.json'),
      getJson('./model/preprocess.json'),
      getJson('./data/data.json'),
    ]);
    loaded = { model: createModel(weights, pre), rows: dataset.rows };
    return loaded;
  }

  els.start.onclick = async () => {
    els.start.disabled = true;
    els.error.hidden = true;
    els.stats.hidden = true;                      // 시작하면 지난 결과는 즉시 치운다
    els.table.hidden = true;
    els.run.hidden = true;
    try {
      const { model, rows } = await load();
      await evaluate(model, rows);
    } catch (e) {
      els.status.textContent = '오류';
      els.error.textContent = e.message;
      els.error.hidden = false;
    } finally {
      els.start.disabled = false;
    }
  };

  // 전처리 오류로 실패한 행은 빗나간 것으로 세어 분모에 남긴다 — sLLM 탭이 실패 행을 평균에
  // 포함하는 것과 같은 기준. 실패 행을 빼면 정확도가 실제보다 높게 보인다.
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

  // 한 행 추론 → 표 행 하나. 정확도는 그때까지 돈 행 기준으로 매 행 갱신한다.
  function predictRow(model, row, i) {
    const truth = row.final_label;
    const tr = document.createElement('tr');
    let ranked;
    try { ranked = model.predict(row).ranked; }
    catch (e) {
      tr.className = 'miss';
      tr.innerHTML = `<td>${i + 1}</td><td>${esc(truth)}</td>
        <td colspan="4" class="fail">추론 실패 · ${esc(e.message)}</td>`;
      return { tr, ok1: false, ok3: false };
    }
    const [a, b, c] = ranked;
    const ok1 = a.label === truth;
    const ok3 = ok1 || b.label === truth || c.label === truth;
    tr.className = ok1 ? 'hit' : ok3 ? 'near' : 'miss';
    // 확률은 적지 않는다. 순위별 예측 라벨과 1위 모드의 권장 force/time 만 보여 준다.
    tr.innerHTML = `<td>${i + 1}</td><td>${esc(truth)}</td>
      <td class="pred">${esc(a.label)}</td><td>${esc(b.label)}</td><td>${esc(c.label)}</td>
      <td>${a.force_kg} kg / ${a.time_min} min</td>`;
    return { tr, ok1, ok3 };
  }

  async function evaluate(model, rows) {
    const total = rows.length;
    let hit1 = 0, hit3 = 0;
    els.tbody.replaceChildren();
    els.n.textContent = `0 / ${total}`;
    els.top1.textContent = '–';
    els.top3.textContent = '–';
    setBar(els.runbar, 0, `0 / ${total}`);
    els.run.hidden = false;
    els.stats.hidden = false;
    els.table.hidden = false;
    els.status.textContent = '평가 중…';

    for (let i = 0; i < total; i++) {
      const { tr, ok1, ok3 } = predictRow(model, rows[i], i);
      hit1 += ok1; hit3 += ok3;
      const done = i + 1;
      els.tbody.append(tr);
      els.n.textContent = `${done} / ${total}`;
      els.top1.textContent = pct(hit1 / done);
      els.top3.textContent = pct(hit3 / done);
      setBar(els.runbar, done / total, `${done} / ${total}`);
      // 표 안에서만 따라간다. 페이지 전체를 끌어내리면 위의 정확도 카드가 안 보인다.
      els.wrap.scrollTop = els.wrap.scrollHeight;
      await sleep(ROW_GAP_MS);
    }
    els.n.textContent = total;
    els.status.textContent = `완료 · ${total}건`;
  }
}
