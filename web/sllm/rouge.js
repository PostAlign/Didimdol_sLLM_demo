/**
 * ROUGE-1 (unigram) — 프로젝트 토크나이저 기반.
 *
 * rouge_score 의 기본 토크나이저는 `[^a-z0-9]+` 로 치환하므로 한글이 통째로 사라진다.
 * 이 데이터의 정답 문장을 넣으면 남는 토큰이 ['39','119'] 두 개뿐이고, 0 점이 아니라
 * F1 0.667 같은 그럴듯한 허수가 나온다. 그래서 생성에 쓰는 Gemma 토크나이저를 그대로
 * 재사용해 예측/정답을 같은 방식으로 쪼갠다.
 *
 * 비교 단위는 토큰 문자열이다. 문자·숫자를 하나도 포함하지 않는 토큰(공백 '▁', 구두점)은
 * 버린다. 이걸 안 하면 아무 문장 쌍이나 공백 겹침만으로 점수가 붙는다.
 */

const HAS_WORD = /[\p{L}\p{N}]/u;

export function makeRouge1(tokenizer) {
  const specials = new Set(tokenizer.all_special_tokens ?? []);
  const keepCache = new Map(); // token -> boolean

  function pieces(text) {
    const out = [];
    for (const t of tokenizer.tokenize(text, { add_special_tokens: false })) {
      let keep = keepCache.get(t);
      if (keep === undefined) {
        keep = !specials.has(t) && HAS_WORD.test(t.replaceAll('▁', ''));
        keepCache.set(t, keep);
      }
      if (keep) out.push(t);
    }
    return out;
  }

  /** @returns {{p:number,r:number,f1:number,nPred:number,nRef:number}} */
  return function rouge1(pred, ref) {
    const P = pieces(pred), R = pieces(ref);
    if (P.length === 0 || R.length === 0) {
      return { p: 0, r: 0, f1: 0, nPred: P.length, nRef: R.length };
    }
    const rc = new Map();
    for (const t of R) rc.set(t, (rc.get(t) ?? 0) + 1);

    // clipped overlap: 예측의 각 토큰은 정답에 나온 횟수까지만 인정
    let overlap = 0;
    const seen = new Map();
    for (const t of P) {
      const used = seen.get(t) ?? 0;
      if (used < (rc.get(t) ?? 0)) { overlap++; seen.set(t, used + 1); }
    }
    const p = overlap / P.length;
    const r = overlap / R.length;
    const f1 = p + r === 0 ? 0 : (2 * p * r) / (p + r);
    return { p, r, f1, nPred: P.length, nRef: R.length };
  };
}
