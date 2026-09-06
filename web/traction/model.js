/**
 * 허리 견인기 모드 추천 모델 — 순수 JavaScript 추론기 (의존성 없음, 프레임워크 무관 ES module)
 *
 * export_model.py 가 뽑은 model/weights.json + model/preprocess.json 만 있으면 브라우저/Node 어디서든 동작한다.
 * 순전파는 np_model.py 의 forward() 와 1:1 대응한다.
 *
 *   import { loadModel } from './traction_model.js';
 *   const model = await loadModel('./model');          // fetch 기반 (정적 서빙)
 *   const out = model.predict({ age_group:'50s', sex:'F', ... });
 *   out.top1.label, out.top1.force_kg, out.probs, out.vsnWeights ...
 *
 * Node 에서는 파일을 직접 읽어 createModel(weights, preprocess) 를 쓰면 된다.
 */

// ------------------------------------------------------------------ 수치 연산
function dense(x, d) {
  const { kernel, bias } = d;
  const out = bias.slice();
  const inDim = kernel.length;
  for (let i = 0; i < inDim; i++) {
    const xi = x[i];
    if (xi === 0) continue;
    const row = kernel[i];
    for (let j = 0; j < row.length; j++) out[j] += xi * row[j];
  }
  return out;
}

function matNoBias(x, kernel) {
  const out = new Array(kernel[0].length).fill(0);
  for (let i = 0; i < kernel.length; i++) {
    const xi = x[i];
    if (xi === 0) continue;
    const row = kernel[i];
    for (let j = 0; j < row.length; j++) out[j] += xi * row[j];
  }
  return out;
}

const elu = (v) => (v > 0 ? v : Math.expm1(v));
const sigmoid = (v) => 1 / (1 + Math.exp(-v));

function softmax(x) {
  const m = Math.max(...x);
  const e = x.map((v) => Math.exp(v - m));
  const s = e.reduce((a, b) => a + b, 0);
  return e.map((v) => v / s);
}

function layerNorm(x, ln, eps) {
  const n = x.length;
  let mean = 0;
  for (const v of x) mean += v;
  mean /= n;
  let variance = 0;
  for (const v of x) variance += (v - mean) * (v - mean);
  variance /= n;
  const inv = 1 / Math.sqrt(variance + eps);
  const out = new Array(n);
  for (let i = 0; i < n; i++) out[i] = (x[i] - mean) * inv * ln.gamma[i] + ln.beta[i];
  return out;
}

/** GRN (Gated Residual Network), 추론 모드: dropout 없음 */
function grn(x, g, eps) {
  const skip = g.skip_proj ? matNoBias(x, g.skip_proj.kernel) : x;
  let h = dense(x, g.dense1).map(elu);
  h = dense(h, g.dense2);
  const gate = dense(h, g.gate).map(sigmoid);
  const y = new Array(h.length);
  for (let i = 0; i < h.length; i++) y[i] = skip[i] + gate[i] * h[i];
  return layerNorm(y, g.layernorm, eps);
}

// ------------------------------------------------------------------ 모델
export function createModel(weights, pre) {
  if (weights.format !== 'traction-mode-classifier/v1' || pre.format !== 'traction-mode-classifier/v1') {
    throw new Error('weights/preprocess 포맷 버전이 맞지 않습니다');
  }
  const eps = pre.layernorm_epsilon;
  const catIndex = {};
  for (const c of pre.categorical_order) {
    catIndex[c] = new Map(pre.categorical[c].map((v, i) => [v, i]));
  }
  const nPain = pre.pain_cols.length;

  /** 원시 환자 객체 → { catIdx[11], binary[19], cont[8] }. 잘못된 값은 throw. */
  function preprocess(patient) {
    const catIdx = pre.categorical_order.map((c) => {
      const raw = patient[c];
      const idx = catIndex[c].get(String(raw));
      if (idx === undefined) {
        throw new Error(`${c}: 허용되지 않는 값 "${raw}" (가능: ${pre.categorical[c].join(', ')})`);
      }
      return idx;
    });

    const binary = [];
    for (const c of pre.pain_cols) {
      const v = Number(patient[c]);
      if (v !== 0 && v !== 1) throw new Error(`${c}: 0 또는 1 이어야 합니다 (받은 값 ${patient[c]})`);
      binary.push(v);
    }
    for (const field of ['aggravating_posture', 'relieving_posture']) {
      const raw = String(patient[field]);
      if (!pre.posture_options.includes(raw)) {
        throw new Error(`${field}: 허용되지 않는 값 "${raw}" (가능: ${pre.posture_options.join(', ')})`);
      }
      for (const o of pre.posture_options) binary.push(raw === o ? 1 : 0);
    }

    const cont = pre.continuous.order.map((c, i) => {
      const v = Number(patient[c]);
      if (!Number.isFinite(v)) throw new Error(`${c}: 숫자가 아닙니다 (받은 값 ${patient[c]})`);
      return (v - pre.continuous.mean[i]) / pre.continuous.scale[i];
    });
    return { catIdx, binary, cont };
  }

  /** 전처리된 입력 → { probs[6], vsnWeights[21] } */
  function forward({ catIdx, binary, cont }) {
    const feats = [];
    weights.categorical.forEach((c, j) => {
      feats.push(dense(c.embedding[catIdx[j]], c.proj));
    });
    feats.push(dense(binary.slice(0, nPain), weights.pain_proj));
    feats.push(dense(binary.slice(nPain), weights.posture_proj));
    weights.continuous.forEach((c, i) => {
      feats.push(dense([cont[i]], c.proj));
    });

    // Variable Selection Network
    const flat = feats.flat();
    const sel = grn(flat, weights.vsn.select_grn, eps);
    const vsnWeights = softmax(dense(sel, weights.vsn.select_dense));
    const dModel = weights.d_model;
    const x = new Array(dModel).fill(0);
    feats.forEach((f, i) => {
      const t = grn(f, weights.vsn.feature_grns[i], eps);
      const w = vsnWeights[i];
      for (let k = 0; k < dModel; k++) x[k] += w * t[k];
    });

    let h = x;
    for (const g of weights.backbone) h = grn(h, g, eps);
    const probs = softmax(dense(h, weights.logits));
    return { probs, vsnWeights };
  }

  function describe(classIndex, prob) {
    const label = pre.classes[classIndex];
    return { index: classIndex, label, prob, ...pre.mode_params[label] };
  }

  /** 원시 환자 객체 → 추천 결과 */
  function predict(patient) {
    const { probs, vsnWeights } = forward(preprocess(patient));
    const ranked = probs
      .map((p, i) => [p, i])
      .sort((a, b) => b[0] - a[0])
      .map(([p, i]) => describe(i, p));
    const importance = pre.vsn_feature_names
      .map((name, i) => ({ name, weight: vsnWeights[i] }))
      .sort((a, b) => b.weight - a.weight);
    return { probs, ranked, top1: ranked[0], top2: ranked[1], vsnWeights, importance };
  }

  return {
    classes: pre.classes,
    modeParams: pre.mode_params,
    inputFields: pre.input_fields,
    preprocessSpec: pre,
    preprocess,
    forward,
    predict,
    predictBatch: (patients) => patients.map(predict),
  };
}

/** 정적 서빙 환경: baseUrl/weights.json, baseUrl/preprocess.json 을 fetch */
export async function loadModel(baseUrl = './model') {
  const [weights, pre] = await Promise.all([
    fetch(`${baseUrl}/weights.json`).then((r) => {
      if (!r.ok) throw new Error(`weights.json 로드 실패: ${r.status}`);
      return r.json();
    }),
    fetch(`${baseUrl}/preprocess.json`).then((r) => {
      if (!r.ok) throw new Error(`preprocess.json 로드 실패: ${r.status}`);
      return r.json();
    }),
  ]);
  return createModel(weights, pre);
}

/** 키·몸무게로 bmi 계산 (데이터 생성기와 동일: 소수 1자리 반올림) */
export function computeBmi(heightCm, weightKg) {
  const h = heightCm / 100;
  return Math.round((weightKg / (h * h)) * 10) / 10;
}
