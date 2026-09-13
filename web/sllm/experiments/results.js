export const isResident = kind => ['resident', 'resident-opfs', 'resident-opfs-tokenizer'].includes(kind);
// The OPFS pair uses the same application worker and JS imports. Neither creates
// an ORT WASM instance; only one retains the prepared tokenizer.
export const isSimpleProbe = kind => kind === 'resident' || kind === 'runtime';
export const canRepeat = kind => ['resident', 'resident-opfs', 'resident-opfs-tokenizer', 'session-only', 'load', 'runtime-resident'].includes(kind);
export const applicationOperation = kind => ['tokenizer', 'session-only', 'resident-opfs', 'resident-opfs-tokenizer', 'runtime-resident'].includes(kind) ? kind : 'load';

export function executionSettings(kind, config = {}) {
  return { runtimeMode: isResident(kind) ? null : config.mode || 'asyncify',
    idleSeconds: kind === 'runtime' ? config.idleSeconds ?? 120 : kind === 'session-only' ? config.idleSeconds ?? 0 : 0,
    inputSource: kind === 'resident' ? 'synthetic' : ['resident-opfs', 'resident-opfs-tokenizer', 'runtime-resident'].includes(kind) ? 'opfs-cache' : null };
}

/** Describe evidence, never promote screen preferences to completed work. Old exports remain readable. */
export function executionEvidence(result, run = null) {
  const kind = result.kind, summary = run?.summary || result;
  const environment = run?.environment || result.environment || {};
  const observes = ['runtime', 'session-only'].includes(kind), prefix = kind === 'session-only' ? 'session' : 'runtime';
  const idleStart = run?.milestones?.[`${prefix}-idle-start`];
  const idleComplete = run?.milestones?.[`${prefix}-idle-complete`];
  const idleProgress = [...(run?.records || [])].reverse().find(record => record.stage === `${prefix}-idle`);
  const idleRequestedSeconds = observes ? idleComplete?.idleSeconds ?? idleStart?.idleSeconds
    ?? summary.idleSeconds ?? environment.idleSeconds ?? null : 0;
  const idleElapsedMs = observes ? idleComplete?.idleElapsedMs ?? summary.idleElapsedMs
    ?? idleProgress?.idleElapsedMs ?? idleStart?.idleElapsedMs ?? null : 0;
  const idleAcceptanceCompleted = observes && idleElapsedMs >= 120000 && idleRequestedSeconds >= 120
    && (summary.idleAcceptanceCompleted === true || !!idleComplete);
  let completedScope = null;
  if (result.success && !run?.fault && run?.cleanup?.success !== false && !run?.cleanupError) {
    if (kind === 'runtime-resident' && summary.allBytesUsed && summary.inferenceVerified && summary.ortWasmInstantiated &&
      summary.sameDevice && summary.smallSessionRetained && summary.modelSessionCreated === false) completedScope = 'runtime-and-gpu-residency';
    else if (kind === 'resident-opfs-tokenizer' && summary.allBytesUsed && summary.tokenizerPrepared && summary.modelSessionCreated === false) {
      completedScope = 'tokenizer-and-gpu-residency';
    } else if (kind !== 'resident-opfs-tokenizer' && isResident(kind) && summary.allBytesUsed) completedScope = 'gpu-residency';
    else if (kind === 'session-only' && summary.modelSessionCreated && summary.tokenizerPrepared === false) {
      completedScope = idleAcceptanceCompleted ? 'model-session-and-idle' : 'model-session';
    }
    else if (kind === 'tokenizer' && summary.tokenizerPrepared && summary.modelSessionCreated === false) completedScope = 'tokenizer-preparation';
    else if (kind === 'runtime' && (summary.inferenceVerified || run?.milestones?.['runtime-inference-complete'])) {
      completedScope = idleAcceptanceCompleted ? 'small-runtime-and-idle' : 'small-runtime-inference';
    } else if (['load', 'warm-load'].includes(kind)) completedScope = 'model-load';
    else if (kind === 'probe') completedScope = 'short-long-inference';
    else if (kind === 'evaluation') completedScope = 'two-evaluations';
  }
  return { runtimeMode: isResident(kind) ? null : environment.runtimeMode ?? result.mode ?? null,
    loadOrder: environment.loadOrder ?? summary.tokenizer?.loadOrder ?? null,
    tokenizerBuild: environment.build?.tokenizer ?? summary.tokenizer?.tokenizerBuild ?? null,
    tokenizerPrepared: summary.tokenizerPrepared ?? (run?.milestones?.['tokenizer-ready'] ? true : run?.milestones?.['session-create']?.tokenizerPrepared ?? null),
    tokenizerFormat: summary.tokenizer?.tokenizerFormat ?? run?.milestones?.['tokenizer-ready']?.tokenizerFormat ?? null,
    requestedTokenizerFormat: environment.tokenizerFormat ?? result.tokenizerFormat ?? null,
    modelSessionCreated: summary.modelSessionCreated ?? (run?.milestones?.['session-create-complete'] ? true : null),
    ortJavaScriptLoaded: environment.ortJavaScriptLoaded ?? null,
    ortJavaScriptMode: environment.ortJavaScriptMode ?? null,
    ortWasmInstantiated: summary.ortWasmInstantiated ?? (summary.modelSessionCreated === true || run?.milestones?.['ort-wasm-complete'] ? true : null),
    diagnosticsMode: run?.persistence?.mode ?? environment.diagnosticsMode ?? null,
    sameDevice: summary.sameDevice ?? null, smallSessionRetained: summary.smallSessionRetained ?? null,
    inputSource: summary.inputSource ?? executionSettings(kind).inputSource,
    verification: summary.verification ?? null, allocationOrder: summary.allocationOrder ?? null,
    idleRequestedSeconds, idleElapsedMs, idleAcceptanceCompleted, completedScope };
}

export const scopeLabel = scope => ({ 'gpu-residency': 'GPU 상주 검증 완료',
  'runtime-and-gpu-residency': '작은 ORT 세션·GPU 상주 검증 완료',
  'tokenizer-and-gpu-residency': '토크나이저·GPU 상주 검증 완료', 'model-session': '모델 세션 생성 완료 (토크나이저 없음)',
  'tokenizer-preparation': '토크나이저 준비 완료',
  'model-session-and-idle': '모델 세션 생성·120초 관찰 완료 (토크나이저 없음)',
  'small-runtime-and-idle': '작은 모델 추론·120초 관찰 완료', 'small-runtime-inference': '작은 모델 추론 완료',
  'model-load': '모델 로딩 완료', 'short-long-inference': '짧은·긴 입력 추론 완료',
  'two-evaluations': '100건 평가 2회 완료' }[scope] || '성공');

export function seriesSummary(results, active = null) {
  const groups = new Map();
  for (const result of [...results, ...(active ? [active] : [])]) {
    const key = result.seriesId || result.runIds?.[0] || result.runId;
    if (!groups.has(key)) groups.set(key, { seriesId: key, kind: result.kind,
      requestedRuns: result.requestedRuns ?? null, startedRuns: 0, successfulRuns: 0, runIds: [] });
    const series = groups.get(key), id = result.runIds?.[0] || result.runId;
    if (series.runIds.includes(id)) continue;
    series.runIds.push(id); series.startedRuns++;
    if (result.success) series.successfulRuns++;
  }
  return [...groups.values()];
}
