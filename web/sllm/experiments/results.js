export const isResident = kind => ['resident', 'resident-opfs', 'resident-opfs-tokenizer'].includes(kind);
// The OPFS pair uses the same application worker and JS imports. Neither creates
// an ORT WASM instance; only one retains the prepared tokenizer.
export const isSimpleProbe = kind => kind === 'resident' || kind === 'runtime';
export const canRepeat = kind => ['resident', 'resident-opfs', 'resident-opfs-tokenizer', 'session-only', 'load', 'runtime-resident'].includes(kind);
export const applicationOperation = kind => ['tokenizer', 'session-only', 'resident-opfs', 'resident-opfs-tokenizer', 'runtime-resident'].includes(kind) ? kind : 'load';

/** OS and browser versions from the user agent; the tester adds the device model. */
export function describeDevice(userAgent = '') {
  const ios = /(?:iPhone|CPU) OS (\d+)_(\d+)(?:_(\d+))?/.exec(userAgent);
  const android = /Android ([\d.]+)/.exec(userAgent);
  const family = /iPad/.test(userAgent) ? 'iPad' : /iPhone/.test(userAgent) ? 'iPhone' : android ? 'Android'
    : /Macintosh/.test(userAgent) ? 'Mac' : /Windows/.test(userAgent) ? 'Windows' : /Linux/.test(userAgent) ? 'Linux' : '';
  const os = ios ? `iOS ${[ios[1], ios[2], ios[3]].filter(Boolean).join('.')}` : android ? `Android ${android[1]}` : '';
  const browsers = [[/CriOS\/([\d.]+)/, 'Chrome'], [/FxiOS\/([\d.]+)/, 'Firefox'], [/EdgiOS\/([\d.]+)/, 'Edge'], [/Edg\/([\d.]+)/, 'Edge'],
    [/Chrome\/([\d.]+)/, 'Chrome'], [/Version\/([\d.]+).*Safari/, 'Safari'], [/Firefox\/([\d.]+)/, 'Firefox']];
  const browser = browsers.map(([pattern, name]) => { const match = pattern.exec(userAgent); return match && `${name} ${match[1]}`; }).find(Boolean) || '';
  return [family, os, browser].filter(Boolean).join(' / ');
}

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
    modelExecution: environment.modelExecution ?? summary.modelExecution ?? result.modelExecution ?? null,
    // Output buffers of the streamed projection (2 overlap reads with compute, 1 is the serial comparison); null before this field existed.
    outputBuffers: environment.outputBuffers ?? summary.outputBuffers ?? summary.sessionMetrics?.outputBuffers ?? result.outputBuffers ?? null,
    streaming: [run?.last, ...(run?.records || []).slice().reverse()].find(record => record?.streaming)?.streaming ?? summary.streaming ?? null,
    loadOrder: environment.loadOrder ?? summary.tokenizer?.loadOrder ?? null,
    tokenizerBuild: environment.build?.tokenizer ?? summary.tokenizer?.tokenizerBuild ?? null,
    tokenizerPrepared: summary.tokenizerPrepared ?? (run?.milestones?.['tokenizer-ready'] ? true : run?.milestones?.['session-create']?.tokenizerPrepared ?? null),
    tokenizerFormat: summary.tokenizer?.tokenizerFormat ?? run?.milestones?.['tokenizer-ready']?.tokenizerFormat ?? null,
    requestedTokenizerFormat: environment.tokenizerFormat ?? result.tokenizerFormat ?? null,
    modelSessionCreated: summary.modelSessionCreated ?? (run?.milestones?.['session-create-complete'] ? true : null),
    ortJavaScriptLoaded: environment.ortJavaScriptLoaded ?? null,
    ortJavaScriptMode: environment.ortJavaScriptMode ?? null,
    // A failure between `ort-wasm-start` and `ort-wasm-complete` is recorded as an `ort-wasm-error` fault; the runtime never came up.
    ortWasmInstantiated: summary.ortWasmInstantiated ?? (summary.modelSessionCreated === true || run?.milestones?.['ort-wasm-complete'] ? true
      : run?.fault?.stage === 'ort-wasm-error' || run?.milestones?.['ort-wasm-error'] ? false : null),
    wasmFailure: summary.wasmFailure ?? run?.milestones?.['ort-wasm-error'] ?? null,
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

/**
 * What the page knows about the runs before this one. A process kill cannot be
 * observed directly, so the count of results appended since the last interrupted
 * row and the gap since the previous run ended are the closest proxies for
 * "how much this renderer process has already done". Results are durable
 * across tabs and browser restarts; the gap says whether the previous run was
 * moments ago or long before.
 */
export function runContext(results = [], now = Date.now()) {
  const previous = results.at(-1) ?? null;
  let resultsSinceInterruption = 0;
  for (let i = results.length - 1; i >= 0 && !results[i].interrupted; i--) resultsSinceInterruption++;
  const endedAt = previous?.endedAt ?? (Number.isFinite(previous?.startedAt) && Number.isFinite(previous?.durationMs)
    ? previous.startedAt + previous.durationMs : null);
  return { resultsSinceInterruption,
    gapSincePreviousMs: endedAt == null ? null : Math.max(0, now - endedAt),
    previous: previous ? { kind: previous.kind ?? null, modelExecution: previous.modelExecution ?? null,
      success: previous.success ?? null, interrupted: previous.interrupted ?? false, endedAt,
      gpuRequestedCurrent: previous.comparison?.gpuRequestedCurrent ?? null,
      gpuWeightAllocated: previous.comparison?.gpuWeightAllocated ?? null } : null };
}

/** Free-text device log note ("JetsamEvent-2026-09-13-163104.ips · per-process-limit · 1,024 MiB") with the parts the analysis needs. */
export function parseDeviceLogNote(note) {
  const text = String(note ?? '').trim();
  if (!text) return null;
  const file = /[\w.-]*\.ips\b/i.exec(text)?.[0] ?? null;
  const reason = /per-process-limit|vm-pageshortage|highwater|fc-thrashing|jettisoned|memory limit|EXC_RESOURCE/i.exec(text)?.[0]?.toLowerCase() ?? null;
  const size = /(\d+(?:,\d{3})*(?:\.\d+)?)\s*(MiB|MB|GiB|GB)\b/i.exec(text);
  let footprintMiB = null;
  if (size) {
    const value = Number(size[1].replace(/,/g, ''));
    footprintMiB = { mib: value, mb: value * 1e6 / 2 ** 20, gib: value * 1024, gb: value * 1e9 / 2 ** 20 }[size[2].toLowerCase()];
  } else {
    // JetsamEvent bodies write `"rpages": 65536`; a hand-written note may put the number first. 16 KiB pages.
    const pages = /(\d+(?:,\d{3})*)\s*(?:rpages|pages)\b/i.exec(text) || /rpages\D{0,4}(\d+(?:,\d{3})*)/i.exec(text);
    if (pages) footprintMiB = Number(pages[1].replace(/,/g, '')) * 16384 / 2 ** 20;
  }
  return { note: text, file, reason, footprintMiB: footprintMiB == null ? null : Math.round(footprintMiB * 10) / 10 };
}
