import { RunDiagnostics, newRunId, buildIdentity, saveCheckpoint } from '../diagnostics.js';
import { STAGING_MIB } from '../range-loader.js';
import { FIXTURE } from './fixture.js';
import { runtimeRelease } from '../ort-runtime.js';
import { executionSettings } from './results.js';
import { residentProbe, opfsResidentProbe, runtimeProbe } from './probes.js';

self.onmessage = async ({ data }) => {
  const stagingMiB = data.stagingMiB ?? 8, mode = data.mode || 'asyncify';
  const settings = executionSettings(data.kind, data), { idleSeconds } = settings;
  const journal = new RunDiagnostics(data.runId || newRunId(), { ...data.environment, userAgent: navigator.userAgent,
    experiment: data.kind, ...settings, stagingMiB,
    fixture: !!data.fixture, workerURL: self.location.href });
  const checkpoint = async record => {
    await journal.checkpoint(record);
    self.postMessage({ type: 'progress', record });
  };
  const start = performance.now();
  try {
    const { build } = await runtimeRelease();
    journal.state.environment.build = buildIdentity(build);
    await checkpoint({ stage: 'probe-start' });
    if (!STAGING_MIB.includes(stagingMiB) || !Number.isInteger(idleSeconds) || idleSeconds < 0 || idleSeconds > 600) throw new Error('Invalid probe settings');
    let result;
    if (['resident', 'resident-opfs'].includes(data.kind)) {
      const manifest = data.fixture ? FIXTURE.manifest : await fetch(new URL('../../../model/initializers.json', import.meta.url)).then(r => r.json());
      if (data.kind === 'resident') result = await residentProbe(manifest, checkpoint, stagingMiB);
      else {
        const execute = async lock => {
          if (!lock) throw new Error('다른 탭에서 모델을 준비 중입니다. 해당 작업이 끝난 후 다시 시도해 주세요.');
          return opfsResidentProbe(manifest, checkpoint, stagingMiB, !!data.fixture);
        };
        result = navigator.locks ? await navigator.locks.request('didimdol-model-load', { ifAvailable: true }, execute) : await execute(true);
      }
    } else if (data.kind === 'runtime' || data.kind === 'runtime-resident') {
      const options = { onCleanup: cleanup => saveCheckpoint(cleanup, `cleanup:${journal.state.runId}`) };
      if (data.kind === 'runtime-resident') {
        if (!data.fixture) throw new Error('Production combined comparison uses the application worker');
        options.residentManifest = FIXTURE.manifest; options.fixture = true;
      }
      const execute = () => runtimeProbe(mode, checkpoint, idleSeconds, stagingMiB, options);
      result = data.kind === 'runtime-resident' && navigator.locks
        ? await navigator.locks.request('didimdol-model-load', execute) : await execute();
    }
    else throw new Error(`Unknown experiment: ${data.kind}`);
    result = { ...result, success: !journal.state.fault, durationMs: performance.now() - start,
      releaseId: build.releaseId, stagingMiB, environment: journal.state.environment };
    await journal.finish('complete', result);
    self.postMessage({ type: 'result', result });
  } catch (error) {
    const result = { success: false, error: String(error.stack || error), durationMs: performance.now() - start };
    await journal.finish('failed', result);
    self.postMessage({ type: 'result', result });
  }
};
