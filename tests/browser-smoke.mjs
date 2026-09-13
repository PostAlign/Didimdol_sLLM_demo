import { chromium } from 'playwright';
import { createServer } from 'node:http';
import { createReadStream } from 'node:fs';
import { stat, writeFile, mkdir, readFile } from 'node:fs/promises';
import path from 'node:path';
import assert from 'node:assert/strict';

const root = path.resolve(import.meta.dirname, '..');
const types = { '.mjs': 'text/javascript', '.js': 'text/javascript', '.wasm': 'application/wasm', '.json': 'application/json', '.html': 'text/html' };
const server = createServer(async (request, response) => {
  response.setHeader('Access-Control-Allow-Origin', '*');
  if (request.url === '/') { response.setHeader('Content-Type', 'text/html'); response.end('<!doctype html><title>ORT test</title>'); return; }
  const pathname = decodeURIComponent(new URL(request.url, 'http://localhost').pathname);
  const file = path.resolve(root, '.' + pathname);
  if (!file.startsWith(root + path.sep)) { response.writeHead(403).end(); return; }
  try {
    const info = await stat(file);
    response.writeHead(200, { 'Content-Type': types[path.extname(file)] || 'application/octet-stream', 'Content-Length': info.size });
    if (request.method === 'HEAD') { response.end(); return; }
    createReadStream(file).pipe(response);
  } catch { response.writeHead(404).end(); }
});
await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
const origin = `http://127.0.0.1:${server.address().port}`;
let browser;
const results = [];
try {
  browser = await chromium.launch({ headless: true, args: ['--enable-unsafe-webgpu', '--use-angle=swiftshader', '--disable-gpu-sandbox'] });
  {
    const page = await browser.newPage();
    await page.goto(origin);
    const atomic = await page.evaluate(async () => {
      const { RunDiagnostics, readRun, saveJournalEntries, readCheckpoint } = await import('/web/sllm/diagnostics.js');
      const run = new RunDiagnostics('atomic-journal');
      await run.checkpoint({ stage: 'gpu-wait', phase: 'before-call', destinationOffset: 8 });
      const key = 'journal:atomic-journal:head', before = await readCheckpoint(key);
      const success = await saveJournalEntries([[key, { ...before, recordCount: 2 }], ['cannot-clone', () => {}]]);
      return { success, before, after: await readCheckpoint(key), run: await readRun(run.state.runId) };
    });
    assert.equal(atomic.success, false);
    assert.deepEqual(atomic.after, atomic.before, 'a synchronous put failure must abort earlier writes in the same transaction');
    assert.equal(atomic.run.last.destinationOffset, 8);
    await page.reload();
    const recovered = await page.evaluate(async () => (await import('/web/sllm/diagnostics.js')).readRun('atomic-journal'));
    assert.equal(recovered.last.phase, 'before-call');
    assert.equal(recovered.recordCount, 1);
    results.push({ id: 'atomic-journal-recovery', success: true });
    await page.close();
  }
  for (const mode of (process.env.ORT_MODES ?? 'asyncify,jspi,stock').split(',').filter(Boolean)) {
    const page = await browser.newPage();
    page.on('pageerror', error => console.error('pageerror:', error.message));
    page.on('console', message => { if (message.type() === 'error') console.error('browser:', message.text()); });
    await page.goto(origin);
    const result = await page.evaluate(async ({ origin, mode }) => {
      const { asset } = await (await import(`${origin}/web/sllm/ort-runtime.js`)).runtimeRelease();
      const worker = new Worker(asset('web/sllm/experiments/worker.js'), { type: 'module' });
      return await new Promise((resolve, reject) => {
        const timeout = setTimeout(() => { worker.terminate(); reject(new Error('Browser test timed out')); }, 240000);
        worker.onerror = event => { clearTimeout(timeout); worker.terminate(); reject(new Error(event.message)); };
        worker.onmessage = ({ data }) => {
          if (data.type === 'result') { clearTimeout(timeout); worker.terminate(); resolve(data.result); }
        };
        worker.postMessage({ mode, storage: mode === 'stock' ? 'blob' : 'opfs', source: `${origin}/.work/test-experiments/experiments.json`, verifyFixture: true,
          experiment: { id: `smoke-${mode}`, kind: mode === 'stock' ? 'stock' : 'range', fileMiB: 32, stagingMiB: 8 } });
      });
    }, { origin, mode });
    results.push(result);
    console.log(JSON.stringify(result, null, 2));
    assert.equal(result.success, true, result.error);
    assert.equal(result.inferenceVerified, true);
    if (mode !== 'stock') {
      assert.equal(result.metrics.wasmTempPeak, 0);
      assert.equal(result.metrics.cpuStagingCurrent, 0);
      assert.equal(result.metrics.loadedInitializerCount, 2);
      assert.ok(result.metrics.cpuStagingPeak <= 16 * 2**20);
      assert.equal(result.sessionGpuLedger.mappedUploadRequested, 0, 'no full-tensor mapped GPU staging');
      assert.equal(result.sessionGpuLedger.tracking.status, 'complete', 'bind actual ORT device allocations');
      assert.ok(result.sessionGpuLedger.bufferCount >= result.metrics.gpuWeightBufferCount);
      assert.ok(result.sessionGpuLedger.requestedCurrent >= result.metrics.gpuWeightAllocated);
      assert.equal(result.metrics.gpuWriteReturnedBytes, result.metrics.totalExternalTensorBytes);
      assert.equal(result.metrics.gpuQueueCompletedBytes, result.metrics.totalExternalTensorBytes);
      assert.equal(result.metrics.gpuValidatedInitializerCount, 2);
      assert.equal(result.storage.peakOpenHandles, 1);
      assert.equal(result.storage.openHandles, 0);
      for (const kind of (mode === 'asyncify' ? ['resident', 'runtime', 'resident-opfs', 'runtime-resident'] : ['runtime', 'runtime-resident'])) {
        const probePage = await browser.newPage();
        const runtimeRequests = [];
        probePage.on('request', request => { if (/\/web\/vendor\/ort[.-].*\.(mjs|wasm)/.test(request.url())) runtimeRequests.push(request.url()); });
        await probePage.goto(origin);
        const probe = await probePage.evaluate(async ({ origin, mode, kind, stagingMiB, idleSeconds }) => {
          const { asset } = await (await import(`${origin}/web/sllm/ort-runtime.js`)).runtimeRelease();
          const worker = new Worker(asset('web/sllm/experiments/device-probes.js'), { type: 'module' });
          return new Promise((resolve, reject) => {
            const timer = setTimeout(() => { worker.terminate(); reject(new Error('Device probe timed out')); }, 240000);
            worker.onerror = e => { clearTimeout(timer); worker.terminate(); reject(new Error(e.message)); };
            worker.onmessage = ({ data }) => {
              if (data.type === 'result') { clearTimeout(timer); worker.terminate(); resolve(data.result); }
            };
            worker.postMessage({ kind, mode, fixture: true, idleSeconds, stagingMiB });
          });
        }, { origin, mode, kind, stagingMiB: 4, idleSeconds: Number(process.env.TEST_IDLE_SECONDS || 0) });
        results.push({ id: `${kind}-${mode}`, ...probe });
        console.log(JSON.stringify({ id: `${kind}-${mode}`, ...probe }, null, 2));
        assert.equal(probe.success, true, probe.error);
        assert.match(probe.releaseId, /^[a-f0-9]{64}$/);
        if (kind === 'runtime') {
          assert.equal(probe.inferenceVerified, true);
          assert.equal(probe.metrics.cpuStagingPeak, 4 * 2**20, 'OPFS reads straight into the reusable scratch');
          assert.equal(probe.storage.downloadedFiles, 1, 'the diagnostic generates and verifies its own cold-cache fixture');
          assert.equal(probe.idleAcceptanceCompleted, Number(process.env.TEST_IDLE_SECONDS || 0) >= 120);
        } else if (kind === 'runtime-resident') {
          assert.equal(probe.sameDevice, true);
          assert.equal(probe.runtimeDeviceId, probe.residentDeviceId);
          assert.equal(probe.gpuLedger.tracking.deviceCount, 1);
          assert.equal(probe.smallSessionRetained, true);
          assert.equal(probe.ortWasmInstantiated, true);
          assert.equal(probe.modelSessionCreated, false);
          assert.equal(probe.loadedInitializerCount, 2);
          assert.equal(probe.gpuWeightAllocated, probe.expectedBytes);
          const categories = Object.fromEntries(probe.gpuLedger.categories.map(item => [item.role, item]));
          assert.equal(categories.weight.requestedCurrent, probe.expectedBytes);
          assert.equal(categories['runtime-weight'].requestedCurrent, probe.runtimeMetrics.gpuWeightAllocated);
          assert.ok(probe.gpuLedger.requestedCurrent >= probe.expectedBytes + probe.runtimeMetrics.gpuWeightAllocated);
          assert.ok(runtimeRequests.some(url => url.endsWith('.wasm')));
          assert.equal(probe.storage.openHandles, 0);
        } else {
          assert.equal(probe.gpuWeightAllocated, probe.expectedBytes);
          assert.equal(probe.environment.runtimeMode, null);
          assert.equal(probe.environment.idleSeconds, 0);
          assert.equal(probe.allBytesUsed, true);
          assert.equal(probe.verification, 'u32-fnv1a-64-lanes-v1');
          assert.equal(probe.gpuLedger.tracking.status, 'complete');
          assert.deepEqual(runtimeRequests, [], 'resident comparisons must not import ORT binaries');
          if (kind === 'resident-opfs') {
            assert.equal(probe.storage.rangeReadBytes, probe.expectedBytes);
            assert.equal(probe.storage.openHandles, 0);
            assert.equal(probe.storage.peakOpenHandles, 1);
          }
        }
        await probePage.close();
      }
    }
    await page.close();
  }
  // Exercise the real tokenizer diagnostic through the UI and export its durable stages.
  {
    const page = await browser.newPage();
    const modelRequests = [];
    page.on('request', request => { if (/model\.onnx|\.wasm(?:\?|$)/.test(request.url())) modelRequests.push(request.url()); });
    let releaseBootstrap;
    const bootstrapGate = new Promise(resolve => { releaseBootstrap = resolve; });
    await page.route(`${origin}/web/vendor/build.json`, async route => { await bootstrapGate; await route.continue(); });
    await page.goto(`${origin}/web/sllm/experiments/index.html`, { waitUntil: 'commit' });
    await page.locator('#kind').waitFor();
    for (const id of ['kind', 'staging', 'repeats', 'start', 'export']) {
      assert.equal(await page.locator(`#${id}`).isDisabled(), true, 'controls wait for release bootstrap');
    }
    releaseBootstrap();
    await page.locator('#kind').selectOption('tokenizer');
    assert.equal(await page.locator('#staging').isDisabled(), true);
    await page.locator('#start').click();
    await page.locator('#status').getByText('토크나이저 준비 완료', { exact: true }).waitFor({ timeout: 60000 });
    const stored = await page.evaluate(() => JSON.parse(localStorage.getItem('didimdol.device-experiments.results.v3')));
    const result = stored.results.at(-1);
    assert.equal(result.navigation.reason, 'experiment-start');
    assert.equal(result.navigation.runId, result.runId);
    assert.equal(new URL(page.url()).searchParams.get('next'), result.navigation.id);
    assert.equal(result.success, true);
    assert.equal(result.execution.completedScope, 'tokenizer-preparation');
    assert.equal(result.execution.loadOrder, 'tokenizer-only');
    assert.equal(result.execution.tokenizerBuild.implementation, 'compact-bpe-v2');
    assert.equal(result.comparison.gpuWeightAllocated, null);
    const run = await page.evaluate(async runId => {
      const { asset } = await (await import('/web/sllm/ort-runtime.js')).runtimeRelease();
      return (await import(asset('web/sllm/diagnostics.js'))).readRun(runId);
    }, result.runId);
    assert.equal(run.summary.tokenizer.vocabSize, 262145, 'vocabulary includes the added token at ID 262144');
    assert.equal(run.summary.tokenizer.mergeCount, 514906);
    assert.equal(run.summary.tokenizer.memoryLayout.sourceReleased, true);
    assert.ok(run.preparation['tokenizer.json']['tokenizer-parse-start']);
    assert.ok(run.milestones['tokenizer-ready']);
    assert.equal(run.milestones['session-create'], undefined);
    assert.equal(run.milestones.ready, undefined);
    assert.deepEqual(modelRequests, [], 'tokenizer diagnostic must not fetch model weights or instantiate WASM');
    results.push({ id: 'tokenizer-ui', success: true, tokenizer: run.summary.tokenizer,
      preparation: run.preparation, persistence: run.persistence, diagnosticCharacters: JSON.stringify(run).length,
      noModelOrWasmRequests: true });
    await page.evaluate(async runId => {
      const state = JSON.parse(sessionStorage.getItem('didimdol.device-experiments.v2'));
      state.active = { kind: 'tokenizer', runId, startedAt: Date.now() };
      sessionStorage.setItem('didimdol.device-experiments.v2', JSON.stringify(state));
    }, result.runId);
    await page.reload();
    await page.locator('#status').getByText('이전 실험 완료 기록을 복구했습니다.').waitFor();
    assert.equal(await page.evaluate(() => JSON.parse(localStorage.getItem('didimdol.device-experiments.results.v3')).results.at(-1).success), true);
    await page.evaluate(async () => {
      const { asset } = await (await import('/web/sllm/ort-runtime.js')).runtimeRelease();
      const { RunDiagnostics } = await import(asset('web/sllm/diagnostics.js'));
      const run = new RunDiagnostics('tokenizer-parse-interrupted');
      await run.checkpoint({ stage: 'tokenizer-parse-start', file: 'tokenizer.json' });
      const state = JSON.parse(sessionStorage.getItem('didimdol.device-experiments.v2'));
      state.active = { kind: 'tokenizer', runId: run.state.runId, startedAt: run.state.startedAt };
      sessionStorage.setItem('didimdol.device-experiments.v2', JSON.stringify(state));
    });
    await page.reload();
    await page.locator('#status').getByText('이전 실험 기록을 복구했습니다. 진단 JSON을 저장해 주세요.').waitFor();
    const recovered = await page.evaluate(() => JSON.parse(localStorage.getItem('didimdol.device-experiments.results.v3')).results.at(-1));
    assert.equal(recovered.interrupted, true);
    assert.equal(recovered.success, false);
    assert.equal(recovered.comparison.file, 'tokenizer.json');
    assert.equal(recovered.comparison.stage, 'tokenizer-parse-start');
    assert.match(await page.locator('#rows').innerText(), /tokenizer-parse-start · tokenizer.json/);
    assert.deepEqual(modelRequests, [], 'recovery must not restart loading');
    results.push({ id: 'tokenizer-recovery-ui', success: true, completedRunRecovered: true, interruptedParseRecovered: true });
    // The comparison selector must reach the worker and retain the same schema.
    await page.locator('#diagnosticsMode').selectOption('snapshot');
    await page.locator('#start').click();
    await page.locator('#status').getByText('토크나이저 준비 완료', { exact: true }).waitFor({ timeout: 60000 });
    const snapshot = await page.evaluate(async () => {
      const state = JSON.parse(localStorage.getItem('didimdol.device-experiments.results.v3'));
      const { readRun } = await import('/web/sllm/diagnostics.js');
      return readRun(state.results.at(-1).runId);
    });
    assert.equal(snapshot.persistence.mode, 'snapshot');
    assert.equal(snapshot.summary.tokenizerPrepared, true);
    const preparedRequests = [];
    page.on('request', request => { if (/tokenizer.*\.(json|bin)$/.test(request.url())) preparedRequests.push(request.url()); });
    await page.locator('#diagnosticsMode').selectOption('compact');
    await page.locator('#tokenizerFormat').selectOption('prepared');
    assert.equal(new URL(await page.locator('#evaluationLink').getAttribute('href')).searchParams.get('tokenizerFormat'), 'prepared');
    await page.locator('#start').click();
    await page.locator('#status').getByText('토크나이저 준비 완료', { exact: true }).waitFor({ timeout: 60000 });
    const prepared = await page.evaluate(async () => {
      const state = JSON.parse(localStorage.getItem('didimdol.device-experiments.results.v3'));
      return (await import('/web/sllm/diagnostics.js')).readRun(state.results.at(-1).runId);
    });
    assert.equal(prepared.summary.tokenizer.tokenizerFormat, 'prepared');
    assert.equal(prepared.summary.tokenizer.memoryLayout.preparedFormat, 'didimdol-bpe-v1');
    assert.equal(preparedRequests.some(url => url.endsWith('/tokenizer/tokenizer.json')), false);
    assert.ok(preparedRequests.some(url => url.endsWith('/tokenizer-ranks.bin')));
    assert.deepEqual(modelRequests, []);
    results.push({ id: 'prepared-tokenizer-ui', success: true, summary: prepared.summary, persistence: prepared.persistence,
      diagnosticCharacters: JSON.stringify(prepared).length });
    const brokenRanks = Buffer.from(await readFile(path.join(root, 'web/vendor/tokenizer/tokenizer-ranks.bin')));
    brokenRanks[0] ^= 1;
    await page.route('**/vendor/tokenizer/tokenizer-ranks.bin', route => route.fulfill({ body: brokenRanks }));
    await page.locator('#start').click();
    await page.locator('#status').getByText('실험 실패 · 진단 JSON을 저장해 주세요.').waitFor({ timeout: 60000 });
    assert.match(await page.locator('#last').innerText(), /hash mismatch/);
    assert.equal(preparedRequests.some(url => url.endsWith('/tokenizer/tokenizer.json')), false, 'a corrupt prepared asset cannot fall back to JSON');
    await page.unroute('**/vendor/tokenizer/tokenizer-ranks.bin');
    results.push({ id: 'prepared-tokenizer-corruption', success: true });
    // A saved model result does not establish that an interrupted cleanup finished.
    await page.evaluate(async () => {
      const { RunDiagnostics, saveCheckpoint } = await import('/web/sllm/diagnostics.js');
      const run = new RunDiagnostics('cleanup-reentry');
      await run.finish('complete', { modelSessionCreated: true, tokenizerPrepared: false });
      await saveCheckpoint({ stage: 'cleanup-start', startedAt: Date.now() }, 'cleanup:cleanup-reentry');
      const state = JSON.parse(sessionStorage.getItem('didimdol.device-experiments.v2'));
      state.active = { kind: 'session-only', runId: 'cleanup-reentry', phase: 'cleanup', startedAt: run.state.startedAt };
      sessionStorage.setItem('didimdol.device-experiments.v2', JSON.stringify(state));
    });
    await page.reload();
    await page.locator('#rows').getByText('자원 정리 중 재진입 · 정리 완료 미확인').waitFor();
    const cleanupRecovery = await page.evaluate(() => JSON.parse(localStorage.getItem('didimdol.device-experiments.results.v3')).results.at(-1));
    assert.equal(cleanupRecovery.interrupted, true);
    assert.equal(cleanupRecovery.success, false);
    assert.equal(cleanupRecovery.execution.modelSessionCreated, true);
    assert.equal(cleanupRecovery.execution.completedScope, null);
    assert.deepEqual(modelRequests, []);
    results.push({ id: 'cleanup-reentry-ui', success: true });
    await page.locator('#tokenizerFormat').selectOption('json');
    // Cancel while the application worker awaits a file read, then recover the
    // persisted cancellation. Native-session cleanup has a separate deadline test.
    let releaseTokenRead;
    const tokenGate = new Promise(resolve => { releaseTokenRead = resolve; });
    await page.route('**/tokenizer/tokenizer.json', async route => {
      await tokenGate;
      await route.continue().catch(() => {});
    });
    await page.locator('#diagnosticsMode').selectOption('compact');
    await page.locator('#start').click();
    await page.waitForFunction(() => {
      try { return JSON.parse(document.getElementById('last').textContent).stage === 'tokenizer-read-start'; }
      catch { return false; }
    });
    await page.locator('#stop').click();
    await page.locator('#status').getByText('사용자가 중단했습니다.', { exact: true }).waitFor({ timeout: 10000 });
    releaseTokenRead();
    const cancelled = await page.evaluate(async () => {
      const state = JSON.parse(localStorage.getItem('didimdol.device-experiments.results.v3'));
      return (await import('/web/sllm/diagnostics.js')).readRun(state.results.at(-1).runId);
    });
    assert.equal(cancelled.status, 'cancelled');
    assert.equal(cancelled.cleanup.success, true);
    await page.reload();
    const stopResult = await page.evaluate(() => JSON.parse(localStorage.getItem('didimdol.device-experiments.results.v3')).results.at(-1));
    assert.equal(stopResult.cancelled, true);
    assert.notEqual(stopResult.interrupted, true);
    results.push({ id: 'snapshot-and-cooperative-cancel', success: true });
    await page.close();
  }
  if (process.env.TEST_APP_LOAD === '1') {
    const page = await browser.newPage();
    await page.goto(origin);
    page.on('console', message => { if (message.type() === 'error') console.error('app:', message.text()); });
    const cdp = await page.context().newCDPSession(page);
    await cdp.send('Storage.overrideQuotaForOrigin', { origin, quotaSize: 8 * 2**30 });
    // Exercise the real app worker/from_pretrained path while serving the same
    // SHA-verified HF graph/weights locally, without another 1 GB network fetch.
    await page.context().route('https://huggingface.co/**', async route => {
      const url = new URL(route.request().url());
      if (url.pathname.includes('/api/models/') && url.pathname.includes('/tree/')) {
        const manifest = JSON.parse(await (await import('node:fs/promises')).readFile(path.join(root, 'model/initializers.json'), 'utf8'));
        await route.fulfill({ headers: { 'Access-Control-Allow-Origin': '*' }, json: [{ type: 'file', path: 'model.onnx' }, ...manifest.files.map(f =>
          ({ type: 'file', path: f.location, size: f.minimumBytes }))] });
      } else {
        const name = url.pathname.split('/').at(-1);
        if (name === 'model.onnx' || /^model\.onnx_data(_\d+)?$/.test(name)) {
          // Redirect binary transfers to the streaming HTTP server instead of
          // materializing 128 MiB responses in the DevTools transport.
          await route.fulfill({ status: 302, headers: {
            'Access-Control-Allow-Origin': '*', location: `${origin}/.work/full-model/${name}`,
          }, body: '' });
        } else await route.continue();
      }
    });
    let result;
    for (const format of ['json', 'prepared']) {
      result = await page.evaluate(async ({ origin, format }) => {
        const { asset } = await (await import(`${origin}/web/sllm/ort-runtime.js`)).runtimeRelease();
        const worker = new Worker(asset('web/sllm/worker.js'), { type: 'module' });
        const loadRunId = crypto.randomUUID();
        const probeRunId = crypto.randomUUID();
        return await new Promise((resolve, reject) => {
          let sessionResult, probeResult;
          const timer = setTimeout(() => { worker.terminate(); reject(new Error('App load/probe timed out')); }, 600000);
          worker.onerror = e => { clearTimeout(timer); worker.terminate(); reject(new Error(e.message)); };
          worker.onmessage = async ({ data }) => {
            if (data.type === 'worker-ready') worker.postMessage({ type: 'load', device: 'webgpu', stagingMiB: 2, runId: loadRunId,
              environment: { tokenizerFormat: format } });
            if (data.type === 'session-result') sessionResult = data.result;
            if (data.type === 'fatal') { clearTimeout(timer); worker.terminate(); reject(new Error(data.error)); }
            if (data.type === 'ready') worker.postMessage({ type: 'probe', runId: probeRunId, maxNewTokens: 2, sampled: true });
            if (data.type === 'probe-result') { probeResult = data.result; worker.postMessage({ type: 'dispose' }); }
            if (data.type === 'disposed') {
              clearTimeout(timer); worker.terminate();
              if (data.error) reject(new Error(data.error));
              else {
                try {
                  const { readRun } = await import(asset('web/sllm/diagnostics.js'));
                  const run = await readRun(probeRunId);
                  const loaded = await readRun(loadRunId);
                  resolve({ id: `app-from-pretrained-${format}`, ...sessionResult, ...probeResult, cleanupVerified: true,
                    loadStatus: loaded.status, loadMilestones: loaded.milestones,
                    probeEnvironment: run.environment, probeMilestones: run.milestones });
                } catch (error) { reject(error); }
              }
            }
          };
        });
      }, { origin, format });
      results.push(result);
      console.log(JSON.stringify(result, null, 2));
      assert.equal(result.success, true, result.error);
      assert.equal(result.metrics.cpuStagingPeak, 2 * 2**20);
      assert.equal(result.probeEnvironment.stagingMiB, 2, 'evaluation diagnostics keep the loaded session settings');
      assert.deepEqual(result.outputs.map(output => output.promptLen), [35, 266]);
      assert.deepEqual(result.outputs.map(output => output.tokens), [[238789, 236764], [238789, 236764]], 'recorded FP32 real-prompt baseline');
      assert.equal(result.metrics.loadedInitializerCount, 251);
      assert.equal(result.loadStatus, 'ready');
      assert.equal(result.loadOrder, 'tokenizer-before-session');
      assert.ok(result.loadMilestones['tokenizer-ready'].timestamp < result.loadMilestones['session-create'].timestamp);
      assert.equal(result.loadMilestones['tokenizer-ready'].vocabSize, 262145);
      assert.equal(result.metrics.gpuWeightUploaded, result.metrics.totalExternalTensorBytes);
      assert.equal(result.sessionMetrics?.gpuLedger?.tracking?.status || result.gpuLedger?.tracking?.status, 'complete');
      assert.equal(result.storage.rangeReadBytes, result.metrics.totalExternalTensorBytes);
      assert.ok(result.timings.modelLoadCallMs >= 0);
      assert.equal(result.loadMilestones['tokenizer-ready'].tokenizerFormat, format);
      // Sampling creates transformers.js's top_k session after the range loader is sealed.
      // It must be recorded as an auxiliary session, never refused or mistaken for the model session.
      assert.equal(result.sampledOutput?.tokens.length, 1, 'the sampled probe exercises the evaluation sampling path');
      assert.equal(result.probeMilestones['auxiliary-session-start']?.auxiliarySession, 1, 'one auxiliary ORT session after seal');
      assert.equal(result.probeMilestones['auxiliary-session-complete']?.auxiliarySession, 1);
      assert.equal(result.probeMilestones['auxiliary-session-complete'].gpuLedger.requestedCurrent,
        result.probeMilestones['auxiliary-session-start'].gpuLedger.requestedCurrent, 'auxiliary sessions request no GPU memory');
      assert.equal(result.probeMilestones['ort-session-start'], undefined, 'auxiliary phases never overwrite model session milestones');
      assert.equal(result.loadMilestones['ort-session-start'].sessionDiagnosticsVersion, 1);
      assert.equal(result.loadMilestones['auxiliary-session-start'], undefined);
    }
    // Exercise A/B/C through the actual UI, sharing only verified OPFS storage.
    // Every case creates a fresh page/worker and uses the production JS imports.
    await page.goto(`${origin}/web/sllm/experiments/index.html`);
    const requests = [];
    page.on('request', request => requests.push(request.url()));
    for (const kind of ['resident-opfs', 'resident-opfs-tokenizer', 'runtime-resident', 'session-only']) {
      await page.locator('#kind').selectOption(kind);
      await page.locator('#staging').selectOption('2');
      await page.locator('#repeats').selectOption('1');
      if (kind === 'session-only') await page.locator('#sessionIdle').selectOption(process.env.TEST_SESSION_IDLE || '0');
      requests.length = 0;
      const priorCount = await page.evaluate(() => JSON.parse(localStorage.getItem('didimdol.device-experiments.results.v3') || '{}').results?.length || 0);
      await page.locator('#start').click();
      await page.waitForFunction(priorCount => {
        const tab = JSON.parse(sessionStorage.getItem('didimdol.device-experiments.v2') || '{}');
        const durable = JSON.parse(localStorage.getItem('didimdol.device-experiments.results.v3') || '{"results":[]}');
        return !tab.active && durable.results.length > priorCount;
      }, priorCount, { timeout: 600000 });
      const exported = await page.evaluate(async () => {
        const state = JSON.parse(localStorage.getItem('didimdol.device-experiments.results.v3'));
        const result = state.results.at(-1);
        const { asset } = await (await import('/web/sllm/ort-runtime.js')).runtimeRelease();
        const { readRun } = await import(asset('web/sllm/diagnostics.js'));
        return { result, run: await readRun(result.runId) };
      });
      const { result: comparison, run } = exported;
      assert.equal(comparison.kind, kind, 'bootstrap must preserve the selected experiment');
      assert.equal(comparison.stagingMiB, 2);
      assert.equal(comparison.success, true, comparison.error);
      assert.equal(run.status, 'complete');
      assert.equal(run.cleanup.success, true, JSON.stringify(run.cleanup));
      assert.equal(run.cleanup.gpuLedger?.requestedCurrent ?? 0, 0);
      assert.equal(run.persistence.mode, 'compact');
      if (kind === 'session-only' || kind === 'runtime-resident') {
        for (const stage of ['ort-wasm-start', 'ort-wasm-complete', 'ort-session-start', 'ort-initializers-start',
          'ort-initializers-complete', 'ort-kernels-start', 'ort-kernels-complete', 'ort-session-complete']) assert.ok(run.milestones[stage], stage);
        assert.equal(run.milestones['ort-session-start'].sessionDiagnosticsVersion, 1);
      }
      assert.equal(run.summary.modelSessionCreated, kind === 'session-only');
      assert.equal(run.summary.tokenizerPrepared, kind === 'resident-opfs-tokenizer');
      assert.equal(run.milestones.ready, undefined, 'a comparison must never announce application readiness');
      assert.equal(comparison.comparison.loadedInitializerCount, 251);
      assert.equal(comparison.comparison.gpuQueueCompletedBytes, result.metrics.totalExternalTensorBytes);
      assert.equal(comparison.comparison.storage.cacheHits, 9);
      assert.equal(comparison.comparison.storage.downloadedFiles, 0);
      assert.equal(comparison.execution.ortJavaScriptLoaded, true);
      assert.equal(comparison.execution.ortWasmInstantiated, ['session-only', 'runtime-resident'].includes(kind));
      assert.equal(comparison.execution.completedScope, { 'resident-opfs': 'gpu-residency',
        'resident-opfs-tokenizer': 'tokenizer-and-gpu-residency', 'runtime-resident': 'runtime-and-gpu-residency',
        'session-only': process.env.TEST_SESSION_IDLE === '120' ? 'model-session-and-idle' : 'model-session' }[kind]);
      if (kind === 'session-only' && process.env.TEST_SESSION_IDLE === '120') {
        assert.ok(run.milestones['session-idle-complete'].idleElapsedMs >= 120000);
        assert.ok(run.milestones['session-idle-complete'].gpuLedger.requestedCurrent >= result.metrics.totalExternalTensorBytes);
        assert.equal(comparison.execution.idleAcceptanceCompleted, true);
      }
      assert.equal(requests.some(url => url.endsWith('/tokenizer/tokenizer.json')), kind === 'resident-opfs-tokenizer');
      assert.equal(requests.some(url => url.endsWith('.wasm')), ['session-only', 'runtime-resident'].includes(kind));
      if (kind !== 'session-only') assert.equal(requests.some(url => url.includes('huggingface.co')), false);
      await page.evaluate(comparison => {
        const state = JSON.parse(sessionStorage.getItem('didimdol.device-experiments.v2'));
        state.active = { kind: comparison.kind, runId: comparison.runId, runIds: [comparison.runId],
          startedAt: comparison.startedAt, seriesId: comparison.seriesId, requestedRuns: 1, remaining: 1,
          mode: comparison.mode, stagingMiB: comparison.stagingMiB };
        sessionStorage.setItem('didimdol.device-experiments.v2', JSON.stringify(state));
      }, comparison);
      requests.length = 0;
      await page.reload();
      await page.locator('#status').getByText('이전 실험 완료 기록을 복구했습니다.').waitFor();
      const recovered = await page.evaluate(() => JSON.parse(localStorage.getItem('didimdol.device-experiments.results.v3')).results.at(-1));
      assert.equal(recovered.success, true);
      assert.equal(recovered.execution.completedScope, comparison.execution.completedScope);
      assert.equal(requests.some(url => /\.wasm$|\/tokenizer\/tokenizer.json$/.test(url)), false, 'recovery must not start a new worker');
      results.push({ id: `comparison-${kind}`, ...exported, recoveryVerified: true });
      console.log(JSON.stringify({ id: `comparison-${kind}`, success: true, execution: comparison.execution,
        uploadedBytes: comparison.comparison.gpuQueueCompletedBytes, storage: comparison.comparison.storage }));
      if (kind === 'session-only') {
        await page.locator('#sessionIdle').selectOption('120');
        await page.locator('#start').click();
        await page.locator('#status').getByText('세션 생성 완료 · 세션을 유지하며 관찰 중…', { exact: true }).waitFor({ timeout: 600000 });
        // Playwright's waitForFunction predicate is synchronous; a Promise is
        // truthy even when it resolves false. Await the durable read explicitly.
        await page.evaluate(async () => {
          const { readRun } = await import('/web/sllm/diagnostics.js');
          for (let attempt = 0; attempt < 100; attempt++) {
            const state = JSON.parse(sessionStorage.getItem('didimdol.device-experiments.v2'));
            const run = await readRun(state.active?.runId);
            if (run?.milestones?.['session-idle-start']) return;
            await new Promise(resolve => setTimeout(resolve, 20));
          }
          throw new Error('Session observation start was not persisted');
        });
        await page.locator('#stop').click();
        await page.locator('#status').getByText('사용자가 중단했습니다.', { exact: true }).waitFor({ timeout: 15000 });
        const stopped = await page.evaluate(async () => {
          const state = JSON.parse(localStorage.getItem('didimdol.device-experiments.results.v3'));
          const result = state.results.at(-1);
          return { result, run: await (await import('/web/sllm/diagnostics.js')).readRun(result.runId) };
        });
        results.push({ id: 'session-observation-cancel', result: stopped.result, run: stopped.run });
        assert.equal(stopped.run.status, 'cancelled');
        assert.equal(stopped.run.cleanup?.success, true, JSON.stringify({ result: stopped.result, last: stopped.run.last, summary: stopped.run.summary, cleanup: stopped.run.cleanup }));
        assert.equal(stopped.run.cleanup.gpuLedger.requestedCurrent, 0);
        assert.equal(stopped.result.execution.modelSessionCreated, true);
        assert.equal(stopped.result.execution.idleAcceptanceCompleted, false);
        assert.equal(stopped.run.milestones['session-idle-complete'], undefined);
        results.at(-1).success = true;
      }
    }
    await page.close();
  }
  if (process.env.TEST_FULL_MODEL === '1') {
    const page = await browser.newPage();
    await page.goto(origin);
    // Incognito's artificial quota can be smaller than the 1.07 GB test model.
    const cdp = await page.context().newCDPSession(page);
    await cdp.send('Storage.overrideQuotaForOrigin', { origin, quotaSize: 8 * 2**30 });
    page.on('console', message => { if (message.type() === 'error') console.error('full-model:', message.text()); });
    const result = await page.evaluate(async origin => {
      const { asset } = await (await import(`${origin}/web/sllm/ort-runtime.js`)).runtimeRelease();
      const worker = new Worker(asset('web/sllm/experiments/worker.js'), { type: 'module' });
      return await new Promise((resolve, reject) => {
        const timer = setTimeout(() => { worker.terminate(); reject(new Error('Full model timed out')); }, 600000);
        worker.onerror = e => { clearTimeout(timer); worker.terminate(); reject(new Error(e.message)); };
        worker.onmessage = ({ data }) => {
          if (data.type === 'result') { clearTimeout(timer); worker.terminate(); resolve(data.result); }
        };
        worker.postMessage({ storage: 'opfs', source: `${origin}/.work/full-experiments/experiments.json`, verifyGemma: true,
          experiment: { id: 'Gemma-FP32-8MiB', kind: 'range', fileMiB: 128, stagingMiB: 8 } });
      });
    }, origin);
    results.push(result);
    result.testStorageQuotaOverrideBytes = 8 * 2**30;
    console.log(JSON.stringify(result, null, 2));
    assert.equal(result.success, true, result.error);
    assert.equal(result.inferenceVerified, true);
    assert.deepEqual(result.decode.map(step => step.token), [1106, 4940], 'same greedy tokens as the recorded FP32 baseline');
    await page.close();
  }
  if (process.env.TEST_MATRIX === '1') {
    const matrix = [
      ...[128, 64, 32].map((fileMiB, i) => ({ id: 'ABC'[i], kind: 'stock', fileMiB })),
      ...[64, 32, 16, 8].map((stagingMiB, i) => ({ id: 'DEFG'[i], kind: 'range', fileMiB: 128, stagingMiB })),
    ];
    for (const experiment of matrix) {
      const page = await browser.newPage();
      await page.goto(origin);
      const cdp = await page.context().newCDPSession(page);
      await cdp.send('Storage.overrideQuotaForOrigin', { origin, quotaSize: 8 * 2**30 });
      const result = await page.evaluate(async ({ origin, experiment }) => {
        const { asset } = await (await import(`${origin}/web/sllm/ort-runtime.js`)).runtimeRelease();
        const worker = new Worker(asset('web/sllm/experiments/worker.js'), { type: 'module' });
        return await new Promise((resolve, reject) => {
          const timer = setTimeout(() => { worker.terminate(); reject(new Error('Matrix timed out')); }, 180000);
          worker.onerror = event => { clearTimeout(timer); worker.terminate(); reject(new Error(event.message)); };
          worker.onmessage = ({ data }) => {
            if (data.type === 'result') { clearTimeout(timer); worker.terminate(); resolve(data.result); }
          };
          worker.postMessage({ source: `${origin}/.work/full-experiments/experiments.json`, experiment });
        });
      }, { origin, experiment });
      result.testStorageQuotaOverrideBytes = 8 * 2**30;
      results.push(result);
      console.log(JSON.stringify(result, null, 2));
      assert.equal(result.success, true, result.error);
      await page.close();
    }
  }
  // Verify that the app's local transformers bundle imports and shares the same
  // Tensor class as the selected ORT, including in a dedicated worker.
  const recoveryContext = await browser.newContext();
  const page = await recoveryContext.newPage();
  await page.goto(origin);
  // Exercise real WebGPU validation errors and late binding through cached native
  // methods, which bypass interception in the same way a missed wrapper can.
  const gpuTracking = await page.evaluate(async origin => {
    const { asset } = await (await import(`${origin}/web/sllm/ort-runtime.js`)).runtimeRelease();
    const { installGpuTracking } = await import(asset('web/sllm/gpu-device.js'));
    const adapter = await navigator.gpu.requestAdapter();
    const request = adapter.requestDevice, create = GPUDevice.prototype.createBuffer;
    let current = 'allocation-under-test';
    const faults = [];
    const tracked = await installGpuTracking(16, record => {
      if (record.stage === 'gpu-error') faults.push(record);
    }, { context: () => ({ initializerName: current }) });
    try {
      const device = await request.call(adapter);
      const buffer = create.call(device, { size: 16, usage: GPUBufferUsage.COPY_DST });
      tracked.observeBuffer(device, buffer); tracked.observeBuffer(device, buffer);
      const bound = tracked.ledger;
      const invalid = device.createBuffer({ size: 16, usage: 0 });
      const synchronous = invalid instanceof GPUBuffer;
      current = 'later-operation';
      await tracked.flush();
      const error = tracked.ledger.firstError;
      device.destroy(); buffer.destroy();
      await tracked.flush();
      return { bound, synchronous, error, faults, remaining: tracked.ledger.requestedCurrent };
    } finally { tracked.restore(); }
  }, origin);
  assert.equal(gpuTracking.bound.tracking.status, 'partial');
  assert.equal(gpuTracking.bound.requestedCurrent, 16);
  assert.equal(gpuTracking.bound.bufferCount, 1);
  assert.equal(gpuTracking.bound.requestedPeak, null);
  assert.equal(gpuTracking.synchronous, true);
  assert.equal(gpuTracking.error.errorType, 'GPUValidationError');
  assert.equal(gpuTracking.error.initializerName, 'allocation-under-test');
  assert.equal(gpuTracking.error.operation, 'createBuffer');
  assert.equal(gpuTracking.faults.length, 1);
  assert.equal(gpuTracking.remaining, 0);
  results.push({ id: 'gpu-tracking-browser', success: true, lateBindingVerified: true, allocationErrorVerified: true });
  const corruptedResidency = await page.evaluate(async () => {
    const { asset } = await (await import('/web/sllm/ort-runtime.js')).runtimeRelease();
    const { residentProbe } = await import(asset('web/sllm/experiments/probes.js'));
    const { FIXTURE } = await import(asset('web/sllm/experiments/fixture.js'));
    const original = GPUQueue.prototype.writeBuffer;
    let corrupted = false;
    GPUQueue.prototype.writeBuffer = function (buffer, offset, data, ...args) {
      if (corrupted) return original.call(this, buffer, offset, data, ...args);
      corrupted = true;
      const words = new Uint32Array(data.buffer, data.byteOffset, data.byteLength / 4);
      words[0] ^= 1;
      try { return original.call(this, buffer, offset, data, ...args); }
      finally { words[0] ^= 1; }
    };
    try { await residentProbe(FIXTURE.manifest, async () => {}, 2); return null; }
    catch (error) { return error.message; }
    finally { GPUQueue.prototype.writeBuffer = original; }
  });
  assert.match(corruptedResidency, /Resident buffer readback mismatch/);
  results.push({ id: 'resident-corruption', success: true, corruptedUploadRejected: true });
  await page.evaluate(async origin => {
    const { asset } = await (await import(`${origin}/web/sllm/ort-runtime.js`)).runtimeRelease();
    await new Promise((resolve, reject) => {
      const worker = new Worker(asset('web/sllm/worker.js'), { type: 'module' });
      worker.onerror = event => { worker.terminate(); reject(new Error(event.message)); };
      worker.onmessage = ({ data }) => { if (data.type === 'worker-ready') { worker.terminate(); resolve(); } };
      setTimeout(() => { worker.terminate(); reject(new Error('App worker import timed out')); }, 30000);
    });
  }, origin);
  // A persisted checkpoint must be associated with this tab's run, and survive reload/export.
  await page.goto(`${origin}/index.html`);
  // goto() waits for page load, but pickDevice()/reportBrokenAttempt() can still
  // be running. Seed only after boot finishes, otherwise this first page can
  // consume and clear the marker before the reload under test.
  await page.locator('#start:not([disabled])').waitFor();
  await page.evaluate(async () => {
    const { RunDiagnostics } = await import('/web/sllm/diagnostics.js');
    // Original schema-2 exports stay readable after the schema-3 upgrade.
    const { saveCheckpoint, runKey } = await import('/web/sllm/diagnostics.js');
    await saveCheckpoint({ schemaVersion: 2, runId: 'legacy-test', status: 'running', fault: null,
      last: { stage: 'gpu-wait', initializerName: 'legacy-weight' }, records: [] }, runKey('legacy-test'));
    const run = new RunDiagnostics('reload-test');
    await run.checkpoint({ stage: 'gpu-wait', initializerName: 'test-weight',
      metrics: { gpuWeightAllocated: 123456, loadedInitializerCount: 1 },
      gpuLedger: { requestedCurrent: 0, bufferCount: 0 } });
    sessionStorage.setItem('didimdol.testLifecycle', JSON.stringify([{ event: 'pagehide', timestamp: run.state.startedAt - 4162 }]));
    sessionStorage.setItem('didimdol.activeRun.v2', JSON.stringify({ runId: 'reload-test', device: 'webgpu', phase: 'load', t: Date.now() }));
    sessionStorage.setItem('didimdol.runHistory.v2', JSON.stringify(['reload-test']));
  });
  await page.addInitScript(() => {
    // Seed before the next app boot: the outgoing page owns an in-memory
    // lifecycle list and correctly writes it again during pagehide.
    const seed = sessionStorage.getItem('didimdol.testLifecycle');
    if (seed) { sessionStorage.setItem('didimdol.lifecycle.v2', seed); sessionStorage.removeItem('didimdol.testLifecycle'); }
  });
  await page.reload();
  await page.locator('#rows').getByText(/gpu-wait/).waitFor();
  // The seeded pagehide precedes this run, so the re-entry is silent: the row must say so without claiming a memory kill.
  assert.match(await page.locator('#rows').innerText(), /언로드 이벤트 없이 \d+\.\d초 뒤 다시 열림 · 탐색 유형 reload\. .*기기 기록으로만 확정됩니다/);
  assert.match(await page.locator('#rows').innerText(), /GPU 계측 부분 관측/);
  const recovery = await page.evaluate(async () => {
    const { readRun, recordRecovery } = await import('/web/sllm/diagnostics.js');
    await recordRecovery(await readRun('legacy-test'));
    return { current: await readRun('reload-test'), legacy: await readRun('legacy-test') };
  });
  assert.equal(recovery.current.recovery.cause, 'unknown');
  assert.equal(recovery.current.recovery.unloadEvidence, 'no-unload-event');
  assert.equal(recovery.current.recovery.navigationType, 'reload');
  assert.ok(recovery.current.recovery.reentryGapMs >= 0);
  assert.ok(recovery.current.recovery.lifecycle.every(event => event.timestamp >= recovery.current.startedAt));
  assert.ok(recovery.current.recovery.lifecycleHistory.some(event => event.timestamp < recovery.current.startedAt));
  assert.equal(recovery.current.last.stage, 'gpu-wait');
  assert.equal(recovery.legacy.schemaVersion, 2);
  assert.equal(recovery.legacy.last.initializerName, 'legacy-weight');
  assert.equal(recovery.legacy.recovery.classification, 'interrupted');
  const other = await page.context().newPage();
  await other.goto(`${origin}/index.html`);
  await other.locator('#start:not([disabled])').waitFor();
  assert.doesNotMatch(await other.locator('#rows').innerText(), /test-weight/);
  await other.close();
  // Recover a completed experiment and an interrupted one without restarting either.
  await page.evaluate(async () => {
    const { RunDiagnostics } = await import('/web/sllm/diagnostics.js');
    await new RunDiagnostics('experiment-complete').finish('complete', { success: true });
    sessionStorage.setItem('didimdol.device-experiments.v2', JSON.stringify({
      results: [], device: 'Local browser validation', mode: 'asyncify',
      active: { kind: 'runtime', runId: 'experiment-complete', startedAt: Date.now() },
    }));
  });
  await page.goto(`${origin}/web/sllm/experiments/index.html`);
  await page.locator('#status').getByText('이전 실험 완료 기록을 복구했습니다.').waitFor();
  assert.match(await page.locator('#rows').innerText(), /성공/);
  await page.evaluate(async () => {
    const { RunDiagnostics } = await import('/web/sllm/diagnostics.js');
    const run = new RunDiagnostics('experiment-device-lost');
    await run.finish('complete', { success: true });
    await run.checkpoint({ stage: 'device-lost', message: 'Late GPU fault' });
    const state = JSON.parse(sessionStorage.getItem('didimdol.device-experiments.v2'));
    state.active = { kind: 'runtime', runId: 'experiment-device-lost', startedAt: Date.now() };
    sessionStorage.setItem('didimdol.device-experiments.v2', JSON.stringify(state));
  });
  await page.reload();
  await page.locator('#rows').getByText('실패', { exact: true }).waitFor();
  await page.evaluate(async () => {
    const { RunDiagnostics } = await import('/web/sllm/diagnostics.js');
    await new RunDiagnostics('experiment-interrupted').checkpoint({ stage: 'gpu-wait' });
    const state = JSON.parse(sessionStorage.getItem('didimdol.device-experiments.v2'));
    state.active = { kind: 'warm-load', runId: 'experiment-interrupted', startedAt: Date.now(), remaining: 4 };
    sessionStorage.setItem('didimdol.device-experiments.v2', JSON.stringify(state));
  });
  await page.reload();
  await page.locator('#rows').getByText(/중단 \(원인 미확인 · 언로드 이벤트 없이 \d+\.\d초 뒤 다시 열림 · 탐색 유형 reload\)/).waitFor();
  assert.equal(await page.evaluate(() => JSON.parse(sessionStorage.getItem('didimdol.device-experiments.v2')).active), null);
  // Exercise the actual experiment controls, then stop during the intended idle period.
  await page.locator('#kind').selectOption('resident');
  assert.equal(await page.locator('#mode').isDisabled(), true);
  assert.equal(await page.locator('#repeats').isDisabled(), false);
  await page.locator('#kind').selectOption('runtime');
  assert.equal(await page.locator('#mode').isDisabled(), false);
  assert.equal(await page.locator('#repeats').isDisabled(), true);
  await page.locator('#staging').selectOption('2');
  await page.locator('#inspector').selectOption('detached');
  await page.locator('#start').click();
  await page.locator('#status').getByText('runtime-idle', { exact: true }).waitFor({ timeout: 120000 });
  await page.locator('#stop').click();
  await page.locator('#status').getByText('사용자가 중단했습니다.').waitFor();
  const downloadReady = page.waitForEvent('download');
  await page.locator('#export').click();
  const download = await downloadReady;
  const exported = JSON.parse(await readFile(await download.path(), 'utf8'));
  assert.equal(exported.results.at(-1).cancelled, true);
  assert.equal(exported.results.at(-1).comparison.trackingStatus, 'complete');
  assert.equal(exported.results.at(-1).comparison.gpuWeightAllocated, 10496000);
  assert.equal(exported.results.at(-1).comparison.storage.downloadedFiles, 1);
  assert.equal(exported.results.at(-1).execution.idleAcceptanceCompleted, false);
  assert.equal(exported.results.at(-1).execution.completedScope, null);
  assert.equal(exported.results.at(-1).execution.idleRequestedSeconds, 120);
  assert.equal(exported.series.at(-1).startedRuns, 1);
  assert.equal(exported.series.at(-1).successfulRuns, 0);
  assert.equal(exported.series.at(-1).requestedRuns, 1);
  assert.equal(exported.repeats, undefined, 'screen preferences are not experiment evidence');
  assert.ok(exported.screenSettings);
  assert.match(await page.locator('#rows').innerText(), /정상/);
  assert.equal(exported.runs.find(run => run.runId === exported.results.at(-1).runId).status, 'cancelled');
  const interrupted = exported.runs.find(run => run.runId === 'experiment-interrupted');
  assert.equal(interrupted.status, 'running');
  assert.equal(interrupted.last.stage, 'gpu-wait');
  assert.equal(interrupted.recovery.classification, 'interrupted');
  const stopped = exported.runs.find(run => run.runId === exported.results.at(-1).runId);
  assert.equal(stopped.environment.stagingMiB, 2);
  assert.equal(stopped.environment.inspector, 'detached');
  assert.equal(stopped.environment.idleSeconds, 120);
  assert.match(stopped.environment.build.releaseId, /^[a-f0-9]{64}$/);
  assert.equal(stopped.milestones['runtime-inference-complete'].metrics.gpuWeightUploaded, 10496000);
  assert.equal(exported.schemaVersion, 4);
  // The real OPFS comparison needs prepared model files; it must not fetch a
  // gigabyte or treat a generated small fixture as the full model.
  const modelRequests = [];
  page.on('request', request => { if (request.url().includes('huggingface.co')) modelRequests.push(request.url()); });
  await page.locator('#kind').selectOption('resident-opfs');
  await page.locator('#start').click();
  await page.locator('#status').getByText('실험 실패 · 진단 JSON을 저장해 주세요.').waitFor();
  assert.match(await page.locator('#last').innerText(), /저장된 가중치가 없습니다/);
  assert.deepEqual(modelRequests, []);
  const missingCache = await page.evaluate(() => JSON.parse(localStorage.getItem('didimdol.device-experiments.results.v3')).results.at(-1));
  assert.equal(missingCache.execution.runtimeMode, null);
  assert.equal(missingCache.execution.idleRequestedSeconds, 0);
  assert.equal(missingCache.requestedRuns, 1);
  // A repeated experiment must not silently cross a deployment boundary.
  await page.evaluate(() => {
    const state = JSON.parse(sessionStorage.getItem('didimdol.device-experiments.v2'));
    state.continue = { kind: 'load', remaining: 2, mode: 'asyncify', stagingMiB: 4, releaseId: '0'.repeat(64),
      navigation: { id: 'repeat-test', reason: 'repeat', timestamp: Date.now() } };
    sessionStorage.setItem('didimdol.device-experiments.v2', JSON.stringify(state));
  });
  await page.goto(`${origin}/web/sllm/experiments/index.html?next=repeat-test`);
  await page.locator('#status').getByText('실험 실패 · 진단 JSON을 저장해 주세요.').waitFor();
  assert.match(await page.locator('#last').innerText(), /반복 실행 중 빌드가 변경/);
  assert.equal(await page.evaluate(() => JSON.parse(sessionStorage.getItem('didimdol.device-experiments.v2')).active), null);
  await page.evaluate(() => {
    const state = JSON.parse(sessionStorage.getItem('didimdol.device-experiments.v2'));
    state.continue = { kind: 'load', remaining: 1, navigation: { id: 'expected-navigation' } };
    sessionStorage.setItem('didimdol.device-experiments.v2', JSON.stringify(state));
  });
  modelRequests.length = 0;
  await page.goto(`${origin}/web/sllm/experiments/index.html?next=unrelated`);
  await page.locator('#status').getByText('예정된 페이지 이동을 확인하지 못해 자동 실행을 중단했습니다.', { exact: false }).waitFor();
  assert.deepEqual(modelRequests, []);
  assert.equal(await page.evaluate(() => JSON.parse(sessionStorage.getItem('didimdol.device-experiments.v2')).continue), null);
  // Results outlive the tab; live progress does not leak into a fresh tab.
  const freshTab = await page.context().newPage();
  await freshTab.goto(`${origin}/web/sllm/experiments/index.html`);
  await freshTab.locator('#kind').waitFor();
  const durable = await freshTab.evaluate(() => ({
    results: JSON.parse(localStorage.getItem('didimdol.device-experiments.results.v3')).results.length,
    tab: JSON.parse(sessionStorage.getItem('didimdol.device-experiments.v2') || 'null') }));
  assert.ok(durable.results > 0, 'a new tab lists the stored results');
  assert.equal(durable.tab?.active ?? null, null);
  assert.match(await freshTab.locator('#rows').innerText(), /resident-opfs/);
  await freshTab.close();
  results.push({ id: 'diagnostic-ui', success: true, tabIsolation: true, recoveryVerified: true, stopExportVerified: true, durableResults: durable.results });
} finally {
  await mkdir(path.join(root, 'test-results'), { recursive: true });
  await writeFile(path.join(root, 'test-results/browser-smoke.json'), JSON.stringify(results, null, 2));
  await browser?.close();
  server.closeAllConnections();
  await new Promise(resolve => server.close(resolve));
}
