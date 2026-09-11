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
  for (const mode of (process.env.ORT_MODES ?? 'asyncify,jspi,stock').split(',').filter(Boolean)) {
    const page = await browser.newPage();
    page.on('pageerror', error => console.error('pageerror:', error.message));
    page.on('console', message => { if (message.type() === 'error') console.error('browser:', message.text()); });
    await page.goto(origin);
    const result = await page.evaluate(async ({ origin, mode }) => {
      const worker = new Worker(`${origin}/web/sllm/experiments/worker.js`, { type: 'module' });
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
      assert.equal(result.storage.peakOpenHandles, 1);
      assert.equal(result.storage.openHandles, 0);
      for (const kind of (mode === 'asyncify' ? ['resident', 'runtime'] : ['runtime'])) {
        const probePage = await browser.newPage();
        await probePage.goto(origin);
        const probe = await probePage.evaluate(async ({ origin, mode, kind }) => {
          const worker = new Worker(`${origin}/web/sllm/experiments/device-probes.js`, { type: 'module' });
          return new Promise((resolve, reject) => {
            const timer = setTimeout(() => { worker.terminate(); reject(new Error('Device probe timed out')); }, 120000);
            worker.onerror = e => { clearTimeout(timer); worker.terminate(); reject(new Error(e.message)); };
            worker.onmessage = ({ data }) => {
              if (data.type === 'result') { clearTimeout(timer); worker.terminate(); resolve(data.result); }
            };
            worker.postMessage({ kind, mode, fixture: true, idleSeconds: 0 });
          });
        }, { origin, mode, kind });
        results.push({ id: `${kind}-${mode}`, ...probe });
        console.log(JSON.stringify({ id: `${kind}-${mode}`, ...probe }, null, 2));
        assert.equal(probe.success, true, probe.error);
        if (kind === 'runtime') {
          assert.equal(probe.inferenceVerified, true);
          assert.equal(probe.metrics.cpuStagingPeak, 8 * 2**20, 'OPFS reads straight into the reusable scratch');
          assert.equal(probe.storage.downloadedFiles, 1, 'the diagnostic generates and verifies its own cold-cache fixture');
        } else assert.equal(probe.gpuWeightAllocated, probe.expectedBytes);
        await probePage.close();
      }
    }
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
    const result = await page.evaluate(async origin => {
      const worker = new Worker(`${origin}/web/sllm/worker.js`, { type: 'module' });
      return await new Promise((resolve, reject) => {
        let sessionResult, probeResult;
        const timer = setTimeout(() => { worker.terminate(); reject(new Error('App load/probe timed out')); }, 600000);
        worker.onerror = e => { clearTimeout(timer); worker.terminate(); reject(new Error(e.message)); };
        worker.onmessage = ({ data }) => {
          if (data.type === 'worker-ready') worker.postMessage({ type: 'load', device: 'webgpu' });
          if (data.type === 'session-result') sessionResult = data.result;
          if (data.type === 'fatal') { clearTimeout(timer); worker.terminate(); reject(new Error(data.error)); }
          if (data.type === 'ready') worker.postMessage({ type: 'probe', maxNewTokens: 2 });
          if (data.type === 'probe-result') { probeResult = data.result; worker.postMessage({ type: 'dispose' }); }
          if (data.type === 'disposed') {
            clearTimeout(timer); worker.terminate();
            if (data.error) reject(new Error(data.error));
            else resolve({ id: 'app-from-pretrained', ...sessionResult, ...probeResult, cleanupVerified: true });
          }
        };
      });
    }, origin);
    results.push(result);
    console.log(JSON.stringify(result, null, 2));
    assert.equal(result.success, true, result.error);
    assert.equal(result.metrics.cpuStagingPeak, 8 * 2**20);
    assert.deepEqual(result.outputs.map(output => output.promptLen), [35, 266]);
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
      const worker = new Worker(`${origin}/web/sllm/experiments/worker.js`, { type: 'module' });
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
        const worker = new Worker(`${origin}/web/sllm/experiments/worker.js`, { type: 'module' });
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
  await page.evaluate(async origin => {
    await new Promise((resolve, reject) => {
      const worker = new Worker(`${origin}/web/sllm/worker.js`, { type: 'module' });
      worker.onerror = event => { worker.terminate(); reject(new Error(event.message)); };
      worker.onmessage = ({ data }) => { if (data.type === 'worker-ready') { worker.terminate(); resolve(); } };
      setTimeout(() => { worker.terminate(); reject(new Error('App worker import timed out')); }, 30000);
    });
  }, origin);
  // A persisted checkpoint must be associated with this tab's run, and survive reload/export.
  await page.goto(`${origin}/index.html`);
  await page.evaluate(async () => {
    const { RunDiagnostics } = await import('/web/sllm/diagnostics.js');
    const run = new RunDiagnostics('reload-test');
    await run.checkpoint({ stage: 'gpu-wait', initializerName: 'test-weight',
      metrics: { gpuWeightAllocated: 123456, loadedInitializerCount: 1 } });
    sessionStorage.setItem('didimdol.activeRun.v2', JSON.stringify({ runId: 'reload-test', device: 'webgpu', phase: 'load', t: Date.now() }));
    sessionStorage.setItem('didimdol.runHistory.v2', JSON.stringify(['reload-test']));
  });
  await page.reload();
  await page.locator('#rows').getByText(/gpu-wait/).waitFor();
  assert.match(await page.locator('#rows').innerText(), /메모리 부족은 아직 확인되지 않았습니다/);
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
  await page.locator('#rows').getByText('중단 (원인 미확인)').waitFor();
  assert.equal(await page.evaluate(() => JSON.parse(sessionStorage.getItem('didimdol.device-experiments.v2')).active), null);
  // Exercise the actual experiment controls, then stop during the intended idle period.
  await page.locator('#kind').selectOption('runtime');
  await page.locator('#start').click();
  await page.locator('#status').getByText('runtime-idle', { exact: true }).waitFor({ timeout: 120000 });
  await page.locator('#stop').click();
  await page.locator('#status').getByText('사용자가 중단했습니다.').waitFor();
  const downloadReady = page.waitForEvent('download');
  await page.locator('#export').click();
  const download = await downloadReady;
  const exported = JSON.parse(await readFile(await download.path(), 'utf8'));
  assert.equal(exported.results.at(-1).cancelled, true);
  assert.equal(exported.runs.find(run => run.runId === exported.results.at(-1).runId).status, 'cancelled');
  assert.ok(exported.runs.some(run => run.runId === 'experiment-interrupted'));
  results.push({ id: 'diagnostic-ui', success: true, tabIsolation: true, recoveryVerified: true, stopExportVerified: true });
} finally {
  await mkdir(path.join(root, 'test-results'), { recursive: true });
  await writeFile(path.join(root, 'test-results/browser-smoke.json'), JSON.stringify(results, null, 2));
  await browser?.close();
  server.closeAllConnections();
  await new Promise(resolve => server.close(resolve));
}
