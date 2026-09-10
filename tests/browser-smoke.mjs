import { chromium } from 'playwright';
import { createServer } from 'node:http';
import { createReadStream } from 'node:fs';
import { stat, writeFile, mkdir } from 'node:fs/promises';
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
        worker.postMessage({ mode, source: `${origin}/.work/test-experiments/experiments.json`, verifyFixture: true,
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
        let sessionResult;
        const timer = setTimeout(() => { worker.terminate(); reject(new Error('App load timed out')); }, 300000);
        worker.onerror = e => { clearTimeout(timer); worker.terminate(); reject(new Error(e.message)); };
        worker.onmessage = ({ data }) => {
          if (data.type === 'worker-ready') worker.postMessage({ type: 'load', device: 'webgpu', stagingMiB: 16 });
          if (data.type === 'session-result') sessionResult = data.result;
          if (data.type === 'fatal') { clearTimeout(timer); worker.terminate(); reject(new Error(data.error)); }
          if (data.type === 'ready') { clearTimeout(timer); worker.terminate(); resolve({ id: 'app-from-pretrained', success: true, ...sessionResult }); }
        };
      });
    }, origin);
    results.push(result);
    console.log(JSON.stringify(result, null, 2));
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
        worker.postMessage({ source: `${origin}/.work/full-experiments/experiments.json`, verifyGemma: true,
          experiment: { id: 'Gemma-FP32-16MiB', kind: 'range', fileMiB: 128, stagingMiB: 16 } });
      });
    }, origin);
    results.push(result);
    result.testStorageQuotaOverrideBytes = 8 * 2**30;
    console.log(JSON.stringify(result, null, 2));
    assert.equal(result.success, true, result.error);
    assert.equal(result.inferenceVerified, true);
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
  const page = await browser.newPage();
  await page.goto(origin);
  await page.evaluate(async origin => {
    await new Promise((resolve, reject) => {
      const worker = new Worker(`${origin}/web/sllm/worker.js`, { type: 'module' });
      worker.onerror = event => { worker.terminate(); reject(new Error(event.message)); };
      worker.onmessage = ({ data }) => { if (data.type === 'worker-ready') { worker.terminate(); resolve(); } };
      setTimeout(() => { worker.terminate(); reject(new Error('App worker import timed out')); }, 30000);
    });
  }, origin);
} finally {
  await mkdir(path.join(root, 'test-results'), { recursive: true });
  await writeFile(path.join(root, 'test-results/browser-smoke.json'), JSON.stringify(results, null, 2));
  await browser?.close();
  server.closeAllConnections();
  await new Promise(resolve => server.close(resolve));
}
