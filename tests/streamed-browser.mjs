import { chromium } from 'playwright';
import { createServer } from 'node:http';
import { createReadStream } from 'node:fs';
import { stat, readFile, writeFile, mkdir } from 'node:fs/promises';
import path from 'node:path';
import assert from 'node:assert/strict';

const root = path.resolve(import.meta.dirname, '..');
const types = { '.js': 'text/javascript', '.mjs': 'text/javascript', '.wasm': 'application/wasm', '.json': 'application/json', '.html': 'text/html' };
const server = createServer(async (req, res) => {
  if (req.url === '/__test') { res.setHeader('Content-Type', 'text/html'); res.end('<!doctype html><title>Streamed FP32 test</title>'); return; }
  const file = path.resolve(root, '.' + new URL(req.url, 'http://localhost').pathname);
  if (!file.startsWith(root + path.sep)) { res.writeHead(403).end(); return; }
  try {
    const info = await stat(file);
    res.writeHead(200, { 'Access-Control-Allow-Origin': '*', 'Content-Type': types[path.extname(file)] || 'application/octet-stream', 'Content-Length': info.size });
    createReadStream(file).pipe(res);
  } catch { res.writeHead(404).end(); }
});
await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
const origin = `http://127.0.0.1:${server.address().port}`;
let browser;
const results = [];
try {
  browser = await chromium.launch({ headless: true, args: ['--enable-unsafe-webgpu', '--use-angle=swiftshader', '--disable-gpu-sandbox'] });
  const context = await browser.newContext();
  await context.route('https://huggingface.co/**', async route => {
    const url = new URL(route.request().url()), name = url.pathname.split('/').at(-1);
    if (url.pathname.includes('/tree/')) {
      const manifest = JSON.parse(await readFile(path.join(root, 'model/initializers.json')));
      await route.fulfill({ headers: { 'Access-Control-Allow-Origin': '*' }, json: [{ type: 'file', path: 'model.onnx' },
        ...manifest.files.map(file => ({ type: 'file', path: file.location, size: file.bytes }))] });
    } else if (name === 'model.onnx' || /^model\.onnx_data(_\d+)?$/.test(name)) {
      await route.fulfill({ status: 302, headers: { 'Access-Control-Allow-Origin': '*', location: `${origin}/.work/full-model/${name}` }, body: '' });
    } else await route.continue();
  });
  for (const execution of (process.env.MODEL_EXECUTIONS ?? 'resident,streamed').split(',').filter(Boolean)) {
    const page = await context.newPage();
    page.on('console', message => console.log(message.text()));
    await page.goto(`${origin}/__test`);
    const cdp = await context.newCDPSession(page);
    await cdp.send('Storage.overrideQuotaForOrigin', { origin, quotaSize: 8 * 2**30 });
    const result = await page.evaluate(async ({ execution, tokens, mode }) => {
      const { asset, build } = await (await import('/web/sllm/ort-runtime.js')).runtimeRelease();
      const url = new URL(asset('web/sllm/worker.js')); url.searchParams.set('ortMode', mode);
      const worker = new Worker(url, { type: 'module' });
      const loadId = crypto.randomUUID(); let probeId = crypto.randomUUID();
      let session, probe, firstProbe, probing = false;
      return await new Promise((resolve, reject) => {
        const progress = setInterval(async () => {
          const { readRun } = await import('/web/sllm/diagnostics.js');
          const run = await readRun(probing ? probeId : loadId);
          console.log(execution, 'checkpoint', run?.last?.stage, run?.last?.initializerName);
        }, 30000);
        const timeout = setTimeout(() => { worker.terminate(); reject(new Error('Streamed browser timeout')); }, 600000);
        worker.onerror = event => { clearTimeout(timeout); worker.terminate(); reject(new Error(event.message)); };
        worker.onmessage = async ({ data }) => {
          if (['phase', 'ready', 'session-result'].includes(data.type)) console.log(execution, data.type, data.text || data.result?.stage || '');
          if (data.type === 'worker-ready') worker.postMessage({ type: 'load', device: 'webgpu', runId: loadId,
            environment: { modelExecution: execution, tokenizerFormat: 'prepared' } });
          if (data.type === 'session-result') session = data.result;
          if (data.type === 'ready') { probing = true; worker.postMessage({ type: 'probe', runId: probeId, maxNewTokens: tokens }); }
          if (data.type === 'probe-result') {
            probe = data.result;
            if (execution === 'streamed' && !firstProbe) {
              firstProbe = probe; probeId = crypto.randomUUID();
              worker.postMessage({ type: 'probe', runId: probeId, maxNewTokens: tokens });
            } else worker.postMessage({ type: 'dispose' });
          }
          if (data.type === 'fatal') { clearTimeout(timeout); clearInterval(progress); worker.terminate(); reject(new Error(data.error)); }
          if (data.type === 'disposed') {
            clearTimeout(timeout); clearInterval(progress); worker.terminate();
            const { readRun } = await import('/web/sllm/diagnostics.js');
            const run = await readRun(probeId);
            resolve({ execution, mode, releaseId: build.releaseId, session, probe, firstProbe, cleanup: data.cleanup, steps: run.records.filter(x => x.stage === 'streamed-step-complete'),
              samples: run.records.filter(x => x.stage === 'inference-sample').map(({ reason, phase, row, inferenceElapsedMs, sampleIndex, gpuLedger, metrics }) =>
                ({ reason, phase, row, inferenceElapsedMs, sampleIndex, gpuRequestedCurrent: gpuLedger?.requestedCurrent ?? null,
                  computePipelines: gpuLedger?.programs?.computePipelines ?? null, wasmHeapBytes: metrics?.wasmHeapBytes ?? null })),
              inferences: run.records.filter(x => x.inference).map(x => ({ stage: x.stage, row: x.row, ...x.inference })) });
          }
        };
      });
    }, { execution, tokens: Number(process.env.STREAMED_TEST_TOKENS || 2), mode: process.env.ORT_MODE || 'asyncify' });
    results.push(result);
    assert.equal(result.probe.success, true);
    assert.equal(result.cleanup.success, true);
    // Every generate() call is sampled: the last probe carries a summary and its first token was marked.
    assert.ok(result.inferences.length >= 1, 'probe-inference-complete carries the sampling summary');
    const inference = result.inferences.at(-1);
    assert.equal(inference.stage, 'probe-inference-complete');
    assert.equal(inference.phase, 'probe');
    assert.ok(inference.samples >= 1 && inference.marks >= 1, JSON.stringify(inference));
    assert.ok(inference.lastSample.gpuRequestedCurrent > 0);
    assert.ok(result.samples.some(sample => sample.reason === 'first-token'));
    assert.ok(result.samples.every(sample => sample.phase === 'probe' && sample.sampleIndex >= 1 && sample.gpuRequestedCurrent > 0));
    console.log(JSON.stringify({ execution, inference, sampleReasons: result.samples.map(sample => sample.reason) }));
    if (execution === 'streamed') {
      assert.equal(result.session.metrics.gpuWeightAllocated, 401304064);
      assert.equal(result.session.streaming.gpuBufferBytes, 41943040);
      assert.equal(result.session.gpuLedger.tracking.deviceCount, 1);
      const categories = Object.fromEntries(result.probe.gpuLedger.categories.map(value => [value.role, value]));
      assert.equal(categories['streamed-weight'].createdCount, 1);
      assert.equal(categories['streamed-weight'].requestedCurrent, 41943040);
      assert.deepEqual(result.probe.outputs.map(x => x.tokens), result.firstProbe.outputs.map(x => x.tokens));
      assert.ok(result.probe.gpuLedger.requestedCurrent <= result.firstProbe.gpuLedger.requestedCurrent + 2**20,
        'repeating the same prompts must not accumulate GPU allocations');
      assert.equal(result.steps.at(-1).streaming.outputReadBytes,
        result.steps.at(-1).streaming.projections * 671088640);
      const resident = results.find(value => value.execution === 'resident');
      if (resident) assert.deepEqual(result.probe.outputs.map(x => x.tokens), resident.probe.outputs.map(x => x.tokens));
    }
    console.log(JSON.stringify({ execution, outputs: result.probe.outputs,
      gpuPeak: result.probe.gpuLedger.requestedPeak, streaming: result.steps.at(-1)?.streaming }));
    await page.close();
  }
  if (process.env.STREAMED_TEST_UI !== '0') {
    const page = await context.newPage();
    const cdp = await context.newCDPSession(page);
    await cdp.send('Storage.overrideQuotaForOrigin', { origin, quotaSize: 8 * 2**30 });
    await page.goto(`${origin}/web/sllm/experiments/index.html`);
    await page.locator('#kind').selectOption('session-only');
    await page.locator('#modelExecution').selectOption('streamed');
    const idle = process.env.STREAMED_TEST_IDLE || '0';
    await page.locator('#sessionIdle').selectOption(idle);
    assert.equal(await page.locator('#staging').isDisabled(), true);
    assert.match(await page.locator('#evaluationLink').getAttribute('href'), /modelExecution=streamed/);
    await page.locator('#start').click();
    const progress = setInterval(async () => console.log('UI:', await page.locator('#status').textContent()), 30000);
    try {
      await page.locator('#status').getByText(idle === '120' ? '모델 세션 생성·120초 관찰 완료 (토크나이저 없음)' :
        '모델 세션 생성 완료 (토크나이저 없음)', { exact: true }).waitFor({ timeout: 180000 });
    } catch (error) {
      const state = await page.evaluate(() => ({ status: document.querySelector('#status').textContent,
        state: JSON.parse(sessionStorage.getItem('didimdol.device-experiments.v2')) }));
      await writeFile(path.join(root, '.work/streamed-ui-failure.json'), JSON.stringify(state, null, 2));
      throw error;
    } finally { clearInterval(progress); }
    const complete = await page.evaluate(() => JSON.parse(localStorage.getItem('didimdol.device-experiments.results.v3')).results.at(-1));
    assert.equal(complete.execution.modelExecution, 'streamed');
    assert.equal(complete.stagingMiB, 2);
    assert.equal(complete.comparison.cleanup.success, true);
    if (idle === '120') assert.equal(complete.execution.idleAcceptanceCompleted, true);
    await page.locator('#sessionIdle').selectOption('120');
    await page.locator('#start').click();
    await page.locator('#status').getByText('세션 생성 완료 · 세션을 유지하며 관찰 중…', { exact: true }).waitFor({ timeout: 120000 });
    await page.locator('#stop').click();
    await page.locator('#status').getByText('사용자가 중단했습니다.', { exact: true }).waitFor({ timeout: 15000 });
    const cancelled = await page.evaluate(() => JSON.parse(localStorage.getItem('didimdol.device-experiments.results.v3')).results.at(-1));
    assert.equal(cancelled.cancelled, true);
    assert.equal(cancelled.comparison.cleanup.success, true);
    const leaseFree = await page.evaluate(() => navigator.locks.request('didimdol-model-load', { ifAvailable: true }, lock => !!lock));
    assert.equal(leaseFree, true);
    results.push({ id: 'streamed-ui-session-and-cancel', success: true, complete, cancelled });
    await page.close();
  }
  await mkdir(path.join(root, '.work'), { recursive: true });
  await writeFile(path.join(root, process.env.STREAMED_REPORT || '.work/streamed-browser-results.json'), JSON.stringify(results, null, 2));
} finally { await browser?.close(); await new Promise(resolve => server.close(resolve)); }
