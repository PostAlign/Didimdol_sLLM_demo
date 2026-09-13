// Isolated Node comparison of BPE initialization, using the same loader in both cases.
// This is not an iPhone/WebKit process-memory measurement.
import { readFile } from 'node:fs/promises';
import { execFile } from 'node:child_process';
import { promisify } from 'node:util';
import path from 'node:path';
import { prepareTokenizer } from '../web/sllm/tokenizer-loader.js';
import { patchTokenizerSource } from './tokenizer-patch.mjs';

const root = path.resolve(import.meta.dirname, '..');
const mode = process.argv[2];
if (['baseline', 'incremental', 'patched', 'prepared'].includes(mode)) {
  const moduleURL = new URL('../node_modules/@huggingface/tokenizers/dist/tokenizers.mjs', import.meta.url);
  const loadClass = async () => {
    if (mode === 'baseline') return (await import(moduleURL)).Tokenizer;
    const source = patchTokenizerSource('tokenizers', await readFile(moduleURL, 'utf8'), '0.1.3', { compact: ['patched', 'prepared'].includes(mode) });
    return (await import(`data:text/javascript;base64,${Buffer.from(source).toString('base64')}`)).Tokenizer;
  };
  const Tokenizer = await loadClass();
  const build = mode === 'prepared' ? JSON.parse(await readFile(path.join(root, 'web/vendor/build.json'), 'utf8')) : undefined;
  global.gc();
  const baseline = process.memoryUsage();
  const samples = [];
  const result = await prepareTokenizer({ baseURL: new URL('../web/sllm/worker.js', import.meta.url),
    format: mode === 'prepared' ? 'prepared' : 'json', build,
    createTokenizer: (data, config) => ({ _tokenizer: new Tokenizer(data, config) }),
    fetchFile: async url => new Response(await readFile(url)),
    checkpoint: async record => samples.push({ stage: record.stage, file: record.file, ...process.memoryUsage() }),
  });
  global.gc();
  console.log(JSON.stringify({ mode, node: process.version, baseline, maxRSSBytes: process.resourceUsage().maxRSS * 1024,
    retained: process.memoryUsage(), durationMs: result.summary.durationMs, vocabSize: result.tokenizer._tokenizer.model.vocab.length,
    memoryLayout: result.summary.memoryLayout, samples }));
} else {
  const runs = [];
  for (let repeat = 1; repeat <= 3; repeat++) for (const mode of ['patched', 'prepared']) {
    const { stdout } = await promisify(execFile)(process.execPath, ['--expose-gc', path.join(root, 'tools/measure-tokenizer.mjs'), mode]);
    runs.push({ repeat, ...JSON.parse(stdout) });
  }
  console.log(JSON.stringify({ note: 'Node-only BPE comparison; same preparation loader, fresh process per run. Not iPhone RSS.', runs }, null, 2));
}
