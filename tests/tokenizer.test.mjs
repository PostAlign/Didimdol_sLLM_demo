import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile, mkdir, mkdtemp, rm } from 'node:fs/promises';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import { build } from 'esbuild';
import { GemmaTokenizer } from '@huggingface/transformers';
import { tokenizerPatch, patchTokenizerSource } from '../tools/tokenizer-patch.mjs';
import { prepareTokenizer, readPreparationFile, jsMemory } from '../web/sllm/tokenizer-loader.js';
import { RunDiagnostics, diagnosticSummary, recoveryEvidence } from '../web/sllm/diagnostics.js';
import { releaseBpeSource } from '../web/sllm/tokenizer-memory.js';
import { makeRouge1 } from '../web/sllm/rouge.js';
import { serializeTokenizer } from '../tools/prepare-tokenizer.mjs';
import { PREPARED_TOKENIZER_FORMAT, validatePreparedManifest, decodePreparedVocabulary, decodePreparedRanks } from '../web/sllm/prepared-tokenizer.js';
import { createHash } from 'node:crypto';

const root = path.resolve(import.meta.dirname, '..');
const assetInfo = value => ({ bytes: value.length, sha256: createHash('sha256').update(value).digest('hex') });

function preparedFixture(tokenizer, data, config) {
  const sourceBytes = { 'tokenizer.json': Buffer.from(JSON.stringify(data)), 'tokenizer_config.json': Buffer.from(JSON.stringify(config)) };
  const sources = Object.fromEntries(Object.entries(sourceBytes).map(([name, value]) => [name, assetInfo(value)]));
  const assets = serializeTokenizer(tokenizer, data, sources);
  const files = { ...assets, ...sourceBytes };
  const build = { tokenizer: { preparedFormat: PREPARED_TOKENIZER_FORMAT }, assets: Object.fromEntries(Object.entries(files)
    .map(([name, value]) => [(name in sourceBytes ? 'tokenizer/' : 'web/vendor/tokenizer/') + name, assetInfo(value)])) };
  return { assets, files, build };
}

test('patched class selection, tokens, decoding and chat templates match upstream on all evaluation rows', async t => {
  await mkdir(path.join(root, '.work'), { recursive: true });
  const directory = await mkdtemp(path.join(root, '.work/tokenizer-test-'));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const patch = await tokenizerPatch(root);
  const outfile = path.join(directory, 'auto.mjs');
  await build({ entryPoints: [path.join(root, 'node_modules/@huggingface/transformers/src/models/auto/tokenization_auto.js')],
    outfile, bundle: true, format: 'esm', platform: 'node', plugins: [patch.plugin],
    external: ['onnxruntime-node', 'onnxruntime-web/webgpu', 'onnxruntime-common', 'sharp'] });
  const { AutoTokenizer } = await import(pathToFileURL(outfile));
  const data = JSON.parse(await readFile(path.join(root, 'tokenizer/tokenizer.json'), 'utf8'));
  const config = JSON.parse(await readFile(path.join(root, 'tokenizer/tokenizer_config.json'), 'utf8'));
  const baseline = new GemmaTokenizer(data, config);
  const patched = AutoTokenizer.from_json(data, config);
  const fixture = preparedFixture(AutoTokenizer.from_json(data, config), data, config), requests = [], checkpoints = [];
  const { tokenizer: prepared, summary } = await prepareTokenizer({ format: 'prepared', build: fixture.build,
    baseURL: 'https://example.com/web/sllm/worker.js', createTokenizer: (data, config) => AutoTokenizer.from_json(data, config),
    checkpoint: async record => checkpoints.push(record), fetchFile: async url => {
      const name = String(url).split('/').at(-1); requests.push(name); return new Response(fixture.files[name]);
    } });
  assert.equal(summary.tokenizerFormat, 'prepared');
  assert.equal(summary.memoryLayout.preparedFormat, PREPARED_TOKENIZER_FORMAT);
  assert.equal(requests.includes('tokenizer.json'), false);
  assert.equal(prepared._tokenizerJSON.model._didimdolPreparedBpe, undefined);
  assert.equal(prepared._tokenizer.model.merges, null);
  assert.equal(prepared._tokenizer.model.bpe_ranks.keys.buffer, prepared._tokenizer.model.bpe_ranks.ranks.buffer);
  assert.equal(checkpoints.some(record => record.file === 'tokenizer.json'), false);
  const layout = releaseBpeSource(patched);
  assert.equal(layout.sourceReleased, true);
  assert.equal(layout.mergeCount, 514906);
  assert.equal(layout.fallbackRankCount, 0);
  assert.equal(layout.rankTableBytes, 12 * 2**20);
  assert.equal(patched._tokenizer.model.merges, null);
  assert.equal(patched._tokenizer.model.config.vocab, undefined);
  assert.equal(patched._tokenizerJSON.model.merges, undefined);
  assert.equal(baseline._tokenizerJSON, data, 'compaction must not mutate a shared input object');
  assert.equal(data.model.merges.length, 514906);
  for (const [left, right] of data.model.merges) {
    assert.equal(patched._tokenizer.model.rank_for_pair(left, right),
      baseline._tokenizer.model.bpe_ranks.get(JSON.stringify([left, right])), 'all merge ranks, including hash collisions');
    assert.equal(prepared._tokenizer.model.rank_for_pair(left, right), patched._tokenizer.model.rank_for_pair(left, right));
  }
  assert.equal(patched.constructor.name, 'GemmaTokenizer');
  assert.equal(AutoTokenizer.from_json(data, { ...config, tokenizer_class: 'GemmaTokenizerFast' }).constructor.name, 'GemmaTokenizer');
  assert.deepEqual(patched.all_special_ids, baseline.all_special_ids);
  assert.deepEqual(patched.all_special_tokens, baseline.all_special_tokens);
  assert.deepEqual(prepared.all_special_ids, baseline.all_special_ids);
  assert.deepEqual(prepared.all_special_tokens, baseline.all_special_tokens);
  const rows = (await readFile(path.join(root, 'web/sllm/data.jsonl'), 'utf8')).trim().split('\n').map(JSON.parse);
  const chat_template = await readFile(path.join(root, 'tokenizer/chat_template.jinja'), 'utf8');
  const examples = ['', '안녕하세요. 한글과 English 123', '  공백\t\n\n유지  ', '😀 👩‍💻 e\u0301 가나다',
    '<bos><start_of_turn>user\n질문<end_of_turn>\n', ...rows.flatMap(row => row.messages.map(message => message.content))];
  for (const variant of [patched, prepared]) for (const text of examples) {
    const expected = baseline.encode(text), actual = variant.encode(text);
    assert.deepEqual(actual, expected, text.slice(0, 60));
    assert.equal(variant.decode(actual), baseline.decode(expected));
    assert.equal(variant.decode(actual, { skip_special_tokens: true }), baseline.decode(expected, { skip_special_tokens: true }));
    assert.deepEqual(variant.tokenize(text, { add_special_tokens: false }), baseline.tokenize(text, { add_special_tokens: false }));
  }
  const expectedRouge = makeRouge1(baseline);
  for (const variant of [patched, prepared]) for (const row of rows) {
    const actualRouge = makeRouge1(variant);
    const reference = row.messages.at(-1).content;
    assert.deepEqual(actualRouge('견인 치료 123 😀', reference), expectedRouge('견인 치료 123 😀', reference));
    const messages = row.messages.slice(0, -1);
    const options = { chat_template, add_generation_prompt: true, return_dict: true };
    const expected = baseline.apply_chat_template(messages, options);
    const actual = variant.apply_chat_template(messages, options);
    for (const key of Object.keys(expected)) {
      assert.deepEqual(actual[key].dims, expected[key].dims);
      assert.deepEqual(actual[key].data, expected[key].data);
      actual[key].dispose(); expected[key].dispose();
    }
  }
});

test('compact rank table preserves duplicate ranks, absent pairs, large IDs and legacy merge format', async () => {
  const source = await readFile(path.join(root, 'node_modules/@huggingface/tokenizers/dist/tokenizers.mjs'), 'utf8');
  const { Tokenizer } = await import(`data:text/javascript;base64,${Buffer.from(patchTokenizerSource('tokenizers', source, '0.1.3')).toString('base64')}`);
  const merges = [['a', 'b'], ['missing', 'a'], ['ab', 'ab'], ['a', 'b'], ['b', 'a']];
  for (const rules of [merges, merges.map(pair => pair.join(' '))]) {
    const data = { model: { type: 'BPE', vocab: { a: 0, b: 1, ab: 300000 }, merges: rules },
      normalizer: null, pre_tokenizer: null, post_processor: null, decoder: null,
      added_tokens: [{ content: 'missing', id: 2, special: false }] };
    const tokenizer = { _tokenizer: new Tokenizer(data, {}) };
    const model = tokenizer._tokenizer.model;
    const layout = releaseBpeSource(tokenizer);
    assert.equal(layout.fallbackRankCount, 1);
    assert.equal(model.rank_for_pair('a', 'b'), 3, 'last duplicate rule wins');
    assert.equal(model.rank_for_pair('missing', 'a'), 1);
    assert.equal(model.rank_for_pair('ab', 'ab'), 2, 'keys above 32 bits stay exact');
    assert.equal(model.rank_for_pair('b', 'a'), 4);
    assert.equal(model.rank_for_pair('a', 'ab'), undefined);
    assert.equal(model.rank_for_pair('missing', 'missing'), undefined);
    assert.equal(data.model.merges, rules);
  }
  const data = { model: { type: 'BPE', vocab: { a: 0, b: 1 }, merges: [['a', 'b']] },
    normalizer: null, pre_tokenizer: null, post_processor: null, decoder: null,
    added_tokens: [{ content: 'a', id: 2, special: true }] };
  assert.throws(() => releaseBpeSource({ _tokenizer: new Tokenizer(data, {}) }), /changes a base BPE/);
  data.added_tokens = [{ content: 'alias', id: 0, special: false }];
  assert.throws(() => releaseBpeSource({ _tokenizer: new Tokenizer(data, {}) }), /changes a base BPE/);
});

test('build patch rejects changed source or dependency version', async () => {
  const source = await readFile(path.join(root, 'node_modules/@huggingface/tokenizers/dist/tokenizers.mjs'), 'utf8');
  assert.throws(() => patchTokenizerSource('tokenizers', source, '0.1.4'), /source\/version mismatch/);
  assert.throws(() => patchTokenizerSource('tokenizers', source + '\n', '0.1.3'), /source\/version mismatch/);
});

test('prepared assets preserve BOM tokens and reject corrupt, truncated, incompatible or mismatched data', async () => {
  const source = await readFile(path.join(root, 'node_modules/@huggingface/tokenizers/dist/tokenizers.mjs'), 'utf8');
  const { Tokenizer } = await import(`data:text/javascript;base64,${Buffer.from(patchTokenizerSource('tokenizers', source, '0.1.3')).toString('base64')}`);
  const data = { model: { type: 'BPE', vocab: { a: 0, b: 1, ab: 2, '\uFEFFa': 3 }, merges: [['a', 'b'], ['a', 'b']] },
    normalizer: null, pre_tokenizer: null, post_processor: null, decoder: null, added_tokens: [] };
  const { files, build } = preparedFixture({ _tokenizer: new Tokenizer(data, {}) }, data, {});
  const array = value => value.buffer.slice(value.byteOffset, value.byteOffset + value.byteLength);
  const manifest = JSON.parse(files['tokenizer-prepared.json']);
  assert.equal(decodePreparedVocabulary(array(files['tokenizer-vocab.bin']), 4)[3], '\uFEFFa');
  const sources = Object.fromEntries(['tokenizer.json', 'tokenizer_config.json'].map(file => [file, build.assets[`tokenizer/${file}`]]));
  assert.throws(() => validatePreparedManifest({ ...manifest, format: 'future' }, sources), /Unsupported/);
  assert.throws(() => validatePreparedManifest({ ...manifest, sources: {} }, sources), /source mismatch/);
  assert.throws(() => validatePreparedManifest({ ...manifest, capacity: 3 }, sources), /layout/);
  assert.throws(() => decodePreparedVocabulary(new ArrayBuffer(2), 4), /Truncated/);
  const corruptOffsets = array(files['tokenizer-vocab.bin']); new DataView(corruptOffsets).setUint32(4, 0xffffffff, true);
  assert.throws(() => decodePreparedVocabulary(corruptOffsets, 4), /range/);
  assert.throws(() => decodePreparedRanks(new ArrayBuffer(4), manifest), /size/);
  const corruptRanks = array(files['tokenizer-ranks.bin']); new Uint32Array(corruptRanks, manifest.capacity * 8).fill(0);
  assert.throws(() => decodePreparedRanks(corruptRanks, manifest), /count/);
  for (const broken of ['tokenizer-prepared.json', 'tokenizer-vocab.bin', 'tokenizer-ranks.bin']) {
    const altered = Buffer.from(files[broken]); altered[altered.length - 1] ^= 1;
    const run = new RunDiagnostics('corrupt-prepared', { diagnosticsMode: 'snapshot' }, async () => true);
    await assert.rejects(prepareTokenizer({ format: 'prepared', build, baseURL: 'https://example.com/web/sllm/worker.js',
      checkpoint: run.checkpoint, createTokenizer: () => assert.fail('corrupt data must never construct a tokenizer'),
      fetchFile: async url => { const name = String(url).split('/').at(-1); return new Response(name === broken ? altered : files[name]); },
    }), /hash mismatch/);
    assert.equal(run.state.fault.file, broken);
    assert.equal(run.state.fault.observedDuring, 'tokenizer-verify-start');
  }
  let constructed = false;
  await assert.rejects(prepareTokenizer({ format: 'prepared', build: { tokenizer: {} }, createTokenizer: () => { constructed = true; } }), /Rebuild/);
  assert.equal(constructed, false);
});

test('preparation awaits durable markers and retains file stages after model events evict recent records', async () => {
  let saved;
  const run = new RunDiagnostics('tokenizer', { loadOrder: 'tokenizer-before-session', diagnosticsMode: 'snapshot' }, async value => {
    await Promise.resolve(); saved = structuredClone(value); return true;
  });
  const requests = [];
  const result = await prepareTokenizer({ baseURL: 'https://example.com/web/sllm/worker.js', checkpoint: run.checkpoint,
    fetchFile: async url => {
      const file = String(url).split('/').at(-1); requests.push(file);
      assert.equal(saved.last.stage, 'tokenizer-read-start');
      assert.equal(saved.last.file, file);
      return new Response(JSON.stringify(file === 'tokenizer_config.json' ? { tokenizer_class: 'GemmaTokenizer' } : { model: {} }));
    },
    createTokenizer: () => {
      assert.equal(saved.last.stage, 'tokenizer-create-start');
      assert.ok(saved.preparation['tokenizer.json']['tokenizer-parse-complete']);
      return { _tokenizer: { model: { vocab: ['a'], merges: [] } } };
    },
  });
  assert.deepEqual(requests, ['tokenizer_config.json', 'tokenizer.json']);
  assert.equal(result.summary.vocabSize, 1);
  for (let i = 0; i < 70; i++) await run.checkpoint({ stage: 'gpu-wait' });
  assert.equal(saved.records.some(record => record.stage === 'tokenizer-ready'), false);
  assert.ok(saved.preparation['tokenizer.json']['tokenizer-decode-start']);
  assert.equal(diagnosticSummary(saved).tokenizer.vocabSize, 1);
  assert.equal(diagnosticSummary(saved).loadOrder, 'tokenizer-before-session');
  assert.equal(jsMemory().usedJSHeapBytes, null, 'Node has no performance.memory');
});

test('HTTP, decode and JSON failures retain the actual file and failing operation', async () => {
  for (const [response, stage] of [
    [() => new Response('', { status: 404 }), 'tokenizer-read-start'],
    [() => new Response(new Uint8Array([255])), 'tokenizer-decode-start'],
    [() => new Response('{broken'), 'tokenizer-parse-start'],
  ]) {
    const run = new RunDiagnostics('failed-tokenizer', { diagnosticsMode: 'snapshot' }, async () => true);
    await assert.rejects(readPreparationFile({ file: 'tokenizer.json', url: 'https://example.com/tokenizer.json', json: true,
      fetchFile: response, checkpoint: run.checkpoint }));
    assert.equal(run.state.fault.file, 'tokenizer.json');
    assert.equal(run.state.fault.observedDuring, stage);
    await run.finish('failed');
    assert.equal(diagnosticSummary(run.state).observedDuring, stage);
  }
});

test('an interruption at parse-start stays cause unknown, and cancellation prevents construction', async () => {
  const run = new RunDiagnostics('parse-interrupted', { diagnosticsMode: 'snapshot' }, async () => true);
  await run.checkpoint({ stage: 'tokenizer-parse-start', file: 'tokenizer.json' });
  const recovery = recoveryEvidence(run.state);
  const summary = diagnosticSummary({ ...run.state, recovery });
  assert.equal(recovery.cause, 'unknown');
  assert.equal(summary.stage, 'tokenizer-parse-start');
  assert.equal(summary.file, 'tokenizer.json');
  assert.equal(summary.gpuWeightAllocated, null);
  const controller = new AbortController();
  await assert.rejects(prepareTokenizer({ baseURL: 'https://example.com/web/sllm/worker.js', signal: controller.signal,
    checkpoint: async record => { if (record.stage === 'tokenizer-create-start') controller.abort(); },
    fetchFile: async () => new Response('{}'),
    createTokenizer: () => { assert.fail('cancelled initialization must not construct the tokenizer'); },
  }), { name: 'AbortError' });
});
