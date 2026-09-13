import { readFile, mkdir, writeFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import { patchTokenizerSource } from './tokenizer-patch.mjs';
import { releaseBpeSource } from '../web/sllm/tokenizer-memory.js';
import { PREPARED_TOKENIZER_FORMAT } from '../web/sllm/prepared-tokenizer.js';

const sha = data => createHash('sha256').update(data).digest('hex');

export function serializeTokenizer(tokenizer, source, sources) {
  const layout = releaseBpeSource(tokenizer), model = tokenizer._tokenizer.model;
  if (!layout || layout.fallbackRankCount) throw new Error('Prepared tokenizer requires numeric BPE ranks');
  const vocabCount = model.rankStride;
  if (vocabCount < 1 || !model.vocab.slice(0, vocabCount).every(value => typeof value === 'string') ||
      Object.keys(source.model.vocab).length !== vocabCount) throw new Error('Prepared vocabulary must have contiguous base IDs');
  const tokens = Array.from({ length: vocabCount }, (_, id) => Buffer.from(model.vocab[id], 'utf8'));
  if (tokens.some((token, id) => token.toString('utf8') !== model.vocab[id])) throw new Error('Prepared vocabulary requires lossless UTF-8 tokens');
  const headerBytes = (vocabCount + 1) * 4;
  const vocabulary = Buffer.alloc(headerBytes + tokens.reduce((sum, token) => sum + token.length, 0));
  let offset = 0;
  for (let id = 0; id < vocabCount; id++) {
    vocabulary.writeUInt32LE(offset, id * 4); tokens[id].copy(vocabulary, headerBytes + offset); offset += tokens[id].length;
  }
  vocabulary.writeUInt32LE(offset, vocabCount * 4);
  const capacity = model.bpe_ranks.keys.length, ranks = Buffer.alloc(capacity * 12);
  for (let i = 0; i < capacity; i++) {
    ranks.writeDoubleLE(model.bpe_ranks.keys[i], i * 8);
    ranks.writeUInt32LE(model.bpe_ranks.ranks[i], capacity * 8 + i * 4);
  }
  const { vocab, merges, ...modelConfig } = source.model;
  const metadata = { format: PREPARED_TOKENIZER_FORMAT, sources, vocabCount, stride: model.rankStride,
    capacity, mergeCount: model.mergeCount, numericRankCount: model.bpe_ranks.size,
    tokenizer: { ...source, model: modelConfig } };
  return { 'tokenizer-prepared.json': Buffer.from(JSON.stringify(metadata)), 'tokenizer-vocab.bin': vocabulary,
    'tokenizer-ranks.bin': ranks };
}

export async function prepareTokenizerAssets(root) {
  const sources = {}, bytes = {};
  for (const file of ['tokenizer.json', 'tokenizer_config.json']) {
    bytes[file] = await readFile(path.join(root, 'tokenizer', file));
    sources[file] = { bytes: bytes[file].length, sha256: sha(bytes[file]) };
  }
  const source = await readFile(path.join(root, 'node_modules/@huggingface/tokenizers/dist/tokenizers.mjs'), 'utf8');
  const { Tokenizer } = await import(`data:text/javascript;base64,${Buffer.from(patchTokenizerSource('tokenizers', source, '0.1.3')).toString('base64')}`);
  const data = JSON.parse(bytes['tokenizer.json']), config = JSON.parse(bytes['tokenizer_config.json']);
  const assets = serializeTokenizer({ _tokenizer: new Tokenizer(data, config) }, data, sources);
  const directory = path.join(root, 'web/vendor/tokenizer');
  await mkdir(directory, { recursive: true });
  for (const [file, value] of Object.entries(assets)) await writeFile(path.join(directory, file), value);
  return Object.keys(assets).map(file => `web/vendor/tokenizer/${file}`);
}

if (process.argv[1] && import.meta.url === pathToFileURL(path.resolve(process.argv[1])).href) {
  console.log(await prepareTokenizerAssets(path.resolve(import.meta.dirname, '..')));
}
