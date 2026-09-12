import { readFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import path from 'node:path';

export const TOKENIZER_IMPLEMENTATION = 'compact-bpe-v2';
const sha = value => createHash('sha256').update(value).digest('hex');
const sources = {
  tokenizers: { version: '0.1.3', file: 'dist/tokenizers.mjs',
    sha256: '6d92e25f9576e67124b3a3f910f5cc1df95a42bde4df4f5387f62dee4554f301' },
  transformers: { version: '4.2.0', file: 'src/models/auto/tokenization_auto.js',
    sha256: 'd1ef852e81052053c459b3c0ad424cea54cde9e078615d613cce012e43b0962a' },
};
const retentionSource = { file: 'src/tokenization_utils.js',
  sha256: 'b119e0be7088a6ef6fd56e0821c510559f5f6c6c1275a0805ec77a3dc4a5f68e' };

export function patchTokenizerSource(kind, source, version, { compact = true } = {}) {
  const expected = sources[kind];
  if (!expected || version !== expected.version || sha(source) !== expected.sha256) {
    throw new Error(`Tokenizer patch source/version mismatch: ${kind} ${version}`);
  }
  if (kind === 'transformers') {
    // Reuse upstream class selection without hiding file decode/parse inside from_pretrained.
    return source.replace('        // Some tokenizers are saved with the "Fast" suffix',
      '        return this.from_json(tokenizerJSON, tokenizerConfig);\n    }\n\n' +
      '    static from_json(tokenizerJSON, tokenizerConfig) {\n' +
      '        // Some tokenizers are saved with the "Fast" suffix');
  }
  const incremental = source.replace('var object_to_map = (obj) => new Map(Object.entries(obj));',
    `var object_to_map = (obj) => {
  const result = new Map();
  for (const key in obj) if (Object.hasOwn(obj, key)) result.set(key, obj[key]);
  return result;
};`).replace('    this.bpe_ranks = new Map(this.merges.map((x, i) => [JSON.stringify(x), i]));',
    `    this.bpe_ranks = new Map();
    for (let i = 0; i < this.merges.length; i++) {
      this.bpe_ranks.set(JSON.stringify(this.merges[i]), i);
    }`);
  // Retain v1 solely as the previous-release baseline for the memory comparison.
  if (!compact) return incremental;
  return incremental.replace('var BPE = class extends TokenizerModel_default {', `// Fixed-size open addressing avoids boxed Number keys and Map growth peaks.
var CompactBpeRanks = class {
  constructor(count, stride) {
    let capacity = 1;
    while (capacity < count * 2) capacity *= 2;
    this.keys = new Float64Array(capacity);
    this.ranks = new Uint32Array(capacity);
    this.mask = capacity - 1;
    this.stride = stride;
    this.size = 0;
  }
  slot(a, b) {
    const hash = Math.imul(a, 0x9e3779b1) ^ Math.imul(b, 0x85ebca6b);
    let slot = (hash ^ (hash >>> 16)) & this.mask;
    const key = a * this.stride + b;
    while (this.ranks[slot] && this.keys[slot] !== key) slot = (slot + 1) & this.mask;
    return slot;
  }
  set(a, b, rank) {
    const slot = this.slot(a, b);
    if (!this.ranks[slot]) this.size++;
    this.keys[slot] = a * this.stride + b;
    this.ranks[slot] = rank + 1;
  }
  get(a, b) {
    const value = this.ranks[this.slot(a, b)];
    return value ? value - 1 : undefined;
  }
};
var BPE = class extends TokenizerModel_default {`).replace(`    this.bpe_ranks = new Map();
    for (let i = 0; i < this.merges.length; i++) {
      this.bpe_ranks.set(JSON.stringify(this.merges[i]), i);
    }`, `    this.rankKeyFormat = 'token-id-pair-v2';
    this.rankStride = this.vocab.length;
    if (!Number.isSafeInteger(this.rankStride * this.rankStride)) {
      throw new Error('BPE vocabulary exceeds exact numeric pair-key range');
    }
    this.bpe_ranks = new CompactBpeRanks(this.merges.length, this.rankStride);
    this.fallback_ranks = new Map();
    this.mergeCount = this.merges.length;
    for (let i = 0; i < this.merges.length; i++) {
      const [left, right] = this.merges[i];
      const a = this.tokens_to_ids.get(left), b = this.tokens_to_ids.get(right);
      if (Number.isInteger(a) && a >= 0 && a < this.rankStride &&
          Number.isInteger(b) && b >= 0 && b < this.rankStride) {
        this.bpe_ranks.set(a, b, i);
      } else {
        this.fallback_ranks.set(JSON.stringify([left, right]), i);
      }
    }`).replace(`    const rank = this.bpe_ranks.get(
      JSON.stringify([node.token, node.next.token])
    );`, `    const rank = this.rank_for_pair(node.token, node.next.token);`)
    .replace('  add_node(queue, node) {', `  rank_for_pair(left, right) {
    const a = this.tokens_to_ids.get(left), b = this.tokens_to_ids.get(right);
    const rank = Number.isInteger(a) && a >= 0 && a < this.rankStride &&
      Number.isInteger(b) && b >= 0 && b < this.rankStride
      ? this.bpe_ranks.get(a, b) : undefined;
    return rank ?? (this.fallback_ranks.size ? this.fallback_ranks.get(JSON.stringify([left, right])) : undefined);
  }
  add_node(queue, node) {`);
}

/** Checked transforms are applied in memory; node_modules is never modified. */
export async function tokenizerPatch(root) {
  const files = new Map();
  const retention = await readFile(path.join(root, 'node_modules/@huggingface/transformers', retentionSource.file), 'utf8');
  if (sha(retention) !== retentionSource.sha256) throw new Error('Tokenizer source-retention audit mismatch');
  for (const [kind, spec] of Object.entries(sources)) {
    const directory = path.join(root, 'node_modules/@huggingface', kind);
    const { version } = JSON.parse(await readFile(path.join(directory, 'package.json'), 'utf8'));
    const file = path.join(directory, spec.file);
    files.set(file, patchTokenizerSource(kind, await readFile(file, 'utf8'), version));
  }
  return {
    metadata: { implementation: TOKENIZER_IMPLEMENTATION, version: sources.tokenizers.version,
      transformersVersion: sources.transformers.version,
      sourceSha256: sources.tokenizers.sha256, autoTokenizerSourceSha256: sources.transformers.sha256,
      retentionSourceSha256: retentionSource.sha256,
      patchSha256: sha(await readFile(new URL(import.meta.url))) },
    plugin: { name: 'didimdol-tokenizer', setup(builder) {
      builder.onLoad({ filter: /(?:tokenizers\.mjs|tokenization_auto\.js)$/ }, ({ path: file }) => {
        if (files.has(file)) return { contents: files.get(file), loader: 'js', resolveDir: path.dirname(file) };
      });
    } },
  };
}
