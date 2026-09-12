import { readFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import path from 'node:path';

export const TOKENIZER_IMPLEMENTATION = 'incremental-bpe-v1';
const sha = value => createHash('sha256').update(value).digest('hex');
const sources = {
  tokenizers: { version: '0.1.3', file: 'dist/tokenizers.mjs',
    sha256: '6d92e25f9576e67124b3a3f910f5cc1df95a42bde4df4f5387f62dee4554f301' },
  transformers: { version: '4.2.0', file: 'src/models/auto/tokenization_auto.js',
    sha256: 'd1ef852e81052053c459b3c0ad424cea54cde9e078615d613cce012e43b0962a' },
};

export function patchTokenizerSource(kind, source, version) {
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
  return source.replace('var object_to_map = (obj) => new Map(Object.entries(obj));',
    `var object_to_map = (obj) => {
  const result = new Map();
  for (const key in obj) if (Object.hasOwn(obj, key)) result.set(key, obj[key]);
  return result;
};`).replace('    this.bpe_ranks = new Map(this.merges.map((x, i) => [JSON.stringify(x), i]));',
    `    this.bpe_ranks = new Map();
    for (let i = 0; i < this.merges.length; i++) {
      this.bpe_ranks.set(JSON.stringify(this.merges[i]), i);
    }`);
}

/** Checked transforms are applied in memory; node_modules is never modified. */
export async function tokenizerPatch(root) {
  const files = new Map();
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
      patchSha256: sha(await readFile(new URL(import.meta.url))) },
    plugin: { name: 'didimdol-tokenizer', setup(builder) {
      builder.onLoad({ filter: /(?:tokenizers\.mjs|tokenization_auto\.js)$/ }, ({ path: file }) => {
        if (files.has(file)) return { contents: files.get(file), loader: 'js', resolveDir: path.dirname(file) };
      });
    } },
  };
}
