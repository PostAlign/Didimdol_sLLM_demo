/** Release construction-only BPE source data in this application's pinned runtime.
 * Audited tokenizers 0.1.3 / transformers 4.2.0 encode, decode, chat-template and
 * ROUGE paths use the runtime model, not these private source-JSON fields.
 * Do not mutate the caller's JSON: another tokenizer may still own it.
 */
export function releaseBpeSource(tokenizer) {
  const core = tokenizer._tokenizer, model = core?.model;
  if (model?.rankKeyFormat !== 'token-id-pair-v2') return null;
  const source = core.tokenizer, config = source.model;
  // Added tokens may extend the vocabulary, but changing a base ID would change
  // numeric merge keys. Refuse an incompatible tokenizer instead of misencoding.
  for (const token in config.vocab) {
    if (Object.hasOwn(config.vocab, token) && (model.tokens_to_ids.get(token) !== config.vocab[token] ||
        model.vocab[config.vocab[token]] !== token)) {
      throw new Error('Added token changes a base BPE vocabulary ID');
    }
  }
  const runtimeConfig = {};
  for (const key of Object.keys(config)) {
    if (key !== 'vocab' && key !== 'merges' && key !== '_didimdolPreparedBpe') runtimeConfig[key] = config[key];
  }
  model.config = runtimeConfig;
  model.merges = null;
  core.tokenizer = { ...source, model: runtimeConfig };
  if ('_tokenizerJSON' in tokenizer) tokenizer._tokenizerJSON = core.tokenizer;
  return { sourceReleased: true, rankKeyFormat: model.rankKeyFormat, preparedFormat: model.preparedFormat ?? null,
    mergeCount: model.mergeCount, numericRankCount: model.bpe_ranks.size,
    rankTableBytes: model.bpe_ranks.keys.byteLength + model.bpe_ranks.ranks.byteLength,
    fallbackRankCount: model.fallback_ranks.size };
}
