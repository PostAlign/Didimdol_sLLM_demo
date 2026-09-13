// Versioned, build-time BPE data. No ORT dependency and no source merge objects.
export const PREPARED_TOKENIZER_FORMAT = 'didimdol-bpe-v1';
export const PREPARED_TOKENIZER_FILES = ['tokenizer-prepared.json', 'tokenizer-vocab.bin', 'tokenizer-ranks.bin'];

export function validatePreparedManifest(value, sources) {
  if (value?.format !== PREPARED_TOKENIZER_FORMAT || value.tokenizer?.model?.type !== 'BPE') {
    throw new Error('Unsupported prepared tokenizer format');
  }
  for (const file of ['tokenizer.json', 'tokenizer_config.json']) {
    if (!sources?.[file]?.sha256 || value.sources?.[file]?.sha256 !== sources[file].sha256) {
      throw new Error(`Prepared tokenizer source mismatch: ${file}`);
    }
  }
  const { vocabCount, stride, capacity, mergeCount, numericRankCount } = value;
  if (!Number.isSafeInteger(vocabCount) || vocabCount < 1 || vocabCount > 2**24 || stride !== vocabCount ||
      !Number.isSafeInteger(stride * stride) || !Number.isSafeInteger(capacity) || capacity < 2 ||
      capacity > 2**26 || !Number.isInteger(Math.log2(capacity)) ||
      !Number.isSafeInteger(mergeCount) || mergeCount < 0 || mergeCount >= 2**32 - 1 ||
      !Number.isSafeInteger(numericRankCount) || numericRankCount < 0 || numericRankCount > mergeCount ||
      numericRankCount >= capacity || value.tokenizer.model.vocab || value.tokenizer.model.merges) {
    throw new Error('Invalid prepared tokenizer layout');
  }
  return value;
}

export function decodePreparedVocabulary(buffer, count) {
  const headerBytes = (count + 1) * 4;
  if (buffer.byteLength < headerBytes) throw new Error('Truncated prepared vocabulary');
  const offsets = new DataView(buffer, 0, headerBytes), bytes = new Uint8Array(buffer, headerBytes);
  // A token may begin with U+FEFF. It is token content, not a file BOM.
  const decoder = new TextDecoder('utf-8', { fatal: true, ignoreBOM: true }), vocab = new Array(count);
  if (offsets.getUint32(0, true) !== 0 || offsets.getUint32(count * 4, true) !== bytes.byteLength) {
    throw new Error('Invalid prepared vocabulary offsets');
  }
  for (let i = 0; i < count; i++) {
    const start = offsets.getUint32(i * 4, true), end = offsets.getUint32((i + 1) * 4, true);
    if (end < start || end > bytes.byteLength) throw new Error('Invalid prepared vocabulary range');
    vocab[i] = decoder.decode(bytes.subarray(start, end));
  }
  return vocab;
}

export function decodePreparedRanks(buffer, manifest) {
  const { capacity, stride, mergeCount, numericRankCount } = manifest;
  if (buffer.byteLength !== capacity * 12) throw new Error('Invalid prepared rank table size');
  // Files are little-endian. Current supported browsers use little-endian typed arrays.
  if (new Uint8Array(new Uint32Array([1]).buffer)[0] !== 1) throw new Error('Unsupported tokenizer byte order');
  const keys = new Float64Array(buffer, 0, capacity), ranks = new Uint32Array(buffer, capacity * 8, capacity);
  let count = 0;
  for (let i = 0; i < capacity; i++) {
    if (!ranks[i]) continue;
    count++;
    if (ranks[i] > mergeCount || !Number.isSafeInteger(keys[i]) || keys[i] < 0 || keys[i] >= stride * stride) {
      throw new Error('Invalid prepared rank entry');
    }
  }
  if (count !== numericRankCount) throw new Error('Invalid prepared rank count');
  return { keys, ranks, stride, mergeCount, numericRankCount };
}
