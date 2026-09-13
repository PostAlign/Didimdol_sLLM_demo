// No ORT/GPU imports: both the application and tokenizer diagnostic use this loader.
import { releaseBpeSource } from './tokenizer-memory.js';
import { PREPARED_TOKENIZER_FORMAT, validatePreparedManifest, decodePreparedVocabulary, decodePreparedRanks } from './prepared-tokenizer.js';
export const LOAD_ORDER = 'tokenizer-before-session';

export function jsMemory() {
  const memory = globalThis.performance?.memory;
  return { usedJSHeapBytes: Number.isFinite(memory?.usedJSHeapSize) ? memory.usedJSHeapSize : null,
    totalJSHeapBytes: Number.isFinite(memory?.totalJSHeapSize) ? memory.totalJSHeapSize : null,
    source: memory ? 'performance.memory' : null };
}

function observer({ checkpoint, signal, emit = () => {} }) {
  return async (stage, details = {}) => {
    signal?.throwIfAborted();
    const record = { stage, ...details, jsMemory: jsMemory() };
    await checkpoint(record);
    signal?.throwIfAborted();
    emit(record);
  };
}

/** HTTP caching uses immutable release URLs; no Response.clone or CacheStorage copy of the JSON. */
export async function readPreparationFile({ file, url, expected, prefix = 'tokenizer', json = false, binary = false, verifyHash = false,
  checkpoint, signal, emit, fetchFile = fetch }) {
  const observe = observer({ checkpoint, signal, emit });
  const started = performance.now();
  const details = { file, expectedBytes: expected?.bytes ?? null, expectedSha256: expected?.sha256 ?? null };
  let buffer, text, observedDuring;
  const mark = async suffix => { observedDuring = `${prefix}-${suffix}`; await observe(observedDuring, details); };
  try {
    await mark('read-start');
    let response = await fetchFile(url, { signal });
    if (!response.ok) throw new Error(`${file}: HTTP ${response.status}`);
    buffer = await response.arrayBuffer();
    response = null;
    details.bytes = buffer.byteLength;
    if (expected && details.bytes !== expected.bytes) throw new Error(`${file}: release size mismatch`);
    await mark('read-complete');
    if (verifyHash) {
      await mark('verify-start');
      if (!expected?.sha256) throw new Error(`${file}: missing release hash`);
      const digest = await crypto.subtle.digest('SHA-256', buffer);
      const actual = Array.from(new Uint8Array(digest), byte => byte.toString(16).padStart(2, '0')).join('');
      if (actual !== expected.sha256) throw new Error(`${file}: release hash mismatch`);
      await mark('verify-complete');
    }
    if (binary) return { value: buffer, details: { ...details, durationMs: performance.now() - started } };
    await mark('decode-start');
    text = new TextDecoder('utf-8', { fatal: true }).decode(buffer);
    buffer = null;
    details.characters = text.length;
    await mark('decode-complete');
    if (!json) return { value: text, details: { ...details, durationMs: performance.now() - started } };
    await mark('parse-start');
    const value = JSON.parse(text);
    text = null;
    await mark('parse-complete');
    return { value, details: { ...details, durationMs: performance.now() - started } };
  } catch (error) {
    if (!signal?.aborted) await observe(`${prefix}-error`, { ...details, observedDuring, message: String(error), errorType: error.name });
    throw error;
  } finally { buffer = null; text = null; }
}

export async function prepareTokenizer({ createTokenizer, baseURL, build, checkpoint, signal, emit, fetchFile,
  loadOrder = LOAD_ORDER, format = 'json' }) {
  if (!['json', 'prepared'].includes(format)) throw new Error('Unknown tokenizer format');
  if (format === 'prepared' && build?.tokenizer?.preparedFormat !== PREPARED_TOKENIZER_FORMAT) {
    throw new Error('Rebuild the runtime for the prepared tokenizer');
  }
  const observe = observer({ checkpoint, signal, emit });
  const started = performance.now();
  const identity = { loadOrder, tokenizerFormat: format, tokenizerBuild: build?.tokenizer ?? null };
  await observe('tokenizer-load', identity);
  const read = file => readPreparationFile({ file, url: new URL(`../../tokenizer/${file}`, baseURL),
    expected: build?.assets?.[`tokenizer/${file}`], json: true, verifyHash: format === 'prepared', checkpoint, signal, emit, fetchFile });
  // Sequential scopes let the byte buffer and decoded string become collectible before construction.
  let config = await read('tokenizer_config.json');
  let data;
  const files = [config.details];
  if (format === 'json') { data = await read('tokenizer.json'); files.push(data.details); }
  else {
    const preparedRead = (file, binary = false) => readPreparationFile({ file,
      url: new URL(`../vendor/tokenizer/${file}`, baseURL),
      expected: build.assets?.[`web/vendor/tokenizer/${file}`], binary, json: !binary, verifyHash: true,
      checkpoint, signal, emit, fetchFile });
    const metadata = await preparedRead('tokenizer-prepared.json'); files.push(metadata.details);
    let manifest;
    try {
      manifest = validatePreparedManifest(metadata.value, Object.fromEntries(
        ['tokenizer.json', 'tokenizer_config.json'].map(file => [file, build.assets?.[`tokenizer/${file}`]])));
      const vocabulary = await preparedRead('tokenizer-vocab.bin', true); files.push(vocabulary.details);
      await observe('tokenizer-vocabulary-start', identity);
      const vocab = decodePreparedVocabulary(vocabulary.value, manifest.vocabCount);
      vocabulary.value = null;
      await observe('tokenizer-vocabulary-complete', { ...identity, vocabCount: vocab.length });
      const table = await preparedRead('tokenizer-ranks.bin', true); files.push(table.details);
      const ranks = decodePreparedRanks(table.value, manifest);
      const value = manifest.tokenizer;
      value.model._didimdolPreparedBpe = { ...ranks, vocab, addedTokens: value.added_tokens };
      data = { value };
    } catch (error) {
      // File read/hash errors already preserve their precise operation.
      await observe('tokenizer-error', { ...identity, observedDuring: 'tokenizer-prepared-validation', message: String(error), errorType: error.name });
      throw error;
    }
  }
  await observe('tokenizer-create-start', { ...identity, tokenizerClass: config.value.tokenizer_class });
  let tokenizer;
  let memoryLayout;
  try {
    tokenizer = createTokenizer(data.value, config.value);
    memoryLayout = releaseBpeSource(tokenizer);
  }
  catch (error) {
    await observe('tokenizer-error', { ...identity, observedDuring: 'tokenizer-create-start',
      message: String(error), errorType: error.name });
    throw error;
  }
  // The compact runtime now owns the required lookup structures. Dropping these
  // references makes construction data collectible; it does not force browser GC.
  data = null; config = null;
  const summary = { ...identity, files, durationMs: performance.now() - started,
    vocabSize: tokenizer._tokenizer.model.vocab.length,
    mergeCount: tokenizer._tokenizer.model.mergeCount ?? tokenizer._tokenizer.model.merges?.length ?? null,
    memoryLayout };
  await observe('tokenizer-create-complete', summary);
  await observe('tokenizer-ready', summary);
  return { tokenizer, summary };
}
