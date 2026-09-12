// No ORT/GPU imports: both the application and tokenizer diagnostic use this loader.
import { releaseBpeSource } from './tokenizer-memory.js';
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
export async function readPreparationFile({ file, url, expected, prefix = 'tokenizer', json = false,
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
  loadOrder = LOAD_ORDER }) {
  const observe = observer({ checkpoint, signal, emit });
  const started = performance.now();
  const identity = { loadOrder, tokenizerBuild: build?.tokenizer ?? null };
  await observe('tokenizer-load', identity);
  const read = file => readPreparationFile({ file, url: new URL(`../../tokenizer/${file}`, baseURL),
    expected: build?.assets?.[`tokenizer/${file}`], json: true, checkpoint, signal, emit, fetchFile });
  // Sequential scopes let the byte buffer and decoded string become collectible before construction.
  let config = await read('tokenizer_config.json');
  let data = await read('tokenizer.json');
  const files = [config.details, data.details];
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
