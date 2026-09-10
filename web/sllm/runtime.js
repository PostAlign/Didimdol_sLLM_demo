const query = new URL(import.meta.url).searchParams;
// Asyncify is the default even on JSPI-capable devices, for reproducible comparisons.
const mode = query.get('mode') || 'asyncify';
if (!['asyncify', 'jspi', 'stock'].includes(mode)) throw new Error('Invalid ORT mode');
if (mode === 'jspi' && !(typeof WebAssembly.Suspending === 'function' && typeof WebAssembly.promising === 'function')) {
  throw new Error('This browser does not support JSPI. Use Asyncify.');
}
const ort = await import(`../vendor/ort.${mode}.mjs`);
const file = mode === 'stock' ? 'stock-ort-wasm-simd-threaded.asyncify' : `ort-wasm-simd-threaded.${mode}`;
ort.env.wasm.wasmPaths = {
  mjs: new URL(`../vendor/${file}.mjs`, import.meta.url).href,
  wasm: new URL(`../vendor/${file}.wasm`, import.meta.url).href,
};
ort.env.wasm.numThreads = 1;
ort.env.wasm.proxy = false;
// Use the browser backend branch in transformers.js so its supportedDevices
// list is populated. Its Symbol.for('onnxruntime') branch skips that setup.
globalThis.__didimdolOrt = ort;
export { ort, mode };
const transformers = await import('../vendor/transformers.mjs');
export const { AutoModelForCausalLM, AutoTokenizer, BaseStreamer, InterruptableStoppingCriteria, env, random } = transformers;
