const query = new URL(import.meta.url).searchParams;
// Asyncify is the default even on JSPI-capable devices, for reproducible comparisons.
const mode = query.get('mode') || 'asyncify';
import { loadOrt } from './ort-runtime.js';
const { ort, build, asset } = await loadOrt(mode);
// Use the browser backend branch in transformers.js so its supportedDevices
// list is populated. Its Symbol.for('onnxruntime') branch skips that setup.
globalThis.__didimdolOrt = ort;
export { ort, mode, build };
const transformers = await import(asset('web/vendor/transformers.mjs'));
export const { AutoModelForCausalLM, AutoTokenizer, AutoConfig, Gemma3ForCausalLM, BaseStreamer, InterruptableStoppingCriteria, env, random } = transformers;
