import { readRelease } from './release-manifest.js';
let releasePromise;
export const runtimeRelease = () => releasePromise ||= readRelease();

/** Resolve factory, WASM and JS from the same immutable release. */
export async function loadOrt(mode = 'asyncify') {
  if (!['asyncify', 'jspi', 'stock'].includes(mode)) throw new Error('Invalid ORT mode');
  if (mode === 'jspi' && !(typeof WebAssembly.Suspending === 'function' && typeof WebAssembly.promising === 'function')) {
    throw new Error('This browser does not support JSPI. Use Asyncify.');
  }
  const { build, asset } = await runtimeRelease();
  if (mode !== 'stock' && (build.rangeLoaderVersion !== 2 || !build.builds?.[mode])) {
    throw new Error(`Build the ${mode} runtime with range-loader ABI 2`);
  }
  const ort = await import(asset(`web/vendor/ort.${mode}.mjs`));
  const file = mode === 'stock' ? 'stock-ort-wasm-simd-threaded.asyncify' : `ort-wasm-simd-threaded.${mode}`;
  ort.env.wasm.wasmPaths = { mjs: asset(`web/vendor/${file}.mjs`), wasm: asset(`web/vendor/${file}.wasm`) };
  ort.env.wasm.numThreads = 1; ort.env.wasm.proxy = false;
  return { ort, build, asset };
}
