/** Match factory, WASM and build metadata across static-site updates. */
export async function loadOrt(mode = 'asyncify') {
  if (!['asyncify', 'jspi', 'stock'].includes(mode)) throw new Error('Invalid ORT mode');
  if (mode === 'jspi' && !(typeof WebAssembly.Suspending === 'function' && typeof WebAssembly.promising === 'function')) {
    throw new Error('This browser does not support JSPI. Use Asyncify.');
  }
  const response = await fetch(new URL('../vendor/build.json', import.meta.url), { cache: 'no-store' });
  if (!response.ok) throw new Error('Runtime build metadata unavailable');
  const build = await response.json();
  if (mode !== 'stock' && (build.rangeLoaderVersion !== 2 || !build.builds?.[mode])) {
    throw new Error(`Build the ${mode} runtime with range-loader ABI 2`);
  }
  const version = build.builds?.[mode]?.wasmSha256 || build.ortVersion;
  const asset = name => {
    const url = new URL(`../vendor/${name}`, import.meta.url);
    url.searchParams.set('v', version); return url.href;
  };
  const ort = await import(asset(`ort.${mode}.mjs`));
  const file = mode === 'stock' ? 'stock-ort-wasm-simd-threaded.asyncify' : `ort-wasm-simd-threaded.${mode}`;
  ort.env.wasm.wasmPaths = { mjs: asset(`${file}.mjs`), wasm: asset(`${file}.wasm`) };
  ort.env.wasm.numThreads = 1; ort.env.wasm.proxy = false;
  return { ort, build };
}
