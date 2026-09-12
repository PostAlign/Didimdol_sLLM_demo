import { build } from 'esbuild';
import { mkdir, cp, readFile, writeFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import path from 'node:path';
import { git, packageRelease } from './release.mjs';

const root = path.resolve(import.meta.dirname, '..');
const ortRoot = path.resolve(process.env.ORT_SOURCE || path.join(root, '.work/onnxruntime'));
const output = path.join(root, 'web/vendor');
await mkdir(output, { recursive: true });
const mode = process.env.ORT_MODE || 'asyncify';
const profile = process.env.ORT_PROFILE || 'mobile';
const threads = process.env.ORT_THREADS === '1';
if (!['asyncify', 'jspi'].includes(mode)) throw new Error('ORT_MODE must be asyncify or jspi');
const artifact = `ort-wasm-simd-threaded.${mode}`;
const sourceArtifact = `ort-wasm-simd${threads ? '-threaded' : ''}.${mode}`;
const artifacts = path.resolve(process.env.ORT_ARTIFACTS || path.join(root, `.work/ort-build-${mode}-${profile}-t${Number(threads)}/Release`));
// Require matching patched binaries; never quietly serve stock WASM with custom JS.
const glue = await readFile(path.join(artifacts, `${sourceArtifact}.mjs`), 'utf8');
if (!/ortRangeLoaderVersion["']?\]?\s*=\s*2/.test(glue)) throw new Error('WASM factory requires range-loader ABI 2');
// Stable public filenames; wasmPaths explicitly selects the matching binary.
for (const ext of ['mjs', 'wasm']) await cp(path.join(artifacts, `${sourceArtifact}.${ext}`), path.join(output, `${artifact}.${ext}`));
const define = Object.fromEntries(Object.entries({
  DISABLE_WEBGL: true, DISABLE_JSEP: true, DISABLE_WASM: false, DISABLE_WASM_PROXY: true,
  ENABLE_JSPI: mode === 'jspi', ENABLE_BUNDLE_WASM_JS: false, DISABLE_WEBGPU: false,
  DISABLE_WEBNN: true, IS_ESM: true,
}).map(([key, value]) => [`BUILD_DEFS.${key}`, String(value)]));
define['BUILD_DEFS.ESM_IMPORT_META_URL'] = 'import.meta.url';
await build({
  entryPoints: [path.join(ortRoot, 'js/web/lib/index.ts')],
  outfile: path.join(output, `ort.${mode}.mjs`), bundle: true, format: 'esm', platform: 'browser',
  target: 'es2022', minify: true, sourcemap: true, define,
  nodePaths: [path.join(root, 'node_modules')], external: ['node:*', 'module', 'worker_threads'],
  alias: { 'onnxruntime-common': path.join(root, 'node_modules/onnxruntime-common/dist/esm/index.js') },
});
// Transformers loads the ORT instance registered by runtime.js. Keep its stock
// import external to avoid a second backend and a mismatched common Tensor class.
await build({
  entryPoints: [path.join(root, 'node_modules/@huggingface/transformers/src/transformers.js')],
  outfile: path.join(output, 'transformers.mjs'), bundle: true, format: 'esm', platform: 'browser',
  target: 'es2022', minify: true, external: ['onnxruntime-web/webgpu', 'onnxruntime-common'],
  plugins: [{ name: 'browser-only', setup(builder) {
    builder.onResolve({ filter: /^(onnxruntime-node|sharp|node:.*)$/ }, args => ({ path: args.path, namespace: 'empty-node' }));
    builder.onLoad({ filter: /.*/, namespace: 'empty-node' }, () => ({ contents: 'module.exports = {};', loader: 'js' }));
  } }],
});
// Bare package imports are not resolved inside browser workers.
const tf = path.join(output, 'transformers.mjs');
await writeFile(tf, (await readFile(tf, 'utf8'))
  .replaceAll('"onnxruntime-web/webgpu"', '"./selected-ort.mjs"')
  .replaceAll('"onnxruntime-common"', '"./selected-ort.mjs"'));
await writeFile(path.join(output, 'selected-ort.mjs'),
  'const ort = globalThis.__didimdolOrt; if (!ort) throw new Error("Load runtime.js first"); export default ort; export const { Tensor, env, InferenceSession } = ort;\n');
await cp(path.join(root, 'node_modules/onnxruntime-web/dist/ort.webgpu.min.mjs'), path.join(output, 'ort.stock.mjs'));
for (const ext of ['mjs', 'wasm']) await cp(path.join(root, `node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.asyncify.${ext}`), path.join(output, `stock-ort-wasm-simd-threaded.asyncify.${ext}`));
await cp(path.join(ortRoot, 'LICENSE'), path.join(output, 'LICENSE-onnxruntime'));
await cp(path.join(ortRoot, 'ThirdPartyNotices.txt'), path.join(output, 'ThirdPartyNotices-onnxruntime.txt'));
await cp(path.join(root, 'node_modules/@huggingface/transformers/LICENSE'), path.join(output, 'LICENSE-transformers'));
const sha = data => createHash('sha256').update(data).digest('hex');
let nativeBuild = null;
try { nativeBuild = JSON.parse(await readFile(path.join(artifacts, 'didimdol-build.json'), 'utf8')); } catch {}
let builds = {};
try { builds = JSON.parse(await readFile(path.join(output, 'build.json'), 'utf8')).builds || {}; } catch {}
builds[mode] = { profile, threads, sourceArtifact, ortCommit: git(ortRoot, 'rev-parse', 'HEAD'), nativeBuild,
  wasmBytes: (await readFile(path.join(output, `${artifact}.wasm`))).byteLength,
  wasmSha256: sha(await readFile(path.join(output, `${artifact}.wasm`))),
  patchSha256: sha(await readFile(path.join(root, 'patches/ort-session-range-loader.patch'))),
  operatorsSha256: profile === 'mobile' ? sha(await readFile(path.join(root, 'model/required-operators.config'))) : null };
const runtimeBuild = {
  ortVersion: '1.26.0-dev.20260416-b7804b056c', transformersVersion: '4.2.0', rangeLoaderVersion: 2,
  modes: Object.keys(builds), builds,
};
await packageRelease(root, runtimeBuild);
console.log(`Built browser runtime: ${mode} → web/vendor`);
