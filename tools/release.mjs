import { createHash } from 'node:crypto';
import { createReadStream } from 'node:fs';
import { mkdir, readFile, writeFile, rename, cp, stat } from 'node:fs/promises';
import { execFileSync } from 'node:child_process';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { validateRelease, releaseDescriptor } from '../web/sllm/release-manifest.js';
import { prepareTokenizerAssets } from './prepare-tokenizer.mjs';

export const sha = value => createHash('sha256').update(value).digest('hex');
export const releaseId = manifest => sha(JSON.stringify(releaseDescriptor(manifest)));
export function git(root, ...args) { return execFileSync('git', ['-C', root, ...args], { encoding: 'utf8' }).trim(); }
export async function hashStream(stream) {
  const hash = createHash('sha256'); let bytes = 0;
  for await (const chunk of stream) { bytes += chunk.length; hash.update(chunk); }
  return { bytes, sha256: hash.digest('hex') };
}

/** Repackage cached native binaries with CURRENT application files on every deployment. */
export async function packageRelease(root, runtimeBuild) {
  const vendor = path.join(root, 'web/vendor');
  const previous = runtimeBuild || JSON.parse(await readFile(path.join(vendor, 'build.json'), 'utf8'));
  const names = git(root, 'ls-files', '--cached', '--others', '--exclude-standard', '-z').split('\0').filter(Boolean)
    .filter(name => /^(web\/|model\/|tokenizer\/|index\.html$|manifest\.webmanifest$)/.test(name) && !name.startsWith('web/vendor/'));
  const vendorNames = ['transformers.mjs', 'selected-ort.mjs', 'ort.stock.mjs',
    'stock-ort-wasm-simd-threaded.asyncify.mjs', 'stock-ort-wasm-simd-threaded.asyncify.wasm',
    'LICENSE-onnxruntime', 'ThirdPartyNotices-onnxruntime.txt', 'LICENSE-transformers'];
  for (const mode of previous.modes) vendorNames.push(`ort.${mode}.mjs`,
    `ort-wasm-simd-threaded.${mode}.mjs`, `ort-wasm-simd-threaded.${mode}.wasm`);
  names.push(...vendorNames.map(name => `web/vendor/${name}`));
  if (previous.tokenizer?.preparedFormat === 'didimdol-bpe-v1') names.push(...await prepareTokenizerAssets(root));
  const assets = {};
  for (const name of [...new Set(names)].sort()) {
    // Deleted tracked files must not reappear in a local snapshot.
    if (!(await stat(path.join(root, name)).catch(() => null))?.isFile()) continue;
    assets[name] = await hashStream(createReadStream(path.join(root, name)));
  }
  const build = { releaseSchema: 1, ortVersion: previous.ortVersion, transformersVersion: previous.transformersVersion,
    ...(previous.tokenizer ? { tokenizer: previous.tokenizer } : {}),
    rangeLoaderVersion: previous.rangeLoaderVersion, modes: previous.modes, builds: previous.builds,
    provenance: { appCommit: git(root, 'rev-parse', 'HEAD'), appDirty: !!git(root, 'status', '--porcelain'),
      nodeVersion: process.version }, assets };
  build.releaseId = releaseId(build);
  build.assetBase = `releases/${build.releaseId}/`;
  validateRelease(build);
  const destination = path.join(vendor, build.assetBase);
  for (const name of Object.keys(assets)) {
    const target = path.join(destination, name);
    await mkdir(path.dirname(target), { recursive: true });
    // Existing releases are immutable. A corrupted existing snapshot is an error.
    try { await cp(path.join(root, name), target, { force: false, errorOnExist: true }); }
    catch (error) { if (error.code !== 'ERR_FS_CP_EEXIST') throw error; }
  }
  await writeFile(path.join(destination, 'web/vendor/build.json'), JSON.stringify({ ...build, assetBase: '../../' }, null, 2));
  await verifyRelease(pathToFileURL(destination + path.sep), { ...build, assetBase: './' });
  // Publish the pointer only after every file in the new snapshot has been verified.
  const pending = path.join(vendor, 'build.json.tmp');
  await writeFile(pending, JSON.stringify(build, null, 2));
  await rename(pending, path.join(vendor, 'build.json'));
  return build;
}

/** Sequential streaming verification avoids materializing model/tokenizer/runtime copies. */
export async function verifyRelease(baseURL, suppliedBuild) {
  const base = new URL(baseURL);
  const metadataURL = new URL('web/vendor/build.json', base);
  const build = validateRelease(suppliedBuild || JSON.parse(base.protocol === 'file:'
    ? await readFile(metadataURL, 'utf8') : await fetch(metadataURL, { cache: 'no-store' }).then(async r => {
      if (!r.ok) throw new Error(`Build metadata HTTP ${r.status}`); return r.text();
    })));
  if (releaseId(build) !== build.releaseId) throw new Error('Release identity mismatch');
  const assetBase = suppliedBuild ? base : new URL(build.assetBase, metadataURL);
  if (!suppliedBuild && (assetBase.origin !== base.origin || !assetBase.pathname.endsWith(`/releases/${build.releaseId}/`))) {
    throw new Error('Release asset URL mismatch');
  }
  async function verifyAsset(name, expected, from = assetBase) {
    const url = new URL(name, from);
    let stream;
    if (url.protocol === 'file:') stream = createReadStream(fileURLToPath(url));
    else {
      const response = await fetch(url, { cache: 'no-store' });
      if (!response.ok || !response.body) throw new Error(`Release asset HTTP ${response.status}: ${name}`);
      stream = response.body;
    }
    const actual = await hashStream(stream);
    if (actual.bytes !== expected.bytes || actual.sha256 !== expected.sha256) throw new Error(`Release hash mismatch: ${name}`);
  }
  for (const [name, expected] of Object.entries(build.assets)) await verifyAsset(name, expected);
  // The runtime reads this nested manifest; checking only the outer pointer misses mixed manifests.
  const nestedURL = new URL('web/vendor/build.json', assetBase);
  const nested = validateRelease(JSON.parse(nestedURL.protocol === 'file:' ? await readFile(nestedURL, 'utf8')
    : await fetch(nestedURL, { cache: 'no-store' }).then(async response => {
      if (!response.ok) throw new Error(`Nested metadata HTTP ${response.status}`); return response.text();
    })));
  if (nested.assetBase !== '../../' || nested.releaseId !== build.releaseId || releaseId(nested) !== build.releaseId) {
    throw new Error('Nested release identity mismatch');
  }
  if (!suppliedBuild) {
    for (const name of ['index.html', 'manifest.webmanifest', 'web/app.js', 'web/sllm/release-manifest.js',
      'web/sllm/experiments/index.html', 'web/sllm/experiments/bootstrap.js']) {
      if (build.assets[name]) await verifyAsset(name, build.assets[name], base);
    }
  }
  return { releaseId: build.releaseId, verifiedAssets: Object.keys(build.assets).length };
}

if (process.argv[1] && import.meta.url === pathToFileURL(path.resolve(process.argv[1])).href) {
  const root = path.resolve(import.meta.dirname, '..');
  if (process.argv[2] === 'verify') {
    const location = process.argv[3];
    const base = location && /^https?:/.test(location) ? new URL(location.endsWith('/') ? location : location + '/')
      : pathToFileURL(path.resolve(location || root) + path.sep);
    console.log(JSON.stringify(await verifyRelease(base), null, 2));
  } else console.log(JSON.stringify({ releaseId: (await packageRelease(root)).releaseId }, null, 2));
}
