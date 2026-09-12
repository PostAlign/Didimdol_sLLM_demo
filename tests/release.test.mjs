import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, mkdir, readFile, writeFile, rm, chmod } from 'node:fs/promises';
import { execFileSync } from 'node:child_process';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import { createServer } from 'node:http';
import { createReadStream } from 'node:fs';
import { validateRelease } from '../web/sllm/release-manifest.js';
import { releaseId, sha, verifyRelease, packageRelease } from '../tools/release.mjs';

async function fixture(t) {
  const root = await mkdtemp(path.join(tmpdir(), 'didimdol-release-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  const assets = {};
  const files = ['web/app-main.js', 'web/sllm/worker.js', 'web/vendor/transformers.mjs', 'web/vendor/selected-ort.mjs',
    'web/vendor/ort.asyncify.mjs', 'web/vendor/ort-wasm-simd-threaded.asyncify.mjs', 'web/vendor/ort-wasm-simd-threaded.asyncify.wasm'];
  for (const name of files) assets[name] = { bytes: Buffer.byteLength(name), sha256: sha(name) };
  const wasm = assets[files.at(-1)];
  const build = { releaseSchema: 1, ortVersion: 'test', transformersVersion: 'test', rangeLoaderVersion: 2,
    modes: ['asyncify'], builds: { asyncify: { wasmBytes: wasm.bytes, wasmSha256: wasm.sha256 } }, provenance: { appCommit: 'test' }, assets };
  build.releaseId = releaseId(build); build.assetBase = `releases/${build.releaseId}/`;
  const destination = path.join(root, 'web/vendor', build.assetBase);
  for (const name of files) {
    await mkdir(path.dirname(path.join(destination, name)), { recursive: true });
    await writeFile(path.join(destination, name), name);
  }
  await writeFile(path.join(root, 'web/vendor/build.json'), JSON.stringify(build));
  await writeFile(path.join(destination, 'web/vendor/build.json'), JSON.stringify({ ...build, assetBase: '../../' }));
  return { root, destination, build, url: pathToFileURL(root + path.sep) };
}

test('release verification rejects mixed factory/transformers bytes even if WASM is unchanged', async t => {
  const f = await fixture(t);
  assert.equal((await verifyRelease(f.url)).releaseId, f.build.releaseId);
  await writeFile(path.join(f.destination, 'web/vendor/transformers.mjs'), 'older transformers');
  await assert.rejects(verifyRelease(f.url), /Release hash mismatch: web\/vendor\/transformers/);
});

test('manifest rejects incorrect binary metadata and paths escaping the release', async t => {
  const { build } = await fixture(t);
  const mismatched = structuredClone(build); mismatched.builds.asyncify.wasmBytes++;
  assert.throws(() => validateRelease(mismatched), /Runtime asset mismatch/);
  const unsafe = structuredClone(build); unsafe.assets['../outside.js'] = { bytes: 1, sha256: sha('x') };
  assert.throws(() => validateRelease(unsafe), /Invalid release asset/);
});

test('tokenizer implementation and patch hashes participate in the release identity', async t => {
  const { build } = await fixture(t);
  const original = releaseId(build);
  build.tokenizer = { implementation: 'incremental-bpe-v1', patchSha256: sha('patch-one') };
  const patched = releaseId(build);
  assert.notEqual(patched, original);
  build.tokenizer.patchSha256 = sha('patch-two');
  assert.notEqual(releaseId(build), patched);
});

test('a mismatched nested manifest is rejected before it can select a different runtime', async t => {
  const f = await fixture(t);
  await writeFile(path.join(f.destination, 'web/vendor/build.json'), JSON.stringify({ ...f.build, assetBase: '../' }));
  await assert.rejects(verifyRelease(f.url), /Nested release identity mismatch/);
});

test('application changes create a new immutable release and failed packaging preserves the old pointer', async t => {
  const f = await fixture(t);
  for (const name of Object.keys(f.build.assets)) {
    await mkdir(path.dirname(path.join(f.root, name)), { recursive: true });
    await writeFile(path.join(f.root, name), name);
  }
  execFileSync('git', ['init', '-q', f.root]);
  execFileSync('git', ['-C', f.root, 'add', 'web/app-main.js', 'web/sllm/worker.js']);
  execFileSync('git', ['-C', f.root, '-c', 'user.name=Release Test', '-c', 'user.email=test@example.com', 'commit', '-qm', 'fixture']);
  const first = await packageRelease(f.root);
  await writeFile(path.join(f.root, 'web/sllm/worker.js'), 'new worker');
  const second = await packageRelease(f.root);
  assert.notEqual(second.releaseId, first.releaseId);
  const original = path.join(f.root, 'web/vendor', first.assetBase, 'web/sllm/worker.js');
  assert.equal(await readFile(original, 'utf8'), 'web/sllm/worker.js');
  // Repackaging an unchanged release is idempotent.
  assert.equal((await packageRelease(f.root)).releaseId, second.releaseId);
  await writeFile(path.join(f.root, 'web/vendor/ort-wasm-simd-threaded.asyncify.wasm'), 'different native binary');
  await assert.rejects(packageRelease(f.root), /Runtime asset mismatch/);
  const pointer = JSON.parse(await readFile(path.join(f.root, 'web/vendor/build.json'), 'utf8'));
  assert.equal(pointer.releaseId, second.releaseId);
});

test('remote verification supports a Pages subpath and rejects a modified release identity', async t => {
  const f = await fixture(t);
  const server = createServer((req, res) => {
    const file = path.join(f.root, decodeURIComponent(new URL(req.url, 'http://localhost').pathname).slice('/demo/'.length));
    const stream = createReadStream(file); stream.on('error', () => res.writeHead(404).end()); stream.pipe(res);
  });
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  t.after(() => { server.closeAllConnections(); return new Promise(resolve => server.close(resolve)); });
  const url = new URL(`http://127.0.0.1:${server.address().port}/demo/`);
  assert.equal((await verifyRelease(url)).verifiedAssets, Object.keys(f.build.assets).length);
  f.build.provenance.appCommit = 'changed';
  await writeFile(path.join(f.root, 'web/vendor/build.json'), JSON.stringify(f.build));
  await assert.rejects(verifyRelease(url), /Release identity mismatch/);
});

test('native provenance uses the compiler recorded by CMake and preserves build arguments', async t => {
  const f = await fixture(t);
  const compiler = path.join(f.root, 'compiler with spaces');
  await writeFile(compiler, '#!/usr/bin/env node\nconsole.log("emcc fixture compiler 1.2.3");\n');
  await chmod(compiler, 0o755);
  await mkdir(path.join(f.root, 'CMakeFiles/1.2.3'), { recursive: true });
  await writeFile(path.join(f.root, 'CMakeFiles/1.2.3/CMakeCCompiler.cmake'), `set(CMAKE_C_COMPILER "${compiler}")`);
  const root = path.resolve(import.meta.dirname, '..');
  const command = ['python3', 'build.py', '--config', 'Release', '--use_webgpu'];
  execFileSync(process.execPath, [path.join(root, 'tools/record-ort-build.mjs'), f.root, root, ...command]);
  const record = JSON.parse(await readFile(path.join(f.root, 'didimdol-build.json'), 'utf8'));
  assert.equal(record.emscriptenVersion, 'emcc fixture compiler 1.2.3');
  assert.deepEqual(record.command, command);
  assert.match(record.ortCommit, /^[a-f0-9]{40}$/);
});
