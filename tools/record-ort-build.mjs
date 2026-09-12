import { readFile, writeFile, readdir } from 'node:fs/promises';
import { execFileSync } from 'node:child_process';
import path from 'node:path';
import { git } from './release.mjs';

const [artifacts, source, ...command] = process.argv.slice(2);
// build.py's Emscripten toolchain is the compiler of record, not a different PATH installation.
let compiler;
for (const entry of await readdir(path.join(artifacts, 'CMakeFiles'))) {
  if (!/^\d+\.\d+\.\d+$/.test(entry)) continue;
  const text = await readFile(path.join(artifacts, 'CMakeFiles', entry, 'CMakeCCompiler.cmake'), 'utf8').catch(() => null);
  if (text) { if (compiler) throw new Error('Ambiguous CMake compiler metadata'); compiler = text; }
}
if (!compiler) throw new Error('Native build compiler metadata missing');
const emcc = compiler.match(/set\(CMAKE_C_COMPILER "([^"]+)"\)/)?.[1];
if (!emcc) throw new Error('Cannot identify the native build compiler');
await writeFile(path.join(artifacts, 'didimdol-build.json'), JSON.stringify({
  ortCommit: git(source, 'rev-parse', 'HEAD'),
  emscriptenVersion: execFileSync(emcc, ['--version'], { encoding: 'utf8' }).trim(),
  command,
}, null, 2));
