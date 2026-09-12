const HASH = /^[a-f0-9]{64}$/;
export const releaseDescriptor = manifest => ({ releaseSchema: manifest.releaseSchema,
  ortVersion: manifest.ortVersion, transformersVersion: manifest.transformersVersion,
  rangeLoaderVersion: manifest.rangeLoaderVersion, modes: manifest.modes, builds: manifest.builds,
  provenance: manifest.provenance, assets: manifest.assets });
export function validateRelease(build) {
  if (build?.releaseSchema !== 1 || !HASH.test(build.releaseId) || !build.assets || !build.modes?.length) {
    throw new Error('Runtime release metadata missing. Rebuild the site runtime.');
  }
  for (const [name, file] of Object.entries(build.assets)) {
    if (!/^[\w./-]+$/.test(name) || name.startsWith('/') || name.split('/').some(part => !part || part === '.' || part === '..') ||
        !HASH.test(file.sha256) || !Number.isSafeInteger(file.bytes) || file.bytes < 0) throw new Error(`Invalid release asset: ${name}`);
  }
  for (const name of ['web/app-main.js', 'web/sllm/worker.js', 'web/vendor/transformers.mjs', 'web/vendor/selected-ort.mjs']) {
    if (!build.assets[name]) throw new Error(`Release asset missing: ${name}`);
  }
  for (const mode of build.modes) {
    if (!['asyncify', 'jspi'].includes(mode) || build.rangeLoaderVersion !== 2) throw new Error('Unsupported runtime release');
    const stem = `web/vendor/ort-wasm-simd-threaded.${mode}`;
    if (!build.assets[`web/vendor/ort.${mode}.mjs`] || !build.assets[`${stem}.mjs`] ||
        build.assets[`${stem}.wasm`]?.sha256 !== build.builds?.[mode]?.wasmSha256 ||
        build.assets[`${stem}.wasm`]?.bytes !== build.builds?.[mode]?.wasmBytes) throw new Error(`Runtime asset mismatch: ${mode}`);
  }
  return build;
}

/** Both the landing page and an immutable snapshot resolve this same release. */
export async function readRelease(metadataURL = new URL('../vendor/build.json', import.meta.url)) {
  const response = await fetch(metadataURL, { cache: 'no-store' });
  if (!response.ok) throw new Error(`Runtime build metadata HTTP ${response.status}`);
  const build = validateRelease(await response.json());
  const fingerprint = Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',
    new TextEncoder().encode(JSON.stringify(releaseDescriptor(build))))), byte => byte.toString(16).padStart(2, '0')).join('');
  if (fingerprint !== build.releaseId) throw new Error('Runtime release identity mismatch');
  const base = new URL(build.assetBase, metadataURL);
  if (!base.pathname.endsWith(`/releases/${build.releaseId}/`) || base.origin !== new URL(metadataURL).origin) {
    throw new Error('Runtime release URL mismatch');
  }
  return { build, asset(name) {
    if (!build.assets[name]) throw new Error(`Asset absent from release: ${name}`);
    return new URL(name, base).href;
  } };
}
