import test from 'node:test';
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { OpfsWeightStore } from '../web/sllm/opfs-store.js';

function folder() {
  const entries = new Map();
  let live = 0, peak = 0;
  return { entries, get live() { return live; }, get peak() { return peak; },
    async removeEntry(name) { entries.delete(name); },
    async getFileHandle(name, options) {
      if (!entries.has(name)) {
        if (!options?.create) throw new Error('Not found');
        entries.set(name, new Uint8Array());
      }
      return {
        async getFile() { return { size: entries.get(name).length, text: async () => new TextDecoder().decode(entries.get(name)) }; },
        async createSyncAccessHandle() {
          live++; peak = Math.max(peak, live);
          let closed = false;
          return {
            getSize: () => entries.get(name).length,
            truncate(n) { const bytes = new Uint8Array(n); bytes.set(entries.get(name).subarray(0, n)); entries.set(name, bytes); },
            read(dst, { at }) { const part = entries.get(name).subarray(at, at + Math.min(3, dst.length)); dst.set(part); return part.length; },
            write(src, { at }) {
              const n = Math.min(3, src.length);
              const bytes = new Uint8Array(Math.max(entries.get(name).length, at + n));
              bytes.set(entries.get(name)); bytes.set(src.subarray(0, n), at); entries.set(name, bytes); return n;
            },
            flush() {}, close() { assert.equal(closed, false); closed = true; live--; },
          };
        },
      };
    },
  };
}
function fixture() {
  const bytes = Uint8Array.from({ length: 35 }, (_, i) => i);
  const hashes = [];
  for (let i = 0; i < bytes.length; i += 8) hashes.push(createHash('sha256').update(bytes.subarray(i, i + 8)).digest('hex'));
  const file = { location: 'weights', bytes: bytes.length, minimumBytes: bytes.length, blockBytes: 8, blockSha256: hashes };
  const directory = folder();
  const store = new OpfsWeightStore(directory, { graphSha256: 'test-graph' });
  return { bytes, file, directory, store };
}

test('OPFS verifies blocks, handles short I/O and reuses completed files without fetching', async () => {
  const f = fixture();
  await f.store.prepare(f.file, { openResponse: async () => new Response(f.bytes) });
  assert.equal(await f.store.complete(f.file), true);
  await f.store.prepare(f.file, { openResponse: () => { throw new Error('must not fetch'); } });
  const dst = new Uint8Array(17);
  await f.store.descriptor(f.file).readRangeInto(7, 17, dst);
  assert.deepEqual(dst, f.bytes.subarray(7, 24));
  assert.equal(f.store.metrics.cacheHits, 1);
  f.store.close();
  assert.equal(f.directory.live, 0);
  await assert.rejects(f.store.readRangeInto(f.file, 0, 1, new Uint8Array(1)), /after session/);
});

test('truncation and corrupt cached bytes never acquire a completion marker; retry repairs the file', async () => {
  const f = fixture();
  await assert.rejects(f.store.prepare(f.file, { openResponse: async () => new Response(f.bytes.subarray(0, 10)) }), /Truncated/);
  assert.equal(await f.store.complete(f.file), false);
  let removed = false;
  await assert.rejects(f.store.prepare(f.file, { url: 'cached',
    cache: { match: async () => new Response(new Uint8Array(35)), delete: async () => { removed = true; } },
    openResponse: () => { throw new Error('must not fetch'); } }), /SHA-256 mismatch/);
  assert.equal(removed, true);
  assert.equal(await f.store.complete(f.file), false);
  assert.equal(f.directory.live, 0);
  await f.store.prepare(f.file, { openResponse: async () => new Response(f.bytes) });
  assert.equal(await f.store.complete(f.file), true);
});

test('source switches retain only one read handle, and cancellation leaves files incomplete', async () => {
  const f = fixture();
  const second = { ...f.file, location: 'weights2' };
  for (const file of [f.file, second]) await f.store.prepare(file, { openResponse: async () => new Response(f.bytes) });
  for (const file of [f.file, second, f.file]) await f.store.descriptor(file).readRangeInto(0, 4, new Uint8Array(4));
  assert.equal(f.directory.live, 1);
  assert.equal(f.directory.peak, 1);
  f.store.close();
  const g = fixture(); const controller = new AbortController();
  await assert.rejects(g.store.prepare(g.file, { signal: controller.signal,
    openResponse: async () => new Response(g.bytes), progress() { controller.abort(); } }), /abort/i);
  assert.equal(await g.store.complete(g.file), false);
  assert.equal(g.directory.live, 0);
});
