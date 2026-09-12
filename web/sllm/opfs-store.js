import { checkedRange } from './external-source.js';

const hex = buffer => Array.from(new Uint8Array(buffer), b => b.toString(16).padStart(2, '0')).join('');
const abort = signal => signal?.throwIfAborted();
const safeName = name => {
  if (!/^[a-zA-Z0-9_.-]+$/.test(name) || name === '.' || name === '..') throw new Error(`Invalid weight filename: ${name}`);
  return name;
};

export function readFully(handle, destination, offset) {
  let done = 0;
  while (done < destination.byteLength) {
    const n = handle.read(destination.subarray(done), { at: offset + done });
    if (!Number.isInteger(n) || n <= 0 || n > destination.byteLength - done) throw new Error('Short OPFS read');
    done += n;
  }
}

function writeFully(handle, data, offset) {
  let done = 0;
  while (done < data.byteLength) {
    const n = handle.write(data.subarray(done), { at: offset + done });
    if (!Number.isInteger(n) || n <= 0 || n > data.byteLength - done) throw new Error('Short OPFS write');
    done += n;
  }
}

/** No full-file Blob/ArrayBuffer. Sync handles are only used inside a dedicated worker. */
export class OpfsWeightStore {
  static async open(manifest) {
    if (!globalThis.navigator?.storage?.getDirectory) throw new Error('이 브라우저에서 모델 파일 저장소(OPFS)를 사용할 수 없습니다.');
    const root = await navigator.storage.getDirectory();
    const base = await root.getDirectoryHandle('didimdol-weights-v2', { create: true });
    const folder = await base.getDirectoryHandle(safeName(manifest.graphSha256), { create: true });
    return new OpfsWeightStore(folder, manifest);
  }
  constructor(folder, manifest) {
    this.folder = folder;
    this.manifest = manifest;
    this.active = null;
    this.closed = false;
    this.verificationScratch = null;
    this.metrics = { totalFiles: manifest.files?.length || 0, cacheHits: 0, migratedFiles: 0, downloadedFiles: 0,
      verifiedBytes: 0, openHandles: 0, peakOpenHandles: 0 };
  }
  async openHandle(name) {
    const file = await this.folder.getFileHandle(safeName(name), { create: true });
    if (!file.createSyncAccessHandle) throw new Error('이 브라우저에서 OPFS 범위 읽기를 지원하지 않습니다.');
    return file.createSyncAccessHandle();
  }
  identity(file) {
    return JSON.stringify([this.manifest.graphSha256, file.location, file.bytes, file.blockBytes, file.blockSha256]);
  }
  async complete(file) {
    try {
      const marker = await this.folder.getFileHandle(`${safeName(file.location)}.complete`);
      const text = await (await marker.getFile()).text(); // small integrity metadata only
      if (text !== this.identity(file)) return false;
      const data = await this.folder.getFileHandle(file.location);
      return (await data.getFile()).size === file.bytes;
    } catch { return false; }
  }
  async prepare(file, { openResponse, cache, url, signal, progress = () => {}, checkpoint = async () => {} }) {
    if (this.closed) throw new Error('OPFS store is closed');
    safeName(file.location);
    if (!Number.isSafeInteger(file.minimumBytes) || file.minimumBytes < 0 ||
        !Number.isSafeInteger(file.bytes) || file.bytes <= 0 || file.bytes < file.minimumBytes ||
        !Number.isSafeInteger(file.blockBytes) || file.blockBytes <= 0 || file.blockBytes > 8 * 2**20 ||
        file.blockSha256?.length !== Math.ceil(file.bytes / file.blockBytes) ||
        file.blockSha256.some(hash => !/^[a-f0-9]{64}$/.test(hash))) {
      throw new Error(`블록 검증 정보가 없는 가중치: ${file.location}`);
    }
    abort(signal);
    if (await this.complete(file)) {
      this.metrics.cacheHits++; progress(file.bytes, 'cache');
      await checkpoint({ stage: 'weight-ready', location: file.location, length: file.bytes, source: 'opfs-cache', storage: { ...this.metrics } });
      return;
    }
    // Never accept a partial file left by a terminated worker as a cache hit.
    await this.folder.removeEntry(`${file.location}.complete`).catch(() => {});
    await checkpoint({ stage: 'weight-download', location: file.location, length: file.bytes });
    let response = await cache?.match(url);
    const migrated = !!response;
    response ??= await openResponse();
    if (!response.ok || !response.body) throw new Error(`${file.location}: HTTP ${response.status}`);
    let handle, reader;
    try {
      handle = await this.openHandle(file.location);
      handle.truncate(0);
      reader = response.body.getReader();
      let offset = 0;
      for (;;) {
        abort(signal);
        const { done, value } = await reader.read();
        if (done) break;
        abort(signal);
        if (offset + value.byteLength > file.bytes) throw new Error(`Oversized weights: ${file.location}`);
        writeFully(handle, value, offset);
        offset += value.byteLength;
        progress(offset, migrated ? 'cache' : 'net');
      }
      if (offset !== file.bytes) throw new Error(`Truncated weights: ${file.location} (${offset}/${file.bytes})`);
      handle.flush();
      await checkpoint({ stage: 'weight-verify', location: file.location, length: file.bytes });
      if (!this.verificationScratch || this.verificationScratch.byteLength !== file.blockBytes) {
        this.verificationScratch = new Uint8Array(file.blockBytes);
      }
      const scratch = this.verificationScratch;
      for (let i = 0; i < file.blockSha256.length; i++) {
        abort(signal);
        const offset = i * file.blockBytes;
        const data = scratch.subarray(0, Math.min(file.blockBytes, file.bytes - offset));
        readFully(handle, data, offset);
        if (hex(await crypto.subtle.digest('SHA-256', data)) !== file.blockSha256[i]) {
          // A corrupt legacy cache must not poison the next attempt.
          if (migrated) await cache.delete(url);
          throw new Error(`Weight SHA-256 mismatch: ${file.location}, block ${i}`);
        }
        this.metrics.verifiedBytes += data.byteLength;
      }
    } finally {
      await reader?.cancel().catch(() => {});
      reader?.releaseLock();
      if (!reader) await response.body.cancel().catch(() => {});
      handle?.close();
    }
    abort(signal);
    const marker = await this.openHandle(`${file.location}.complete`);
    try {
      marker.truncate(0);
      writeFully(marker, new TextEncoder().encode(this.identity(file)), 0);
      marker.flush();
    } finally { marker.close(); }
    if (migrated) { this.metrics.migratedFiles++; await cache.delete(url).catch(() => {}); }
    else this.metrics.downloadedFiles++;
    await checkpoint({ stage: 'weight-ready', location: file.location, length: file.bytes,
      source: migrated ? 'legacy-cache' : 'network', storage: { ...this.metrics } });
  }
  descriptor(file) {
    return { ortRangeSource: 2, size: file.bytes,
      readRangeInto: (offset, length, destination) => this.readRangeInto(file, offset, length, destination) };
  }
  finishPreparation() { this.verificationScratch = null; }
  async readRangeInto(file, offset, length, destination) {
    if (this.closed) throw new Error('Weight read after session creation');
    checkedRange(offset, length, file.bytes);
    if (destination.byteLength !== length) throw new RangeError('Destination size mismatch');
    if (this.active?.name !== file.location) {
      this.closeActive();
      const handle = await this.openHandle(file.location);
      this.active = { name: file.location, handle };
      this.metrics.openHandles = 1;
      this.metrics.peakOpenHandles = 1;
      if (handle.getSize() !== file.bytes) throw new Error(`OPFS file size changed: ${file.location}`);
    }
    readFully(this.active.handle, destination, offset);
  }
  closeActive() { this.active?.handle.close(); this.active = null; this.metrics.openHandles = 0; }
  close() { this.closeActive(); this.verificationScratch = null; this.closed = true; }
}
