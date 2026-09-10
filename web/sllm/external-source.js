// Only range-sized streams are materialized. Offsets larger than JS's exact
// integer range are rejected instead of silently rounding ONNX metadata.
export function exactInteger(value, label = 'offset') {
  const n = Number(value);
  if (!Number.isSafeInteger(n) || n < 0) throw new RangeError(`Invalid ${label}: ${value}`);
  return n;
}

export function checkedRange(offset, length, size) {
  offset = exactInteger(offset);
  length = exactInteger(length, 'length');
  if (!Number.isSafeInteger(offset + length) || offset + length > size) {
    throw new RangeError(`External range ${offset}+${length} exceeds ${size}`);
  }
  return { offset, length };
}

async function copyStream(stream, destination, observe = () => {}) {
  const reader = stream.getReader();
  let position = 0;
  try {
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      observe(value.byteLength);
      try {
        if (position + value.byteLength > destination.byteLength) throw new Error('Oversized external range');
        destination.set(value, position);
        position += value.byteLength;
      } finally { observe(-value.byteLength); }
    }
    if (position !== destination.byteLength) throw new Error(`Short external range: ${position}/${destination.byteLength}`);
  } finally {
    await reader.cancel().catch(() => {});
    reader.releaseLock();
  }
}

/** Blob, File, and Blobs retrieved from IndexedDB or Cache Storage. */
export class BlobTensorSource {
  constructor(files, observe) { this.files = files; this.observe = observe; }
  async readRangeInto(location, offset, length, destination) {
    const file = this.files.get(location);
    if (!(file instanceof Blob)) throw new TypeError(`Missing Blob: ${location}`);
    const range = checkedRange(offset, length, file.size);
    if (destination.byteLength !== range.length) throw new RangeError('Destination size mismatch');
    await copyStream(file.slice(range.offset, range.offset + range.length).stream(), destination, this.observe);
  }
}

/** Strict HTTP Range transport. A server that ignores Range is never buffered. */
export class HttpTensorSource {
  constructor(files, { fetchImpl = fetch, observe } = {}) {
    this.files = files; this.fetch = fetchImpl; this.observe = observe;
  }
  async readRangeInto(location, offset, length, destination) {
    const file = this.files.get(location);
    if (!file) throw new Error(`Missing HTTP source: ${location}`);
    const r = checkedRange(offset, length, file.size);
    if (destination.byteLength !== r.length) throw new RangeError('Destination size mismatch');
    if (!r.length) return;
    const response = await this.fetch(file.url, { headers: { Range: `bytes=${r.offset}-${r.offset + r.length - 1}` } });
    const expected = `bytes ${r.offset}-${r.offset + r.length - 1}/${file.size}`;
    if (response.status !== 206 || response.headers.get('content-range') !== expected ||
        ![null, 'identity'].includes(response.headers.get('content-encoding')) || !response.body) {
      await response.body?.cancel();
      throw new Error(`Server must return exact, uncompressed HTTP 206 range: ${location}`);
    }
    await copyStream(response.body, destination, this.observe);
  }
}
