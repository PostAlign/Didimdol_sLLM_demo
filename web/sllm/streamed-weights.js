/** OPFS-backed FP32 embedding lookup and reusable output-weight GPU buffers. */
export const DEFAULT_SCRATCH_BYTES = 8 * 2**20;
// One output buffer serializes each chunk's OPFS read, upload and projection.
// Two let the next chunk's read and writeBuffer run while the GPU projects the
// current chunk; on the September 14 phone that overlap cost 30-36% per token
// (projection wall 39.9 s against 30.0 s per 64 tokens), so it stays optional
// until the reordered loop in StreamedSession is measured there.
export const DEFAULT_OUTPUT_BUFFERS = 1;
export class StreamedWeights {
  constructor({ ort, device, tracker, store, manifest, descriptor, signal, scratchBytes = DEFAULT_SCRATCH_BYTES,
    outputBuffers = DEFAULT_OUTPUT_BUFFERS }) {
    Object.assign(this, { ort, device, tracker, store, descriptor, signal });
    this.check();
    if (!Number.isInteger(scratchBytes) || scratchBytes < 4 || scratchBytes % 4 || scratchBytes > descriptor.chunkBytes) {
      throw new RangeError('Streamed scratch must be a multiple of four bytes no larger than one chunk');
    }
    if (![1, 2].includes(outputBuffers)) throw new RangeError('Streamed output buffers must be 1 or 2');
    this.files = new Map(manifest.files.map(file => [file.location, file]));
    this.scratch = new Uint8Array(scratchBytes);
    this.buffers = Array.from({ length: outputBuffers }, () => device.createBuffer({ size: descriptor.chunkBytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST }));
    for (const buffer of this.buffers) tracker.observeBuffer(device, buffer, 'streamed-weight');
    this.tensors = this.buffers.map(buffer => ort.Tensor.fromGpuBuffer(buffer, { dataType: 'float32', dims: [descriptor.chunkRows, descriptor.hiddenSize] }));
    // `uploadMs` covers reading, writing and the queue wait of every chunk;
    // `outputReadMs` is the OPFS share of it, `queueWaitMs` the wait share.
    // With two buffers uploads overlap projections, so `uploadMs` and
    // `outputComputeMs` add up to more than `projectionMs`, the wall time of
    // the output loop; the difference is the time the overlap saved.
    this.metrics = { gpuBufferBytes: descriptor.chunkBytes * outputBuffers, gpuBufferCount: outputBuffers,
      cpuScratchBytes: this.scratch.byteLength,
      embeddingReadBytes: 0, outputReadBytes: 0, uploadedBytes: 0, embeddingMs: 0,
      uploadMs: 0, outputReadMs: 0, queueWaitMs: 0, outputComputeMs: 0, projectionMs: 0, projections: 0, chunks: 0,
      readCalls: 0, writeCalls: 0, queueWaits: 0 };
  }
  get buffer() { return this.buffers[0]; }
  get tensor() { return this.tensors[0]; }
  slot(index) { return index % this.buffers.length; }
  check() {
    this.signal?.throwIfAborted();
    if (this.disposed) throw new Error('Streamed weights disposed');
    if (this.tracker.ledger.deviceLost || this.tracker.ledger.lastError) throw new Error('Streamed weights lost their GPU device');
  }
  async embeddings(ids) {
    this.check();
    if (ids.dims[0] !== 1 || ids.dims[1] < 1) throw new Error('Streamed FP32 supports one nonempty prompt at a time');
    const { hiddenSize, chunkRows, vocabSize, chunks } = this.descriptor;
    const values = ids.data;
    const output = new Float32Array(values.length * hiddenSize);
    const start = performance.now();
    for (let i = 0; i < values.length; i++) {
      this.check();
      const id = Number(values[i]);
      if (!Number.isInteger(id) || id < 0 || id >= vocabSize) throw new RangeError(`Invalid token ID: ${id}`);
      const chunk = chunks[Math.floor(id / chunkRows)];
      this.position = { operation: 'embedding-read', tokenIndex: i, tokenId: id, location: chunk.location };
      const bytes = new Uint8Array(output.buffer, i * hiddenSize * 4, hiddenSize * 4);
      await this.store.readRangeInto(this.files.get(chunk.location), chunk.offset + (id % chunkRows) * hiddenSize * 4, bytes.length, bytes);
      this.metrics.embeddingReadBytes += bytes.length;
    }
    this.metrics.embeddingMs += performance.now() - start;
    this.check();
    return new this.ort.Tensor('float32', output, [1, values.length, hiddenSize]);
  }
  /**
   * One output chunk into the buffer of its slot: read scratch-sized pieces
   * from OPFS and queue each write immediately. `writeBuffer` copies the data
   * synchronously, so the scratch can be refilled while the queue works; one
   * queue wait per chunk then bounds the driver backlog. The September 13 phone
   * paid a queue wait per 2 MiB piece, which was 320 waits and 320 reads per
   * generated token.
   *
   * Error scopes bracket each `writeBuffer` call synchronously. An upload may
   * run while the head session's `run()` is pending on the same device, and a
   * scope held across an `await` could cross one the runtime pushes and pops
   * around its own dispatches.
   *
   * `queueWait: false` skips the chunk's queue wait. The overlapped loop uses
   * it: its writes follow the previous chunk's readback copy in the queue, and
   * the next projection's readback is the boundary that bounds the backlog, so
   * a wait here would only stall the thread behind the projection in flight.
   * Resolves with the chunk's timing for the per-chunk timeline.
   */
  async upload(index, { queueWait = true } = {}) {
    this.check();
    const chunk = this.descriptor.chunks[index];
    const buffer = this.buffers[this.slot(index)];
    const start = performance.now();
    let failure, written = 0, readMs = 0, waitMs = 0;
    const scopes = [];
    try {
      for (let offset = 0; offset < chunk.bytes; offset += this.scratch.length) {
        this.check();
        const size = Math.min(this.scratch.length, chunk.bytes - offset), data = this.scratch.subarray(0, size);
        this.position = { operation: 'output-read', chunk: index, slot: this.slot(index), offset, location: chunk.location };
        const readStart = performance.now();
        await this.store.readRangeInto(this.files.get(chunk.location), chunk.offset + offset, size, data);
        readMs += performance.now() - readStart;
        this.metrics.outputReadBytes += size;
        this.metrics.readCalls++;
        this.check();
        this.position.operation = 'output-upload';
        this.device.pushErrorScope('out-of-memory');
        this.device.pushErrorScope('validation');
        try { this.device.queue.writeBuffer(buffer, offset, data); }
        finally { scopes.push(this.device.popErrorScope(), this.device.popErrorScope()); }
        this.metrics.writeCalls++;
        written += size;
      }
      if (queueWait) {
        this.position = { operation: 'output-upload-wait', chunk: index, slot: this.slot(index), offset: written, location: chunk.location };
        const waitStart = performance.now();
        await this.device.queue.onSubmittedWorkDone();
        waitMs = performance.now() - waitStart;
        this.metrics.queueWaits++;
      }
      this.metrics.uploadedBytes += written;
    } catch (error) { failure = error; }
    finally {
      for (const scope of scopes) {
        try {
          const error = await scope;
          if (error) failure ||= new Error(`Streamed upload: ${error.message}`);
        } catch (error) { failure ||= error; }
      }
    }
    this.metrics.outputReadMs += readMs;
    this.metrics.queueWaitMs += waitMs;
    if (failure) throw failure;
    const end = performance.now();
    this.metrics.uploadMs += end - start;
    this.metrics.chunks++;
    return { start, end, readMs, waitMs };
  }
  dispose() {
    if (this.disposed) return;
    this.disposed = true;
    for (const tensor of this.tensors) tensor.dispose();
    for (const buffer of this.buffers) buffer.destroy();
    this.scratch = null;
    this.metrics.gpuBufferBytes = 0; this.metrics.cpuScratchBytes = 0;
  }
}
