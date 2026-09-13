/** OPFS-backed FP32 embedding lookup and one reusable output-weight GPU buffer. */
export const DEFAULT_SCRATCH_BYTES = 8 * 2**20;
export class StreamedWeights {
  constructor({ ort, device, tracker, store, manifest, descriptor, signal, scratchBytes = DEFAULT_SCRATCH_BYTES }) {
    Object.assign(this, { ort, device, tracker, store, descriptor, signal });
    this.check();
    if (!Number.isInteger(scratchBytes) || scratchBytes < 4 || scratchBytes % 4 || scratchBytes > descriptor.chunkBytes) {
      throw new RangeError('Streamed scratch must be a multiple of four bytes no larger than one chunk');
    }
    this.files = new Map(manifest.files.map(file => [file.location, file]));
    this.scratch = new Uint8Array(scratchBytes);
    this.buffer = device.createBuffer({ size: descriptor.chunkBytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    tracker.observeBuffer(device, this.buffer, 'streamed-weight');
    this.tensor = ort.Tensor.fromGpuBuffer(this.buffer, { dataType: 'float32', dims: [descriptor.chunkRows, descriptor.hiddenSize] });
    // `uploadMs` covers reading, writing and the queue wait of every chunk;
    // `outputReadMs` is the OPFS share of it, `queueWaitMs` the wait share.
    this.metrics = { gpuBufferBytes: descriptor.chunkBytes, cpuScratchBytes: this.scratch.byteLength,
      embeddingReadBytes: 0, outputReadBytes: 0, uploadedBytes: 0, embeddingMs: 0,
      uploadMs: 0, outputReadMs: 0, queueWaitMs: 0, outputComputeMs: 0, projections: 0, chunks: 0,
      readCalls: 0, writeCalls: 0, queueWaits: 0 };
  }
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
   * One output chunk: read scratch-sized pieces from OPFS and queue each write
   * immediately. `writeBuffer` copies the data synchronously, so the scratch can
   * be refilled while the queue works; one queue wait per chunk then bounds the
   * driver backlog. The September 13 phone paid a queue wait per 2 MiB piece,
   * which was 320 waits and 320 reads per generated token.
   */
  async upload(index) {
    this.check();
    const chunk = this.descriptor.chunks[index];
    const start = performance.now();
    this.device.pushErrorScope('out-of-memory');
    this.device.pushErrorScope('validation');
    let failure, written = 0;
    try {
      for (let offset = 0; offset < chunk.bytes; offset += this.scratch.length) {
        this.check();
        const size = Math.min(this.scratch.length, chunk.bytes - offset), data = this.scratch.subarray(0, size);
        this.position = { operation: 'output-read', chunk: index, offset, location: chunk.location };
        const readStart = performance.now();
        await this.store.readRangeInto(this.files.get(chunk.location), chunk.offset + offset, size, data);
        this.metrics.outputReadMs += performance.now() - readStart;
        this.metrics.outputReadBytes += size;
        this.metrics.readCalls++;
        this.check();
        this.position.operation = 'output-upload';
        this.device.queue.writeBuffer(this.buffer, offset, data);
        this.metrics.writeCalls++;
        written += size;
      }
      this.position = { operation: 'output-upload-wait', chunk: index, offset: written, location: chunk.location };
      const waitStart = performance.now();
      await this.device.queue.onSubmittedWorkDone();
      this.metrics.queueWaitMs += performance.now() - waitStart;
      this.metrics.queueWaits++;
      this.metrics.uploadedBytes += written;
    } catch (error) { failure = error; }
    finally {
      for (let i = 0; i < 2; i++) {
        try {
          const error = await this.device.popErrorScope();
          if (error) failure ||= new Error(`Streamed upload: ${error.message}`);
        } catch (error) { failure ||= error; }
      }
    }
    if (failure) throw failure;
    this.metrics.uploadMs += performance.now() - start;
    this.metrics.chunks++;
  }
  dispose() {
    if (this.disposed) return;
    this.disposed = true;
    this.tensor.dispose(); this.buffer.destroy(); this.scratch = null;
    this.metrics.gpuBufferBytes = 0; this.metrics.cpuScratchBytes = 0;
  }
}
