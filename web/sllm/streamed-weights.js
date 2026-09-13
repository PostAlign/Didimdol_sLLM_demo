/** OPFS-backed FP32 embedding lookup and one reusable output-weight GPU buffer. */
export class StreamedWeights {
  constructor({ ort, device, tracker, store, manifest, descriptor, signal }) {
    Object.assign(this, { ort, device, tracker, store, descriptor, signal });
    this.check();
    this.files = new Map(manifest.files.map(file => [file.location, file]));
    this.scratch = new Uint8Array(2 * 2**20);
    this.buffer = device.createBuffer({ size: descriptor.chunkBytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    tracker.observeBuffer(device, this.buffer, 'streamed-weight');
    this.tensor = ort.Tensor.fromGpuBuffer(this.buffer, { dataType: 'float32', dims: [descriptor.chunkRows, descriptor.hiddenSize] });
    this.metrics = { gpuBufferBytes: descriptor.chunkBytes, cpuScratchBytes: this.scratch.byteLength,
      embeddingReadBytes: 0, outputReadBytes: 0, uploadedBytes: 0, embeddingMs: 0,
      uploadMs: 0, outputComputeMs: 0, projections: 0, chunks: 0 };
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
  async upload(index) {
    this.check();
    const chunk = this.descriptor.chunks[index];
    const start = performance.now();
    this.device.pushErrorScope('out-of-memory');
    this.device.pushErrorScope('validation');
    let failure;
    try {
      for (let offset = 0; offset < chunk.bytes; offset += this.scratch.length) {
        this.check();
        const size = Math.min(this.scratch.length, chunk.bytes - offset), data = this.scratch.subarray(0, size);
        this.position = { operation: 'output-read', chunk: index, offset, location: chunk.location };
        await this.store.readRangeInto(this.files.get(chunk.location), chunk.offset + offset, size, data);
        this.metrics.outputReadBytes += size;
        this.check();
        this.position.operation = 'output-upload';
        this.device.queue.writeBuffer(this.buffer, offset, data);
        await this.device.queue.onSubmittedWorkDone();
        this.metrics.uploadedBytes += size;
      }
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
