import { StreamedWeights } from './streamed-weights.js';

export function bodyManifest(source, descriptor) {
  if (descriptor.format !== 'didimdol-streamed-fp32-v1' || descriptor.sourceGraphSha256 !== source.graphSha256 ||
      descriptor.sourceRevision !== source.revision) throw new Error('Streamed model/source mismatch');
  const names = new Set(descriptor.bodyInitializerNames);
  const initializers = source.initializers.filter(value => names.has(value.name));
  const bytes = initializers.filter(value => value.location).reduce((sum, value) => sum + value.bytes, 0);
  if (initializers.length !== names.size || bytes !== descriptor.bodyWeightBytes ||
      initializers.some(value => value.name.startsWith('embed_tokens.chunk'))) throw new Error('Invalid streamed body manifest');
  if (descriptor.chunks.length !== 16 || descriptor.chunkBytes !== 40 * 2**20 || descriptor.chunkRows !== 16384 ||
      descriptor.hiddenSize !== 640 || descriptor.vocabSize !== 262144) throw new Error('Invalid streamed embedding layout');
  for (const [index, chunk] of descriptor.chunks.entries()) {
    const original = source.initializers.find(value => value.name === `embed_tokens.chunk${index}`);
    if (!original || ['name', 'location', 'offset', 'bytes', 'dtype'].some(key => chunk[key] !== original[key]) ||
        original.bytes !== descriptor.chunkBytes || original.dtype !== 'FLOAT') throw new Error('Streamed embedding/source mismatch');
  }
  return { ...source, graphSha256: descriptor.graphs.body.sha256, initializers,
    totalExternalTensorBytes: bytes, largestInitializerBytes: Math.max(...initializers.map(value => value.bytes)) };
}

export async function verifiedStreamedGraph(descriptor, name, baseURL) {
  const graph = descriptor.graphs[name];
  if (graph?.file !== `${name}.onnx`) throw new Error('Invalid streamed graph path');
  const response = await fetch(new URL(graph.file, baseURL));
  if (!response.ok) throw new Error(`Streamed graph HTTP ${response.status}`);
  const bytes = new Uint8Array(await response.arrayBuffer());
  const hash = Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256', bytes)), b => b.toString(16).padStart(2, '0')).join('');
  if (bytes.length !== graph.bytes || hash !== graph.sha256) throw new Error(`Streamed ${name} graph hash mismatch`);
  return bytes;
}

/** ORT session facade: preserves Transformers' generation/cache/sampling machinery. */
export class StreamedSession {
  constructor({ ort, body, head, descriptor, store, manifest, tracker, signal, checkpoint = async () => {} }) {
    Object.assign(this, { ort, body, head, descriptor, checkpoint });
    this.weights = new StreamedWeights({ ort, device: tracker.device, tracker, store, manifest, descriptor, signal });
    this.inputNames = [...new Set(['input_ids', ...body.inputNames.filter(name => name !== 'streamed_embeddings')])];
    this.inputMetadata = this.inputNames.map(name => body.inputMetadata.find(value => value.name === name) ||
      { name, type: 'int64', shape: ['batch_size', 'sequence_length'], isTensor: true });
    this.outputNames = ['logits', ...body.outputNames.filter(name => name !== descriptor.hiddenOutput)];
    this.outputMetadata = this.outputNames.map(name => body.outputMetadata.find(value => value.name === name) ||
      { name, type: 'float32', shape: [1, 1, descriptor.vocabSize], isTensor: true });
  }
  async run(feeds) {
    if (this.busy || this.released) throw new Error('Streamed session is busy or released');
    this.busy = true;
    let embeddings, outputs, returned = false;
    const start = performance.now();
    try {
      embeddings = await this.weights.embeddings(feeds.input_ids);
      this.weights.check();
      const bodyFeeds = Object.fromEntries(this.body.inputNames.map(name => [name, name === 'streamed_embeddings' ? embeddings : feeds[name]]));
      outputs = await this.body.run(bodyFeeds);
      this.weights.check();
      const hidden = outputs[this.descriptor.hiddenOutput];
      const logits = new Float32Array(this.descriptor.vocabSize);
      for (let i = 0; i < this.descriptor.chunks.length; i++) {
        await this.weights.upload(i);
        this.weights.check();
        this.weights.position = { operation: 'output-compute', chunk: i };
        const computeStart = performance.now();
        const result = await this.head.run({ hidden, weight: this.weights.tensor });
        try {
          const values = await result.chunk_logits.getData();
          logits.set(values, i * this.descriptor.chunkRows);
          // No overwrite of the shared weight buffer before this projection finishes.
          await this.weights.device.queue.onSubmittedWorkDone();
        } finally { result.chunk_logits.dispose(); }
        this.weights.metrics.outputComputeMs += performance.now() - computeStart;
      }
      this.weights.check();
      this.weights.metrics.projections++;
      await this.checkpoint({ stage: 'streamed-step-complete', durationMs: performance.now() - start,
        streaming: { ...this.weights.metrics } });
      hidden.dispose(); delete outputs[this.descriptor.hiddenOutput];
      outputs.logits = new this.ort.Tensor('float32', logits, [1, 1, this.descriptor.vocabSize]);
      returned = true;
      return outputs;
    } catch (error) {
      await this.checkpoint({ stage: 'streamed-error', message: String(error), position: this.weights.position,
        streaming: { ...this.weights.metrics } });
      throw error;
    } finally {
      embeddings?.dispose();
      if (!returned && outputs) for (const tensor of Object.values(outputs)) tensor.dispose();
      this.busy = false;
    }
  }
  async release() {
    if (this.released) return;
    this.released = true;
    try { await this.body.release(); }
    finally { try { await this.head.release(); } finally { this.weights.dispose(); } }
  }
}
