"""Split the pinned, already exported FP32 graph without rewriting weight bytes."""
import argparse
import copy
import hashlib
import json
from pathlib import Path

import onnx
from onnx import TensorProto, helper


def sha(data):
    return hashlib.sha256(data).hexdigest()


def prepare(source, manifest_path, output):
    raw = source.read_bytes()
    manifest = json.loads(manifest_path.read_text())
    if sha(raw) != manifest['graphSha256']:
        raise ValueError('Source graph/manifest mismatch')
    model = onnx.load_model_from_string(raw)
    chunks = [x for x in manifest['initializers'] if x['name'].startswith('embed_tokens.chunk')]
    chunks.sort(key=lambda x: int(x['name'].split('chunk')[1]))
    if len(chunks) != 16 or any(x['dtype'] != 'FLOAT' or x['shape'] != [16384, 640] for x in chunks):
        raise ValueError('Expected the pinned 16 x [16384, 640] FP32 embedding')
    nodes = {n.name: n for n in model.graph.node}
    embed_output = nodes['/model/embed_tokens/split/Where15'].output[0]
    flat = nodes['/lm_head/split/Reshape'].output[0]
    # Fail on an unfamiliar head rather than silently discarding postprocessing.
    if list(nodes['/lm_head/split/Reshape_out'].output) != ['logits']:
        raise ValueError('Unsupported logits graph')
    for i, chunk in enumerate(chunks):
        n = nodes[f'/lm_head/split/Gemm{i}']
        if list(n.input) != [flat, chunk['name']] or {a.name: helper.get_attribute_value(a) for a in n.attribute} != {'transB': 1}:
            raise ValueError('Unsupported output projection')

    body = copy.deepcopy(model)
    kept = [n for n in body.graph.node if not n.name.startswith('/model/embed_tokens/split/')
            and (not n.name.startswith('/lm_head/split/') or n.name in ['/lm_head/split/Slice', '/lm_head/split/Reshape'])]
    for node in kept:
        for i, name in enumerate(node.input):
            if name == embed_output:
                node.input[i] = 'streamed_embeddings'
    # Keep original input_ids if used for shape/position calculations.
    used = {name for n in kept for name in n.input}
    inputs = [x for x in body.graph.input if x.name in used]
    inputs.append(helper.make_tensor_value_info('streamed_embeddings', TensorProto.FLOAT, ['batch_size', 'sequence_length', 640]))
    outputs = [x for x in body.graph.output if x.name != 'logits']
    outputs.append(helper.make_tensor_value_info(flat, TensorProto.FLOAT, ['batch_size', 640]))
    initializers = [x for x in body.graph.initializer if x.name in used]
    body.graph.ClearField('node'); body.graph.node.extend(kept)
    body.graph.ClearField('input'); body.graph.input.extend(inputs)
    body.graph.ClearField('output'); body.graph.output.extend(outputs)
    body.graph.ClearField('initializer'); body.graph.initializer.extend(initializers)
    body.graph.ClearField('value_info')
    if any(x.name.startswith('embed_tokens.chunk') for x in initializers):
        raise ValueError('Embedding was retained in the body')
    head = helper.make_model(helper.make_graph([
        helper.make_node('Gemm', ['hidden', 'weight'], ['chunk_logits'], transB=1)
    ], 'streamed-output', [helper.make_tensor_value_info('hidden', TensorProto.FLOAT, [1, 640]),
        helper.make_tensor_value_info('weight', TensorProto.FLOAT, [16384, 640])],
        [helper.make_tensor_value_info('chunk_logits', TensorProto.FLOAT, [1, 16384])]),
        opset_imports=list(model.opset_import), ir_version=model.ir_version)
    onnx.checker.check_model(head)
    output.mkdir(parents=True, exist_ok=True)
    graphs = {}
    for name, graph in [('body', body), ('head', head)]:
        data = graph.SerializeToString()
        (output / f'{name}.onnx').write_bytes(data)
        graphs[name] = {'file': f'{name}.onnx', 'bytes': len(data), 'sha256': sha(data)}
    names = {x.name for x in initializers}
    body_initializers = [x for x in manifest['initializers'] if x['name'] in names]
    descriptor = {'format': 'didimdol-streamed-fp32-v1', 'sourceGraphSha256': manifest['graphSha256'],
        'sourceRevision': manifest['revision'], 'graphs': graphs, 'hiddenOutput': flat,
        'vocabSize': 262144, 'hiddenSize': 640, 'chunkRows': 16384, 'chunkBytes': 41943040,
        'embeddingBytes': sum(x['bytes'] for x in chunks), 'chunks': chunks,
        'bodyInitializerNames': [x['name'] for x in body_initializers],
        'bodyWeightBytes': sum(x['bytes'] for x in body_initializers if x['location'])}
    (output / 'manifest.json').write_text(json.dumps(descriptor, indent=2) + '\n')
    return descriptor


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('.work/full-model/model.onnx'))
    parser.add_argument('--manifest', type=Path, default=Path('model/initializers.json'))
    parser.add_argument('--output', type=Path, default=Path('model/streamed'))
    args = parser.parse_args()
    result = prepare(args.source, args.manifest, args.output)
    print(json.dumps({k: result[k] for k in ['format', 'graphs', 'bodyWeightBytes', 'embeddingBytes']}))
