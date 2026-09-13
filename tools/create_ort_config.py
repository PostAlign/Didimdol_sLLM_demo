"""Generate a pinned operator list from the deployed graph, without loading weights."""
import argparse
from pathlib import Path
import onnx


def configuration(path):
    model = onnx.load(str(path), load_external_data=False)
    versions = {entry.domain: entry.version for entry in model.opset_import}
    operations = {}

    def visit(graph):
        for node in graph.node:
            operations.setdefault(node.domain, set()).add(node.op_type)
            for attr in node.attribute:
                if attr.type == onnx.AttributeProto.GRAPH:
                    visit(attr.g)
                elif attr.type == onnx.AttributeProto.GRAPHS:
                    for child in attr.graphs:
                        visit(child)
    visit(model.graph)
    lines = ['# Generated from the FP32 deployment graph. Also includes the smoke fixture.',
             '!globally_allowed_types;bool,int32_t,uint32_t,int64_t,uint64_t,float',
             'ai.onnx;17;Add,MatMul']
    for domain, names in sorted(operations.items()):
        lines.append(f'{domain or "ai.onnx"};{versions[domain]};{",".join(sorted(names))}')
    # transformers.js sampling (do_sample with top_k/top_p) lazily creates a
    # 73-byte TopK graph (TensorOpRegistry.top_k, opset 21) on the first sampled
    # token. It runs on the CPU/WASM provider and is not part of the model graph.
    lines.append('# transformers.js TensorOpRegistry.top_k helper used by sampling')
    lines.append('ai.onnx;21;TopK')
    return '\n'.join(lines) + '\n'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('graph', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    args.output.write_text(configuration(args.graph))
