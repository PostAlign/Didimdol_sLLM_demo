"""Inspect tensor metadata and hash external files in bounded 8 MiB blocks."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import onnx


def inspect(path, revision=None):
    model = onnx.load(str(path), load_external_data=False)
    consumers = {}
    for node in model.graph.node:
        for name in node.input:
            consumers.setdefault(name, []).append(node)
    rows = []
    for index, tensor in enumerate(model.graph.initializer):
        dtype = onnx.TensorProto.DataType.Name(tensor.data_type)
        size = math.prod(tensor.dims) * onnx.helper.tensor_dtype_to_np_dtype(tensor.data_type).itemsize
        external = {entry.key: entry.value for entry in tensor.external_data}
        offset = int(external.get("offset", 0))
        declared = int(external.get("length", 0))
        if declared not in (0, size):
            raise ValueError(f"{tensor.name}: external length {declared} != {size}")
        names = " ".join([tensor.name] + [node.name for node in consumers.get(tensor.name, [])]).lower()
        category = "other"
        for match, label in [("embed", "token_embedding/tied_lm_head"), ("q_proj", "attention_q"),
                             ("k_proj", "attention_k"), ("v_proj", "attention_v"), ("o_proj", "attention_o"),
                             ("gate_proj", "mlp_gate"), ("up_proj", "mlp_up"), ("down_proj", "mlp_down"),
                             ("norm", "rmsnorm"), ("lm_head", "lm_head")]:
            if match in names:
                category = label
                break
        if dtype not in ("FLOAT", "DOUBLE", "FLOAT16", "BFLOAT16"):
            category = "other_constants"
        rows.append(dict(index=index, name=tensor.name, shape=list(tensor.dims), dtype=dtype, bytes=size,
                         category=category, location=external.get("location"), offset=offset,
                         length=size if external else None, declaredLength=declared if external else None,
                         consumers=[node.name for node in consumers.get(tensor.name, [])]))
    external_rows = [row for row in rows if row["location"]]
    files = {}
    for row in external_rows:
        files[row["location"]] = max(files.get(row["location"], 0), row["offset"] + row["bytes"])
    file_rows = []
    for name, minimum in files.items():
        entry = dict(location=name, minimumBytes=minimum)
        source = path.parent / name
        if Path(name).is_absolute() or '..' in Path(name).parts:
            raise ValueError(f'External file escapes model directory: {name}')
        if source.is_file():
            block_size = 8 * 2**20
            hashes = []
            with source.open('rb') as stream:
                while block := stream.read(block_size):
                    hashes.append(hashlib.sha256(block).hexdigest())
            entry.update(bytes=source.stat().st_size, blockBytes=block_size, blockSha256=hashes)
        file_rows.append(entry)
    return dict(schemaVersion=2, revision=revision, graphSha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                graphBytes=path.stat().st_size, totalInitializerBytes=sum(row["bytes"] for row in rows),
                totalExternalTensorBytes=sum(row["bytes"] for row in external_rows),
                largestInitializerBytes=max((row["bytes"] for row in rows), default=0),
                expectedGpuResidentBytes=sum((row["bytes"] + 15) // 16 * 16 for row in external_rows),
                # Actual CPU placement, driver memory and allocator overhead require runtime instrumentation.
                expectedCpuStagingBytes={str(n): {"scratch": n * 2**20, "streamChunkUpperBound": n * 2**20}
                                         for n in (8, 16, 32, 64)},
                files=file_rows, initializers=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("graph", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--revision")
    args = parser.parse_args()
    report = inspect(args.graph, args.revision)
    data = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(data)
        print(json.dumps({k: v for k, v in report.items() if k not in ("initializers", "files")}, indent=2))
    else:
        print(data)


if __name__ == "__main__":
    main()
