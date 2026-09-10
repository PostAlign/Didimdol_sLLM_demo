"""Create 128/64/32 MiB transport shards without changing the ONNX graph/tensors.

Input directory must contain model.onnx and its external data files. The browser
joins shard Blobs into logical files; it never joins ArrayBuffers. This lets a
40 MiB initializer span two 32 MiB physical transport shards.
"""
import argparse
import json
import shutil
from pathlib import Path
from inspect_initializers import inspect

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("source", type=Path)
parser.add_argument("output", type=Path)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
manifest = inspect(args.source / "model.onnx")
shutil.copyfile(args.source / "model.onnx", args.output / "model.onnx")
(args.output / "initializers.json").write_text(json.dumps(manifest, indent=2))
variants = {}
for mib in (128, 64, 32):
    folder = args.output / str(mib)
    folder.mkdir(exist_ok=True)
    files = []
    for entry in manifest["files"]:
        path = args.source / entry["location"]
        # Reject manifest paths that escape the supplied model directory.
        if not path.resolve().is_relative_to(args.source.resolve()):
            raise ValueError(f"External location escapes source: {path}")
        if path.stat().st_size < entry["minimumBytes"]:
            raise ValueError(f"Truncated weights: {path}")
        shards = []
        with path.open("rb") as source:
            i = 0
            while True:
                part = source.read(mib * 2**20)
                if not part:
                    break
                name = f"{len(files)}-{i}.part"
                (folder / name).write_bytes(part)
                shards.append(dict(url=f"{mib}/{name}", bytes=len(part)))
                i += 1
        files.append(dict(location=entry["location"], bytes=path.stat().st_size, shards=shards))
    variants[str(mib)] = files
(args.output / "experiments.json").write_text(json.dumps(dict(graph="model.onnx", manifest="initializers.json", variants=variants), indent=2))
print(args.output / "experiments.json")
