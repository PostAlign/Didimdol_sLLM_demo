"""Generate a multi-chunk FP32 external-data model for browser integration tests."""
from pathlib import Path
import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto
from inspect_initializers import inspect
import json

out = Path(".work/test-model")
out.mkdir(parents=True, exist_ok=True)
weights = numpy_helper.from_array(np.ones((1024, 2560), dtype=np.float32), "W")
bias = numpy_helper.from_array(np.full((2560,), 0.5, dtype=np.float32), "bias")
graph = helper.make_graph([
    helper.make_node("MatMul", ["X", "W"], ["M"], name="projection"),
    helper.make_node("Add", ["M", "bias"], ["Y"], name="bias_add"),
], "range-loader-test", [helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 1024])],
    [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 2560])], [weights, bias])
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)], ir_version=10)
with (out / "weights.bin").open("wb") as stream:
    stream.write(b"external-test")  # unaligned external file offset (13 bytes)
    for tensor in model.graph.initializer:
        raw = tensor.raw_data
        onnx.external_data_helper.set_external_data(tensor, location="weights.bin", offset=stream.tell(), length=len(raw))
        tensor.ClearField("raw_data")
        stream.write(raw)
onnx.save(model, out / "model.onnx")
(out / "initializers.json").write_text(json.dumps(inspect(out / "model.onnx"), indent=2))
