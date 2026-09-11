import sys
import tempfile
import unittest
import hashlib
from pathlib import Path
import numpy as np
import onnx
from onnx import helper, numpy_helper

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools'))
from build_web_models import save_external
from inspect_initializers import inspect


class ModelToolsTest(unittest.TestCase):
    def test_external_length_and_roundtrip(self):
        with tempfile.TemporaryDirectory() as folder:
            weights = np.arange(300, dtype=np.float32)
            tensor = numpy_helper.from_array(weights, 'W')
            model = helper.make_model(helper.make_graph([], 'test', [], [], [tensor]))
            path = Path(folder) / 'model.onnx'
            save_external(model, path, 2048)
            metadata = inspect(path)
            self.assertEqual(metadata['initializers'][0]['declaredLength'], weights.nbytes)
            self.assertEqual(metadata['largestInitializerBytes'], weights.nbytes)
            files_bytes = (path.parent / metadata['files'][0]['location']).read_bytes()
            self.assertEqual(metadata['files'][0]['blockSha256'], [hashlib.sha256(files_bytes).hexdigest()])
            self.assertEqual(metadata['files'][0]['bytes'], len(files_bytes))
            np.testing.assert_array_equal(numpy_helper.to_array(onnx.load(path).graph.initializer[0]), weights)

    def test_file_cap_accounts_for_alignment(self):
        with tempfile.TemporaryDirectory() as folder:
            tensors = [numpy_helper.from_array(np.ones(257, dtype=np.float32), name) for name in ['A', 'B']]
            model = helper.make_model(helper.make_graph([], 'test', [], [], tensors))
            files = save_external(model, Path(folder) / 'model.onnx', 2060)
            self.assertEqual(len(files), 2)
            self.assertTrue(all(file.stat().st_size <= 2060 for file in files))


if __name__ == '__main__':
    unittest.main()
