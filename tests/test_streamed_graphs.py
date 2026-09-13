import hashlib
import json
from pathlib import Path
import unittest

import onnx
from onnx import helper

ROOT = Path(__file__).resolve().parent.parent


class StreamedGraphsTest(unittest.TestCase):
    def test_graph_contract_and_weight_budget(self):
        folder = ROOT / 'model/streamed'
        manifest = json.loads((folder / 'manifest.json').read_text())
        source = json.loads((ROOT / 'model/initializers.json').read_text())
        self.assertEqual(manifest['sourceGraphSha256'], source['graphSha256'])
        graphs = {}
        for name in ['body', 'head']:
            data = (folder / f'{name}.onnx').read_bytes()
            self.assertEqual(hashlib.sha256(data).hexdigest(), manifest['graphs'][name]['sha256'])
            graphs[name] = onnx.load_model_from_string(data).graph
        body, head = graphs['body'], graphs['head']
        self.assertEqual({x.name for x in body.initializer}, set(manifest['bodyInitializerNames']))
        self.assertFalse(any(x.name.startswith('embed_tokens.chunk') for x in body.initializer))
        self.assertIn('streamed_embeddings', {x.name for x in body.input})
        self.assertIn(manifest['hiddenOutput'], {x.name for x in body.output})
        self.assertEqual(len([x for x in body.output if x.name.startswith('present.')]), 36)
        self.assertEqual(len(head.initializer), 0)
        self.assertEqual([x.name for x in head.input], ['hidden', 'weight'])
        self.assertEqual([x.op_type for x in head.node], ['Gemm'])
        self.assertEqual({a.name: helper.get_attribute_value(a) for a in head.node[0].attribute}, {'transB': 1})
        self.assertEqual(manifest['bodyWeightBytes'] + manifest['chunkBytes'], 443247104)
        # The reduced native runtime must include every generated graph operator.
        supported = set()
        for line in (ROOT / 'model/required-operators.config').read_text().splitlines():
            if line.startswith('ai.onnx;'):
                supported.update(line.split(';')[2].split(','))
        self.assertTrue({x.op_type for graph in graphs.values() for x in graph.node} <= supported)


if __name__ == '__main__':
    unittest.main()
