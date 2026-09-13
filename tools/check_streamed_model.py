"""Compare real FP32 resident/streamed logits, KV outputs and greedy decoding on CPU."""
import json
import os
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from build_web_models import _prompt_ids


def check(root=Path('.'), steps=4):
    source = (root / '.work/full-model').resolve()
    generated = root / 'model/streamed'
    manifest = json.loads((generated / 'manifest.json').read_text())
    work = root / '.work/streamed-check'
    work.mkdir(parents=True, exist_ok=True)
    for path in source.glob('model.onnx_data*'):
        target = work / path.name
        if target.is_symlink():
            target.unlink()
        if not target.exists():
            os.link(path, target)
    body_path = work / 'body.onnx'
    body_path.write_bytes((generated / 'body.onnx').read_bytes())
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.intra_op_num_threads = 4
    resident = ort.InferenceSession(str(source / 'model.onnx'), options, providers=['CPUExecutionProvider'])
    body = ort.InferenceSession(str(body_path), options, providers=['CPUExecutionProvider'])
    head = ort.InferenceSession(str(generated / 'head.onnx'), options, providers=['CPUExecutionProvider'])
    tokenizer = Tokenizer.from_file(str(root / 'tokenizer/tokenizer.json'))
    rows = [json.loads(line) for line in (root / 'web/sllm/data.jsonl').read_text().splitlines() if line]
    prompts = sorted((_prompt_ids(tokenizer, row) for row in rows), key=len)
    results = []
    for prompt in [prompts[0], prompts[-1]]:
        old_past = {f'past_key_values.{i}.{kind}': np.zeros((1, 1, 0, 256), np.float32)
                    for i in range(18) for kind in ['key', 'value']}
        new_past = {k: v.copy() for k, v in old_past.items()}
        ids = prompt
        total = len(ids)
        tokens = []
        max_logit_error = max_cache_error = 0.0
        for step in range(steps):
            inputs = {'input_ids': np.array([ids], np.int64), 'attention_mask': np.ones((1, total), np.int64)}
            reference = dict(zip([x.name for x in resident.get_outputs()], resident.run(None, {**inputs, **old_past})))
            embeddings = np.empty((1, len(ids), 640), np.float32)
            for i, token in enumerate(ids):
                chunk = manifest['chunks'][token // 16384]
                with (source / chunk['location']).open('rb') as file:
                    file.seek(chunk['offset'] + token % 16384 * 2560)
                    embeddings[0, i] = np.frombuffer(file.read(2560), np.float32)
            all_feeds = {**inputs, **new_past, 'streamed_embeddings': embeddings}
            actual = dict(zip([x.name for x in body.get_outputs()], body.run(None, {x.name: all_feeds[x.name] for x in body.get_inputs()})))
            hidden = actual.pop(manifest['hiddenOutput'])
            logits = np.empty((1, 1, 262144), np.float32)
            for i, chunk in enumerate(manifest['chunks']):
                with (source / chunk['location']).open('rb') as file:
                    file.seek(chunk['offset'])
                    weight = np.frombuffer(file.read(chunk['bytes']), np.float32).reshape(16384, 640)
                logits[0, 0, i * 16384:(i + 1) * 16384] = head.run(None, {'hidden': hidden, 'weight': weight})[0][0]
                del weight
            assert np.isfinite(logits).all(), 'Non-finite logits'
            error = float(np.max(np.abs(logits - reference['logits'])))
            max_logit_error = max(max_logit_error, error)
            assert error <= 1e-3, error
            token = int(logits.argmax())
            assert token == int(reference['logits'].argmax()), 'Greedy token mismatch'
            for key in actual:
                error = float(np.max(np.abs(actual[key] - reference[key])))
                max_cache_error = max(max_cache_error, error)
                assert error <= 1e-3, (key, error)
            old_past = {key.replace('present.', 'past_key_values.'): value for key, value in reference.items() if key != 'logits'}
            new_past = {key.replace('present.', 'past_key_values.'): value for key, value in actual.items()}
            tokens.append(token)
            ids = [token]
            total += 1
        result = {'promptTokens': len(prompt), 'tokens': tokens, 'maxLogitError': max_logit_error, 'maxCacheError': max_cache_error}
        results.append(result)
        print(json.dumps(result), flush=True)
    (root / '.work/streamed-cpu-results.json').write_text(json.dumps(results, indent=2) + '\n')


if __name__ == '__main__':
    check(steps=int(os.environ.get('STREAMED_CHECK_STEPS', '4')))
