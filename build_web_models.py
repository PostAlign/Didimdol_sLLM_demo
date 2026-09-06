"""model/model.onnx (fp32 단일 파일) → 브라우저 배포용 model/web/ 산출물.

  model/web/model.onnx        + model.onnx_data        fp32 (그래프 + 외부 가중치, 1.07 GB)
  model/web/model_fp16.onnx   + model_fp16.onnx_data   fp16 (그래프 + 외부 가중치, 0.54 GB)

왜 외부 데이터 포맷인가
  ORT 웹은 단일 파일 모델을 받으면 JS 버퍼(1 GB)를 wasm 힙에 통째로 복사한 뒤 파싱한다.
  그 순간 같은 바이트가 두 벌이라 iPhone 의 WebContent 한계(약 2 GB, jetsam ActiveHard)를 넘어
  오류 없이 죽는다. 외부 데이터 파일은 텐서 단위로 잘라 WebGPU 버퍼에 바로 올리므로
  (transformers.js 4.2.0 이 묶은 ORT 의 로더, case 1 = GPU 업로드) wasm 힙 사본이 없다.
  그래프 파일(model.onnx)은 초기값이 빠져 수 MB 라 복사돼도 무방하다.

왜 fp16 도 만드나
  fp32 가 그래도 안 되는 기기를 위한 대체 경로다. 파일이 절반이라 로드 피크도 절반이다.
  Gemma 3 는 fp16 에서 잔차 스트림이 65504 를 넘을 수 있어 정확도는 fp32 로 재야 한다.
  --check 로 두 모델이 같은 프롬프트에서 같은 텍스트를 내는지, 로짓에 inf/NaN 이 없는지 본다.

  python build_web_models.py            # 산출물 생성
  python build_web_models.py --check    # 생성 후 fp32·fp16 탐욕 디코딩 비교 (CPU, 1~2분)

업로드는 upload_to_hf.py 가 model/web/ 을 레포 루트에 평평하게 올린다.
"""

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
import onnx

ROOT = Path(__file__).parent
SRC = ROOT / "model" / "model.onnx"
OUT = ROOT / "model" / "web"
EOS = {1, 106}   # <eos>, <end_of_turn>


def save_external(model: onnx.ModelProto, path: Path, location: str) -> None:
    onnx.save_model(
        model, str(path),
        save_as_external_data=True, all_tensors_to_one_file=True,
        location=location, size_threshold=1024, convert_attribute=False,
    )
    data = path.parent / location
    print(f"저장: {path.name} {path.stat().st_size / 1e6:.1f} MB · {location} {data.stat().st_size / 1e6:.1f} MB",
          flush=True)


def build_fp32() -> None:
    t0 = time.time()
    model = onnx.load(str(SRC))
    save_external(model, OUT / "model.onnx", "model.onnx_data")
    print(f"fp32 {time.time() - t0:.0f}s", flush=True)


def build_fp16() -> None:
    from onnxruntime.transformers.float16 import convert_float_to_float16

    t0 = time.time()
    model = onnx.load(str(SRC))
    # keep_io_types: input_ids/attention_mask 는 int64 그대로, past/present KV 와 logits 는 fp32 로 남긴다.
    # transformers.js 가 present → past 로 되돌려 넣을 때 타입이 맞아야 하고, 로짓 샘플링도 fp32 로 한다.
    model = convert_float_to_float16(model, keep_io_types=True, disable_shape_infer=True)
    save_external(model, OUT / "model_fp16.onnx", "model_fp16.onnx_data")
    print(f"fp16 {time.time() - t0:.0f}s", flush=True)


# ── 검증 ────────────────────────────────────────────────────────────────────
def greedy(sess, ids, max_new=48):
    import onnxruntime as ort  # noqa: F401
    names = [o.name for o in sess.get_outputs()]
    past = {f"past_key_values.{i}.{kv}": np.zeros((1, 1, 0, 256), np.float32)
            for i in range(18) for kv in ("key", "value")}
    cur = np.array([ids], np.int64)
    total = len(ids)
    out, bad = [], 0
    for _ in range(max_new):
        res = sess.run(None, {"input_ids": cur, "attention_mask": np.ones((1, total), np.int64), **past})
        logits = res[names.index("logits")][0, -1]
        if not np.isfinite(logits).all():
            bad += 1
            logits = np.nan_to_num(logits, nan=-1e9, posinf=1e9, neginf=-1e9)
        nxt = int(logits.argmax())
        if nxt in EOS:
            break
        out.append(nxt)
        cur = np.array([[nxt]], np.int64)
        total += 1
        past = {n.replace("present", "past_key_values"): r for n, r in zip(names, res) if n.startswith("present")}
    return out, bad


def check() -> None:
    import onnxruntime as ort
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(ROOT / "tokenizer" / "tokenizer.json"))
    row = json.loads((ROOT / "data.jsonl").read_text().splitlines()[0])
    q = row["messages"][0]["content"]
    prompt = f"<bos><start_of_turn>user\n{q}<end_of_turn>\n<start_of_turn>model\n"
    ids = tok.encode(prompt, add_special_tokens=False).ids
    print(f"\n프롬프트 {len(ids)}토큰: {q[:60]}…")

    for name in ("model.onnx", "model_fp16.onnx"):
        s = ort.InferenceSession(str(OUT / name), providers=["CPUExecutionProvider"])
        t0 = time.time()
        out, bad = greedy(s, ids)
        print(f"\n[{name}] {len(out)}토큰 · {time.time() - t0:.1f}s · 비정상 로짓 스텝 {bad}")
        print("  " + tok.decode(out))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--skip-build", action="store_true", help="이미 만든 산출물로 --check 만 실행")
    args = ap.parse_args()

    if not args.skip_build:
        if OUT.exists():
            shutil.rmtree(OUT)
        OUT.mkdir(parents=True)
        build_fp32()
        build_fp16()
    if args.check:
        check()


if __name__ == "__main__":
    main()
