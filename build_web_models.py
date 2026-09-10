"""model/model.onnx (fp32 단일 파일) → 브라우저 배포용 model/web/ 산출물 (fp32 전용).

  model/web/model.onnx  +  model.onnx_data, model.onnx_data_1, …   그래프 + 외부 가중치 (합계 1.07 GB)

fp16 은 만들지 않는다. 품질이 fp32 에 못 미쳐(Gemma 3 는 fp16 에서 잔차 스트림이 65504 를 넘을 수 있다)
평가 기준이 될 수 없으므로 fp32 로 확정했다. 대신 fp32 가 스마트폰에서도 올라가도록 그래프를 세 가지
손본다. 모두 모바일 WebGPU 메모리 때문이다.

1. 임베딩 분할
   tied embedding [262144, 640] 이 단일 텐서(671 MB)다. WebGPU 는 텐서 하나가 스토리지 버퍼 하나라,
   어댑터의 maxStorageBufferBindingSize 보다 큰 텐서는 올릴 수 없다. 스마트폰 GPU(Adreno·Mali)는 이
   한계가 스펙 기본값인 128 MiB 그대로인 경우가 대부분이라 원본 그대로는 WebGPU 를 못 쓰고 WASM 으로
   밀려나며, WASM 은 가중치를 wasm 힙에 한 벌 더 복사하므로 1 GB 모델이 2 GB 가 되어 죽는다.
   임베딩을 CHUNK_ROWS 행씩 잘라 텐서마다 41.9 MB 로 두면 어떤 WebGPU 어댑터에도 들어간다.
     Gather   : 청크마다 Gather 한 뒤 토큰이 속한 청크의 결과만 Where 로 고른다.
     lm_head  : 청크마다 Gemm(transB) 한 뒤 Concat 한다. Gemm 은 [rows, 640] 저장 그대로 읽으므로
                전치 사본이 생기지 않는다. 원본의 Transpose 노드는 ORT 가 세션 생성 때 상수
                접기(constant folding)로 CPU(wasm 힙)에 전치본 671 MB 를 만들어 두므로 없앤다.

2. 마지막 토큰 로짓만 계산
   원본은 프롬프트 전 위치의 로짓 [1, S, 262144] 를 낸다. S=266 이면 279 MB 라 이것도 버퍼 한계를
   넘는다. transformers.js 의 generate 는 마지막 위치만 쓰므로 lm_head 앞에서 Slice 해 [1, 1, 262144]
   (1 MB) 만 낸다. 프리필 시간도 그만큼 준다.

3. 외부 데이터 파일 분할
   가중치를 FILE_CAP 이하의 파일 여러 개로 나눈다. 브라우저 워커(web/sllm/worker.js)가 파일 단위로
   받아 캐시하고 Blob 으로 마운트하므로, 한 번에 메모리에 있는 가중치는 파일 하나 이하다.
   파일 이름은 transformers.js 의 규약(model.onnx_data, model.onnx_data_1, …)을 따른다.

  python build_web_models.py            # 산출물 생성
  python build_web_models.py --check    # 생성 후 원본과 로짓·탐욕 디코딩 비교 (CPU, 1~2분)

업로드는 upload_to_hf.py 가 model/web/ 을 레포 루트에 평평하게 올린다. 올린 뒤 web/sllm/worker.js 의
REVISION 을 새 커밋 SHA 로 바꿔야 브라우저가 새 파일을 받는다.
"""

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from onnx.external_data_helper import set_external_data

ROOT = Path(__file__).parent
SRC = ROOT / "model" / "model.onnx"
OUT = ROOT / "model" / "web"
EOS = {1, 106}   # <eos>, <end_of_turn>

# 임베딩 청크 행 수. 262144 / 16384 = 16 청크, 텐서당 16384 × 640 × 4 B = 41.9 MB.
# web/sllm/index.js 의 NEED 가 같은 수치로 GPU 버퍼 한계를 검사하므로 바꾸면 거기도 맞춘다.
CHUNK_ROWS = 16384
# 외부 데이터 파일 하나의 상한. 워커가 파일 단위로 내려받아 캐시·마운트한다.
FILE_CAP = 128 * 2**20
# 이 크기 미만 텐서는 그래프 파일 안에 그대로 둔다 (onnx 의 size_threshold 와 같은 뜻).
INLINE_THRESHOLD = 1024


# ── 그래프 수술 ─────────────────────────────────────────────────────────────
def _consumers(graph, name):
    return [n for n in graph.node if name in n.input]


def split_tied_embedding(model: onnx.ModelProto, rows: int) -> None:
    """tied embedding 을 rows 행 청크로 나누고, Gather 와 lm_head 를 청크 단위로 다시 쓴다.
    lm_head 는 마지막 토큰만 계산하도록 Slice 를 넣는다. model 을 제자리에서 고친다."""
    g = model.graph

    # 임베딩: 2D 초기값 중 Gather 의 data 이면서 Transpose 의 입력이기도 한 것 (tied).
    embed = None
    for t in g.initializer:
        if len(t.dims) == 2 and {"Gather", "Transpose"} <= {n.op_type for n in _consumers(g, t.name)}:
            embed = t
            break
    if embed is None:
        raise SystemExit("tied embedding(Gather + Transpose 가 함께 읽는 2D 초기값)을 찾지 못했습니다")
    V, H = embed.dims
    if V % rows:
        raise SystemExit(f"vocab {V} 가 CHUNK_ROWS {rows} 로 나누어떨어지지 않습니다")
    K = V // rows

    gather = next(n for n in _consumers(g, embed.name) if n.op_type == "Gather")
    transpose = next(n for n in _consumers(g, embed.name) if n.op_type == "Transpose")
    matmul = next(n for n in _consumers(g, transpose.output[0]) if n.op_type == "MatMul")
    ids_name, embed_out = gather.input[1], gather.output[0]
    hidden_name, logits_name = matmul.input[0], matmul.output[0]
    if hidden_name == transpose.output[0]:
        hidden_name = matmul.input[1]
    print(f"임베딩 {embed.name} [{V}, {H}] → {K} 청크 × [{rows}, {H}] · "
          f"Gather={gather.name} · lm_head={matmul.name}", flush=True)

    W = numpy_helper.to_array(embed)          # [V, H]
    chunk_names = [f"embed_tokens.chunk{k}" for k in range(K)]
    new_inits = [numpy_helper.from_array(np.ascontiguousarray(W[k * rows:(k + 1) * rows]), chunk_names[k])
                 for k in range(K)]
    del W

    def const_i32(name, v):
        return helper.make_tensor(name, TensorProto.INT32, [], [v])

    def const_i64(name, vals):
        return helper.make_tensor(name, TensorProto.INT64, [len(vals)], vals)

    consts = [
        const_i32("embed_tokens.rows", rows),
        const_i64("embed_tokens.unsq_axes", [-1]),
        const_i64("lm_head.last_starts", [-1]),
        const_i64("lm_head.last_ends", [np.iinfo(np.int64).max]),
        const_i64("lm_head.last_axes", [1]),
        const_i64("lm_head.flat_shape", [-1, H]),
        const_i64("lm_head.logits_shape", [-1, 1, V]),
    ] + [const_i32(f"embed_tokens.k{k}", k) for k in range(1, K)]

    # ── Gather: ids → (청크 번호, 청크 안 행) ──
    # 청크마다 같은 행 번호로 Gather 하고, 토큰이 속한 청크의 결과만 Where 로 남긴다.
    # int32 로 내려서 계산한다. WebGPU EP 의 정수 산술은 int32 가 가장 넓게 지원된다.
    p = "/model/embed_tokens/split/"
    embed_nodes = [
        helper.make_node("Cast", [ids_name], [p + "ids32"], to=TensorProto.INT32, name=p + "Cast"),
        helper.make_node("Div", [p + "ids32", "embed_tokens.rows"], [p + "chunk"], name=p + "Div"),
        helper.make_node("Mul", [p + "chunk", "embed_tokens.rows"], [p + "base"], name=p + "Mul"),
        helper.make_node("Sub", [p + "ids32", p + "base"], [p + "local"], name=p + "Sub"),
        helper.make_node("Unsqueeze", [p + "chunk", "embed_tokens.unsq_axes"], [p + "chunk_u"], name=p + "Unsqueeze"),
    ]
    acc = None
    for k in range(K):
        gk = f"{p}g{k}"
        embed_nodes.append(helper.make_node("Gather", [chunk_names[k], p + "local"], [gk], axis=0, name=f"{p}Gather{k}"))
        if k == 0:
            acc = gk
            continue
        embed_nodes.append(helper.make_node("Equal", [p + "chunk_u", f"embed_tokens.k{k}"], [f"{p}is{k}"], name=f"{p}Equal{k}"))
        out = embed_out if k == K - 1 else f"{p}acc{k}"
        embed_nodes.append(helper.make_node("Where", [f"{p}is{k}", gk, acc], [out], name=f"{p}Where{k}"))
        acc = out

    # ── lm_head: 마지막 토큰만 → 청크별 Gemm(transB) → Concat ──
    q = "/lm_head/split/"
    head_nodes = [
        helper.make_node("Slice", [hidden_name, "lm_head.last_starts", "lm_head.last_ends", "lm_head.last_axes"],
                         [q + "last"], name=q + "Slice"),
        helper.make_node("Reshape", [q + "last", "lm_head.flat_shape"], [q + "flat"], name=q + "Reshape"),
    ]
    for k in range(K):
        head_nodes.append(helper.make_node("Gemm", [q + "flat", chunk_names[k]], [f"{q}l{k}"], transB=1, name=f"{q}Gemm{k}"))
    head_nodes += [
        helper.make_node("Concat", [f"{q}l{k}" for k in range(K)], [q + "cat"], axis=1, name=q + "Concat"),
        helper.make_node("Reshape", [q + "cat", "lm_head.logits_shape"], [logits_name], name=q + "Reshape_out"),
    ]

    # ── 그래프에 반영 ──
    # onnx.checker 는 위상 정렬을 요구한다. 임베딩 노드는 옛 Gather 자리에, lm_head 노드는 옛 MatMul
    # 자리에 넣고 Transpose 는 뺀다.
    new_nodes = []
    for n in g.node:
        if n.name == gather.name:
            new_nodes += embed_nodes
        elif n.name == matmul.name:
            new_nodes += head_nodes
        elif n.name != transpose.name:
            new_nodes.append(n)
    del g.node[:]
    g.node.extend(new_nodes)
    g.initializer.remove(embed)
    g.initializer.extend(new_inits + consts)
    for o in g.output:
        if o.name == logits_name:
            o.type.tensor_type.shape.dim[1].dim_param = ""
            o.type.tensor_type.shape.dim[1].dim_value = 1


# ── 외부 데이터 저장 ────────────────────────────────────────────────────────
def save_external(model: onnx.ModelProto, path: Path, cap: int) -> list[Path]:
    """INLINE_THRESHOLD 이상 텐서를 cap 이하 파일 여러 개로 나눠 쓴다. 이름은 model.onnx_data,
    model.onnx_data_1, … (transformers.js 규약). 오프셋은 64 바이트 정렬."""
    base = path.name + "_data"
    files, fh, size, name = [], None, 0, None

    for t in model.graph.initializer:
        raw = numpy_helper.to_array(t).tobytes()
        if len(raw) < INLINE_THRESHOLD:
            continue
        if fh is None or size + len(raw) > cap:
            if fh:
                fh.close()
            name = base if not files else f"{base}_{len(files)}"
            files.append(path.parent / name)
            fh, size = open(files[-1], "wb"), 0
        pad = (-size) % 64
        fh.write(b"\0" * pad)
        size += pad
        fh.write(raw)
        set_external_data(t, location=name, offset=size, length=len(raw))
        for f in ("float_data", "int32_data", "int64_data", "double_data", "uint64_data", "string_data"):
            t.ClearField(f)
        t.raw_data = b""
        t.data_location = TensorProto.EXTERNAL
        size += len(raw)
    if fh:
        fh.close()
    onnx.save_model(model, str(path))
    sizes = [f.stat().st_size for f in files]
    print(f"저장: {path.name} {path.stat().st_size / 1e6:.1f} MB · 가중치 {len(files)}개 파일 "
          f"{sum(sizes) / 1e6:.1f} MB (최대 {max(sizes) / 1e6:.1f} MB)", flush=True)
    return files


def build() -> None:
    t0 = time.time()
    model = onnx.load(str(SRC))
    split_tied_embedding(model, CHUNK_ROWS)
    onnx.checker.check_model(model)
    save_external(model, OUT / "model.onnx", FILE_CAP)
    print(f"빌드 {time.time() - t0:.0f}s", flush=True)


# ── 검증 ────────────────────────────────────────────────────────────────────
def _prompt_ids(tok, row):
    s = "<bos>"
    for m in row["messages"][:-1]:
        role = "model" if m["role"] == "assistant" else m["role"]
        s += f"<start_of_turn>{role}\n{m['content']}<end_of_turn>\n"
    s += "<start_of_turn>model\n"
    return tok.encode(s, add_special_tokens=False).ids


def _empty_past():
    return {f"past_key_values.{i}.{kv}": np.zeros((1, 1, 0, 256), np.float32)
            for i in range(18) for kv in ("key", "value")}


def step(sess, ids, total, past):
    names = [o.name for o in sess.get_outputs()]
    res = sess.run(None, {"input_ids": np.array([ids], np.int64),
                          "attention_mask": np.ones((1, total), np.int64), **past})
    logits = res[names.index("logits")][0, -1]           # 원본 [1,S,V]·분할본 [1,1,V] 모두 마지막 위치
    past = {n.replace("present", "past_key_values"): r for n, r in zip(names, res) if n.startswith("present")}
    return logits, past


def greedy(sess, ids, max_new=48):
    past = _empty_past()
    cur, total, out, bad = list(ids), len(ids), [], 0
    for _ in range(max_new):
        logits, past = step(sess, cur, total, past)
        if not np.isfinite(logits).all():
            bad += 1
            logits = np.nan_to_num(logits, nan=-1e9, posinf=1e9, neginf=-1e9)
        nxt = int(logits.argmax())
        if nxt in EOS:
            break
        out.append(nxt)
        cur, total = [nxt], total + 1
    return out, bad


def check() -> None:
    import onnxruntime as ort
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(ROOT / "tokenizer" / "tokenizer.json"))
    rows = [json.loads(l) for l in (ROOT / "web" / "sllm" / "data.jsonl").read_text().splitlines() if l.strip()]
    # 짧은 단일 턴 하나, 가장 긴 다중 턴 하나. 긴 쪽이 청크 경계와 마지막 토큰 Slice 를 더 넓게 훑는다.
    picks = [rows[0], max(rows, key=lambda r: len(_prompt_ids(tok, r)))]
    ref = ort.InferenceSession(str(SRC), providers=["CPUExecutionProvider"])
    new = ort.InferenceSession(str(OUT / "model.onnx"), providers=["CPUExecutionProvider"])
    assert next(o for o in new.get_outputs() if o.name == "logits").shape[1] == 1, "logits 가 마지막 토큰만이 아님"

    for row in picks:
        ids = _prompt_ids(tok, row)
        l_ref, _ = step(ref, ids, len(ids), _empty_past())
        l_new, _ = step(new, ids, len(ids), _empty_past())
        diff = float(np.abs(l_ref - l_new).max())
        t0 = time.time()
        out_ref, _ = greedy(ref, ids)
        out_new, bad = greedy(new, ids)
        print(f"프롬프트 {len(ids):3d}토큰 · 로짓 최대 오차 {diff:.2e} · top-1 {'일치' if l_ref.argmax() == l_new.argmax() else '불일치'} · "
              f"탐욕 {len(out_new)}토큰 {'원본과 동일' if out_ref == out_new else '원본과 다름'} · "
              f"비정상 로짓 스텝 {bad} · {time.time() - t0:.1f}s")
        print("   ", tok.decode(out_new)[:120].replace("\n", " "))
        if diff > 1e-3 or out_ref != out_new:
            raise SystemExit("분할본이 원본과 다릅니다")
    print("검증 통과: 분할본이 원본과 같은 로짓·같은 텍스트를 낸다")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--skip-build", action="store_true", help="이미 만든 산출물로 --check 만 실행")
    args = ap.parse_args()

    if not args.skip_build:
        if OUT.exists():
            shutil.rmtree(OUT)
        OUT.mkdir(parents=True)
        build()
    if args.check:
        check()


if __name__ == "__main__":
    main()
