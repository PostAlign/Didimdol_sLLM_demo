"""model/ · tokenizer/ 산출물을 Hugging Face 레포(PostAlign/Didimdol_sLLM)에 업로드한다.

토큰은 환경변수 HF_ACCOUNT_POSTALIGN 에서 읽는다.

  python upload_to_hf.py                 # 업로드 (이미 올라간 파일은 건너뜀)
  python upload_to_hf.py --dry-run       # 올릴 파일 목록과 원격 상태만 출력
  python upload_to_hf.py --force         # 크기가 같아도 다시 올림
  python upload_to_hf.py --private       # 비공개 레포로 생성

파일마다 커밋을 따로 낸다. 1 GB 를 한 커밋으로 묶으면 중간에 끊겼을 때
아무것도 남지 않기 때문이다. 끊긴 뒤 다시 실행하면 원격 크기가 일치하는
파일은 건너뛰므로 남은 것만 이어서 올라간다.

배포 포맷은 ONNX 외부 데이터 두 벌이다 (build_web_models.py 가 model/web/ 에 만든다).
  model.onnx + model.onnx_data            fp32 (그래프 + 가중치)
  model_fp16.onnx + model_fp16.onnx_data  fp16
레포 루트에 config/토크나이저와 함께 평평하게 올려 optimum 의
ORTModelForCausalLM.from_pretrained 와 브라우저의 transformers.js 가 그대로 읽게 한다.
단일 파일이 아닌 이유는 build_web_models.py 머리말에 있다 (iPhone 메모리 한계).

LICENSE / NOTICE / gemma_terms.md 는 Gemma Terms of Use 3.1 이 요구하는
사본·고지 조건을 맞추기 위한 파일이라 반드시 함께 올린다.
"""

import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi
from huggingface_hub.errors import RepositoryNotFoundError

REPO_ID = "PostAlign/Didimdol_sLLM"
TOKEN_ENV = "HF_ACCOUNT_POSTALIGN"

ROOT = Path(__file__).parent
MODEL = ROOT / "model"
WEB = MODEL / "web"          # build_web_models.py 산출물
TOK = ROOT / "tokenizer"

# (로컬 경로, 레포 내 경로). 작은 파일부터 올려 레포 페이지가 먼저 형태를 갖추게 한다.
FILES = [
    (MODEL / "README.md", "README.md"),
    (MODEL / "NOTICE", "NOTICE"),
    (MODEL / "LICENSE", "LICENSE"),
    (MODEL / "gemma_terms.md", "gemma_terms.md"),
    (MODEL / "config.json", "config.json"),
    (MODEL / "generation_config.json", "generation_config.json"),
    (TOK / "added_tokens.json", "added_tokens.json"),
    (TOK / "special_tokens_map.json", "special_tokens_map.json"),
    (TOK / "chat_template.jinja", "chat_template.jinja"),
    (TOK / "tokenizer_config.json", "tokenizer_config.json"),
    (TOK / "tokenizer.model", "tokenizer.model"),
    (TOK / "tokenizer.json", "tokenizer.json"),
    (WEB / "model.onnx", "model.onnx"),
    (WEB / "model_fp16.onnx", "model_fp16.onnx"),
    (WEB / "model_fp16.onnx_data", "model_fp16.onnx_data"),
    (WEB / "model.onnx_data", "model.onnx_data"),
]


def remote_sizes(api: HfApi, repo: str) -> dict[str, int]:
    """레포에 이미 있는 파일의 {경로: 바이트}. 레포가 없으면 빈 dict."""
    try:
        info = api.repo_info(repo, repo_type="model", files_metadata=True)
    except RepositoryNotFoundError:
        return {}
    return {s.rfilename: (s.size or 0) for s in info.siblings}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", default=REPO_ID)
    ap.add_argument("--private", action="store_true", help="비공개 레포로 생성")
    ap.add_argument("--dry-run", action="store_true", help="업로드 없이 목록만 출력")
    ap.add_argument("--force", action="store_true", help="원격 크기가 같아도 다시 올림")
    args = ap.parse_args()

    token = os.environ.get(TOKEN_ENV)
    if not token:
        raise SystemExit(f"환경변수 {TOKEN_ENV} 가 비어 있습니다.")

    missing = [str(p) for p, _ in FILES if not p.is_file()]
    if missing:
        raise SystemExit("파일 없음:\n  " + "\n  ".join(missing))

    api = HfApi(token=token)
    print(f"인증: {api.whoami()['name']}", flush=True)
    remote = remote_sizes(api, args.repo)

    todo = []
    print(f"\n{args.repo}:", flush=True)
    for p, dst in FILES:
        size = p.stat().st_size
        done = not args.force and remote.get(dst) == size
        print(f"  [{'있음' if done else '올림'}] {dst:26s} {size / 1e6:8.1f} MB", flush=True)
        if not done:
            todo.append((p, dst, size))

    if not todo:
        print("\n모두 최신입니다.", flush=True)
        return
    print(f"\n{len(todo)}개 / {sum(s for _, _, s in todo) / 1e9:.2f} GB 업로드", flush=True)

    if args.dry_run:
        print("--dry-run: 업로드하지 않았습니다.", flush=True)
        return

    api.create_repo(args.repo, repo_type="model", private=args.private, exist_ok=True)

    for i, (p, dst, size) in enumerate(todo, 1):
        print(f"\n[{i}/{len(todo)}] {dst} ({size / 1e6:.1f} MB) ...", flush=True)
        api.upload_file(
            path_or_fileobj=str(p),
            path_in_repo=dst,
            repo_id=args.repo,
            repo_type="model",
            commit_message=f"Upload {dst}",
        )
        print(f"[{i}/{len(todo)}] {dst} 완료", flush=True)

    print(f"\n완료: https://huggingface.co/{args.repo}", flush=True)


if __name__ == "__main__":
    main()
