---
language:
- ko
- en
license: other
license_name: didimdol-sllm-noncommercial
license_link: https://huggingface.co/PostAlign/Didimdol_sLLM/blob/main/LICENSE
base_model: google/gemma-3-270m-it
base_model_relation: finetune
library_name: onnx
pipeline_tag: text-generation
tags:
- korean
- gemma3
- onnx
- on-device
- non-commercial
---

# Didimdol_sLLM

Google **Gemma 3 270M Instruct**(`gemma-3-270m-it`)를 파인튜닝한 한국어 소형 언어모델입니다.
온디바이스 추론에 쓸 수 있도록 **ONNX(fp32)** 로 내보낸 산출물을 배포합니다.

> Gemma is provided under and subject to the Gemma Terms of Use found at
> [ai.google.dev/gemma/terms](https://ai.google.dev/gemma/terms).
> 이 레포의 가중치는 원본 Gemma 3 270M IT 가중치를 파인튜닝하여 **수정한 것**입니다.

## 개요

| | |
| --- | --- |
| 베이스 모델 | [`google/gemma-3-270m-it`](https://huggingface.co/google/gemma-3-270m-it) |
| 아키텍처 | `Gemma3ForCausalLM` (decoder-only, 18층, hidden 640, GQA 4q/1kv, head_dim 256) |
| 컨텍스트 | 32,768 (sliding window 512, 6층마다 full attention) |
| vocab | 262,144 (Gemma 3 토크나이저 그대로) |
| 배포 포맷 | ONNX fp32 (약 1.07 GB), KV 캐시 포함 그래프 (opset 18), 가중치는 외부 데이터 파일 여러 개 |
| 주 언어 | 한국어 |

## 파일

| 파일 | 설명 |
| --- | --- |
| `model.onnx` + `model.onnx_data`, `model.onnx_data_1`, … | 파인튜닝된 모델 (ONNX fp32). 그래프와 가중치(외부 데이터, 파일당 128 MiB 이하)가 분리되어 있으므로 모든 파일을 같은 폴더에 두어야 합니다. tied embedding 은 16 개 청크로 나뉘어 있고 `logits` 는 마지막 위치만 냅니다 (브라우저 WebGPU 의 버퍼 한계 때문. 생성 결과는 원본과 동일) |
| `config.json` / `generation_config.json` | 모델 · 생성 설정 |
| `tokenizer.json`, `tokenizer.model`, `tokenizer_config.json`, `special_tokens_map.json`, `added_tokens.json` | Gemma 3 토크나이저 |
| `chat_template.jinja` | Gemma 3 대화 템플릿 (`<start_of_turn>` / `<end_of_turn>`) |
| `LICENSE`, `NOTICE`, `gemma_terms.md` | 라이선스 및 고지 |

## 사용

### optimum + onnxruntime

```python
from optimum.onnxruntime import ORTModelForCausalLM
from transformers import AutoTokenizer

repo = "PostAlign/Didimdol_sLLM"
tok = AutoTokenizer.from_pretrained(repo)
model = ORTModelForCausalLM.from_pretrained(repo)

ids = tok.apply_chat_template(
    [{"role": "user", "content": "안녕하세요, 자기소개 해주세요."}],
    add_generation_prompt=True,
    return_tensors="pt",
)
out = model.generate(ids, max_new_tokens=256, do_sample=True, top_p=0.95, top_k=64)
print(tok.decode(out[0][ids.shape[-1]:], skip_special_tokens=True))
```

### onnxruntime 직접 사용

그래프 입출력은 아래와 같습니다. `position_ids` 입력은 없고, 위치는 `attention_mask` 로부터 내부에서 계산됩니다.

| | 이름 | shape |
| --- | --- | --- |
| 입력 | `input_ids` | `[batch, seq]` (int64) |
| 입력 | `attention_mask` | `[batch, past + seq]` (int64) |
| 입력 | `past_key_values.{0..17}.{key,value}` | `[batch, 1, past, 256]` (fp32) |
| 출력 | `logits` | `[batch, 1, 262144]` (마지막 위치만) |
| 출력 | `present.{0..17}.{key,value}` | `[batch, 1, past + seq, 256]` |

`logits` 는 입력 길이와 무관하게 마지막 토큰 위치 하나만 나옵니다 (generate 에 필요한 건 그것뿐입니다). 첫 프리필에서는 `past` 를 길이 0 텐서로 넣고, 이후 디코드 스텝에서는 직전 스텝의 `present.*` 를 그대로 `past_key_values.*` 로 넘깁니다. 종료 토큰은 `<eos>`(1) 와 `<end_of_turn>`(106) 입니다.

## 프롬프트 형식

Gemma 3 형식을 그대로 씁니다.

```
<bos><start_of_turn>user
{메시지}<end_of_turn>
<start_of_turn>model
```

## 한계

270M 규모의 소형 모델입니다. 사실 관계가 틀리거나 문맥을 놓칠 수 있고, 긴 추론이 필요한 작업에는 적합하지 않습니다. 출력은 사용자가 검증한 뒤 사용해야 하며, 의료 · 법률 · 금융 등 중요한 판단의 근거로 쓰지 마세요.

## 라이선스

이 모델에는 **두 가지 조건이 함께** 적용됩니다.

1. **Google Gemma Terms of Use** — 본 모델은 Gemma 3 270M IT 의 Model Derivative 이므로 [Gemma Terms of Use](https://ai.google.dev/gemma/terms) 와 [Gemma Prohibited Use Policy](https://ai.google.dev/gemma/prohibited_use_policy) 를 그대로 따릅니다. 전문 사본은 [`gemma_terms.md`](https://huggingface.co/PostAlign/Didimdol_sLLM/blob/main/gemma_terms.md) 에 포함되어 있습니다.
2. **PostAlign 추가 조건 — 비상업적 이용에 한함** — PostAlign 은 본 모델을 **비상업적 목적(개인 이용, 연구, 교육, 평가)** 으로만 사용·복제·수정·배포할 수 있도록 허락합니다. 유료 제품 · 서비스 탑재, 판매, 영리 활동 등 상업적 이용에는 PostAlign 의 별도 서면 허락이 필요합니다.

재배포할 때에는 `LICENSE`, `NOTICE`, `gemma_terms.md` 를 함께 배포하고, 위 두 조건을 다음 이용자에게도 그대로 전달해야 합니다. 전체 내용은 [`LICENSE`](https://huggingface.co/PostAlign/Didimdol_sLLM/blob/main/LICENSE) 를 확인하세요.

상업적 이용 문의: [PostAlign](https://huggingface.co/PostAlign)

---

*Gemma is provided under and subject to the Gemma Terms of Use found at ai.google.dev/gemma/terms*
