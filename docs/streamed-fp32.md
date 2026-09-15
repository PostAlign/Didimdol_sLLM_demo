# FP32 가중치 순차 로딩

`modelExecution=streamed`는 기존 FP32 가중치를 그대로 사용하면서 입력 임베딩과
출력 projection을 본체 ORT 세션에서 분리한다. iPhone의 세션 생성 중 중단에 대응해
GPU 가중치 상주량을 줄이는 비교 경로다. iPhone에서 크래시 해결이 검증됐다는 뜻은 아니다.

실행 진단에서 **모델 실행 → 가중치 순차 로딩**을 선택한다. 2c, 전체 로드,
캐시 로드, 짧은·긴 추론, 100건 평가에 적용된다. 평가 화면 링크가 같은 선택을
전달하며 직접 열 때는 `index.html?modelExecution=streamed`를 사용한다.
기본값은 기존 `resident`다. 토크나이저 형식은 독립적인 설정이다.

## 메모리와 실행

| 구성 | 가중치 요청량 |
| --- | ---: |
| 본체 외부 initializer 235개 | 401,304,064 B (382.713 MiB) |
| 재사용 출력 가중치 GPUBuffer 1개 (기본, `outputBuffers=1`) | 41,943,040 B (40 MiB) |
| 합계 | 443,247,104 B (422.713 MiB) |

`outputBuffers=2`이면 버퍼 둘(80 MiB, 합계 462.713 MiB)로 다음 청크의 읽기·업로드를
현재 청크의 projection과 겹친다. 9월 14일 저녁 실기기에서 이 겹침은 토큰당 30~36%
느렸고(64토큰 projection 벽시계 39.9 s 대 30.0 s, 평가 행 703 ms 대 516 ms), 그래서
기본은 직렬 버퍼 하나다. 겹침 루프는 그 뒤 리드백을 다음 청크 쓰기보다 먼저 큐에
넣도록 바뀌었으며 실기기 재측정 전까지 비교용이다. CPU 전송 scratch는 전송
크기(기본 8 MiB)를 따른다. 입력 임베딩은
토큰당 2,560 B를 OPFS에서 직접 읽어 프롬프트 크기의 배열에 넣는다. 본체 그래프가
기존 임베딩 배율, attention, 위치 계산, 정규화, 마지막 위치 선택과 KV 출력을 수행한다.

출력층은 동적 `weight` 입력을 받는 FP32 `Gemm(transB=1)` 세션이다. 16개 청크를
GPU 버퍼에 차례로 채우고 계산 결과를 전체 어휘 logits에 복사한다. 버퍼가 하나면
청크마다 읽기 → `writeBuffer` → 큐 대기 → projection → 리드백 순서로 직렬이다.
버퍼가 둘이면 청크 *i*의 `run()`이 돌아온 직후 `getData()`로 리드백 복사를 먼저
큐에 넣고, 그 결과를 기다리는 동안 청크 *i+1*을 다른 버퍼로 읽어 올린다. 큐 순서는
compute(i) → readback(i) → writes(i+1)이므로 리드백이 40 MiB 업로드 뒤에 서지
않으며, 겹친 업로드는 자체 큐 대기를 하지 않는다(다음 projection의 리드백이 그
경계다). 한 버퍼는 앞선 projection의 결과를 회수한 뒤에만 다시 쓴다.
`writeBuffer`마다 오류 스코프를 동기적으로 감싸는데, 업로드가 런타임의 `run()`과
같은 장치에서 겹치기 때문이다. 세션당 처음 두 projection은 청크별 타임라인(run
제출·반환, 리드백 제출·완료, 업로드 구간)을 다음 `streamed-step-complete` 기록의
`chunkTimelines`에 남긴다. 본체와 출력 세션은 같은 ORT 런타임과 GPUDevice를
사용하며 세션을 토큰/청크마다 다시 만들지 않는다.

한 번의 모델 forward마다 출력 가중치 640 MiB를 다시 읽고 업로드한다. 읽기는 로컬
OPFS이며 네트워크 재다운로드가 아니다. 생성 속도와 평가 총시간은 기존 방식보다
불리할 수 있다. 메모리 표에는 KV 캐시, 연산 임시 버퍼, 토크나이저와 드라이버 메모리가
포함되지 않는다. 진단의 GPU 카운터는 실제 RSS가 아니다.

## 저장소와 자원 수명

원본 manifest로 OPFS 저장소를 열어 이미 검증된 원본 가중치 파일을 재사용한다.
분리 그래프 해시를 원본 캐시 키에 넣지 않는다. 본체용 SessionRangeLoader에는
임베딩을 제외한 initializer 목록만 전달하며 생성 후 이 로더를 봉인한다.
추론 중 계획된 파일 읽기는 별도 StreamedWeights가 담당한다.

순차 로딩은 준비부터 세션 폐기까지 `didimdol-model-load` origin lock을 유지한다.
다른 탭의 로딩/캐시 이전과 충돌하지 않으며 워커 메시지 처리는 계속 가능하다.
한 번에 읽기 핸들 하나만 열고, 추론 작업 사이에는 핸들을 닫는다. 모델 종료,
실패와 취소 시 세션, 외부 GPU 버퍼, OPFS 저장소와 lock을 정리한다.

## 생성기와 산출물

원본은 `8c50d7686bb1b205c02d42bb4b64505483c41a2b` revision의 배포 그래프다.
`tools/prepare_streamed_model.py`는 `model/initializers.json`의 SHA-256을 확인한 뒤
그래프만 분리한다. 외부 가중치의 파일명·오프셋·바이트는 변경하지 않는다.
작은 `model/streamed/body.onnx`, `head.onnx`, `manifest.json`은 배포 산출물에 포함된다.
기존 릴리스 패키저가 이 파일들의 크기와 해시를 함께 고정한다.

```bash
.venv/bin/python tools/prepare_streamed_model.py --source .work/full-model/model.onnx
npm run release
npm run verify:release
```

실행 시 원본 revision/manifest, 본체 initializer 집합, 임베딩 청크 범위와 생성
그래프의 크기·SHA-256을 검증한다. 지원하지 않는 그래프 구조는 변환 오류로 처리한다.
분리 그래프의 연산자는 현재 모바일 ORT 빌드의 연산자 목록 안에 있으므로 이번
변경은 native ORT 패치/바이너리 재빌드를 요구하지 않는다.

## 생성·진단 호환성

ORT session facade를 기존 Gemma3ForCausalLM에 연결해 Transformers.js 4.2.0의
generate, sampling, EOS, cache 처리를 재사용한다. 기존 평가의 seed 42, temperature
0.3, top-k 64, top-p 0.95와 최대 512토큰 설정을 유지한다.

2c 완료는 본체 세션, 출력 세션과 교체 버퍼가 준비됐다는 의미다. 출력 가중치 전체를
실제로 읽고 계산했는지는 추론 실험에서 확인한다. 원본 전체 상주 실험의 완료 가중치
251개와 순차 로딩 본체의 235개는 서로 다른 범위다.

`modelExecution`, `streaming`을 결과에 기록하고 `streamed-weight` GPU 역할을 추가한다.
`streaming`에는 임베딩/출력 읽기 바이트, 업로드 바이트, 버퍼 크기와 개수, projection 수,
읽기·전송·큐 대기를 합한 uploadMs(outputReadMs, queueWaitMs로 분리), 출력 계산/결과
회수 시간 outputComputeMs, 출력 루프 전체의 벽시계 시간 projectionMs가 있다. 두 버퍼에서는
uploadMs와 outputComputeMs가 겹치므로 `uploadMs + outputComputeMs − projectionMs`가
겹침으로 줄어든 시간이다. 토큰별 완료 시 누적 집계만 저장하며 오류에는 업로드 중이던
청크·오프셋(`position`)과 계산 중이던 청크(`computing`)를 기록한다. 기존 세션 가중치
카운터는 본체만 센다. 전체 요청량은 GPU ledger에서 확인한다.

## 검증

이번 변경의 실제 측정값과 검증 범위는 [검증 결과](streamed-validation.json)에 기록했다.

```bash
npm test
.venv/bin/python -m unittest discover -s tests -p 'test_*.py'
.venv/bin/pip install -r requirements-validation.txt
.venv/bin/python tools/check_streamed_model.py
npm run test:streamed
ORT_MODE=jspi MODEL_EXECUTIONS=streamed STREAMED_TEST_UI=0 npm run test:streamed
MODEL_EXECUTIONS='' STREAMED_TEST_IDLE=120 npm run test:streamed
npm run test:browser
```

CPU/전체 브라우저 검증에는 `.work/full-model/`의 원본 그래프와 외부 가중치가 필요하다.
CPU 검증은 실제 평가의 가장 짧은/긴 프롬프트에서 logits 오차 ≤ 1e-3,
KV 출력 오차 ≤ 1e-3 및 greedy 토큰 일치를 검사한다. 브라우저 검증은 원본/순차
경로의 토큰 비교, 같은 세션 재실행, 40 MiB 버퍼 단일 생성, 저장소 읽기량,
정리, 진단 화면 선택·관찰 취소·lock 반환을 확인한다.

테스트 보고서는 `.work/streamed-cpu-results.json`과
`.work/streamed-browser-results.json`에 저장된다. 데스크톱 SwiftShader의 속도는
iPhone 속도를 대신하지 않는다. 실기기에서는 이 수정 릴리스로 2c/120초 관찰,
캐시 로드 반복, 짧은·긴 추론과 같은 세션의 100건 평가 2회를 확인해야 한다.
