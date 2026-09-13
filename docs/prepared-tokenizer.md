# 토크나이저 준비 메모리와 세션 관찰

2026-09-13 실기기 내보내기 `c3675eab6a18`에서 1d와 2c는 251개 가중치를
모두 처리했다. 토크나이저를 포함한 전체 로드는 188개 완료 후
`embed_tokens.chunk0`의 마지막 8 MiB `writeBuffer` 호출 직전 기록에서 중단됐다.
이 변경은 토크나이저 생성 중 임시 객체를 줄이는 비교 경로를 추가한다.
해당 중단이 메모리 부족 때문이라는 확정이나 실기기 해결 판정은 아니다.

## 사용

실행 진단 화면의 **토크나이저 로더**에서 `기존 JSON` 또는
`미리 변환한 데이터`를 선택한다. 2b, 1c, 전체 로드, 캐시 로드, 짧은 추론,
100건 평가가 동일한 선택을 사용한다. 토크나이저가 없는 실험에서는 선택을 비활성화한다.
`선택한 설정으로 평가 화면 열기` 링크는 런타임·전송 크기·진단 저장·토크나이저
선택을 평가 화면으로 전달한다. 평가 화면에서 직접 선택하려면
`index.html?tokenizerFormat=prepared`를 사용한다. 기본값은 `json`이며
실기기 비교를 통과하기 전에는 자동 전환하지 않는다.

2c에는 `생성 후 120초 유지` 옵션이 있다. 추론 없이 실제 Gemma 세션과 GPU 장치를
유지한다. 생성 완료는 `session-create-complete`, 관찰은 `session-idle-start`,
5초 간격 진행 기록, `session-idle-complete`로 구분한다. 120초 전체 경과와 완료
기록이 있어야 관찰 완료로 표시한다. 짧은 개발용 관찰은 해당 판정을 받지 않는다.
관찰 중 중단 버튼은 동일한 AbortSignal과 기존 자원 정리 경로를 사용한다.

## 변환 데이터

`npm run build`로 JS 런타임을 빌드한다. `npm run release`도 지원되는 런타임을
패키징할 때 현재 원본으로부터 변환 데이터를 다시 생성한다. ORT native 패치는
변경하지 않았다. 생성 파일은 무시되는 `web/vendor/tokenizer/`에 두고, 릴리스
패키저가 명시적으로 포함해 각 파일의 크기와 SHA-256을 검증한다.

| 파일 | 내용 |
| --- | --- |
| `tokenizer-prepared.json` | 포맷 `didimdol-bpe-v1`, 원본 JSON/config 해시, 레이아웃, vocabulary/merges를 제외한 기존 토크나이저 설정 |
| `tokenizer-vocab.bin` | little-endian Uint32 오프셋과 UTF-8 토큰 데이터 |
| `tokenizer-ranks.bin` | 기존 압축 BPE와 같은 Float64 키·Uint32 rank 테이블 |

변환기는 고정한 tokenizers 0.1.3의 동일한 패치 코드를 사용한다. 브라우저는
원본 `tokenizer.json`을 요청하지 않고, 각 변환 파일과 config의 SHA-256을
확인한 뒤 vocabulary와 rank 테이블을 직접 초기화한다. rank 배열 두 개는
하나의 ArrayBuffer를 공유한다. vocabulary 바이너리는 문자열 생성 뒤 참조를
제거하며, 토큰 시작의 U+FEFF도 보존한다. vocabulary·merge 객체를 되살려
기존 생성자에 넣지 않는다. 원본 JSON 경로도 같은 릴리스에서 비교 가능하다.

이 포맷은 연속된 base token ID와 numeric BPE rank를 지원한다. 원본에서
fallback merge, base ID 변경, UTF-8로 무손실 표현할 수 없는 토큰을 발견하면
변환을 거부한다. 버전·원본 해시·파일 해시·크기·배열 범위가 맞지 않으면 로드를
실패 처리한다. 실패 후 JSON을 자동으로 다시 읽지 않는다.

토큰 문자열, ID lookup, 특수 토큰, normalizer/decoder는 계속 필요하다.
이 변경의 목표는 생성 중 피크 감소이며, 전체 토크나이저의 상주 메모리를
rank 테이블 12 MiB만으로 계산하면 안 된다. GC 시점과 iPhone 프로세스 RSS는
앱의 논리 카운터로 알 수 없다.

## 재진입과 진단

의도된 페이지 이동에는 일회성 navigation ID, 사유(`experiment-start`/`repeat`),
이전·다음 실행 ID, 시각을 저장한다. 도착 URL과 대기 중인 ID가 일치할 때만
다음 실행을 시작하고 표식을 소비한다. 표식이 없거나 일치하지 않으면 자동
실행을 중단하며, 이것만으로 OS 종료를 판정하지 않는다.

자원 정리 중에는 실행 표식을 유지하고 `cleanup-start`를 별도 저장한다.
복구는 `completed-run-reentry`, `cleanup-reentry`, 실행 중 중단을 구분한다.
세션 생성은 끝났지만 관찰 중 끊긴 경우 `interruptedPhase: after-session-create`를
기록한다. 취소·실패는 해당 상태를 보존한다. 원래 worker 상태·마지막 위치·fault는
복구 정보로 덮어쓰지 않는다. 준비 완료 요약이 없는 중단 기록도
`tokenizer-ready` milestone에서 토크나이저 준비 여부와 포맷을 복원한다.

새 옵션과 복구 필드는 schema 4의 추가 필드이며 기존 schema 2/3/4 기록을 읽는다.
`totalLoadMs`는 준비와 세션 생성을, 결과의 전체 시간은 관찰과 정리까지 포함한다.
관찰 시간과 cleanup 시간은 별도로 내보낸다. 진단 저장 시간에는 기존처럼
완료된 저장만 포함되며 프로세스 메모리나 직렬화 전체 비용은 포함되지 않는다.

## 검증

```bash
npm test
npm run build
npm run verify:release
node tools/measure-tokenizer.mjs
ORT_MODES=asyncify,jspi,stock TEST_APP_LOAD=1 TEST_FULL_MODEL=1 TEST_SESSION_IDLE=120 npm run test:browser
```

메모리 비교 도구는 각 경로를 새 Node 프로세스에서 3회 실행하고 생성 피크 RSS와
GC 뒤 보유 heap/ArrayBuffer를 보고한다. 실행 시간에는 로컬 파일 읽기와 해시가
포함된다. iPhone 메모리나 처리 속도 측정은 아니다.

이번 로컬 3회 측정 결과는 다음과 같다. 보유량은 실행 전 대비 증가분이다.

| 중앙값 | 기존 JSON | 미리 변환한 데이터 |
| --- | ---: | ---: |
| 프로세스 최대 RSS | 302.22 MiB | 153.24 MiB |
| GC 후 heap + ArrayBuffer 증가분 | 42.93 MiB | 43.03 MiB |

생성 피크가 약 49% 감소했지만 보유량은 거의 동일하다. 이 수치로 iPhone의
실제 메모리 여유나 전체 로드 성공을 예측하지 않는다. 변환 파일 세 개의 합은
16,744,605 bytes이며 공통 `tokenizer_config.json`은 별도다.

정확성 검증은 전체 514,906개 merge rank, 모든 평가 입력의 토큰 ID,
특수 토큰·공백·한국어·이모지·U+FEFF, decode, 채팅 템플릿, 동일 문자열에 대한
ROUGE를 비교한다. 실제 브라우저에서는 두 로더의 전체 가중치 로드 및 기존
짧은·긴 입력의 FP32 출력과 함께 해시 오류, 취소, 복구, 릴리스 일치를 검증한다.

실기기에서는 한 릴리스에서 기존/새 토크나이저를 교차 실행하며 먼저 8 MiB로
조건별 3회 비교한다. 2c 생성 후 120초 관찰, 전체 로드 3회, 캐시 로드 5회,
짧은·긴 입력 추론, 같은 세션의 100건 평가 2회가 필요하다. 기기·브라우저·검사기
연결 상태를 기록하고 중단 시각의 시스템 로그와 대조한다. 4/2 MiB는 별도 비교다.
로컬 기능 검증은 이 실기기 완료 기준을 대신하지 않는다.

실행한 명령, 릴리스 ID, 메모리 표본, 기준 출력과 관찰·취소 결과는
[prepared-tokenizer-validation.json](prepared-tokenizer-validation.json)에 기록했다.
