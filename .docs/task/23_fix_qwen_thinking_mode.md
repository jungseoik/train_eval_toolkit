# TASK 23: Qwen3.5 Thinking 모드 비활성화 및 Soil_v2 평가

## 배경

Qwen3.5 모델 평가 시 전체 프레임에서 pred=0만 출력되는 문제 발생.
9B 모델 Soil_v2_Fire 평가에서 124,474 프레임 전부 pred=0으로, 화재를 전혀 감지하지 못하는 상태.

## 원인 분석

Qwen3.5 4B/9B 모델은 **thinking 모드가 기본 활성화**되어 있어, 응답 첫 토큰이 `"Thinking"`으로 시작.
CLS 모드(`max_tokens=1`)에서 첫 토큰이 "Thinking"으로 잘려 `parse_cls_output`이 항상 0을 반환.

### 모델별 thinking 기본동작 테스트 결과

| 모델 | thinking 기본? | max_tokens=1 응답 | enable_thinking=false 시 |
|------|---------------|-------------------|------------------------|
| 0.8B | No | `"yes"` / `"no"` (정상) | `"yes"` / `"no"` |
| 2B | No | `"yes"` / `"no"` (정상) | `"yes"` / `"no"` |
| 4B | **Yes** | `"The"` (잘림) | `"yes"` (정상) |
| 9B | **Yes** | `"The"` (잘림) | `"yes"` (정상) |

- 4개 모델 모두 이미지 이해능력은 정상 (이미지 설명 테스트 통과)
- 테스트 이미지: 프로젝트 `image.webp` (화재 현장 사진)

## 수정 내용

### 1. YAML config 기반 enable_thinking 옵션 추가

**설계 결정**: 하드코딩 대신 YAML config 방식 채택
- 호환성: 미설정 시 기존 동작 유지 (Cosmos 등 비-Qwen 모델에 영향 없음)
- 확장성: YAML에서 true/false 토글 가능 (향후 reasoning 평가 대응)

변경 파일:
- `src/vllm_pipeline/config.py` — `EvalConfig`에 `enable_thinking: bool | None` 필드 추가 (기본값 `None`)
- `src/vllm_pipeline/evaluator.py` — `ENABLE_THINKING`을 cfg namespace에 전달
- `src/evaluation/vllm_bench_eval.py` — `_infer_frame`에서 `enable_thinking`이 `None`이 아닐 때만 `chat_template_kwargs` payload에 포함

### 2. Qwen3.5 YAML 전체에 enable_thinking: false 추가

대상 YAML 9개:
- `qwen35_08b_fire.yaml`, `qwen35_08b_fire_Soil_v2.yaml`, `qwen35_08b_smoke_Soil_v2.yaml`
- `qwen35_2b_fire.yaml`, `qwen35_2b_smoke.yaml`
- `qwen35_4b_fire.yaml`, `qwen35_4b_smoke.yaml`
- `qwen35_9b_fire.yaml`, `qwen35_9b_smoke.yaml`

### 3. 신규 YAML 생성

- `qwen35_08b_fire_Soil_v2.yaml` — 0.8B Soil_v2_Fire 평가용
- `qwen35_08b_smoke_Soil_v2.yaml` — 0.8B Soil_v2_Smoke 평가용
- `qwen35_9b_fire.yaml` — 9B Soil_v2_Fire 평가용
- `qwen35_9b_smoke.yaml` — 9B Soil_v2_Smoke 평가용

### 4. 프롬프트 통일

Qwen3.5 전 모델 fire/smoke CLS 프롬프트를 간결한 형태로 통일.

## 평가 결과

| 모델 | 벤치마크 | pred=0 | pred=1 | 소요시간 | 제출 |
|------|----------|--------|--------|----------|------|
| 0.8B | Soil_v2_Fire | 142,055 | 10,471 | 19.1분 | 성공 |
| 0.8B | Soil_v2_Smoke | 110,083 | 42,443 | 18.6분 | 성공 |
| 2B | Soil_v2_Fire | 143,234 | 9,292 | 25.0분 | 성공 |
| 2B | Soil_v2_Smoke | (기존 결과 유지) | 18,515 | - | 기제출 |
| 4B | Soil_v2_Fire | 136,415 | 16,111 | 31.1분 | 성공 |
| 4B | Soil_v2_Smoke | 115,941 | 36,585 | 35.4분 | 성공 |
| 9B | Soil_v2_Fire | 136,533 | 15,993 | 46.0분 | 성공 |
| 9B | Soil_v2_Smoke | 123,015 | 29,511 | 52.2분 | 성공 |

## 추가 발견 (참고)

- Qwen3.5-4B 모델 weight가 미다운로드 상태 → HuggingFace xet CAS 서버 + root 권한 캐시 충돌이 원인 → 수동 다운로드로 해결
- 9B Docker 기동 시 HuggingFace 네트워크 타임아웃 발생 → `HF_HUB_OFFLINE=1` 환경변수로 임시 해결 (코드에는 미반영, 호환성 이슈)
