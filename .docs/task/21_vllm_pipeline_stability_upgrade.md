# 21. vLLM 파이프라인 안정성 개선

## 개요
vLLM 파이프라인의 에러 처리, 평가 모드, 진행도 추적을 LMDeploy 파이프라인 수준으로 통일.
모델 사전 검증(model_downloader)은 vLLM 유즈케이스(HF 공개 모델)에 불필요하므로 제외.

## 변경 파일

### 1. `src/evaluation/vllm_bench_eval.py`
| 항목 | Before | After |
|---|---|---|
| 예외 클래스 | `InferenceAbortError` (프레임 단위 abort) | `BenchmarkSkipError` (벤치마크 단위 skip) |
| 추론 실패 처리 | 1회 재시도 → 실패 시 벤치마크 전체 중단 | pred=0 fallback, 경고 출력 후 계속 진행 |
| eval_mode | json만 지원 | json + cls (yes/no) 지원 |
| 벤치마크 경로 없음 | `print("[SKIP]")` + silent return | `raise BenchmarkSkipError` |
| 비디오 페어 없음 | `print("[SKIP]")` + silent return | `raise BenchmarkSkipError` |
| `_evaluate_video_async` | progress 파라미터 없음 | progress_state, bench_idx, video_done, video_total 추가 |
| 프레임 진행도 | 추적 없음 | 10프레임마다 `_update_video_progress` 호출 |

### 2. `src/vllm_pipeline/evaluator.py`
| 항목 | Before | After |
|---|---|---|
| import | `InferenceAbortError` | `BenchmarkSkipError` |
| 예외 핸들링 | abort 메시지 (bench/video/frame/cause) | skip 메시지 |
| EVAL_MODE | SimpleNamespace에 미포함 | 포함 |
| progress 관리 | 인라인 코드 (중복) | `_update_bench_progress()` 헬퍼 |

### 3. `src/vllm_pipeline/config.py`
- `EvalConfig`에 `eval_mode: str = "json"` 필드 추가
- `load_pipeline_config()`에서 YAML `eval_mode` 로딩 추가

## 설계 판단
- vLLM은 범용 모델 여러 개를 비교 평가하는 유즈케이스 → 하나가 실패해도 나머지를 계속 돌려야 함
- LMDeploy의 관대한 에러 처리(pred=0 fallback)가 이 유즈케이스에 더 적합
- `InferenceAbortError`의 상세 정보(어떤 프레임에서 실패)는 디버깅에 유용하나, 전체 중단은 과도함

## 테스트
- 코드 레벨 변경으로, 기존 YAML 설정과 호환 (eval_mode 미지정 시 기본값 "json")
- LMDeploy 파이프라인과 동일한 에러 처리 흐름 확인
