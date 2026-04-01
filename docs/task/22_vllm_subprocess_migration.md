# 22. vLLM 파이프라인 subprocess 분리 마이그레이션

## 배경

LMDeploy 파이프라인에서 발견된 메모리 누적 문제(#20)와 Docker 재시작 기능(#21)을
vLLM 파이프라인에도 동일하게 적용. 두 파이프라인은 동일한 프레임 처리 패턴
(asyncio + httpx + cv2)을 사용하므로 같은 메모리 문제가 발생한다.

## 변경 파일

| 파일 | 변경 내용 |
|------|-----------|
| `src/vllm_pipeline/config.py` | PipelineConfig에 `docker_restart_interval` 추가 |
| `src/vllm_pipeline/evaluator.py` | subprocess 분리 + Docker 재시작 (lmdeploy 구조 이식) |
| `src/vllm_pipeline/runner.py` | run_evaluation에 docker_cfg, docker_restart_interval 전달 |
| `src/evaluation/vllm_bench_eval.py` | `_update_video_progress` + `progress_state` 파라미터 추가 |
| `configs/vllm_pipeline/*.yaml` (전체) | `docker_restart_interval: 1` 추가 |
| `docs/eval/vllm_pipeline.md` | subprocess 분리, Docker 재시작 설명 추가 |
| `docs/eval/eval_process_design.md` | vLLM 적용 범위 명시 |
| `README.md` | vLLM 파이프라인 설명 업데이트 |

## 테스트 결과

Soil_Falldown(77개 영상) × 2회, concurrency=100, subprocess isolation

```
부모 RSS: 17MB → 18MB (+1MB)
결과: 2/2 성공, CSV 77개 생성
소요: 687s
잔여물: Docker + CSV + 스크립트 모두 정리
✓ vllm subprocess 평가 정상 동작
```

## 하위호환

- `evaluate_benchmark(bench_name, cfg)` 기존 호출도 정상 동작 (progress_state=None 기본값)
- CLI 직접 실행 (`python -m src.vllm_pipeline`) 영향 없음
- YAML에 `docker_restart_interval`이 없으면 기본값 1 적용
