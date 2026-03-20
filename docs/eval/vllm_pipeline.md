# vLLM 테스트 자동화 파이프라인 사용 문서

## 개요

Docker 컨테이너 실행, 벤치마크 평가, 결과 제출, 정리까지 이어지는 3단계 수동 프로세스를 YAML 설정 파일 하나로 자동화하는 파이프라인입니다. 모델만 교체하면 동일한 흐름으로 빠르게 테스트할 수 있습니다.

**자동화 흐름**: Docker 컨테이너 기동 → vLLM 서버 준비 대기 → 벤치마크 평가 → 결과 제출 → 컨테이너 정리

---

## 빠른 시작

```bash
# 전체 파이프라인 실행
conda activate llm
python -m src.vllm_pipeline.cli -c configs/vllm_pipeline/qwen35_2b_fire.yaml

# 특정 단계만 실행
python -m src.vllm_pipeline.cli -c configs/vllm_pipeline/qwen35_2b_fire.yaml --steps docker
python -m src.vllm_pipeline.cli -c configs/vllm_pipeline/qwen35_2b_fire.yaml --steps evaluate submit
```

---

## YAML 설정 옵션 전체 레퍼런스

### pipeline 섹션

| 키 | 타입 | 기본값 | 설명 |
|----|------|--------|------|
| `pipeline.name` | str | 필수 | 파이프라인 이름 (로그에 표시) |
| `pipeline.steps.docker` | bool | true | Docker 컨테이너 관리 단계 실행 여부 |
| `pipeline.steps.evaluate` | bool | true | 벤치마크 평가 단계 실행 여부 |
| `pipeline.steps.submit` | bool | true | 결과 제출 단계 실행 여부 |
| `pipeline.cleanup_docker` | bool | true | 종료 후 Docker 컨테이너 자동 제거 (GPU 반환) |

### retry 섹션

| 키 | 타입 | 기본값 | 설명 |
|----|------|--------|------|
| `retry.max_attempts` | int | 3 | 실패 시 최대 재시도 횟수 |
| `retry.wait_seconds` | int | 30 | 재시도 사이 대기 시간 (초) |

### docker 섹션

| 키 | 타입 | 기본값 | 설명 |
|----|------|--------|------|
| `docker.container_name` | str | 필수 | Docker 컨테이너 이름 |
| `docker.image` | str | 필수 | vLLM Docker 이미지 |
| `docker.gpus` | str | "all" | GPU 할당 |
| `docker.port` | int | 8000 | 호스트 포트 |
| `docker.ipc` | str | "host" | IPC 모드 |
| `docker.volumes` | list | [] | 볼륨 마운트 |
| `docker.model` | str | 필수 | HuggingFace 모델 이름 |
| `docker.vllm_args` | dict | {} | vLLM 서버 추가 인자 |
| `docker.startup.timeout_seconds` | int | 300 | 서버 준비 대기 최대 시간 |
| `docker.startup.poll_interval_seconds` | int | 5 | health check 폴링 간격 |
| `docker.startup.stream_logs` | bool | true | 대기 중 Docker 로그 실시간 출력 |

### evaluate 섹션

| 키 | 타입 | 기본값 | 설명 |
|----|------|--------|------|
| `evaluate.eval_config_path` | str | 필수 | 기존 config.py 경로 |
| `evaluate.benchmarks` | list | config.py의 BENCHMARKS | 평가할 벤치마크 목록 |
| `evaluate.overrides` | dict | {} | config.py 속성 런타임 덮어쓰기 |

### submit 섹션

| 키 | 타입 | 기본값 | 설명 |
|----|------|--------|------|
| `submit.gradio_url` | str | 필수 | Gradio 제출 서버 URL |
| `submit.model_name` | str | 필수 | 제출 시 모델 이름 |
| `submit.task_name` | str | 필수 | 제출 시 태스크 이름 |
| `submit.datasets_used` | str | 필수 | 사용 데이터셋 설명 |
| `submit.config_file` | str | "config.json" | 제출 시 첨부할 설정 파일 |
| `submit.results_base_dir` | str | 필수 | 결과 CSV 기본 디렉토리 |
| `submit.interval_seconds` | int | 60 | 벤치마크 간 제출 간격 (초) |

---

## 모델 교체 가이드

기존 YAML 파일을 복사한 후 아래 필드를 수정합니다.

| 필드 | 변경 내용 |
|------|-----------|
| `docker.model` | HuggingFace 모델 이름 |
| `docker.container_name` | 고유한 컨테이너 이름 |
| `docker.vllm_args` | 모델 크기에 맞게 조정 (`tensor-parallel-size` 등) |
| `evaluate.overrides.MODEL` | 평가에 사용할 모델명 |
| `evaluate.overrides.RUN_NAME` | 결과 디렉토리명 |
| `submit.model_name` | 리더보드에 표시될 모델명 |

---

## 단계별 실행

| `--steps` 인자 | 동작 |
|----------------|------|
| `docker` | Docker만 기동 (서버 준비 확인까지) |
| `evaluate` | 평가만 실행 (Docker가 이미 기동된 상태에서) |
| `evaluate submit` | 평가 후 결과 제출 |
| 미지정 | YAML의 `pipeline.steps` 설정을 따름 |

---

## 에러 대응 가이드

| 증상 | 원인 | 대응 방법 |
|------|------|-----------|
| Docker OOM | GPU 메모리 부족 | `docker.vllm_args`에서 `kv-cache-memory-bytes` 축소 또는 `tensor-parallel-size` 증가 |
| 포트 충돌 | 이미 사용 중인 포트 | `docker.port` 변경 또는 기존 컨테이너 정리 |
| Gradio 서버 다운 | 네트워크/서버 일시 장애 | retry가 자동 처리. `retry.max_attempts`와 `retry.wait_seconds` 조정 |
| 벤치마크 경로 없음 | 결과 파일 미생성 | 해당 벤치마크를 skip하고 다음 벤치마크 진행 |

---

## 파일 구조

```
src/vllm_pipeline/
    __init__.py          # 패키지 초기화
    __main__.py          # python -m src.vllm_pipeline 진입점
    config.py            # YAML 파싱 + dataclass
    docker_manager.py    # Docker 생명주기 관리
    evaluator.py         # 평가 래퍼 (기존 vllm_bench_eval.py 활용)
    submitter.py         # Gradio 제출
    runner.py            # 파이프라인 오케스트레이터
    cli.py               # CLI 진입점

configs/vllm_pipeline/
    qwen35_2b_fire.yaml  # Qwen3.5-2B Fire 설정
    qwen35_08b_fire.yaml # Qwen3.5-0.8B Fire 설정
```
