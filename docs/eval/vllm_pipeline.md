# vLLM 테스트 자동화 파이프라인 사용 문서

## 개요

Docker 컨테이너 실행, 벤치마크 평가, 결과 제출, 정리까지 이어지는 3단계 수동 프로세스를 YAML 설정 파일 하나로 자동화하는 파이프라인입니다. 모델만 교체하면 동일한 흐름으로 빠르게 테스트할 수 있습니다.

**자동화 흐름**: MODEL 단계 (HF 선제 다운로드 + Unsloth tokenizer 자동 패치) → Docker 컨테이너 기동 → vLLM 서버 준비 대기 → 벤치마크 평가 (벤치마크별 subprocess 분리 + Docker 재시작) → 결과 제출 → 컨테이너 정리

각 벤치마크는 독립 subprocess에서 실행되어 메모리 누적이 방지됩니다.
`docker_restart_interval: 1`(기본값)이면 매 벤치마크 사이에 Docker를 재시작하여 서버 측 메모리도 초기화합니다.

> 프로세스 설계 배경 및 상세 흐름도: [eval_process_design.md](eval_process_design.md)

## `pipeline.mode` 필수

이 파이프라인의 모든 YAML은 `pipeline.mode: "vllm"` 필드를 반드시 포함해야 합니다. 누락 또는 불일치 시 CLI가 즉시 ValueError로 중단합니다(예: `python -m src.vllm_pipeline`로 `mode: "lmdeploy"` YAML 실행 시 실패).

## MODEL 단계 (Unsloth 대응)

vLLM CLI/API 모두 Docker 기동 전에 다음을 수행합니다.

1. `docker.hf_repo_id`가 지정되어 있으면 `huggingface_hub.snapshot_download`로 HF 캐시에 선제 다운로드합니다.
2. 이후 `src/vllm_pipeline/tokenizer_patcher.py`가 캐시된 `tokenizer_config.json`을 스캔해 `"tokenizer_class": "TokenizersBackend"`가 있으면 `"Qwen2Tokenizer"`로 교체합니다. 공식 Qwen 계열은 `already_ok`로 skip(멱등). 실제 패치는 Docker 데몬을 통한 `docker run --rm alpine sed` 경로로 수행되어 root 소유 캐시 파일도 안전하게 처리합니다.

> Unsloth 파인튜닝 모델 tokenizer 이슈 상세: `.docs/task/26_api_dual_mode.md` 참조.

---

## 빠른 시작

```bash
# 1. 기준 템플릿 복사 후 수정
cp configs/vllm_pipeline/template/template.yaml configs/vllm_pipeline/{모델명}_{카테고리}.yaml

# 2. 전체 파이프라인 실행
conda activate llm
python -m src.vllm_pipeline -c configs/vllm_pipeline/qwen35_2b_fire.yaml

# 특정 단계만 실행
python -m src.vllm_pipeline -c configs/vllm_pipeline/qwen35_2b_fire.yaml --steps docker
python -m src.vllm_pipeline -c configs/vllm_pipeline/qwen35_2b_fire.yaml --steps evaluate submit
```

---

## YAML 설정 옵션 전체 레퍼런스

### pipeline 섹션

| 키 | 타입 | 기본값 | 설명 |
|----|------|--------|------|
| `pipeline.name` | str | 필수 | 파이프라인 이름 (로그에 표시) |
| `pipeline.mode` | str | 필수 | `"vllm"` 고정. `"lmdeploy"` 등을 쓰면 CLI/API가 거절 |
| `pipeline.steps.docker` | bool | true | Docker 컨테이너 관리 단계 실행 여부 |
| `pipeline.steps.evaluate` | bool | true | 벤치마크 평가 단계 실행 여부 |
| `pipeline.steps.submit` | bool | true | 결과 제출 단계 실행 여부 |
| `pipeline.cleanup_docker` | bool | true | 종료 후 Docker 컨테이너 자동 제거 (GPU 반환) |
| `pipeline.docker_restart_interval` | int | 1 | 벤치마크 N개마다 Docker 재시작하여 서버 메모리 해소 (0=비활성) |

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
| `docker.model` | str | 필수 | HuggingFace 모델 이름 (컨테이너가 직접 로드) |
| `docker.hf_repo_id` | str | `""` | 지정 시 MODEL 단계에서 `snapshot_download`로 선제 다운로드. 기본은 `docker.model`을 그대로 검사 |
| `docker.vllm_args` | dict | {} | vLLM 서버 추가 인자 |
| `docker.startup.timeout_seconds` | int | 300 | 서버 준비 대기 최대 시간 |
| `docker.startup.poll_interval_seconds` | int | 5 | health check 폴링 간격 |
| `docker.startup.stream_logs` | bool | true | 대기 중 Docker 로그 실시간 출력 |

### evaluate 섹션

| 키 | 타입 | 기본값 | 설명 |
|----|------|--------|------|
| `evaluate.benchmarks` | list | 필수 | 평가할 벤치마크 목록 |
| `evaluate.model` | str | 필수 | API 호출 시 모델 이름 |
| `evaluate.run_name` | str | 필수 | 결과 디렉토리명 |
| `evaluate.api_base` | str | `http://127.0.0.1:8000/v1` | vLLM 서버 API 주소 |
| `evaluate.bench_base_path` | str | 필수 | 벤치마크 데이터 루트 경로 |
| `evaluate.output_path` | str | 필수 | 평가 결과 저장 경로 |
| `evaluate.eval_mode` | str | `"json"` | 평가 모드: `"json"` (JSON 파싱) 또는 `"cls"` (yes/no 분류) |
| `evaluate.window_size` | int | 15 | 프레임 샘플링 간격 (Fire: 15, Falldown: 30) |
| `evaluate.interpolation` | str | `"forward"` | 인터폴레이션 방법 (`forward` / `backward`) |
| `evaluate.concurrency` | int | 10 | 동시 추론 요청 수 |
| `evaluate.jpeg_quality` | int | 95 | 프레임 JPEG 인코딩 품질 |
| `evaluate.max_tokens` | int | 15 | 최대 응답 토큰 수 (cls 모드 시 1 권장) |
| `evaluate.temperature` | float | 0.0 | 생성 온도 |
| `evaluate.seed` | int | 0 | 랜덤 시드 |
| `evaluate.negative_label` | str | `"normal"` | 미감지 레이블 |
| `evaluate.prompt_templates` | dict | {} | 카테고리별 프롬프트 템플릿 |
| `evaluate.overwrite_results` | bool | `true` | `true`: 항상 덮어쓰기, `false`: 기존 CSV 존재 + GT row 수 일치 시 스킵 |

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
| `submit.benchmark_fail_retry` | int | 2 | 벤치마크 실행 실패 시 총 시도 횟수 |
| `submit.benchmark_fail_wait` | int | 60 | 벤치마크 실패 재시도 전 대기 시간 (초) |

---

## 모델 교체 가이드

기존 YAML 파일을 복사한 후 아래 필드를 수정합니다.

| 필드 | 변경 내용 |
|------|-----------|
| `docker.model` | HuggingFace 모델 이름 |
| `docker.container_name` | 고유한 컨테이너 이름 |
| `docker.vllm_args` | 모델 크기에 맞게 조정 (`tensor-parallel-size` 등) |
| `evaluate.model` | 평가에 사용할 모델명 (`docker.model`과 동일) |
| `evaluate.run_name` | 결과 디렉토리명 |
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
| Gradio 서버 다운 | 네트워크/서버 일시 장애 | API 호출 에러는 `retry` 설정으로 자동 재시도. 벤치마크 실행 실패(예: Google Sheets 503)는 `submit.benchmark_fail_retry`(기본 2회) + `submit.benchmark_fail_wait`(기본 60초)로 자동 재시도 |
| 벤치마크 경로 없음 | 데이터 경로 누락 | `BenchmarkSkipError` 발생 → 해당 벤치마크를 skip하고 다음 벤치마크 진행 |
| 프레임 추론 실패 | 서버 일시 불안정 | 해당 프레임을 pred=0으로 처리하고 계속 진행 (벤치마크 전체가 중단되지 않음) |

---

## 파일 구조

```
src/vllm_pipeline/
    __init__.py          # 패키지 초기화
    __main__.py          # python -m src.vllm_pipeline 진입점
    config.py            # YAML 파싱 + dataclass
    docker_manager.py    # Docker 생명주기 관리
    evaluator.py         # 평가 래퍼 (벤치마크별 subprocess 분리 실행)
    submitter.py         # Gradio 제출
    runner.py            # 파이프라인 오케스트레이터
    cli.py               # CLI 진입점

configs/vllm_pipeline/
    qwen35_2b_fire.yaml  # Qwen3.5-2B Fire 설정
    qwen35_08b_fire.yaml # Qwen3.5-0.8B Fire 설정
```
