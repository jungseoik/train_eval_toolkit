# LMDeploy 벤치마크 평가 파이프라인

파인튜닝 완료된 InternVL3 계열 로컬 모델의 최종 벤치마크 평가를 자동화하는 파이프라인입니다.
테스트셋 평가(Precision/Recall/F1)와는 별개 프로세스이며, PoC 리더보드 벤치마크 테스트에 사용합니다.

## 목차

- [동작 흐름](#동작-흐름)
- [빠른 시작](#빠른-시작)
- [단계별 실행](#단계별-실행)
- [YAML 설정 레퍼런스](#yaml-설정-레퍼런스)
- [모델 교체 가이드](#모델-교체-가이드)
- [vLLM 파이프라인과의 차이점](#vllm-파이프라인과의-차이점)
- [에러 대응 가이드](#에러-대응-가이드)
- [파일 구조](#파일-구조)

---

## 동작 흐름

```
Docker 컨테이너 기동 (LMDeploy API 서버)
        ↓
벤치마크 평가 (프레임 단위 추론)
        ↓
결과 제출 (Gradio 리더보드)
        ↓
컨테이너 정리
```

---

## 빠른 시작

```bash
conda activate llm

# 전체 파이프라인 실행
python -m src.lmdeploy_pipeline -c configs/lmdeploy_pipeline/internvl3_2b_fire.yaml
```

---

## 단계별 실행

`--steps` 인자로 특정 단계만 선택해서 실행할 수 있습니다.

```bash
# Docker만 기동
python -m src.lmdeploy_pipeline -c configs/lmdeploy_pipeline/internvl3_2b_fire.yaml --steps docker

# 평가 후 결과 제출 (Docker 이미 기동 상태)
python -m src.lmdeploy_pipeline -c configs/lmdeploy_pipeline/internvl3_2b_fire.yaml --steps evaluate submit
```

| `--steps` 인자 | 동작 |
|---|---|
| `docker` | Docker만 기동 |
| `evaluate` | 평가만 실행 (Docker 이미 기동 상태에서 사용) |
| `evaluate submit` | 평가 후 결과 제출 |
| 미지정 | YAML의 `pipeline.steps` 설정을 따름 |

---

## YAML 설정 레퍼런스

### pipeline 섹션

| 키 | 타입 | 기본값 | 설명 |
|---|---|---|---|
| `pipeline.name` | str | 필수 | 파이프라인 이름 |
| `pipeline.steps.docker` | bool | `true` | Docker 단계 실행 여부 |
| `pipeline.steps.evaluate` | bool | `true` | 평가 단계 실행 여부 |
| `pipeline.steps.submit` | bool | `true` | 제출 단계 실행 여부 |
| `pipeline.cleanup_docker` | bool | `true` | 종료 후 Docker 자동 제거 여부 |

### retry 섹션

| 키 | 타입 | 기본값 | 설명 |
|---|---|---|---|
| `retry.max_attempts` | int | `3` | 최대 재시도 횟수 |
| `retry.wait_seconds` | int | `30` | 재시도 대기 시간 (초) |

### docker 섹션

| 키 | 타입 | 기본값 | 설명 |
|---|---|---|---|
| `docker.container_name` | str | 필수 | Docker 컨테이너 이름 |
| `docker.image` | str | 필수 | LMDeploy Docker 이미지 |
| `docker.model_path` | str | 필수 | **로컬 파인튜닝 모델 경로** |
| `docker.container_model_path` | str | `"/model"` | 컨테이너 내부 마운트 위치 |
| `docker.gpus` | str | `"all"` | GPU 할당 |
| `docker.port` | int | `23333` | 호스트 포트 (LMDeploy 기본값) |
| `docker.ipc` | str | `"host"` | IPC 모드 |
| `docker.volumes` | list | `[]` | 추가 볼륨 마운트 |
| `docker.lmdeploy_args` | dict | `{}` | LMDeploy 서버 추가 인자 (`tp`, `session-len`, `backend` 등) |
| `docker.startup.timeout_seconds` | int | `300` | 서버 준비 대기 최대 시간 (초) |
| `docker.startup.poll_interval_seconds` | int | `5` | health check 폴링 간격 (초) |
| `docker.startup.stream_logs` | bool | `true` | 대기 중 Docker 로그 실시간 출력 여부 |

### evaluate 섹션

| 키 | 타입 | 기본값 | 설명 |
|---|---|---|---|
| `evaluate.model` | str | 필수 | LMDeploy API에서 사용하는 모델명 |
| `evaluate.run_name` | str | 필수 | 결과 디렉토리명 |
| `evaluate.api_base` | str | `"http://127.0.0.1:23333/v1"` | API 엔드포인트 |
| `evaluate.bench_base_path` | str | 필수 | 벤치마크 데이터셋 루트 경로 |
| `evaluate.output_path` | str | 필수 | 평가 결과 저장 경로 |
| `evaluate.benchmarks` | list | `[]` | 평가 벤치마크 목록 |
| `evaluate.window_size` | int | `15` | 프레임 샘플링 간격 |
| `evaluate.interpolation` | str | `"forward"` | 미샘플 프레임 채우기 방식 |
| `evaluate.concurrency` | int | `10` | 동시 요청 수 |
| `evaluate.jpeg_quality` | int | `95` | JPEG 인코딩 품질 |
| `evaluate.max_tokens` | int | `15` | 최대 생성 토큰 수 |
| `evaluate.temperature` | float | `0.0` | 샘플링 온도 |
| `evaluate.seed` | int | `0` | 재현성을 위한 시드값 |
| `evaluate.negative_label` | str | `"normal"` | 음성(Negative) 클래스 레이블 |
| `evaluate.prompt_templates` | dict | `{}` | 카테고리별 프롬프트 |

### submit 섹션

| 키 | 타입 | 기본값 | 설명 |
|---|---|---|---|
| `submit.gradio_url` | str | 필수 | Gradio 제출 서버 URL |
| `submit.model_name` | str | 필수 | 리더보드 표시 모델 이름 |
| `submit.task_name` | str | 필수 | 제출 시 태스크 이름 |
| `submit.datasets_used` | str | 필수 | 사용 데이터셋 구분 |
| `submit.config_file` | str | `"config.json"` | 제출 시 첨부할 설정 파일 |
| `submit.results_base_dir` | str | 필수 | 결과 CSV 기본 디렉토리 |
| `submit.interval_seconds` | int | `60` | 벤치마크 간 제출 간격 (초) |

---

## 모델 교체 가이드

기존 YAML 파일을 복사한 후 아래 필드만 수정합니다.

| 필드 | 변경 내용 |
|---|---|
| `docker.model_path` | 파인튜닝 모델 로컬 경로 |
| `docker.container_name` | 고유한 컨테이너 이름 |
| `evaluate.run_name` | 결과 디렉토리명 |
| `evaluate.model` | 컨테이너 내부 모델 경로 (보통 `/model` 고정) |
| `submit.model_name` | 리더보드 표시 모델명 |

```bash
# 예시: 새 모델용 설정 파일 생성
cp configs/lmdeploy_pipeline/internvl3_2b_fire.yaml \
   configs/lmdeploy_pipeline/internvl3_8b_fire.yaml
```

---

## vLLM 파이프라인과의 차이점

| 항목 | vLLM | LMDeploy |
|---|---|---|
| 백엔드 | vLLM | LMDeploy (TurboMind / PyTorch) |
| 모델 소스 | HuggingFace 모델 ID | **로컬 파인튜닝 모델 경로** |
| Docker 이미지 | `vllm/vllm-openai` | `openmmlab/lmdeploy` |
| 기본 포트 | 8000 | 23333 |
| 주요 지원 모델 | Qwen, LLaMA 등 | InternVL3, InternVL3.5 등 |

---

## 에러 대응 가이드

| 증상 | 원인 | 대응 |
|---|---|---|
| Docker OOM | GPU 메모리 부족 | `lmdeploy_args.tp` 증가 또는 `max-batch-size` 축소 |
| 포트 충돌 | 이미 사용 중인 포트 | `docker.port` 변경 또는 기존 컨테이너 정리 |
| 모델 로딩 실패 | 경로 오류 또는 모델 파일 손상 | `model_path` 확인, `config.json` 및 `model.safetensors` 존재 여부 점검 |
| Gradio 서버 다운 | 네트워크 또는 서버 일시 장애 | retry가 자동 처리, `retry` 설정 조정 |

---

## 파일 구조

```
src/lmdeploy_pipeline/
    __init__.py          # 패키지 초기화
    __main__.py          # python -m src.lmdeploy_pipeline 진입점
    cli.py               # CLI 진입점
    config.py            # YAML 파싱 및 dataclass 정의
    docker_manager.py    # LMDeploy Docker 생명주기 관리
    evaluator.py         # 평가 래퍼
    submitter.py         # Gradio 제출
    runner.py            # 파이프라인 오케스트레이터

src/evaluation/
    lmdeploy_bench_eval.py  # 프레임 단위 벤치마크 평가

configs/lmdeploy_pipeline/
    internvl3_2b_fire.yaml  # InternVL3-2B Fire 설정 예시

configs/lmdeploy_eval/
    config.py               # standalone 평가 설정
```
