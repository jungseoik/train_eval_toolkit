# Task 12: HuggingFace 모델 자동 다운로드 기능 추가

## 개요

LMDeploy 파이프라인 실행 시 로컬에 모델이 없으면 HuggingFace Hub에서 자동 다운로드하는 기능을 추가했다.

## 배경

기존에는 모델이 `ckpts/` 경로에 미리 존재해야만 파이프라인을 실행할 수 있었다. 모델 다운로드는 문서에 수동 명령어로만 안내되어 있었으며, 파이프라인 자체에서 모델 존재 여부를 확인하거나 자동으로 준비하는 로직이 없었다.

## 변경 내용

### 신규 파일

| 파일 | 설명 |
|------|------|
| `src/lmdeploy_pipeline/model_downloader.py` | 모델 존재 확인 및 HuggingFace 다운로드 모듈 |

### 수정 파일

| 파일 | 변경 내용 |
|------|-----------|
| `src/lmdeploy_pipeline/config.py` | `DockerConfig`에 `hf_repo_id` 필드 추가 |
| `src/lmdeploy_pipeline/runner.py` | Docker 단계 전에 모델 체크/다운로드 단계 삽입, 리포트에 MODEL 항목 추가 |
| `configs/lmdeploy_pipeline/internvl3_2b_fire.yaml` | `hf_repo_id` 주석 예시 추가 |
| `configs/lmdeploy_pipeline/InternVL3-1B_falldown.yaml` | `hf_repo_id` 주석 예시 추가 |
| `configs/lmdeploy_pipeline/InternVL3-2B_hyundai_8_20_falldown.yaml` | `hf_repo_id` 주석 예시 추가 |
| `docs/eval/lmdeploy_pipeline.md` | 사전 준비 섹션 추가, 동작 흐름 업데이트, docker 설정 테이블에 hf_repo_id 추가 |
| `README.md` | LMDeploy 섹션에 자동 다운로드 설명 추가 |
| `tests/test_lmdeploy_pipeline.py` | model_downloader 테스트 7개 추가 |

## 동작 흐름

```
파이프라인 시작
    ↓
[MODEL] model_path에 유효한 모델(config.json) 존재?
    ├─ YES → 그대로 진행
    └─ NO → hf_repo_id 설정됨?
         ├─ YES → HuggingFace에서 model_path로 다운로드 → 진행
         └─ NO → 에러 메시지 출력 후 파이프라인 중단
    ↓
[DOCKER] 컨테이너 기동
    ↓
...
```

## YAML 설정 예시

```yaml
docker:
  model_path: "ckpts/PIA_AI2team_VQA_falldown"
  hf_repo_id: "PIA-SPACE-LAB/PIA_AI2team_VQA_falldown"
```

## 테스트 결과

- 전체 28개 테스트 통과 (기존 21개 + 신규 7개)
- 신규 테스트: import, 모델 존재 확인, 미존재+hf_repo_id 없음, 미존재+hf_repo_id 있음, 유효성 검증, config 기본값, config 설정값

## 작성자

jungseoik
