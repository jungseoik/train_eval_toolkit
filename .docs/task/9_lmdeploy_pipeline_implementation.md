# Task 9: LMDeploy 벤치마크 평가 파이프라인 구현

## 개요

파인튜닝 완료된 InternVL3 계열 로컬 모델의 최종 벤치마크 평가를 위한 LMDeploy 기반 파이프라인을 구현했다. vLLM 파이프라인과 동일한 동작 패턴(Docker -> 평가 -> 제출 -> 정리)이지만, 완전 독립 모듈로 분리하여 관리한다.

## 목적

- 파인튜닝 완료 후 PoC 리더보드 벤치마크 테스트 용도
- 테스트셋 평가(Precision/Recall/F1)와는 별개 프로세스
- InternVL3/3.5 파인튜닝 모델 전용

## 구현 내용

### 신규 파일 (14개)

| 파일 | 역할 |
|------|------|
| `src/lmdeploy_pipeline/__init__.py` | 패키지 초기화 |
| `src/lmdeploy_pipeline/__main__.py` | `python -m` 진입점 |
| `src/lmdeploy_pipeline/cli.py` | CLI 인터페이스 |
| `src/lmdeploy_pipeline/config.py` | LMDeploy 전용 dataclass + YAML 파싱 |
| `src/lmdeploy_pipeline/docker_manager.py` | Docker 생명주기 관리 |
| `src/lmdeploy_pipeline/evaluator.py` | 평가 래퍼 |
| `src/lmdeploy_pipeline/submitter.py` | Gradio 제출 |
| `src/lmdeploy_pipeline/runner.py` | 파이프라인 오케스트레이터 |
| `src/evaluation/lmdeploy_bench_eval.py` | 프레임 단위 벤치마크 평가 |
| `configs/lmdeploy_eval/config.py` | standalone 평가 설정 |
| `configs/lmdeploy_pipeline/internvl3_2b_fire.yaml` | 파이프라인 YAML 설정 |
| `tests/test_lmdeploy_pipeline.py` | 단위 테스트 (21개) |
| `docs/eval/lmdeploy_pipeline.md` | 사용 가이드 |

### 수정 파일

| 파일 | 변경 내용 |
|------|-----------|
| `README.md` | 평가 섹션에 LMDeploy 파이프라인 추가, 도구 매핑 테이블/프로젝트 구조 업데이트 |

## vLLM 파이프라인과의 주요 차이점

| 항목 | vLLM | LMDeploy |
|------|------|----------|
| Docker 이미지 | `vllm/vllm-openai:*` | `openmmlab/lmdeploy:latest-cu12` |
| 기본 포트 | 8000 | 23333 |
| 모델 소스 | HuggingFace ID | 로컬 경로 (볼륨 마운트) |
| 엔트리포인트 | 이미지 기본 | `lmdeploy serve api_server` |
| TP 인자 | `--tensor-parallel-size` | `--tp` |
| 컨텍스트 길이 | `--max-model-len` | `--session-len` |

## 테스트 결과

- 21개 단위 테스트 통과
- CLI 도움말 출력 확인
- YAML config 로드 정상 동작 확인
- Docker 명령어 조립 검증 완료

## 사용법

```bash
# 전체 파이프라인
python -m src.lmdeploy_pipeline -c configs/lmdeploy_pipeline/internvl3_2b_fire.yaml

# 특정 단계만
python -m src.lmdeploy_pipeline -c configs/lmdeploy_pipeline/internvl3_2b_fire.yaml --steps evaluate submit
```

## 향후 계획

- vLLM/LMDeploy 파이프라인 공통 부분 리팩토링 (공통 모듈 추출)
- InternVL3.5 모델 지원 검증
