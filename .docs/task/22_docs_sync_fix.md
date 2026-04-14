# TASK 22: 문서-코드 싱크 수정

## 개요

vLLM pipeline, lmdeploy pipeline의 README, docs, YAML 설정 파일을 전수 점검하여 문서와 실제 코드/설정 간 불일치를 수정.

## 수정 항목

### 1. lmdeploy YAML 파일명 불일치 수정 (HIGH)

파일명이 `internvl3_2b_fire.yaml` → `InternVL3-2B_fire.yaml`로 변경되었으나 문서에 옛 이름이 잔존.

| 파일 | 수정 위치 |
|------|-----------|
| `README.md` | line 260, 263, 300 |
| `docs/eval/lmdeploy_pipeline.md` | line 129, 140, 143, 245, 294 |

### 2. lmdeploy_pipeline.md pipeline 섹션 테이블 누락 필드 추가 (MEDIUM)

`pipeline.docker_restart_interval` 항목이 pipeline 섹션 레퍼런스 테이블에 빠져 있었음. vLLM 문서에는 있고, 실제 코드와 모든 YAML에서 사용 중이므로 추가.

### 3. vLLM YAML configs에 eval_mode 명시 (LOW)

lmdeploy YAML은 모두 `eval_mode: "json"`을 명시하고 있으나, vLLM YAML 4개는 기본값에 의존. 일관성을 위해 명시적으로 추가.

- `qwen35_2b_fire.yaml`
- `qwen35_08b_fire.yaml`
- `cosmos_reason2_2b_fire.yaml`
- `cosmos_reason2_2b_falldown.yaml`

### 4. vLLM CLI 실행 명령 통일 (LOW)

YAML 파일 헤더는 `python -m src.vllm_pipeline`을 사용하지만, README와 docs는 `python -m src.vllm_pipeline.cli`를 사용. `__main__.py`가 존재하므로 `.cli` 없는 짧은 형태로 통일.

- `README.md`: 3곳
- `docs/eval/vllm_pipeline.md`: 3곳

## 점검 후 정상 확인된 항목

- config.py dataclass 필드 vs 문서 레퍼런스 테이블: 일치
- eval_mode "json"/"cls" 코드 구현 vs 문서: 일치
- docker_restart_interval 코드 로직 vs 문서: 일치
- subprocess 분리 실행 구조: 일치
- retry/submit 설정: 일치

## 변경 파일 (7개)

```
README.md
configs/vllm_pipeline/cosmos_reason2_2b_falldown.yaml
configs/vllm_pipeline/cosmos_reason2_2b_fire.yaml
configs/vllm_pipeline/qwen35_08b_fire.yaml
configs/vllm_pipeline/qwen35_2b_fire.yaml
docs/eval/lmdeploy_pipeline.md
docs/eval/vllm_pipeline.md
```
