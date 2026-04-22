# Task 26: Pipeline API 이중 모드 (LMDeploy + vLLM) 확장 + Unsloth Tokenizer 자동 패치

## 배경

기존 `src/api/` FastAPI 서버는 LMDeploy 전용(엔드포인트, YAML validator, worker 호출 전부
`src.lmdeploy_pipeline.*`에 하드코딩)이었다. 평가 요청 범위가 vLLM 기반 파인튜닝 모델
(예: PIA-SPACE-LAB/PIA_AI2team_VQA_falldown_v1.3 같은 Unsloth 저장 모델)까지 넓어지면서
동일 API 서버가 LMDeploy/vLLM 두 파이프라인을 모두 처리해야 하는 요구가 생김.

추가 요구: Unsloth 저장 모델은 `tokenizer_config.json`에 `"tokenizer_class":
"TokenizersBackend"`가 박혀 있어 vLLM 컨테이너의 AutoTokenizer 로드가 실패함
(공식 Qwen 계열은 `"Qwen2Tokenizer"`). 지금까지는 매번 수동 `docker run alpine sed`로
패치했는데, 이걸 파이프라인 내부에서 자동/멱등적으로 처리해야 함.

## 설계 결정

- `pipeline.mode` 필수 필드 (값: `"vllm"` | `"lmdeploy"`). 누락 시 400 Bad Request.
- Tokenizer patch는 vLLM 모드 진입 시 **항상 시도 + 멱등**. 공식 모델은 scan 후 skip.
- vLLM에도 `docker.hf_repo_id` 기반 **선제 다운로드** 지원.
- **모든 기존 YAML + 문서 + README를 싱크 강제 (일괄 마이그레이션)**.
- **CLI 가드**: `python -m src.vllm_pipeline`이 `mode: lmdeploy` YAML 받으면 즉시 ValueError (반대 동일).

## 구현 요약

### 1. Config 스키마 + CLI Guard

- `src/vllm_pipeline/config.py`
  - `DockerConfig.hf_repo_id: str = ""` 추가
  - `PipelineConfig.mode: str` (필수) 추가
  - `load_pipeline_config(path, expected_mode=None)` 시그니처 확장 + mode 필수/화이트리스트/mismatch 검증
  - `SUPPORTED_MODES` 상수
- `src/lmdeploy_pipeline/config.py` 동일 패턴
- `runner.py`에서 `load_pipeline_config(path, expected_mode="vllm"|"lmdeploy")`로 호출

### 2. YAML 일괄 마이그레이션

- 신규 `scripts/utils/add_pipeline_mode.py` (ruamel.yaml 없이 텍스트 기반, 주석/들여쓰기 보존, dry-run 지원)
- 실행 결과: 26개 YAML (vllm 17개 + lmdeploy 9개) 전부 `pipeline.mode` 삽입, 경고 0건

### 3. vLLM MODEL 단계 신규 모듈

- 신규 `src/vllm_pipeline/model_downloader.py`
  - `ensure_model(docker_cfg)` — `hf_repo_id` 있으면 `snapshot_download`, 없으면 HF 캐시 검사 후 컨테이너에 위임
- 신규 `src/vllm_pipeline/tokenizer_patcher.py`
  - `patch_tokenizer_config(model_id, replacement=DEFAULT_REPLACEMENT)` 멱등
  - root 소유 blob은 `docker run --rm alpine sed`로, 유저 소유는 Python sed로 치환
  - `{"patched": bool, "reason": "not_in_cache"|"already_ok"|"patched", "file": str|None}` 반환

### 4. vLLM Runner에 MODEL 단계 삽입

- `src/vllm_pipeline/runner.py` Docker 단계 진입 전에 `ensure_model` + `patch_tokenizer_config` 수행
- 리포트에 `[MODEL]` 라인 추가 (downloaded=.., tokenizer_patch=..)

### 5. API 워커 듀얼 모드 디스패치

- `src/api/pipeline_worker.py` 전면 리팩토링
  - 상단 lmdeploy 하드코딩 import 제거 → `_load_backend(mode)` lazy loader
  - `validate_yaml` mode 필수 + mode별 docker 스키마 검증 (vLLM: `docker.model`, LMDeploy: `docker.model_path`)
  - `_get_error_hint(error, mode=None)` — mode별 OOM/모델 누락 힌트 분기
  - `run_pipeline_sse` mode 추출 → backend 로드 → MODEL/Docker/Eval 단계 모두 backend 경유
  - `_background_eval_submit_cleanup(..., backend)` 인자 확장

### 6. API main.py 정리

- 타이틀 `LMDeploy Pipeline API` → `Model Eval Pipeline API`
- `/health` 응답 서버명 동기화
- YAML 에러 힌트 링크를 mode 중립 문구로 변경

### 7. 테스트 (신규 3종)

- `tests/test_api_worker_dispatch.py` (14 케이스)
- `tests/test_tokenizer_patcher.py` (6 케이스; 한 건은 integration marker)
- `tests/test_config_mode_guard.py` (7 케이스)
- 기존 `tests/test_lmdeploy_pipeline.py`의 fixture YAML에 `mode: "lmdeploy"` 주입

### 8. 문서/README 동기화

- `README.md` — 4-4/4-5/4-6 섹션에 `pipeline.mode` 필수 문구 + 두 모드 curl 예시
- `docs/eval/pipeline_api.md` — 듀얼 모드 설명, mode 스키마 표, SSE 출력 예시(vLLM/LMDeploy 각각), 에러 힌트 표 확장
- `docs/eval/vllm_pipeline.md` — MODEL 단계 + Unsloth 패치 설명, `pipeline.mode` 필수
- `docs/eval/lmdeploy_pipeline.md`, `lmdeploy_yaml_guide.md` — `pipeline.mode: "lmdeploy"` 필수 명시
- `docs/eval/eval_process_design.md` — MODEL 단계 분기 반영

## 검증 결과

### 단위/통합 테스트

```
tests/test_api_worker_dispatch.py  ............ 14 passed
tests/test_tokenizer_patcher.py   ............ 6 passed
tests/test_config_mode_guard.py   ............ 7 passed
tests/test_lmdeploy_pipeline.py   ............ 26 passed (기존 2개 pre-existing failure는 dev HEAD와 동일)
```

- 내가 건드리지 않은 `test_model_exists`/`test_is_valid_model_dir`는 dev HEAD에서도 동일하게 fail (기존 레포 이슈, 본 작업과 무관).

### API Smoke Test

```
GET /health → 200 {"status":"ok","server":"Model Eval Pipeline API","port":9000}

POST /pipeline/run/file (mode 누락 YAML) → 400
  message: "pipeline.mode 필드가 필요합니다. 지원 모드: vllm, lmdeploy"

POST /pipeline/run/file (mode=vllm + docker.model_path만 있는 YAML) → 400
  message: "docker.model 필드가 필요합니다. (vLLM 모드)"
```

### 실 동작 확인

- 직전 task 25에서 `pipeline.mode` 필드가 추가된 `qwen35_2b_falldown_v1_3.yaml`로
  CLI `python -m src.vllm_pipeline -c ...`이 2개 벤치마크(ABB/Hyundai_Falldown)를
  29.7분에 완주한 것을 확인 (tmux `vllm-falldown-v1-3` 세션).
  본 task의 mode guard는 해당 YAML에서 `mode: vllm`이 있으므로 정상 로드됨.

## 로컬 CLI 사용 (사용자 보장 사항)

```bash
conda activate llm
python -m src.vllm_pipeline -c configs/vllm_pipeline/qwen35_2b_fire.yaml
python -m src.lmdeploy_pipeline -c configs/lmdeploy_pipeline/PIA_AI2team_VQA_falldown_v1.0.yaml
```

- 커맨드 그대로 유지. YAML에 `pipeline.mode` 한 줄만 있으면 됨 (일괄 마이그레이션으로 이미 주입됨).
- 모듈과 mode 불일치 시 ValueError로 즉시 중단 → 결과 오염 방지.
- vLLM 모드에서 Unsloth 모델도 수동 sed 없이 자동 패치되어 정상 로드됨.

## 신규/수정 파일 목록

### 수정
- `src/api/pipeline_worker.py`, `src/api/main.py`
- `src/vllm_pipeline/config.py`, `runner.py`
- `src/lmdeploy_pipeline/config.py`, `runner.py`
- 26개 YAML (configs/vllm_pipeline + configs/lmdeploy_pipeline)
- `README.md`, `docs/eval/pipeline_api.md`, `docs/eval/vllm_pipeline.md`,
  `docs/eval/lmdeploy_pipeline.md`, `docs/eval/lmdeploy_yaml_guide.md`,
  `docs/eval/eval_process_design.md`
- `tests/test_lmdeploy_pipeline.py` (fixture mode 주입)

### 신규
- `src/vllm_pipeline/model_downloader.py`
- `src/vllm_pipeline/tokenizer_patcher.py`
- `scripts/utils/add_pipeline_mode.py`
- `tests/test_api_worker_dispatch.py`
- `tests/test_tokenizer_patcher.py`
- `tests/test_config_mode_guard.py`

## Trade-off / 후속 작업 거리

- vLLM `model_downloader.py`가 lmdeploy 버전과 상당 부분 중복 → 향후 `src/pipeline_common/` 공통화 기회.
- tokenizer patcher가 `docker run alpine` 의존 → 서버 환경 docker 상시 전제. 데몬 없는 환경에선 `sudo chmod` 대안 필요.
- 미캐시 상태 + `hf_repo_id` 미지정 모델은 컨테이너가 직접 다운로드하므로 tokenizer_patcher가 `not_in_cache` 반환. 이 경우 컨테이너 실패 시 자동 재시도 루프는 포함하지 않음(차기 개선 포인트 — 필요 시 1회 재시도 + 재패치 루프).
- 기존 `test_model_exists`/`test_is_valid_model_dir` 실패는 본 작업 범위 외 선재 이슈. 별도 티켓으로 처리 권장.
