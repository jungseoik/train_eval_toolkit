# Task 27: Tokenizer Patcher model_type 기반 allowlist 가드

## 배경

Task 26에서 도입된 `src/vllm_pipeline/tokenizer_patcher.py`는 HF 캐시의 `tokenizer_config.json`에 `"tokenizer_class": "TokenizersBackend"`가 발견되면 무조건 `"Qwen2Tokenizer"`로 치환하는 구조였다. 현재 관측된 케이스는 Qwen3.5 base의 Unsloth 파인튜닝 하나뿐이지만, 이론상 다른 base model(Llama, Mistral 등)에서 같은 `TokenizersBackend` 저장 현상이 발생할 경우 잘못된 tokenizer class로 덮어씌워 무음 실패를 일으킬 리스크가 있었다.

→ "허용된 model_type만 건드린다"는 안전 기본값 allowlist 구조로 강화.

## 변경 요약

### `src/vllm_pipeline/tokenizer_patcher.py`

- `MODEL_TYPE_TO_TOKENIZER: dict[str, str]` 모듈 상수 도입. 현재 매핑: `{"qwen3_5": "Qwen2Tokenizer"}`.
- `_resolve_model_type(model_id)` 헬퍼 추가. 같은 snapshot의 `config.json`에서 `model_type` 필드를 읽어 반환. 캐시 누락 / JSON 파싱 실패 시 `None`.
- `patch_tokenizer_config(model_id, model_type_to_tokenizer=None)` 시그니처 변경:
  - 기존 `replacement: str` 인자 제거 (매핑으로 대체).
  - `model_type_to_tokenizer` 선택 인자로 allowlist 덮어쓰기(테스트 / 긴급 대응용).
- 동작:
  1. blob 없음 → `not_in_cache`
  2. `TokenizersBackend` 없음 → `already_ok`
  3. 있지만 `model_type` 판독 실패 → `model_type_unknown` (파일 미변경)
  4. 있지만 allowlist에 없음 → `unsupported_model_type` (파일 미변경)
  5. 허용됨 → 매핑된 tokenizer_class로 치환 → `patched`
- 반환 dict에 `model_type`, `tokenizer_class` 키 추가. SSE 이벤트에 노출 가능.

### `src/api/pipeline_worker.py`

- `_model_step` 예외 fallback 스키마에 `model_type`/`tokenizer_class: None` 추가.
- SSE `model_ready` 이벤트 payload에 `model_type`, `tokenizer_class` 필드 포함.
- `summary` 문자열에 `model_type`이 있으면 덧붙임.

### `tests/test_tokenizer_patcher.py`

기존 6 케이스 → 9 케이스. 주요 신규:
- `test_unsupported_model_type_does_not_patch` (Llama 같은 비허용 model_type 건드리지 않음 검증)
- `test_model_type_unknown_does_not_patch` (config.json 판독 실패 시 safe skip)
- `test_custom_mapping_override` (긴급 대응용 override 인자 검증)
- `test_default_mapping_contains_qwen3_5` (기본 allowlist 명세 회귀 방지)
- 기존 Unsloth 패치 테스트에 `_resolve_model_type="qwen3_5"` 모킹 추가.

### `docs/eval/vllm_pipeline.md`

- MODEL 단계 섹션에 allowlist 기반 동작과 확장 방법 명시.
- 비-qwen3_5 모델은 건드리지 않음을 명확히 기술.

## 검증 결과

### pytest

```
tests/test_tokenizer_patcher.py     ............ 9 passed
tests/test_api_worker_dispatch.py   ............ 14 passed
tests/test_config_mode_guard.py    ............ 7 passed
```

### 실제 캐시 스모크

```
PIA v1.3 (qwen3_5, 이미 패치됨)   → already_ok, 파일 미변경
Qwen/Qwen3.5-2B (공식 qwen3_5)     → already_ok, 파일 미변경
nvidia/Cosmos-Reason2-2B            → not_in_cache (캐시에 없음)
nonexistent/xyz123                  → not_in_cache
```

false positive 없음 확인.

## API 요청 관점 영향

기존 API 엔드포인트/요청 방식 변화 없음. SSE `model_ready` 페이로드에 `model_type`/`tokenizer_class`가 추가되어 운영자가 로그에서 어떤 매핑이 적용됐는지 확인 가능. 기존 클라이언트는 추가 필드를 무시하면 되므로 backward-compat.

## 확장 방법 (향후)

다른 base 모델군에서 같은 이슈가 발견되면:

```python
# src/vllm_pipeline/tokenizer_patcher.py
MODEL_TYPE_TO_TOKENIZER = {
    "qwen3_5": "Qwen2Tokenizer",
    "llama3":  "LlamaTokenizerFast",    # 예시 — 관측 시 한 줄 추가
}
```

코드 한 줄 수정 + 테스트 케이스 하나 추가로 끝.

## 수정/신규 파일

- 수정: `src/vllm_pipeline/tokenizer_patcher.py`, `src/api/pipeline_worker.py`, `tests/test_tokenizer_patcher.py`, `docs/eval/vllm_pipeline.md`
- 신규: `.docs/task/27_tokenizer_patcher_model_type_guard.md`
