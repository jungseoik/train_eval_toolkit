# Task 16: YAML 검증/저장/에러 처리 개선

- 작성자: jungseoik
- 작성일: 2026-03-31
- 브랜치: `worktree-yaml-validation-error-handling`

---

## 배경

외부 내부망 클라이언트가 `POST /pipeline/run/file`로 YAML을 업로드했으나, `bench_base_path`가 `/workspace/PoC_banchmark/...`(오타+잘못된 경로)로 적혀있었다. 실제 서버 경로는 `/mnt/PoC_benchmark/...`이다.

13개 벤치마크 전부 `[SKIP] Benchmark path not found`가 발생했지만, API는 `"평가 완료: 13/13 벤치마크 성공"` + `status: "completed"`로 응답했다. 호출자는 성공으로 인지했고, 실제로는 아무것도 평가되지 않았다.

---

## 근본 원인

1. **YAML 경로 검증 없음** — `validate_yaml()`이 구문/키 존재만 확인하고, 경로 실존 여부를 검증하지 않음
2. **스킵 = 성공 처리** — `evaluate_benchmark()`가 경로 없으면 `print("[SKIP]")` + `return`만 하고 예외를 던지지 않아, 호출자(evaluator.py)가 성공으로 처리
3. **YAML 미보관** — 임시파일(`/tmp`)로 저장 후 삭제되어 사후 디버깅 불가

---

## 수정 내용

### 1. YAML 영구 저장

- **파일**: `src/api/pipeline_worker.py`
- `save_yaml_to_temp()` → `save_yaml_permanent()` 교체
- 저장 경로: `data/uploaded_yamls/{YYYYMMDD_HHMMSS}_{pipeline_name}.yaml`
- 파이프라인 완료 후에도 삭제하지 않음 (디버깅용 영구 보관)
- `_safe_remove()` 함수 제거

### 2. YAML 경로 검증 강화

- **파일**: `src/api/pipeline_worker.py`, `src/api/main.py`
- `validate_paths()` 함수 추가 — 4가지 검증:
  1. `bench_base_path` 존재 여부
  2. `benchmarks` 리스트 비어있는지
  3. 각 벤치마크 폴더 존재 여부 (missing 개수/목록 표시)
  4. `output_path` 쓰기 권한
- `run_from_file()`, `run_from_yaml()` 양쪽 엔드포인트에서 `validate_yaml()` 직후 호출
- 검증 실패 시 HTTP 400 + `validation_error` 상태 + 구체적 에러 메시지 + hint 반환

### 3. 에러 리턴 개선

- **파일**: `src/evaluation/lmdeploy_bench_eval.py`
  - `BenchmarkSkipError(RuntimeError)` 예외 클래스 추가
  - `evaluate_benchmark()` 내 3곳의 `print + return` → `raise BenchmarkSkipError` 변경
    - 벤치마크 경로 미존재
    - dataset 디렉토리 미존재
    - mp4/csv 쌍 없음

- **파일**: `src/lmdeploy_pipeline/evaluator.py`
  - `BenchmarkSkipError` import 추가
  - retry 루프에서 `BenchmarkSkipError`를 별도 catch — 재시도 없이 즉시 실패 처리 (경로 문제는 retry로 해결 불가)

- **파일**: `src/api/pipeline_worker.py`
  - `_background_eval_submit_cleanup()` 상태 처리 개선:
    - 전부 실패 → `status: "failed"` + 에러 메시지 + hint
    - 일부 실패 → `status: "completed"` + 실패 목록 표시
    - 전부 성공 → 기존 동작 유지

### 4. 문서 업데이트

- **`docs/eval/pipeline_api.md`**:
  - YAML 파일 보관 안내 섹션 추가
  - 경로 검증 에러 (HTTP 400) 응답 예시 추가
  - 에러 유형 테이블에 경로 관련 행 2개 추가
  - 전체 벤치마크 실패 시 `/pipeline/status` 응답 예시 추가

- **`docs/eval/lmdeploy_yaml_guide.md`**:
  - `bench_base_path` 필드에 경로 검증 경고 박스 추가

---

## 수정 파일 목록

| 파일 | 변경 유형 |
|------|-----------|
| `src/api/pipeline_worker.py` | 수정 (YAML 저장, 경로 검증, 상태 처리) |
| `src/api/main.py` | 수정 (경로 검증 호출 추가) |
| `src/evaluation/lmdeploy_bench_eval.py` | 수정 (BenchmarkSkipError 추가) |
| `src/lmdeploy_pipeline/evaluator.py` | 수정 (스킵 예외 처리) |
| `docs/eval/pipeline_api.md` | 수정 (에러 문서화) |
| `docs/eval/lmdeploy_yaml_guide.md` | 수정 (경로 경고 추가) |
| `.docs/task/16_yaml_validation_error_handling.md` | 신규 (이 보고서) |

---

## 검증 방법

1. 잘못된 `bench_base_path`로 YAML 전송 → HTTP 400 에러 + 구체적 메시지 확인
2. 존재하지 않는 벤치마크 폴더 포함 → HTTP 400 에러 + missing 목록 확인
3. 정상 YAML 전송 → 기존 동작 유지 확인
4. `data/uploaded_yamls/`에 YAML 파일 저장 확인
5. 전부 실패 시 `/pipeline/status`에서 `status: "failed"` 확인
