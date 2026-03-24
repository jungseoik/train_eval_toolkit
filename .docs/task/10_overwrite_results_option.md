# Task 10: 평가 결과 덮어쓰기 옵션 (overwrite_results)

## 개요
평가 파이프라인에 `overwrite_results` YAML 옵션을 추가하여, 이미 존재하는 결과 CSV를 스킵할 수 있도록 함.
중단된 평가를 재개하거나 누락 파일만 보충 평가할 때 불필요한 재처리를 방지.

## 배경
- InternVL3-2B 평가 중 파이프라인이 비정상 종료되어 PIA_Falldown 1개 파일 + Soil/UGO/Yonsei 3개 벤치마크 누락
- 기존에는 재실행 시 전체 비디오를 처음부터 다시 평가해야 했음 (PIA 50개 중 49개 완료 상태에서도 전부 재처리)

## 변경 사항

### YAML 설정
```yaml
evaluate:
  overwrite_results: false  # false=기존 결과 스킵, true=항상 덮어쓰기 (기본값)
```

### 수정 파일
| 파일 | 변경 내용 |
|------|-----------|
| `src/lmdeploy_pipeline/config.py` | EvalConfig에 `overwrite_results: bool = True` 필드 추가 + YAML 로딩 |
| `src/vllm_pipeline/config.py` | 동일 |
| `src/lmdeploy_pipeline/evaluator.py` | SimpleNamespace에 `OVERWRITE_RESULTS` 전달 |
| `src/vllm_pipeline/evaluator.py` | 동일 |
| `src/evaluation/lmdeploy_bench_eval.py` | 스킵 로직 구현 |
| `src/evaluation/vllm_bench_eval.py` | 동일 |

### 스킵 로직
1. `overwrite_results: false` 일 때만 동작
2. output CSV가 이미 존재하는지 확인
3. output CSV의 row 수가 GT CSV의 row 수와 일치하는지 검증
4. 일치하면 스킵, 불일치하면 `[MISMATCH]` 표시 후 재평가
5. 기본값은 `true` (항상 덮어쓰기) — 기존 동작과 동일

### 출력 예시
```
# overwrite_results: false 일 때
  Skipped 49/50 (existing results with matching rows)

  [50/50] FILE210101-012606F.MOV
  elapsed: 245.4s
  Saved -> results/.../FILE210101-012606F.csv

# overwrite_results: true (기본값) 일 때
  기존과 동일하게 전체 재평가
```

## 작성자
jungseoik
