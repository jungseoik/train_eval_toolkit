# 20. CLS 토큰 기반 평가 모드 추가

## 개요

LMDeploy 벤치마크 파이프라인에 CLS(Classification) 평가 모드를 추가하여, 특수 토큰(`<CLS_FALLDOWN>`)으로 yes/no 응답을 반환하는 파인튜닝 모델을 지원한다.

## 배경

- `PIA_AI2team_VQA_falldown_v1.0` 모델은 기존 JSON 응답 방식과 다르게, `<CLS_FALLDOWN>` 특수 토큰을 프롬프트로 받으면 `yes`/`no`로 응답하는 분류 방식을 사용
- 기존 파이프라인의 `parse_model_output()`은 JSON 파싱만 지원하여 이 모델의 응답을 처리할 수 없었음

## 변경 사항

### 수정 파일

| 파일 | 변경 내용 |
|------|----------|
| `src/lmdeploy_pipeline/config.py` | `EvalConfig`에 `eval_mode: str = "json"` 필드 추가 + YAML 파싱 |
| `src/lmdeploy_pipeline/evaluator.py` | `_build_cfg_namespace()`에 `EVAL_MODE` 전달 |
| `src/evaluation/lmdeploy_bench_eval.py` | `parse_cls_output()` 함수 추가 + `process_frame()` 내 eval_mode 분기 |
| `configs/lmdeploy_pipeline/template/template.yaml` | `eval_mode` 필드 추가 (기본값 문서화) |
| `configs/lmdeploy_pipeline/PIA_AI2team_VQA_falldown_v1.0.yaml` | CLS 모드 전용 YAML 신규 생성 |
| `README.md` | eval_mode 옵션 설명 추가 |
| `docs/eval/lmdeploy_yaml_guide.md` | eval_mode 섹션 추가, CLS 프롬프트 예시 추가 |
| `docs/eval/lmdeploy_pipeline.md` | evaluate 섹션 레퍼런스 테이블에 eval_mode 추가 |

### 동작 방식

| eval_mode | 프롬프트 | 모델 응답 | 파싱 | max_tokens |
|-----------|---------|----------|------|------------|
| `"json"` (기본) | 긴 텍스트 | JSON | JSON 파싱 → category 추출 | 15 |
| `"cls"` | `<CLS_FALLDOWN>` | yes/no | 문자열 비교 (대소문자 무관) | 1 |

### 핵심 로직

```python
# 새로 추가된 CLS 파서
def parse_cls_output(raw_text: str) -> int:
    return 1 if raw_text.strip().lower() == "yes" else 0

# process_frame() 내 분기
if getattr(cfg, 'EVAL_MODE', 'json') == 'cls':
    pred = parse_cls_output(raw_text)
else:
    pred = classify(parse_model_output(raw_text, valid_values), category)
```

## 검증 결과

- CLS 파서 테스트: `yes`, `Yes`, `YES` → 1 / `no`, `No`, `abc`, `""` → 0
- 기존 JSON 모드 호환성: `eval_mode` 미지정 시 기본값 `"json"` → 기존 동작 유지
- YAML 설정 로딩: v1.0 YAML에서 `eval_mode: "cls"`, `max_tokens: 1` 정상 파싱
- evaluator namespace 전달: `EVAL_MODE` 속성 정상 전달 확인

## 설계 결정

- **접근법 1(최소 변경)** 채택: inline if/else 분기. ~70줄 변경, 기존 코드 경로 영향 없음
- vllm_bench_eval.py는 현재 미적용 (추후 필요 시 동일 패턴 적용 가능)
- 접근법 2(Strategy 함수), 접근법 3(Strategy 클래스)은 현재 2개 모드에 비해 과도한 설계로 보류

## 작성자

jungseoik
