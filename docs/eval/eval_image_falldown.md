# 이미지 분류 평가 가이드

이미지 기반 falldown 분류 모델의 정량 평가 방법을 설명합니다.

## 사용 스크립트

| 항목 | 경로 |
|---|---|
| 실행 스크립트 | `scripts/eval/eval_image_falldown/eval.sh` |
| 평가 모듈 | `src/evaluation/evaluate_image_classfication.py` |
| 결과 저장 | `results/eval_result_image/` |

## 기본 실행

```bash
PYTHONPATH="$(pwd)" python src/evaluation/evaluate_image_classfication.py \
  --checkpoint ckpts/InternVL3-2B_hyundai_10 \
  --annotation data/instruction/evaluation/test_hyundai_abb_image_falldown.jsonl \
  --image-root data \
  --out-dir results/eval_result_image/hyundai/InternVL3-2B_hyundai_10 \
  --batch-size 20 \
  --multi-gpu
```

## 옵션

| 옵션 | 설명 | 기본값 |
|---|---|---|
| `--checkpoint` | 모델 체크포인트 경로 | (필수) |
| `--annotation` | 평가용 JSONL 파일 경로 | (필수) |
| `--image-root` | 이미지 루트 디렉토리 | `''` |
| `--out-dir` | 결과 저장 경로 | `results/eval_result` |
| `--batch-size` | 배치 크기 | `8` |
| `--multi-gpu` | 멀티 GPU 분산 로딩 | (flag) |
| `--num-beams` | beam search 크기 | `1` |

## 여러 데이터셋 배치 평가

annotation 파일을 배열로 지정하면 한 번에 여러 데이터셋을 순차 평가할 수 있습니다.

```bash
MODEL_TAG="InternVL3-2B_hyundai_3_5"
CHECKPOINT="ckpts/${MODEL_TAG}"
BATCH_SIZE=20
OUT_DIR="results/eval_result_image/hyundai/${MODEL_TAG}"

ANNOTATIONS=(
  "data/instruction/evaluation/test_hyundai_abb_image_falldown.jsonl"
  "data/instruction/evaluation/test_hyundai_dtro_image_falldown.jsonl"
  "data/instruction/evaluation/test_hyundai_ai_image_falldown.jsonl"
)

for ANN in "${ANNOTATIONS[@]}"; do
  PYTHONPATH="$(pwd)" python src/evaluation/evaluate_image_classfication.py \
    --checkpoint "${CHECKPOINT}" \
    --annotation "${ANN}" \
    --image-root data \
    --out-dir "${OUT_DIR}" \
    --batch-size "${BATCH_SIZE}" \
    --multi-gpu
done
```

자세한 실행 예시는 `scripts/eval/eval_image_falldown/eval.sh`를 참조하세요.

## 결과

평가 완료 후 `--out-dir`에 다음 파일이 생성됩니다.

- `result_*.json` — 샘플별 예측 결과
- `summary_*.txt` — Precision / Recall / F1 / Accuracy 요약
