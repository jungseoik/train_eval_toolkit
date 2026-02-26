#!/usr/bin/env bash
# ============================================================
# 레거시 구성 예시 (참고용 — 실제 사용했던 패턴)
# ============================================================

# [단일 평가 — 가장 기본 스크립트]
PYTHONPATH="$(pwd)" python src/evaluation/evaluate_image_classfication.py \
  --checkpoint ckpts/InternVL3-2B_hyundai_10 \
  --annotation data/instruction/evaluation/test_hyundai_abb_image_falldown.jsonl \
  --image-root data \
  --out-dir results/eval_result_image/hyundai/InternVL3-2B_hyundai_10 \
  --batch-size 20 \
  --multi-gpu


# ============================================================
# Image Classification Evaluation
# evaluate_image_classfication.py 옵션:
#   --checkpoint  : 모델 체크포인트 경로 (required)
#   --annotation  : 평가 JSONL 파일 경로 (required)
#   --image-root  : 이미지 루트 디렉토리 (default: '')
#   --out-dir     : 결과 저장 경로 (default: 'results/eval_result')
#   --batch-size  : 배치 크기 (default: 8)
#   --multi-gpu   : 멀티 GPU 사용 여부 (flag)
#   --num-beams   : beam search 크기 (default: 1)
# ============================================================

MODEL_TAG="InternVL3-2B_hyundai_3_5"
CHECKPOINT="ckpts/${MODEL_TAG}"
BATCH_SIZE=20
OUT_DIR="results/eval_result_image/hyundai/${MODEL_TAG}"

ANNOTATIONS=(
  "data/instruction/evaluation/test_hyundai_abb_image_falldown.jsonl"
  "data/instruction/evaluation/test_hyundai_dtro_image_falldown.jsonl"
  "data/instruction/evaluation/test_hyundai_ai_image_falldown.jsonl"
  "data/instruction/evaluation/test_hyundai_image_gen_ai_1st_image_falldown.jsonl"
  "data/instruction/evaluation/test_hyundai_PoC_5camera_gen_ai_image_falldown.jsonl"
  "data/instruction/evaluation/test_hyundai_PoC_25camera_capture_image_falldown.jsonl"
  "data/instruction/evaluation/test_hyundai_hard_negative_2st_image_falldown.jsonl"
  "data/instruction/evaluation/test_hyundai_image_gen_ai_only_sangrak_image_falldown.jsonl"
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

