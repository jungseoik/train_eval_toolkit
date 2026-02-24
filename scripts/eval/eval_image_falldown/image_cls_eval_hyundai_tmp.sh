###### 현대 백화점! 추가 평가

# MODEL_TAGS=(
#   "InternVL3-2B_hyundai_2_10"
#   "InternVL3-2B_hyundai_2_15"
#   "InternVL3-2B_hyundai_2_20"
#   "InternVL3-2B_hyundai_10"
#   "InternVL3-2B_hyundai_15"
#   "InternVL3-2B"
#   "InternVL3-2B_hyundai"
#   "InternVL3-2B_hyundai_5_5"
#   "InternVL3-2B_hyundai_5_10"
#   "InternVL3-2B_hyundai_5_15"
#   "InternVL3-2B_hyundai_5_20"
#   "InternVL3-2B_hyundai_4_5"
#   "InternVL3-2B_hyundai_4_10"
#   "InternVL3-2B_hyundai_4_15"
#   "InternVL3-2B_hyundai_4_20"
#   "InternVL3-2B_hyundai_3_5"
#   "InternVL3-2B_hyundai_3_10"
#   "InternVL3-2B_hyundai_3_15"
#   "InternVL3-2B_hyundai_3_20"

# )

MODEL_TAGS=(

  "InternVL3-2B_hyundai_7_10"

)

BATCH_SIZE=4
ANNOTATIONS=(
  "data/instruction/evaluation/test_hyundai_01_16_QA_image_falldown.jsonl"
  "data/instruction/evaluation/test_hyundai_01_27_QA_image_falldown.jsonl"
)

for MODEL_TAG in "${MODEL_TAGS[@]}"; do
  CHECKPOINT="ckpts/${MODEL_TAG}"
  OUT_DIR="results/eval_result_image/hyundai/tmp/${MODEL_TAG}"

  for ANN in "${ANNOTATIONS[@]}"; do
    PYTHONPATH="$(pwd)" python src/evaluation/evaluate_image_classfication.py \
      --checkpoint "${CHECKPOINT}" \
      --annotation "${ANN}" \
      --image-root data \
      --out-dir "${OUT_DIR}" \
      --batch-size "${BATCH_SIZE}" \

  done
done
