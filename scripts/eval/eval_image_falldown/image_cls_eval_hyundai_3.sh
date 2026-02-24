
###### 현대 백화점! 3_ 5 에폭 모델

MODEL_TAG="InternVL3-2B_hyundai_3_5"
CHECKPOINT="ckpts/${MODEL_TAG}"
BATCH_SIZE=20
OUT_DIR="results/eval_result_image/hyundai/${MODEL_TAG}"

PYTHONPATH="$(pwd)" python src/evaluation/evaluate_image_classfication.py \
    --checkpoint "${CHECKPOINT}" \
    --annotation data/instruction/evaluation/test_hyundai_abb_image_falldown.jsonl \
    --image-root data \
    --out-dir "${OUT_DIR}" \
    --batch-size "${BATCH_SIZE}" \
    --multi-gpu

PYTHONPATH="$(pwd)" python src/evaluation/evaluate_image_classfication.py \
    --checkpoint "${CHECKPOINT}" \
    --annotation data/instruction/evaluation/test_hyundai_dtro_image_falldown.jsonl \
    --image-root data \
    --out-dir "${OUT_DIR}" \
    --batch-size "${BATCH_SIZE}" \
    --multi-gpu

PYTHONPATH="$(pwd)" python src/evaluation/evaluate_image_classfication.py \
    --checkpoint "${CHECKPOINT}" \
    --annotation data/instruction/evaluation/test_hyundai_ai_image_falldown.jsonl \
    --image-root data \
    --out-dir "${OUT_DIR}" \
    --batch-size "${BATCH_SIZE}" \
    --multi-gpu

PYTHONPATH="$(pwd)" python src/evaluation/evaluate_image_classfication.py \
    --checkpoint "${CHECKPOINT}" \
    --annotation data/instruction/evaluation/test_hyundai_image_gen_ai_1st_image_falldown.jsonl \
    --image-root data \
    --out-dir "${OUT_DIR}" \
    --batch-size "${BATCH_SIZE}" \
    --multi-gpu

PYTHONPATH="$(pwd)" python src/evaluation/evaluate_image_classfication.py \
    --checkpoint "${CHECKPOINT}" \
    --annotation data/instruction/evaluation/test_hyundai_PoC_5camera_gen_ai_image_falldown.jsonl \
    --image-root data \
    --out-dir "${OUT_DIR}" \
    --batch-size "${BATCH_SIZE}" \
    --multi-gpu

PYTHONPATH="$(pwd)" python src/evaluation/evaluate_image_classfication.py \
    --checkpoint "${CHECKPOINT}" \
    --annotation data/instruction/evaluation/test_hyundai_PoC_25camera_capture_image_falldown.jsonl \
    --image-root data \
    --out-dir "${OUT_DIR}" \
    --batch-size "${BATCH_SIZE}" \
    --multi-gpu

PYTHONPATH="$(pwd)" python src/evaluation/evaluate_image_classfication.py \
    --checkpoint "${CHECKPOINT}" \
    --annotation data/instruction/evaluation/test_hyundai_hard_negative_2st_image_falldown.jsonl \
    --image-root data \
    --out-dir "${OUT_DIR}" \
    --batch-size "${BATCH_SIZE}" \
    --multi-gpu

PYTHONPATH="$(pwd)" python src/evaluation/evaluate_image_classfication.py \
    --checkpoint "${CHECKPOINT}" \
    --annotation data/instruction/evaluation/test_hyundai_image_gen_ai_only_sangrak_image_falldown.jsonl \
    --image-root data \
    --out-dir "${OUT_DIR}" \
    --batch-size "${BATCH_SIZE}" \
    --multi-gpu
