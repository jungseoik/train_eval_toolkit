
###### 현대 백화점!
PYTHONPATH="$(pwd)" python src/evaluation/evaluate_image_classfication.py \
    --checkpoint ckpts/InternVL3-2B_hyundai_10 \
    --annotation data/instruction/evaluation/test_hyundai_abb_image_falldown.jsonl \
    --image-root data \
    --out-dir results/eval_result_image/hyundai/InternVL3-2B_hyundai \
    --batch-size 20 \
    --multi-gpu

PYTHONPATH="$(pwd)" python src/evaluation/evaluate_image_classfication.py \
    --checkpoint ckpts/InternVL3-2B_hyundai_10 \
    --annotation data/instruction/evaluation/test_hyundai_dtro_image_falldown.jsonl \
    --image-root data \
    --out-dir results/eval_result_image/hyundai/InternVL3-2B_hyundai \
    --batch-size 20 \
    --multi-gpu

PYTHONPATH="$(pwd)" python src/evaluation/evaluate_image_classfication.py \
    --checkpoint ckpts/InternVL3-2B_hyundai_10 \
    --annotation data/instruction/evaluation/test_hyundai_ai_image_falldown.jsonl \
    --image-root data \
    --out-dir results/eval_result_image/hyundai/InternVL3-2B_hyundai \
    --batch-size 20 \
    --multi-gpu


    # data/instruction/evaluation/test_hyundai_abb_image_falldown.jsonl
    # data/instruction/evaluation/test_hyundai_dtro_image_falldown.jsonl
    # data/instruction/evaluation/test_hyundai_ai_image_falldown.jsonl


# PYTHONPATH="$(pwd)" python src/evaluation/evaluate_image_classfication.py \
#     --checkpoint ckpts/InternVL3-2B \
#     --annotation data/instruction/evaluation/test_hyundai_abb_image_falldown.jsonl \
#     --image-root data \
#     --out-dir results/eval_result_image/hyundai/InternVL3-2B_hyundai \
#     --batch-size 20 \
#     --multi-gpu

# PYTHONPATH="$(pwd)" python src/evaluation/evaluate_image_classfication.py \
#     --checkpoint ckpts/InternVL3-2B \
#     --annotation data/instruction/evaluation/test_hyundai_dtro_image_falldown.jsonl \
#     --image-root data \
#     --out-dir results/eval_result_image/hyundai/InternVL3-2B_hyundai \
#     --batch-size 20 \
#     --multi-gpu

# PYTHONPATH="$(pwd)" python src/evaluation/evaluate_image_classfication.py \
#     --checkpoint ckpts/InternVL3-2B \
#     --annotation data/instruction/evaluation/test_hyundai_ai_image_falldown.jsonl \
#     --image-root data \
#     --out-dir results/eval_result_image/hyundai/InternVL3-2B_hyundai \
#     --batch-size 20 \
#     --multi-gpu