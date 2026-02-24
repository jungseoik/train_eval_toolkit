
###### 현대 백화점! 10 에폭 모델
PYTHONPATH="$(pwd)" python src/evaluation/evaluate_image_classfication.py \
    --checkpoint ckpts/InternVL3-2B_hyundai_10 \
    --annotation data/instruction/evaluation/test_hyundai_abb_image_falldown.jsonl \
    --image-root data \
    --out-dir results/eval_result_image/hyundai/hyundai_10 \
    --batch-size 20 \
    --multi-gpu

PYTHONPATH="$(pwd)" python src/evaluation/evaluate_image_classfication.py \
    --checkpoint ckpts/InternVL3-2B_hyundai_10 \
    --annotation data/instruction/evaluation/test_hyundai_dtro_image_falldown.jsonl \
    --image-root data \
    --out-dir results/eval_result_image/hyundai/hyundai_10 \
    --batch-size 20 \
    --multi-gpu

PYTHONPATH="$(pwd)" python src/evaluation/evaluate_image_classfication.py \
    --checkpoint ckpts/InternVL3-2B_hyundai_10 \
    --annotation data/instruction/evaluation/test_hyundai_ai_image_falldown.jsonl \
    --image-root data \
    --out-dir results/eval_result_image/hyundai/hyundai_10 \
    --batch-size 20 \
    --multi-gpu

PYTHONPATH="$(pwd)" python src/evaluation/evaluate_image_classfication.py \
    --checkpoint ckpts/InternVL3-2B_hyundai_10 \
    --annotation data/instruction/evaluation/test_hyundai_image_gen_ai_1st_image_falldown.jsonl \
    --image-root data \
    --out-dir results/eval_result_image/hyundai/hyundai_10 \
    --batch-size 20 \
    --multi-gpu

PYTHONPATH="$(pwd)" python src/evaluation/evaluate_image_classfication.py \
    --checkpoint ckpts/InternVL3-2B_hyundai_10 \
    --annotation data/instruction/evaluation/test_hyundai_PoC_5camera_gen_ai_image_falldown.jsonl \
    --image-root data \
    --out-dir results/eval_result_image/hyundai/hyundai_10 \
    --batch-size 20 \
    --multi-gpu

PYTHONPATH="$(pwd)" python src/evaluation/evaluate_image_classfication.py \
    --checkpoint ckpts/InternVL3-2B_hyundai_10 \
    --annotation data/instruction/evaluation/test_hyundai_PoC_25camera_capture_image_falldown.jsonl \
    --image-root data \
    --out-dir results/eval_result_image/hyundai/hyundai_10 \
    --batch-size 20 \
    --multi-gpu


    # data/instruction/evaluation/test_hyundai_abb_image_falldown.jsonl
    # data/instruction/evaluation/test_hyundai_dtro_image_falldown.jsonl
    # data/instruction/evaluation/test_hyundai_ai_image_falldown.jsonl

###### 현대 백화점! 기본 베이스 비교
PYTHONPATH="$(pwd)" python src/evaluation/evaluate_image_classfication.py \
    --checkpoint ckpts/InternVL3-2B \
    --annotation data/instruction/evaluation/test_hyundai_abb_image_falldown.jsonl \
    --image-root data \
    --out-dir results/eval_result_image/hyundai/base \
    --batch-size 20 \
    --multi-gpu

PYTHONPATH="$(pwd)" python src/evaluation/evaluate_image_classfication.py \
    --checkpoint ckpts/InternVL3-2B \
    --annotation data/instruction/evaluation/test_hyundai_dtro_image_falldown.jsonl \
    --image-root data \
    --out-dir results/eval_result_image/hyundai/base \
    --batch-size 20 \
    --multi-gpu

PYTHONPATH="$(pwd)" python src/evaluation/evaluate_image_classfication.py \
    --checkpoint ckpts/InternVL3-2B \
    --annotation data/instruction/evaluation/test_hyundai_ai_image_falldown.jsonl \
    --image-root data \
    --out-dir results/eval_result_image/hyundai/base \
    --batch-size 20 \
    --multi-gpu

PYTHONPATH="$(pwd)" python src/evaluation/evaluate_image_classfication.py \
    --checkpoint ckpts/InternVL3-2B \
    --annotation data/instruction/evaluation/test_hyundai_image_gen_ai_1st_image_falldown.jsonl \
    --image-root data \
    --out-dir results/eval_result_image/hyundai/base \
    --batch-size 20 \
    --multi-gpu

PYTHONPATH="$(pwd)" python src/evaluation/evaluate_image_classfication.py \
    --checkpoint ckpts/InternVL3-2B \
    --annotation data/instruction/evaluation/test_hyundai_PoC_5camera_gen_ai_image_falldown.jsonl \
    --image-root data \
    --out-dir results/eval_result_image/hyundai/base \
    --batch-size 20 \
    --multi-gpu

PYTHONPATH="$(pwd)" python src/evaluation/evaluate_image_classfication.py \
    --checkpoint ckpts/InternVL3-2B \
    --annotation data/instruction/evaluation/test_hyundai_PoC_25camera_capture_image_falldown.jsonl \
    --image-root data \
    --out-dir results/eval_result_image/hyundai/base \
    --batch-size 20 \
    --multi-gpu

###### 현대 백화점!  15에폭 비교
PYTHONPATH="$(pwd)" python src/evaluation/evaluate_image_classfication.py \
    --checkpoint ckpts/InternVL3-2B_hyundai_15 \
    --annotation data/instruction/evaluation/test_hyundai_abb_image_falldown.jsonl \
    --image-root data \
    --out-dir results/eval_result_image/hyundai/hyundai_15 \
    --batch-size 20 \
    --multi-gpu

PYTHONPATH="$(pwd)" python src/evaluation/evaluate_image_classfication.py \
    --checkpoint ckpts/InternVL3-2B_hyundai_15 \
    --annotation data/instruction/evaluation/test_hyundai_dtro_image_falldown.jsonl \
    --image-root data \
    --out-dir results/eval_result_image/hyundai/hyundai_15 \
    --batch-size 20 \
    --multi-gpu

PYTHONPATH="$(pwd)" python src/evaluation/evaluate_image_classfication.py \
    --checkpoint ckpts/InternVL3-2B_hyundai_15 \
    --annotation data/instruction/evaluation/test_hyundai_ai_image_falldown.jsonl \
    --image-root data \
    --out-dir results/eval_result_image/hyundai/hyundai_15 \
    --batch-size 20 \
    --multi-gpu

PYTHONPATH="$(pwd)" python src/evaluation/evaluate_image_classfication.py \
    --checkpoint ckpts/InternVL3-2B_hyundai_15 \
    --annotation data/instruction/evaluation/test_hyundai_image_gen_ai_1st_image_falldown.jsonl \
    --image-root data \
    --out-dir results/eval_result_image/hyundai/hyundai_15 \
    --batch-size 20 \
    --multi-gpu

PYTHONPATH="$(pwd)" python src/evaluation/evaluate_image_classfication.py \
    --checkpoint ckpts/InternVL3-2B_hyundai_15 \
    --annotation data/instruction/evaluation/test_hyundai_PoC_5camera_gen_ai_image_falldown.jsonl \
    --image-root data \
    --out-dir results/eval_result_image/hyundai/hyundai_15 \
    --batch-size 20 \
    --multi-gpu

PYTHONPATH="$(pwd)" python src/evaluation/evaluate_image_classfication.py \
    --checkpoint ckpts/InternVL3-2B_hyundai_15 \
    --annotation data/instruction/evaluation/test_hyundai_PoC_25camera_capture_image_falldown.jsonl \
    --image-root data \
    --out-dir results/eval_result_image/hyundai/hyundai_15 \
    --batch-size 20 \
    --multi-gpu

###### 현대 백화점! 전체 영상학습
PYTHONPATH="$(pwd)" python src/evaluation/evaluate_image_classfication.py \
    --checkpoint ckpts/InternVL3-2B_hyundai_2_10 \
    --annotation data/instruction/evaluation/test_hyundai_abb_image_falldown.jsonl \
    --image-root data \
    --out-dir results/eval_result_image/hyundai/hyundai_2_10 \
    --batch-size 20 \
    --multi-gpu

PYTHONPATH="$(pwd)" python src/evaluation/evaluate_image_classfication.py \
    --checkpoint ckpts/InternVL3-2B_hyundai_2_10 \
    --annotation data/instruction/evaluation/test_hyundai_dtro_image_falldown.jsonl \
    --image-root data \
    --out-dir results/eval_result_image/hyundai/hyundai_2_10 \
    --batch-size 20 \
    --multi-gpu

PYTHONPATH="$(pwd)" python src/evaluation/evaluate_image_classfication.py \
    --checkpoint ckpts/InternVL3-2B_hyundai_2_10 \
    --annotation data/instruction/evaluation/test_hyundai_ai_image_falldown.jsonl \
    --image-root data \
    --out-dir results/eval_result_image/hyundai/hyundai_2_10 \
    --batch-size 20 \
    --multi-gpu

PYTHONPATH="$(pwd)" python src/evaluation/evaluate_image_classfication.py \
    --checkpoint ckpts/InternVL3-2B_hyundai_2_10 \
    --annotation data/instruction/evaluation/test_hyundai_image_gen_ai_1st_image_falldown.jsonl \
    --image-root data \
    --out-dir results/eval_result_image/hyundai/hyundai_2_10 \
    --batch-size 20 \
    --multi-gpu

PYTHONPATH="$(pwd)" python src/evaluation/evaluate_image_classfication.py \
    --checkpoint ckpts/InternVL3-2B_hyundai_2_10 \
    --annotation data/instruction/evaluation/test_hyundai_PoC_5camera_gen_ai_image_falldown.jsonl \
    --image-root data \
    --out-dir results/eval_result_image/hyundai/hyundai_2_10 \
    --batch-size 20 \
    --multi-gpu

PYTHONPATH="$(pwd)" python src/evaluation/evaluate_image_classfication.py \
    --checkpoint ckpts/InternVL3-2B_hyundai_2_10 \
    --annotation data/instruction/evaluation/test_hyundai_PoC_25camera_capture_image_falldown.jsonl \
    --image-root data \
    --out-dir results/eval_result_image/hyundai/hyundai_2_10 \
    --batch-size 20 \
    --multi-gpu


############# 생 3에폭 현대 학습
PYTHONPATH="$(pwd)" python src/evaluation/evaluate_image_classfication.py \
    --checkpoint ckpts/InternVL3-2B_hyundai \
    --annotation data/instruction/evaluation/test_hyundai_abb_image_falldown.jsonl \
    --image-root data \
    --out-dir results/eval_result_image/hyundai/hyundai \
    --batch-size 20 \
    --multi-gpu

PYTHONPATH="$(pwd)" python src/evaluation/evaluate_image_classfication.py \
    --checkpoint ckpts/InternVL3-2B_hyundai \
    --annotation data/instruction/evaluation/test_hyundai_dtro_image_falldown.jsonl \
    --image-root data \
    --out-dir results/eval_result_image/hyundai/hyundai \
    --batch-size 20 \
    --multi-gpu

PYTHONPATH="$(pwd)" python src/evaluation/evaluate_image_classfication.py \
    --checkpoint ckpts/InternVL3-2B_hyundai \
    --annotation data/instruction/evaluation/test_hyundai_ai_image_falldown.jsonl \
    --image-root data \
    --out-dir results/eval_result_image/hyundai/hyundai \
    --batch-size 20 \
    --multi-gpu

PYTHONPATH="$(pwd)" python src/evaluation/evaluate_image_classfication.py \
    --checkpoint ckpts/InternVL3-2B_hyundai \
    --annotation data/instruction/evaluation/test_hyundai_image_gen_ai_1st_image_falldown.jsonl \
    --image-root data \
    --out-dir results/eval_result_image/hyundai/hyundai \
    --batch-size 20 \
    --multi-gpu

PYTHONPATH="$(pwd)" python src/evaluation/evaluate_image_classfication.py \
    --checkpoint ckpts/InternVL3-2B_hyundai \
    --annotation data/instruction/evaluation/test_hyundai_PoC_5camera_gen_ai_image_falldown.jsonl \
    --image-root data \
    --out-dir results/eval_result_image/hyundai/hyundai \
    --batch-size 20 \
    --multi-gpu

PYTHONPATH="$(pwd)" python src/evaluation/evaluate_image_classfication.py \
    --checkpoint ckpts/InternVL3-2B_hyundai \
    --annotation data/instruction/evaluation/test_hyundai_PoC_25camera_capture_image_falldown.jsonl \
    --image-root data \
    --out-dir results/eval_result_image/hyundai/hyundai \
    --batch-size 20 \
    --multi-gpu

############# 20 에폭 현대_2 학습
PYTHONPATH="$(pwd)" python src/evaluation/evaluate_image_classfication.py \
    --checkpoint ckpts/InternVL3-2B_hyundai_2_20 \
    --annotation data/instruction/evaluation/test_hyundai_abb_image_falldown.jsonl \
    --image-root data \
    --out-dir results/eval_result_image/hyundai/hyundai_2_20 \
    --batch-size 20 \
    --multi-gpu

PYTHONPATH="$(pwd)" python src/evaluation/evaluate_image_classfication.py \
    --checkpoint ckpts/InternVL3-2B_hyundai_2_20 \
    --annotation data/instruction/evaluation/test_hyundai_dtro_image_falldown.jsonl \
    --image-root data \
    --out-dir results/eval_result_image/hyundai/hyundai_2_20 \
    --batch-size 20 \
    --multi-gpu

PYTHONPATH="$(pwd)" python src/evaluation/evaluate_image_classfication.py \
    --checkpoint ckpts/InternVL3-2B_hyundai_2_20 \
    --annotation data/instruction/evaluation/test_hyundai_ai_image_falldown.jsonl \
    --image-root data \
    --out-dir results/eval_result_image/hyundai/hyundai_2_20 \
    --batch-size 20 \
    --multi-gpu

PYTHONPATH="$(pwd)" python src/evaluation/evaluate_image_classfication.py \
    --checkpoint ckpts/InternVL3-2B_hyundai_2_20 \
    --annotation data/instruction/evaluation/test_hyundai_image_gen_ai_1st_image_falldown.jsonl \
    --image-root data \
    --out-dir results/eval_result_image/hyundai/hyundai \
    --batch-size 20 \
    --multi-gpu

PYTHONPATH="$(pwd)" python src/evaluation/evaluate_image_classfication.py \
    --checkpoint ckpts/InternVL3-2B_hyundai_2_20 \
    --annotation data/instruction/evaluation/test_hyundai_PoC_5camera_gen_ai_image_falldown.jsonl \
    --image-root data \
    --out-dir results/eval_result_image/hyundai/hyundai_2_20 \
    --batch-size 20 \
    --multi-gpu

PYTHONPATH="$(pwd)" python src/evaluation/evaluate_image_classfication.py \
    --checkpoint ckpts/InternVL3-2B_hyundai_2_20 \
    --annotation data/instruction/evaluation/test_hyundai_PoC_25camera_capture_image_falldown.jsonl \
    --image-root data \
    --out-dir results/eval_result_image/hyundai/hyundai_2_20 \
    --batch-size 20 \
    --multi-gpu


############# 15 에폭 현대_2 학습
PYTHONPATH="$(pwd)" python src/evaluation/evaluate_image_classfication.py \
    --checkpoint ckpts/InternVL3-2B_hyundai_2_15 \
    --annotation data/instruction/evaluation/test_hyundai_abb_image_falldown.jsonl \
    --image-root data \
    --out-dir results/eval_result_image/hyundai/hyundai_2_15 \
    --batch-size 10 \
    --multi-gpu

PYTHONPATH="$(pwd)" python src/evaluation/evaluate_image_classfication.py \
    --checkpoint ckpts/InternVL3-2B_hyundai_2_15 \
    --annotation data/instruction/evaluation/test_hyundai_dtro_image_falldown.jsonl \
    --image-root data \
    --out-dir results/eval_result_image/hyundai/hyundai_2_15 \
    --batch-size 10 \
    --multi-gpu

PYTHONPATH="$(pwd)" python src/evaluation/evaluate_image_classfication.py \
    --checkpoint ckpts/InternVL3-2B_hyundai_2_15 \
    --annotation data/instruction/evaluation/test_hyundai_ai_image_falldown.jsonl \
    --image-root data \
    --out-dir results/eval_result_image/hyundai/hyundai_2_15 \
    --batch-size 10 \
    --multi-gpu

PYTHONPATH="$(pwd)" python src/evaluation/evaluate_image_classfication.py \
    --checkpoint ckpts/InternVL3-2B_hyundai_2_15 \
    --annotation data/instruction/evaluation/test_hyundai_image_gen_ai_1st_image_falldown.jsonl \
    --image-root data \
    --out-dir results/eval_result_image/hyundai/hyundai_2_15 \
    --batch-size 10 \
    --multi-gpu

PYTHONPATH="$(pwd)" python src/evaluation/evaluate_image_classfication.py \
    --checkpoint ckpts/InternVL3-2B_hyundai_2_15 \
    --annotation data/instruction/evaluation/test_hyundai_PoC_5camera_gen_ai_image_falldown.jsonl \
    --image-root data \
    --out-dir results/eval_result_image/hyundai/hyundai_2_15 \
    --batch-size 10 \
    --multi-gpu

PYTHONPATH="$(pwd)" python src/evaluation/evaluate_image_classfication.py \
    --checkpoint ckpts/InternVL3-2B_hyundai_2_15 \
    --annotation data/instruction/evaluation/test_hyundai_PoC_25camera_capture_image_falldown.jsonl \
    --image-root data \
    --out-dir results/eval_result_image/hyundai/hyundai_2_15 \
    --batch-size 10 \
    --multi-gpu