#!/bin/bash
# Usage:
#   sh scripts/pipe_line/train_eval_save_sample_scripts.sh
#   GPUS=4 PER_DEVICE_BATCH_SIZE=2 EPOCHS=10 sh scripts/pipe_line/train_eval_save_sample_scripts.sh

set -e
set -x

# ─── 사용자 설정 변수 ────────────────────────────────────────────
BASE_MODEL="ckpts/InternVL3-2B"
META_PATH="scripts/shell/data/hyundai_8.json"
DEEPSPEED_CONFIG="configs/deepspeed/zero_stage1_config.json"
LORA_OUTPUT_DIR="ckpts/lora"

EPOCHS=${EPOCHS:-20}
LEARNING_RATE=4e-5
MAX_SEQ_LENGTH=16384
LORA_RANK=64
SAVE_STEPS=200

MERGE_DIR="ckpts/InternVL3-2B_sample_${EPOCHS}"

# 평가 설정
EVAL_ANNOTATION_DIR="data/instruction/evaluation"
EVAL_BATCH_SIZE=20
IMAGE_ROOT="data"

# 영상 퀄리티 평가 설정
VIDEO_INPUT_ROOT="data/processed/hyundai_backhwajum/hyundai_video_macs_test/falldown"
VIDEO_OUTPUT_ROOT="results/eval_quality/eva_quality_hyundai/${MERGE_DIR##*/}/falldown_poc"
VIDEO_WINDOW_SIZE=15
VIDEO_BATCH_SIZE=40
VIDEO_THRESHOLD=1
# ─────────────────────────────────────────────────────────────────

GPUS=${GPUS:-8}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-512}
PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-4}
GRADIENT_ACC=$((TRAIN_BATCH_SIZE / PER_DEVICE_BATCH_SIZE / GPUS))

export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export MASTER_PORT=34229
export TF_CPP_MIN_LOG_LEVEL=3
export LAUNCHER=pytorch

# ─── 1. 학습 ─────────────────────────────────────────────────────
mkdir -p "${LORA_OUTPUT_DIR}"

torchrun \
  --nnodes=1 \
  --node_rank=0 \
  --master_addr=127.0.0.1 \
  --nproc_per_node=${GPUS} \
  --master_port=${MASTER_PORT} \
  src/training/internvl/train/internvl_chat_finetune.py \
  --model_name_or_path "${BASE_MODEL}" \
  --conv_style "internvl2_5" \
  --output_dir "${LORA_OUTPUT_DIR}" \
  --meta_path "${META_PATH}" \
  --overwrite_output_dir True \
  --force_image_size 448 \
  --max_dynamic_patch 32 \
  --down_sample_ratio 0.5 \
  --drop_path_rate 0.0 \
  --freeze_llm True \
  --freeze_mlp True \
  --freeze_backbone True \
  --use_llm_lora ${LORA_RANK} \
  --vision_select_layer -1 \
  --dataloader_num_workers 4 \
  --bf16 True \
  --num_train_epochs ${EPOCHS} \
  --per_device_train_batch_size ${PER_DEVICE_BATCH_SIZE} \
  --gradient_accumulation_steps ${GRADIENT_ACC} \
  --evaluation_strategy "no" \
  --save_strategy "steps" \
  --save_steps ${SAVE_STEPS} \
  --save_total_limit 1 \
  --learning_rate ${LEARNING_RATE} \
  --weight_decay 0.01 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type "cosine" \
  --logging_steps 1 \
  --max_seq_length ${MAX_SEQ_LENGTH} \
  --do_train True \
  --grad_checkpoint True \
  --group_by_length True \
  --dynamic_image_size True \
  --use_thumbnail True \
  --ps_version 'v2' \
  --deepspeed "${DEEPSPEED_CONFIG}" \
  --report_to "tensorboard" \
  2>&1 | tee -a "${LORA_OUTPUT_DIR}/training_log.txt"

# ─── 2. LoRA 병합 ─────────────────────────────────────────────────
mkdir -p "${MERGE_DIR}"
python src/training/tools/merge_lora.py "${LORA_OUTPUT_DIR}" "${MERGE_DIR}"
cp "${BASE_MODEL}"/*.py "${MERGE_DIR}"/
cp "${BASE_MODEL}"/config.json "${MERGE_DIR}"/
rm -rf "${LORA_OUTPUT_DIR}"

# ─── 3. 이미지 분류 평가 ──────────────────────────────────────────
EVAL_OUT_DIR="results/eval_result_image/hyundai/${MERGE_DIR##*/}"

ANNOTATION_FILES=(
  "test_hyundai_abb_image_falldown.jsonl"
  "test_hyundai_dtro_image_falldown.jsonl"
  "test_hyundai_ai_image_falldown.jsonl"
  "test_hyundai_image_gen_ai_1st_image_falldown.jsonl"
  "test_hyundai_PoC_5camera_gen_ai_image_falldown.jsonl"
  "test_hyundai_PoC_25camera_capture_image_falldown.jsonl"
  "test_hyundai_hard_negative_2st_image_falldown.jsonl"
  "test_hyundai_image_gen_ai_only_sangrak_image_falldown.jsonl"
  "test_hyundai_hard_negative_2st_box_image_falldown.jsonl"
  "test_hyundai_01_16_QA_image_falldown.jsonl"
  "test_hyundai_01_27_QA_image_falldown.jsonl"
)

for ANNOTATION in "${ANNOTATION_FILES[@]}"; do
  python src/evaluation/evaluate_image_classfication.py \
    --checkpoint "${MERGE_DIR}" \
    --annotation "${EVAL_ANNOTATION_DIR}/${ANNOTATION}" \
    --image-root "${IMAGE_ROOT}" \
    --out-dir "${EVAL_OUT_DIR}" \
    --batch-size "${EVAL_BATCH_SIZE}" \
    --multi-gpu
done

# ─── 4. 영상 퀄리티 평가 ─────────────────────────────────────────
python src/evaluation/evaluate_qualitative_video_threshold_image.py \
  --checkpoint "${MERGE_DIR}" \
  --input-root "${VIDEO_INPUT_ROOT}" \
  --output-root "${VIDEO_OUTPUT_ROOT}" \
  --window-size ${VIDEO_WINDOW_SIZE} \
  --batch-size ${VIDEO_BATCH_SIZE} \
  --threshold ${VIDEO_THRESHOLD} \
  --multi-gpu

# ─── 5. Notion용 영상 인코딩 (H.264) ────────────────────────────
find "${VIDEO_OUTPUT_ROOT}" -type f \( -iname "*.mp4" -o -iname "*.mov" -o -iname "*.avi" -o -iname "*.mkv" \) \
  ! -name "notion_*" \
  -exec sh -c '
    in="$1"
    outdir="$(dirname "$in")"
    base="$(basename "${in%.*}")"
    out="$outdir/notion_${base}.mp4"
    [ -f "$out" ] && exit 0
    ffmpeg -y -i "$in" -c:v libx264 -preset medium -crf 23 -c:a aac -b:a 128k -movflags +faststart "$out"
  ' _ {} \;
