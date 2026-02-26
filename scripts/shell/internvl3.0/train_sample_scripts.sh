#!/bin/bash
# Usage:
#   sh scripts/shell/internvl3.0/train_sample_scripts.sh
#   GPUS=4 PER_DEVICE_BATCH_SIZE=2 sh scripts/shell/internvl3.0/train_sample_scripts.sh

set -e
set -x

# ─── 사용자 설정 변수 ────────────────────────────────────────────
BASE_MODEL="ckpts/InternVL3-2B"
META_PATH="scripts/shell/data/hyundai_hard_negative_2st_box.json"
DEEPSPEED_CONFIG="configs/deepspeed/zero_stage1_config.json"
LORA_OUTPUT_DIR="ckpts/lora"
MERGE_DIR="ckpts/InternVL3-2B_hyundai_sample"

EPOCHS=5
LEARNING_RATE=4e-5
MAX_SEQ_LENGTH=16384
LORA_RANK=64
SAVE_STEPS=200
# ─────────────────────────────────────────────────────────────────

GPUS=${GPUS:-8}
BATCH_SIZE=${BATCH_SIZE:-512}
PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-4}
GRADIENT_ACC=$((BATCH_SIZE / PER_DEVICE_BATCH_SIZE / GPUS))

export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export MASTER_PORT=34229
export TF_CPP_MIN_LOG_LEVEL=3
export LAUNCHER=pytorch

# lora 출력 디렉토리 생성
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
  --eval_strategy "no" \
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

# ─── LoRA 병합 ────────────────────────────────────────────────────
mkdir -p "${MERGE_DIR}"
PYTHONPATH="$(pwd)" python src/training/tools/merge_lora.py "${LORA_OUTPUT_DIR}" "${MERGE_DIR}"
cp "${BASE_MODEL}"/*.py "${MERGE_DIR}"/
cp "${BASE_MODEL}"/config.json "${MERGE_DIR}"/
rm -rf "${LORA_OUTPUT_DIR}"
