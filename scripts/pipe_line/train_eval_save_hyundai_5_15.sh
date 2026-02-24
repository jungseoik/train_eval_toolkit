# e.g.
## GPUS=2 PER_DEVICE_BATCH_SIZE=2 sh scripts/shell/internvl3.0/hyundai_3_5.sh
# choose samller PER_DEVICE_BATCH_SIZE to reduce GPU Memory
# h100
# GPUS=4 PER_DEVICE_BATCH_SIZE=1 BATCH_SIZE=256 sh scripts/shell/internvl3.0/hyundai_3_5.sh
# GPUS=4 PER_DEVICE_BATCH_SIZE=2 sh scripts/shell/internvl3.0/hyundai_3_5.sh

set -x
## h 100 누가 사용중
GPUS=4 PER_DEVICE_BATCH_SIZE=1 BATCH_SIZE=256
## h 100 누가 사용중

EPOCHS=15
GPUS=${GPUS:-8}
BATCH_SIZE=${BATCH_SIZE:-512}
PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-4}
GRADIENT_ACC=$((BATCH_SIZE / PER_DEVICE_BATCH_SIZE / GPUS))
MERGE_DIR="InternVL3-2B_hyundai_5_${EPOCHS}"


export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export MASTER_PORT=34229
export TF_CPP_MIN_LOG_LEVEL=3
export LAUNCHER=pytorch

OUTPUT_DIR='ckpts/lora'
mkdir ckpts/lora

if [ ! -d "$OUTPUT_DIR" ]; then
  mkdir -p "$OUTPUT_DIR"
fi

# number of gpus: 8
# batch size per gpu: 4
# gradient accumulation steps: 16
# total batch size: 512
# epoch: 1
torchrun \
  --nnodes=1 \
  --node_rank=0 \
  --master_addr=127.0.0.1 \
  --nproc_per_node=${GPUS} \
  --master_port=${MASTER_PORT} \
  src/training/internvl/train/internvl_chat_finetune.py \
  --model_name_or_path "ckpts/InternVL3-2B" \
  --conv_style "internvl2_5" \
  --output_dir ${OUTPUT_DIR} \
  --meta_path "scripts/shell/data/hyundai_5.json" \
  --overwrite_output_dir True \
  --force_image_size 448 \
  --max_dynamic_patch 32 \
  --down_sample_ratio 0.5 \
  --drop_path_rate 0.0 \
  --freeze_llm True \
  --freeze_mlp True \
  --freeze_backbone True \
  --use_llm_lora 64 \
  --vision_select_layer -1 \
  --dataloader_num_workers 4 \
  --bf16 True \
  --num_train_epochs ${EPOCHS} \
  --per_device_train_batch_size ${PER_DEVICE_BATCH_SIZE} \
  --gradient_accumulation_steps ${GRADIENT_ACC} \
  --evaluation_strategy "no" \
  --save_strategy "steps" \
  --save_steps 200 \
  --save_total_limit 1 \
  --learning_rate 4e-5 \
  --weight_decay 0.01 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type "cosine" \
  --logging_steps 1 \
  --max_seq_length 16384 \
  --do_train True \
  --grad_checkpoint True \
  --group_by_length True \
  --dynamic_image_size True \
  --use_thumbnail True \
  --ps_version 'v2' \
  --deepspeed "configs/deepspeed/zero_stage1_config.json" \
  --report_to "tensorboard" \
  2>&1 | tee -a "${OUTPUT_DIR}/training_log.txt"


mkdir ckpts/$MERGE_DIR
PYTHONPATH="$(pwd)" python src/training/tools/merge_lora.py ckpts/lora ckpts/$MERGE_DIR
cp ckpts/InternVL3-2B/*.py ckpts/$MERGE_DIR/
cp ckpts/InternVL3-2B/config.json ckpts/$MERGE_DIR/
rm -rf ckpts/lora

###### 현대 백화점 평가프로세스

CHECKPOINT="ckpts/${MERGE_DIR}"
BATCH_SIZE=20
OUT_DIR="results/eval_result_image/hyundai/${MERGE_DIR}"

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

## 현대백화점 영상 퀄리티 평가 프로세스

PYTHONPATH="$(pwd)" python src/evaluation/evaluate_qualitative_video.py \
    --checkpoint "ckpts/${MERGE_DIR}" \
    --input-root "data/processed/hyundai_backhwajum/hyundai_video_macs_test/falldown" \
    --output-root "results/eval_quality/eva_quality_hyundai/${MERGE_DIR}/falldown_poc" \
    --window-size 15 \
    --batch-size 20 \
    --multi-gpu

OUTPUT_ROOT="results/eval_quality/eva_quality_hyundai/${MERGE_DIR}/falldown_poc"

find "$OUTPUT_ROOT" -type f \( -iname "*.mp4" -o -iname "*.mov" -o -iname "*.avi" -o -iname "*.mkv" \) -exec sh -c '
  in="$1"
  outdir="$(dirname "$in")"
  base="$(basename "${in%.*}")"
  out="$outdir/notion_${base}.mp4"
  [ -f "$out" ] && exit 0
  ffmpeg -y -i "$in" -c:v libx264 -preset medium -crf 23 -c:a aac -b:a 128k -movflags +faststart "$out"
' _ {} \;

