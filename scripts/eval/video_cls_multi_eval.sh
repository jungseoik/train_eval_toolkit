PYTHONPATH="$(pwd)" torchrun --nproc_per_node=2 src/evaluation/evaluate_video_classfication_multi.py \
    --checkpoint ckpts/InternVL3-2B_split_vio \
    --annotation data/instruction/evaluation/test_gangnam.jsonl \
    --video-root data \
    --out-dir results/eval_result \
    --num-frames 12 \
    --workers-per-gpu 8