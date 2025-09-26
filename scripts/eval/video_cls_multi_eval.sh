PYTHONPATH="$(pwd)" torchrun --nproc_per_node=2 src/evaluation/evaluate_video_classfication_multi.py \
    --checkpoint ckpts/InternVL3-2B_gangnam_vietnam_rwf2000_aihubstore_gj_space_no_split \
    --annotation data/instruction/evaluation/test_gangnam.jsonl \
    --video-root data \
    --out-dir results/eval_result \
    --num-frames 12 \
    --workers-per-gpu 8