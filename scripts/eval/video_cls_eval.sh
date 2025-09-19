PYTHONPATH="$(pwd)" torchrun --nproc_per_node=2 src/evaluation/evaluate_video_classfication.py \
    --checkpoint ckpts/merge_result \
    --annotation test_data.jsonl \
    --video-root data \
    --out-dir ./ \
    --num-frames 12