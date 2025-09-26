############################### space 평가진행

PYTHONPATH="$(pwd)" torchrun --nproc_per_node=2 src/evaluation/evaluate_video_classfication_edit.py \
    --checkpoint ckpts/InternVL3-2B_aihub_space \
    --annotation data/instruction/evaluation/test_gj.jsonl \
    --video-root data \
    --out-dir results/eval_result \
    --num-frames 12 \
    --workers-per-gpu 8

PYTHONPATH="$(pwd)" torchrun --nproc_per_node=2 src/evaluation/evaluate_video_classfication_edit.py \
    --checkpoint ckpts/InternVL3-2B_aihub_space \
    --annotation data/instruction/evaluation/test_rwf2000.jsonl \
    --video-root data \
    --out-dir results/eval_result \
    --num-frames 12 \
    --workers-per-gpu 8

PYTHONPATH="$(pwd)" torchrun --nproc_per_node=2 src/evaluation/evaluate_video_classfication_edit.py \
    --checkpoint ckpts/InternVL3-2B_aihub_space \
    --annotation data/instruction/evaluation/test_gangnam.jsonl \
    --video-root data \
    --out-dir results/eval_result \
    --num-frames 12 \
    --workers-per-gpu 8

PYTHONPATH="$(pwd)" torchrun --nproc_per_node=2 src/evaluation/evaluate_video_classfication_edit.py \
    --checkpoint ckpts/InternVL3-2B_aihub_space \
    --annotation data/instruction/evaluation/test_gangnam_vietnam.jsonl \
    --video-root data \
    --out-dir results/eval_result \
    --num-frames 12 \
    --workers-per-gpu 8


PYTHONPATH="$(pwd)" torchrun --nproc_per_node=2 src/evaluation/evaluate_video_classfication_edit.py \
    --checkpoint ckpts/InternVL3-2B_aihub_space \
    --annotation data/instruction/evaluation/test_scvdALL_NOweapon.jsonl \
    --video-root data \
    --out-dir results/eval_result \
    --num-frames 12 \
    --workers-per-gpu 8