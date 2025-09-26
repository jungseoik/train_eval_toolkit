############################### h100 gangnam_rwf2000_gj

PYTHONPATH="$(pwd)" torchrun --nproc_per_node=4 src/evaluation/evaluate_video_classfication_edit.py \
    --checkpoint ckpts/InternVL3-2B_gangnam_rwf2000_gj \
    --annotation data/instruction/evaluation/test_gj.jsonl \
    --video-root data \
    --out-dir results/eval_result \
    --num-frames 12 \
    --workers-per-gpu 16

PYTHONPATH="$(pwd)" torchrun --nproc_per_node=4 src/evaluation/evaluate_video_classfication_edit.py \
    --checkpoint ckpts/InternVL3-2B_gangnam_rwf2000_gj \
    --annotation data/instruction/evaluation/test_rwf2000.jsonl \
    --video-root data \
    --out-dir results/eval_result \
    --num-frames 12 \
    --workers-per-gpu 16

PYTHONPATH="$(pwd)" torchrun --nproc_per_node=4 src/evaluation/evaluate_video_classfication_edit.py \
    --checkpoint ckpts/InternVL3-2B_gangnam_rwf2000_gj \
    --annotation data/instruction/evaluation/test_gangnam.jsonl \
    --video-root data \
    --out-dir results/eval_result \
    --num-frames 12 \
    --workers-per-gpu 16

PYTHONPATH="$(pwd)" torchrun --nproc_per_node=4 src/evaluation/evaluate_video_classfication_edit.py \
    --checkpoint ckpts/InternVL3-2B_gangnam_rwf2000_gj \
    --annotation data/instruction/evaluation/test_gangnam_vietnam.jsonl \
    --video-root data \
    --out-dir results/eval_result \
    --num-frames 12 \
    --workers-per-gpu 16


PYTHONPATH="$(pwd)" torchrun --nproc_per_node=4 src/evaluation/evaluate_video_classfication_edit.py \
    --checkpoint ckpts/InternVL3-2B_gangnam_rwf2000_gj \
    --annotation data/instruction/evaluation/test_scvdALL_NOweapon.jsonl \
    --video-root data \
    --out-dir results/eval_result \
    --num-frames 12 \
    --workers-per-gpu 16

PYTHONPATH="$(pwd)" torchrun --nproc_per_node=4 src/evaluation/evaluate_video_classfication_edit.py \
    --checkpoint ckpts/InternVL3-2B_gangnam_rwf2000_gj \
    --annotation data/instruction/evaluation/test_cctv.jsonl \
    --video-root data \
    --out-dir results/eval_result \
    --num-frames 12 \
    --workers-per-gpu 16