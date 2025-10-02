############################### h100 gangnam_rwf2000_gj_cctv_scvdALL_NOweapon_no_split

PYTHONPATH="$(pwd)" torchrun --nproc_per_node=4 src/evaluation/evaluate_video_classfication_edit.py \
    --checkpoint ckpts/InternVL3-2B_gangnam_rwf2000_gj_cctv_scvdALL_NOweapon_no_split \
    --annotation data/instruction/evaluation/test_gangnam_yeoksam2st_violence.jsonl \
    --video-root data \
    --out-dir results/eval_quality_gangnam/video_eval \
    --num-frames 12 \
    --workers-per-gpu 16

PYTHONPATH="$(pwd)" torchrun --nproc_per_node=4 src/evaluation/evaluate_video_classfication_edit.py \
    --checkpoint ckpts/InternVL3-2B_gangnam_rwf2000_gj_cctv_scvdALL_NOweapon_no_split \
    --annotation data/instruction/evaluation/test_gangnam_gaepo1_violence.jsonl \
    --video-root data \
    --out-dir results/eval_quality_gangnam/video_eval \
    --num-frames 12 \
    --workers-per-gpu 16

############################### h100 gangnam_rwf2000_gj_cctv_scvdALL_NOweapon_no_split

PYTHONPATH="$(pwd)" torchrun --nproc_per_node=4 src/evaluation/evaluate_video_classfication_edit.py \
    --checkpoint ckpts/InternVL3-2B_gangnam_rwf2000 \
    --annotation data/instruction/evaluation/test_gangnam_yeoksam2st_violence.jsonl \
    --video-root data \
    --out-dir results/eval_quality_gangnam/video_eval \
    --num-frames 12 \
    --workers-per-gpu 16

PYTHONPATH="$(pwd)" torchrun --nproc_per_node=4 src/evaluation/evaluate_video_classfication_edit.py \
    --checkpoint ckpts/InternVL3-2B_gangnam_rwf2000 \
    --annotation data/instruction/evaluation/test_gangnam_gaepo1_violence.jsonl \
    --video-root data \
    --out-dir results/eval_quality_gangnam/video_eval \
    --num-frames 12 \
    --workers-per-gpu 16