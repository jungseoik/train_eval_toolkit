############################### h100 InternVL3-2B_gangnam_yeoksam_v2_gaepo4_1v2_samsung_rwf2000

#### 폭력 비디오 평가

PYTHONPATH="$(pwd)" torchrun --nproc_per_node=4 src/evaluation/evaluate_video_classfication_edit.py \
    --checkpoint ckpts/InternVL3-2B_gangnam_yeoksam_v2_gaepo4_1v2_samsung_rwf2000 \
    --annotation data/instruction/evaluation/test_gangnam_samsung_video_violence.jsonl \
    --video-root data \
    --out-dir results/eval_result \
    --num-frames 12 \
    --prompt-type violence \
    --workers-per-gpu 16
    
PYTHONPATH="$(pwd)" torchrun --nproc_per_node=4 src/evaluation/evaluate_video_classfication_edit.py \
    --checkpoint ckpts/InternVL3-2B_gangnam_yeoksam_v2_gaepo4_1v2_samsung_rwf2000 \
    --annotation data/instruction/evaluation/test_gangnam_gaepo4_video_violence.jsonl \
    --video-root data \
    --out-dir results/eval_result \
    --num-frames 12 \
    --prompt-type violence \
    --workers-per-gpu 16

PYTHONPATH="$(pwd)" torchrun --nproc_per_node=4 src/evaluation/evaluate_video_classfication_edit.py \
    --checkpoint ckpts/InternVL3-2B_gangnam_yeoksam_v2_gaepo4_1v2_samsung_rwf2000 \
    --annotation data/instruction/evaluation/test_gangnam_gaepo1_v2_video_violence.jsonl \
    --video-root data \
    --out-dir results/eval_result \
    --num-frames 12 \
    --prompt-type violence \
    --workers-per-gpu 16

PYTHONPATH="$(pwd)" torchrun --nproc_per_node=4 src/evaluation/evaluate_video_classfication_edit.py \
    --checkpoint ckpts/InternVL3-2B_gangnam_yeoksam_v2_gaepo4_1v2_samsung_rwf2000 \
    --annotation data/instruction/evaluation/test_gangnam_yeoksam2_v2_video_violence.jsonl \
    --video-root data \
    --out-dir results/eval_result \
    --num-frames 12 \
    --prompt-type violence \
    --workers-per-gpu 16


#### 쓰러짐 비디오 평가

PYTHONPATH="$(pwd)" torchrun --nproc_per_node=4 src/evaluation/evaluate_video_classfication_edit.py \
    --checkpoint ckpts/InternVL3-2B_gangnam_yeoksam_v2_gaepo4_1v2_samsung_rwf2000 \
    --annotation data/instruction/evaluation/test_gangnam_samsung_video_falldown.jsonl \
    --video-root data \
    --out-dir results/eval_result \
    --num-frames 12 \
    --prompt-type falldown \
    --workers-per-gpu 16
    
PYTHONPATH="$(pwd)" torchrun --nproc_per_node=4 src/evaluation/evaluate_video_classfication_edit.py \
    --checkpoint ckpts/InternVL3-2B_gangnam_yeoksam_v2_gaepo4_1v2_samsung_rwf2000 \
    --annotation data/instruction/evaluation/test_gangnam_gaepo4_video_falldown.jsonl \
    --video-root data \
    --out-dir results/eval_result \
    --num-frames 12 \
    --prompt-type falldown \
    --workers-per-gpu 16

PYTHONPATH="$(pwd)" torchrun --nproc_per_node=4 src/evaluation/evaluate_video_classfication_edit.py \
    --checkpoint ckpts/InternVL3-2B_gangnam_yeoksam_v2_gaepo4_1v2_samsung_rwf2000 \
    --annotation data/instruction/evaluation/test_gangnam_gaepo1_v2_video_falldown.jsonl \
    --video-root data \
    --out-dir results/eval_result \
    --num-frames 12 \
    --prompt-type falldown \
    --workers-per-gpu 16
    
PYTHONPATH="$(pwd)" torchrun --nproc_per_node=4 src/evaluation/evaluate_video_classfication_edit.py \
    --checkpoint ckpts/InternVL3-2B_gangnam_yeoksam_v2_gaepo4_1v2_samsung_rwf2000 \
    --annotation data/instruction/evaluation/test_gangnam_yeoksam2_v2_video_falldown.jsonl \
    --video-root data \
    --out-dir results/eval_result \
    --num-frames 12 \
    --prompt-type falldown \
    --workers-per-gpu 16

##################################### 기존 평가셋
PYTHONPATH="$(pwd)" torchrun --nproc_per_node=4 src/evaluation/evaluate_video_classfication_edit.py \
    --checkpoint ckpts/InternVL3-2B_gangnam_yeoksam_v2_gaepo4_1v2_samsung_rwf2000 \
    --annotation data/instruction/evaluation/test_gangnam.jsonl \
    --video-root data \
    --out-dir results/eval_result \
    --prompt-type violence \
    --num-frames 12 \
    --workers-per-gpu 16


PYTHONPATH="$(pwd)" torchrun --nproc_per_node=4 src/evaluation/evaluate_video_classfication_edit.py \
    --checkpoint ckpts/InternVL3-2B_gangnam_yeoksam_v2_gaepo4_1v2_samsung_rwf2000 \
    --annotation data/instruction/evaluation/test_gangnam_yeoksam2st_violence.jsonl \
    --video-root data \
    --out-dir results/eval_result \
    --prompt-type violence \
    --num-frames 12 \
    --workers-per-gpu 16

PYTHONPATH="$(pwd)" torchrun --nproc_per_node=4 src/evaluation/evaluate_video_classfication_edit.py \
    --checkpoint ckpts/InternVL3-2B_gangnam_yeoksam_v2_gaepo4_1v2_samsung_rwf2000 \
    --annotation data/instruction/evaluation/test_gangnam_gaepo1_violence.jsonl \
    --video-root data \
    --out-dir results/eval_result \
    --prompt-type violence \
    --num-frames 12 \
    --workers-per-gpu 16