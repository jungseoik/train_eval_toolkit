PYTHONPATH="$(pwd)" torchrun --nproc_per_node=2 src/evaluation/evaluate_video_classfication_edit.py \
    --checkpoint ckpts/InternVL3-2B_gangnam_vietnam_rwf2000_aihubstore_gj_space_no_split \
    --annotation data/instruction/evaluation/test_gangnam.jsonl \
    --video-root data \
    --out-dir results/eval_result \
    --num-frames 12 \
    --workers-per-gpu 8

PYTHONPATH="$(pwd)" torchrun --nproc_per_node=2 src/evaluation/evaluate_video_classfication_edit.py \
    --checkpoint ckpts/InternVL3-2B_total_vio \
    --annotation data/instruction/evaluation/test_gangnam.jsonl \
    --video-root data \
    --out-dir results/eval_result \
    --num-frames 12 \
    --workers-per-gpu 8


PYTHONPATH="$(pwd)" torchrun --nproc_per_node=2 src/evaluation/evaluate_video_classfication_edit.py \
    --checkpoint ckpts/InternVL3-2B_gangnam_vietnam_aihubstore_gj_space_no_split \
    --annotation data/instruction/evaluation/test_gangnam.jsonl \
    --video-root data \
    --out-dir results/eval_result \
    --num-frames 12 \
    --workers-per-gpu 8

PYTHONPATH="$(pwd)" torchrun --nproc_per_node=2 src/evaluation/evaluate_video_classfication_edit.py \
    --checkpoint ckpts/InternVL3-2B_gangnam_vietnam_rwf2000_aihubstore_gj_space_no_split \
    --annotation data/instruction/evaluation/test_gangnam_vietnam.jsonl \
    --video-root data \
    --out-dir results/eval_result \
    --num-frames 12 \
    --workers-per-gpu 8

############################### InternVL3-2B 기본 모델 
PYTHONPATH="$(pwd)" torchrun --nproc_per_node=2 src/evaluation/evaluate_video_classfication_edit.py \
    --checkpoint ckpts/InternVL3-2B \
    --annotation data/instruction/evaluation/test_rwf2000.jsonl \
    --video-root data \
    --out-dir results/eval_result \
    --num-frames 12 \
    --workers-per-gpu 8

PYTHONPATH="$(pwd)" torchrun --nproc_per_node=2 src/evaluation/evaluate_video_classfication_edit.py \
    --checkpoint ckpts/InternVL3-2B \
    --annotation data/instruction/evaluation/test_gangnam_vietnam.jsonl \
    --video-root data \
    --out-dir results/eval_result \
    --num-frames 12 \
    --workers-per-gpu 8


############################### 강남구 데이터 학습 모델 

PYTHONPATH="$(pwd)" torchrun --nproc_per_node=2 src/evaluation/evaluate_video_classfication_edit.py \
    --checkpoint ckpts/InternVL3-2B_gangnam \
    --annotation data/instruction/evaluation/test_gangnam.jsonl \
    --video-root data \
    --out-dir results/eval_result \
    --num-frames 12 \
    --workers-per-gpu 8

PYTHONPATH="$(pwd)" torchrun --nproc_per_node=2 src/evaluation/evaluate_video_classfication_edit.py \
    --checkpoint ckpts/InternVL3-2B_gangnam \
    --annotation data/instruction/evaluation/test_rwf2000.jsonl \
    --video-root data \
    --out-dir results/eval_result \
    --num-frames 12 \
    --workers-per-gpu 8


PYTHONPATH="$(pwd)" torchrun --nproc_per_node=2 src/evaluation/evaluate_video_classfication_edit.py \
    --checkpoint ckpts/InternVL3-2B_gangnam \
    --annotation data/instruction/evaluation/test_gangnam_vietnam.jsonl \
    --video-root data \
    --out-dir results/eval_result \
    --num-frames 12 \
    --workers-per-gpu 8


PYTHONPATH="$(pwd)" torchrun --nproc_per_node=2 src/evaluation/evaluate_video_classfication_edit.py \
   --checkpoint ckpts/InternVL3-2B_gangnam \
    --annotation data/instruction/evaluation/test_gj.jsonl \
    --video-root data \
    --out-dir results/eval_result \
    --num-frames 12 \
    --workers-per-gpu 8

PYTHONPATH="$(pwd)" torchrun --nproc_per_node=2 src/evaluation/evaluate_video_classfication_edit.py \
   --checkpoint ckpts/InternVL3-2B_gangnam \
    --annotation data/instruction/evaluation/test_scvdALL_NOweapon.jsonl \
    --video-root data \
    --out-dir results/eval_result \
    --num-frames 12 \
    --workers-per-gpu 8



############################### RWF2000 데이터 학습 모델 

PYTHONPATH="$(pwd)" torchrun --nproc_per_node=2 src/evaluation/evaluate_video_classfication_edit.py \
    --checkpoint ckpts/InternVL3-2B_rwf2000 \
    --annotation data/instruction/evaluation/test_gangnam.jsonl \
    --video-root data \
    --out-dir results/eval_result \
    --num-frames 12 \
    --workers-per-gpu 8

PYTHONPATH="$(pwd)" torchrun --nproc_per_node=2 src/evaluation/evaluate_video_classfication_edit.py \
    --checkpoint ckpts/InternVL3-2B_rwf2000 \
    --annotation data/instruction/evaluation/test_rwf2000.jsonl \
    --video-root data \
    --out-dir results/eval_result \
    --num-frames 12 \
    --workers-per-gpu 8

PYTHONPATH="$(pwd)" torchrun --nproc_per_node=2 src/evaluation/evaluate_video_classfication_edit.py \
    --checkpoint ckpts/InternVL3-2B_rwf2000 \
    --annotation data/instruction/evaluation/test_gangnam_vietnam.jsonl \
    --video-root data \
    --out-dir results/eval_result \
    --num-frames 12 \
    --workers-per-gpu 8

PYTHONPATH="$(pwd)" torchrun --nproc_per_node=2 src/evaluation/evaluate_video_classfication_edit.py \
    --checkpoint ckpts/InternVL3-2B_rwf2000 \
    --annotation data/instruction/evaluation/test_gj.jsonl \
    --video-root data \
    --out-dir results/eval_result \
    --num-frames 12 \
    --workers-per-gpu 8

PYTHONPATH="$(pwd)" torchrun --nproc_per_node=2 src/evaluation/evaluate_video_classfication_edit.py \
    --checkpoint ckpts/InternVL3-2B_rwf2000 \
    --annotation data/instruction/evaluation/test_scvdALL_NOweapon.jsonl \
    --video-root data \
    --out-dir results/eval_result \
    --num-frames 12 \
    --workers-per-gpu 8


############################### RWF2000 + 강남구 데이터 학습 모델 

PYTHONPATH="$(pwd)" torchrun --nproc_per_node=2 src/evaluation/evaluate_video_classfication_edit.py \
    --checkpoint ckpts/InternVL3-2B_gangnam_rwf2000 \
    --annotation data/instruction/evaluation/test_gangnam.jsonl \
    --video-root data \
    --out-dir results/eval_result \
    --num-frames 12 \
    --workers-per-gpu 8

PYTHONPATH="$(pwd)" torchrun --nproc_per_node=2 src/evaluation/evaluate_video_classfication_edit.py \
    --checkpoint ckpts/InternVL3-2B_gangnam_rwf2000 \
    --annotation data/instruction/evaluation/test_rwf2000.jsonl \
    --video-root data \
    --out-dir results/eval_result \
    --num-frames 12 \
    --workers-per-gpu 8

PYTHONPATH="$(pwd)" torchrun --nproc_per_node=2 src/evaluation/evaluate_video_classfication_edit.py \
    --checkpoint ckpts/InternVL3-2B_gangnam_rwf2000 \
    --annotation data/instruction/evaluation/test_gangnam_vietnam.jsonl \
    --video-root data \
    --out-dir results/eval_result \
    --num-frames 12 \
    --workers-per-gpu 8
    
PYTHONPATH="$(pwd)" torchrun --nproc_per_node=2 src/evaluation/evaluate_video_classfication_edit.py \
    --checkpoint ckpts/InternVL3-2B_gangnam_rwf2000 \
    --annotation data/instruction/evaluation/test_gj.jsonl \
    --video-root data \
    --out-dir results/eval_result \
    --num-frames 12 \
    --workers-per-gpu 8

PYTHONPATH="$(pwd)" torchrun --nproc_per_node=2 src/evaluation/evaluate_video_classfication_edit.py \
    --checkpoint ckpts/InternVL3-2B_gangnam_rwf2000 \
    --annotation data/instruction/evaluation/test_scvdALL_NOweapon.jsonl \
    --video-root data \
    --out-dir results/eval_result \
    --num-frames 12 \
    --workers-per-gpu 8

############################### gj 모델 평가 진행

PYTHONPATH="$(pwd)" torchrun --nproc_per_node=2 src/evaluation/evaluate_video_classfication_edit.py \
    --checkpoint ckpts/InternVL3-2B_gj \
    --annotation data/instruction/evaluation/test_gj.jsonl \
    --video-root data \
    --out-dir results/eval_result \
    --num-frames 12 \
    --workers-per-gpu 8


PYTHONPATH="$(pwd)" torchrun --nproc_per_node=2 src/evaluation/evaluate_video_classfication_edit.py \
    --checkpoint ckpts/InternVL3-2B_gj \
    --annotation data/instruction/evaluation/test_rwf2000.jsonl \
    --video-root data \
    --out-dir results/eval_result \
    --num-frames 12 \
    --workers-per-gpu 8

PYTHONPATH="$(pwd)" torchrun --nproc_per_node=2 src/evaluation/evaluate_video_classfication_edit.py \
    --checkpoint ckpts/InternVL3-2B_gj \
    --annotation data/instruction/evaluation/test_gangnam.jsonl \
    --video-root data \
    --out-dir results/eval_result \
    --num-frames 12 \
    --workers-per-gpu 8

PYTHONPATH="$(pwd)" torchrun --nproc_per_node=2 src/evaluation/evaluate_video_classfication_edit.py \
    --checkpoint ckpts/InternVL3-2B_gj \
    --annotation data/instruction/evaluation/test_gangnam_vietnam.jsonl \
    --video-root data \
    --out-dir results/eval_result \
    --num-frames 12 \
    --workers-per-gpu 8


PYTHONPATH="$(pwd)" torchrun --nproc_per_node=2 src/evaluation/evaluate_video_classfication_edit.py \
    --checkpoint ckpts/InternVL3-2B_gj \
    --annotation data/instruction/evaluation/test_scvdALL_NOweapon.jsonl \
    --video-root data \
    --out-dir results/eval_result \
    --num-frames 12 \
    --workers-per-gpu 8

