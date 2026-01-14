## 현대백화점 영상

PYTHONPATH="$(pwd)" python src/evaluation/evaluate_qualitative_video.py \
    --checkpoint ckpts/InternVL3-2B_hyundai_2_10 \
    --input-root "data/processed/hyundai_backhwajum/hyundai_video_macs_test/falldown" \
    --output-root "results/eval_quality/eva_quality_hyundai/falldown_poc" \
    --window-size 15 \
    --batch-size 40 \
    --multi-gpu

PYTHONPATH="$(pwd)" python src/evaluation/evaluate_qualitative_video.py \
    --checkpoint ckpts/InternVL3-2B_hyundai_2_10 \
    --input-root "data/processed/hyundai_backhwajum/hyundai_video_macs_test/normal" \
    --output-root "results/eval_quality/eva_quality_hyundai/normal_poc" \
    --window-size 15 \
    --batch-size 10 \
    --multi-gpu


