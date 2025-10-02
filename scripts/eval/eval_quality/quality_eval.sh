### ABB falldown video 퀄리티

PYTHONPATH="$(pwd)" python src/evaluation/evaluate_qualitative_video.py \
    --checkpoint ckpts/InternVL3-2B \
    --input-root "data/processed/ABB_banwoldang/PE/Test/falldown/raw" \
    --output-root "results/eval_quality" \
    --window-size 15 \
    --batch-size 20 \
    --multi-gpu


PYTHONPATH="$(pwd)" python src/evaluation/evaluate_qualitative_video.py \
    --checkpoint ckpts/InternVL3-2B \
    --input-root "data/processed/ABB_banwoldang/" \
    --output-root "results/eval_quality" \
    --window-size 15 \
    --batch-size 20 \
    --multi-gpu

### 강남구 이노베이션 퀄리티

PYTHONPATH="$(pwd)" python src/evaluation/evaluate_qualitative_video_violence.py \
    --checkpoint ckpts/InternVL3-2B_gangnam_rwf2000_gj_cctv_scvdALL_NOweapon_no_split \
    --input-root "data/processed/gangnam/gaepo1/Test_dataset_gaepo1st/violence" \
    --output-root "results/eval_quality_gangnam/video_quality" \
    --window-size 12 \
    --batch-size 20 \
    --multi-gpu

PYTHONPATH="$(pwd)" python src/evaluation/evaluate_qualitative_video_violence.py \
    --checkpoint ckpts/InternVL3-2B_gangnam_rwf2000_gj_cctv_scvdALL_NOweapon_no_split \
    --input-root "data/processed/gangnam/yeoksam2/Test_dataset_yeoksam2st/violence" \
    --output-root "results/eval_quality_gangnam/video_quality/yeoksam2st" \
    --window-size 12 \
    --batch-size 20 \
    --multi-gpu


PYTHONPATH="$(pwd)" python src/evaluation/evaluate_qualitative_video_violence.py \
    --checkpoint ckpts/InternVL3-2B_gangnam_rwf2000_gj_cctv_scvdALL_NOweapon_no_split \
    --input-root "data/raw/elevator_violence" \
    --output-root "results/eval_quality_gangnam/video_quality/gaepo4" \
    --window-size 12 \
    --batch-size 1 \
    --multi-gpu


########falldown
PYTHONPATH="$(pwd)" python src/evaluation/evaluate_qualitative_video.py \
    --checkpoint ckpts/InternVL3-2B \
    --input-root "data/processed/gangnam/gaepo1/Train_dataset_gaepo1st/falldown/falldown/raw" \
    --output-root "results/eval_quality_gangnam/video_quality/gaepo1st" \
    --window-size 15 \
    --batch-size 20 \
    --multi-gpu


PYTHONPATH="$(pwd)" python src/evaluation/evaluate_qualitative_video.py \
    --checkpoint ckpts/InternVL3-2B \
    --input-root "data/processed/gangnam/yeoksam2/Train_dataset_yeoksam2st/falldown/falldown/raw" \
    --output-root "results/eval_quality_gangnam/video_quality/yeoksam2st" \
    --window-size 15 \
    --batch-size 20 \
    --multi-gpu