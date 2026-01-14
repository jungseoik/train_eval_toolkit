### 강남구 이노베이션 퀄리티 ### shadow 복싱

### ### ### 2차 테스트 base 모델 5에폭
######
PYTHONPATH="$(pwd)" python src/evaluation/evaluate_qualitative_video.py \
    --checkpoint ckpts/InternVL3-2B_gangnam_yeoksam_v2_gaepo4_1v2_samsung_rwf2000 \
    --input-root "data/processed/gangnam/gaepo1_v2/Test/video/falldown/falldown" \
    --output-root "results/eval_quality_gangnam/video_quality/result_qa2/falldown//1" \
    --window-size 15 \
    --batch-size 10 \
    --multi-gpu

PYTHONPATH="$(pwd)" python src/evaluation/evaluate_qualitative_video.py \
    --checkpoint ckpts/InternVL3-2B_gangnam_yeoksam_v2_gaepo4_1v2_samsung_rwf2000 \
    --input-root "data/processed/gangnam/gaepo1_v2/Test/video/falldown/normal/clip" \
    --output-root "results/eval_quality_gangnam/video_quality/result_qa2/falldown//1" \
    --window-size 15 \
    --batch-size 10 \
    --multi-gpu


PYTHONPATH="$(pwd)" python src/evaluation/evaluate_qualitative_video.py \
    --checkpoint ckpts/InternVL3-2B_gangnam_yeoksam_v2_gaepo4_1v2_samsung_rwf2000 \
    --input-root "data/processed/gangnam/yeoksam2_v2/Test/video/falldown/normal/clip" \
    --output-root "results/eval_quality_gangnam/video_quality/result_qa2/falldown//2" \
    --window-size 15 \
    --batch-size 10 \
    --multi-gpu

PYTHONPATH="$(pwd)" python src/evaluation/evaluate_qualitative_video.py \
    --checkpoint ckpts/InternVL3-2B_gangnam_yeoksam_v2_gaepo4_1v2_samsung_rwf2000 \
    --input-root "data/processed/gangnam/yeoksam2_v2/Test/video/falldown/falldown" \
    --output-root "results/eval_quality_gangnam/video_quality/result_qa2/falldown//2" \
    --window-size 15 \
    --batch-size 10 \
    --multi-gpu

PYTHONPATH="$(pwd)" python src/evaluation/evaluate_qualitative_video.py \
    --checkpoint ckpts/InternVL3-2B_gangnam_yeoksam_v2_gaepo4_1v2_samsung_rwf2000 \
    --input-root "data/processed/gangnam/gaepo4/Test/clean/video/falldown/clip" \
    --output-root "results/eval_quality_gangnam/video_quality/result_qa2/falldown//3" \
    --window-size 15 \
    --batch-size 10 \
    --multi-gpu

# PYTHONPATH="$(pwd)" python src/evaluation/evaluate_qualitative_video.py \
#     --checkpoint ckpts/InternVL3-2B_gangnam_yeoksam_v2_gaepo4_1v2_samsung_rwf2000 \
#     --input-root "data/processed/gangnam/gaepo4/Test/clean/video/violence/raw/violence" \
#     --output-root "results/eval_quality_gangnam/video_quality/result_qa2/falldown//3" \
#     --window-size 15 \
#     --batch-size 10 \
#     --multi-gpu

PYTHONPATH="$(pwd)" python src/evaluation/evaluate_qualitative_video.py \
    --checkpoint ckpts/InternVL3-2B_gangnam_yeoksam_v2_gaepo4_1v2_samsung_rwf2000 \
    --input-root "data/processed/gangnam/samsung/Test/clean/video/falldown/falldown" \
    --output-root "results/eval_quality_gangnam/video_quality/result_qa2/falldown//4" \
    --window-size 15 \
    --batch-size 10 \
    --multi-gpu


PYTHONPATH="$(pwd)" python src/evaluation/evaluate_qualitative_video.py \
    --checkpoint ckpts/InternVL3-2B_gangnam_yeoksam_v2_gaepo4_1v2_samsung_rwf2000 \
    --input-root "data/processed/gangnam/samsung/Test/clean/video/falldown/normal" \
    --output-root "results/eval_quality_gangnam/video_quality/result_qa2/falldown//4" \
    --window-size 15 \
    --batch-size 10 \
    --multi-gpu
### 강남구 이노베이션 퀄리티 ### shadow 복싱

### ### ### 2차 테스트 base 모델 5에폭
######
PYTHONPATH="$(pwd)" python src/evaluation/evaluate_qualitative_video.py \
    --checkpoint ckpts/InternVL3-2B_gangnam_gj_yeoksam_gaepo_rwf2000_fallown_violence \
    --input-root "data/processed/gangnam/gaepo1_v2/Test/video/falldown/falldown" \
    --output-root "results/eval_quality_gangnam/video_quality/result_qa1/falldown//1" \
    --window-size 15 \
    --batch-size 10 \
    --multi-gpu

PYTHONPATH="$(pwd)" python src/evaluation/evaluate_qualitative_video.py \
    --checkpoint ckpts/InternVL3-2B_gangnam_gj_yeoksam_gaepo_rwf2000_fallown_violence \
    --input-root "data/processed/gangnam/gaepo1_v2/Test/video/falldown/normal/clip" \
    --output-root "results/eval_quality_gangnam/video_quality/result_qa1/falldown//1" \
    --window-size 15 \
    --batch-size 10 \
    --multi-gpu


PYTHONPATH="$(pwd)" python src/evaluation/evaluate_qualitative_video.py \
    --checkpoint ckpts/InternVL3-2B_gangnam_gj_yeoksam_gaepo_rwf2000_fallown_violence \
    --input-root "data/processed/gangnam/yeoksam2_v2/Test/video/falldown/normal/clip" \
    --output-root "results/eval_quality_gangnam/video_quality/result_qa1/falldown//2" \
    --window-size 15 \
    --batch-size 10 \
    --multi-gpu

PYTHONPATH="$(pwd)" python src/evaluation/evaluate_qualitative_video.py \
    --checkpoint ckpts/InternVL3-2B_gangnam_gj_yeoksam_gaepo_rwf2000_fallown_violence \
    --input-root "data/processed/gangnam/yeoksam2_v2/Test/video/falldown/falldown" \
    --output-root "results/eval_quality_gangnam/video_quality/result_qa1/falldown//2" \
    --window-size 15 \
    --batch-size 10 \
    --multi-gpu

PYTHONPATH="$(pwd)" python src/evaluation/evaluate_qualitative_video.py \
    --checkpoint ckpts/InternVL3-2B_gangnam_gj_yeoksam_gaepo_rwf2000_fallown_violence \
    --input-root "data/processed/gangnam/gaepo4/Test/clean/video/falldown/clip" \
    --output-root "results/eval_quality_gangnam/video_quality/result_qa1/falldown//3" \
    --window-size 15 \
    --batch-size 10 \
    --multi-gpu

# PYTHONPATH="$(pwd)" python src/evaluation/evaluate_qualitative_video.py \
#     --checkpoint ckpts/InternVL3-2B_gangnam_yeoksam_v2_gaepo4_1v2_samsung_rwf2000 \
#     --input-root "data/processed/gangnam/gaepo4/Test/clean/video/violence/raw/violence" \
#     --output-root "results/eval_quality_gangnam/video_quality/result_qa2/falldown//3" \
#     --window-size 15 \
#     --batch-size 10 \
#     --multi-gpu

PYTHONPATH="$(pwd)" python src/evaluation/evaluate_qualitative_video.py \
    --checkpoint ckpts/InternVL3-2B_gangnam_gj_yeoksam_gaepo_rwf2000_fallown_violence \
    --input-root "data/processed/gangnam/samsung/Test/clean/video/falldown/falldown" \
    --output-root "results/eval_quality_gangnam/video_quality/result_qa1/falldown//4" \
    --window-size 15 \
    --batch-size 10 \
    --multi-gpu


PYTHONPATH="$(pwd)" python src/evaluation/evaluate_qualitative_video.py \
    --checkpoint ckpts/InternVL3-2B_gangnam_gj_yeoksam_gaepo_rwf2000_fallown_violence \
    --input-root "data/processed/gangnam/samsung/Test/clean/video/falldown/normal" \
    --output-root "results/eval_quality_gangnam/video_quality/result_qa1/falldown//4" \
    --window-size 15 \
    --batch-size 10 \
    --multi-gpu
    # --checkpoint ckpts/InternVL3-2B_gangnam_gj_yeoksam_gaepo_rwf2000_fallown_violence \
    # --checkpoint ckpts/InternVL3-2B_gangnam_rwf2000_gj_cctv_scvdALL_NOweapon_no_split \
