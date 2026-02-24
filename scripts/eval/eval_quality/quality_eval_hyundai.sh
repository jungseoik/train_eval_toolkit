## 현대백화점 영상 hyundai_2_10

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


## 현대백화점 영상 hyundai_2_15

PYTHONPATH="$(pwd)" python src/evaluation/evaluate_qualitative_video.py \
    --checkpoint ckpts/InternVL3-2B_hyundai_2_15 \
    --input-root "data/processed/hyundai_backhwajum/hyundai_video_macs_test/falldown" \
    --output-root "results/eval_quality/eva_quality_hyundai/hyundai_2_15/falldown_poc" \
    --window-size 15 \
    --batch-size 10 \
    --multi-gpu


## 현대백화점 영상 hyundai_2_20

PYTHONPATH="$(pwd)" python src/evaluation/evaluate_qualitative_video.py \
    --checkpoint ckpts/InternVL3-2B_hyundai_2_20 \
    --input-root "data/processed/hyundai_backhwajum/hyundai_video_macs_test/falldown" \
    --output-root "results/eval_quality/eva_quality_hyundai/hyundai_2_20/falldown_poc" \
    --window-size 15 \
    --batch-size 15 \
    --multi-gpu

## 현대백화점 영상 InternVL3-2B

PYTHONPATH="$(pwd)" python src/evaluation/evaluate_qualitative_video.py \
    --checkpoint ckpts/InternVL3-2B \
    --input-root "data/processed/hyundai_backhwajum/hyundai_video_macs_test/falldown" \
    --output-root "results/eval_quality/eva_quality_hyundai/InternVL3-2B/falldown_poc" \
    --window-size 15 \
    --batch-size 15 \
    --multi-gpu

## 현대백화점 영상 InternVL3-2B_hyundai_15

PYTHONPATH="$(pwd)" python src/evaluation/evaluate_qualitative_video.py \
    --checkpoint ckpts/InternVL3-2B_hyundai_15 \
    --input-root "data/processed/hyundai_backhwajum/hyundai_video_macs_test/falldown" \
    --output-root "results/eval_quality/eva_quality_hyundai/InternVL3-2B_hyundai_15/falldown_poc" \
    --window-size 15 \
    --batch-size 10 \
    --multi-gpu

## 현대백화점 영상 InternVL3-2B_hyundai_15

PYTHONPATH="$(pwd)" python src/evaluation/evaluate_qualitative_video.py \
    --checkpoint ckpts/InternVL3-2B_hyundai_5_20 \
    --input-root "/mnt/nas_192/Project/hyundai_backhwajum/general_video" \
    --output-root "/mnt/nas_192/Project/hyundai_backhwajum/general_video/seoik_vqa_result" \
    --window-size 15 \
    --batch-size 20 \
    --multi-gpu

PYTHONPATH="$(pwd)" python src/evaluation/evaluate_qualitative_video.py \
    --checkpoint ckpts/InternVL3-2B_hyundai_5_20 \
    --input-root "/mnt/nas_192/Project/hyundai_backhwajum/general_video" \
    --output-root "/mnt/nas_192/Project/hyundai_backhwajum/general_video/seoik_vqa_result" \
    --window-size 15 \
    --batch-size 20 \
    --multi-gpu

## 현대백화점 영상 InternVL3-2B_hyundai_5_20 -> 신규버전

# a6000
PYTHONPATH="$(pwd)" python src/evaluation/evaluate_qualitative_video_thresholde_image.py \
    --checkpoint ckpts/InternVL3-2B_hyundai_5_20 \
    --input-root "/home/piawsa6000/nas192/Project/hyundai_backhwajum/01_16_ collection_data/video_preprocess/gen_ai/videos/" \
    --output-root "/home/piawsa6000/nas192/Project/hyundai_backhwajum/general_video/seoik_vqa_result/gen_ai" \
    --window-size 15 \
    --batch-size 10 \
    --threshold 2 \
    --multi-gpu
# h100
PYTHONPATH="$(pwd)" python src/evaluation/evaluate_qualitative_video_thresholde_image.py \
    --checkpoint ckpts/InternVL3-2B_hyundai_5_20 \
    --input-root "/mnt/nas_192/Project/hyundai_backhwajum/01_16_ collection_data/video_preprocess/gen_ai/videos/" \
    --output-root "/mnt/nas_192/Project/hyundai_backhwajum/general_video/seoik_vqa_result/gen_ai" \
    --window-size 15 \
    --batch-size 10 \
    --threshold 2 \
    --multi-gpu

#### 일반
    # --checkpoint ckpts/InternVL3-2B_hyundai_5_20 \

PYTHONPATH="$(pwd)" python src/evaluation/evaluate_qualitative_video.py \
    --checkpoint ckpts/InternVL3-2B_hyundai_5_20 \
    --input-root "/mnt/nas_192/Project/hyundai_backhwajum/01_16_ collection_data/video_preprocess/gen_ai/videos/" \
    --output-root "/mnt/nas_192/Project/hyundai_backhwajum/general_video/seoik_vqa_result/gen_ai" \
    --window-size 6 \
    --batch-size 40 \
    --multi-gpu

PYTHONPATH="$(pwd)" python src/evaluation/evaluate_qualitative_video.py \
    --checkpoint ckpts/InternVL3-2B_hyundai_5_20 \
    --input-root "/mnt/nas_192/Project/hyundai_backhwajum/general_video" \
    --output-root "/mnt/nas_192/Project/hyundai_backhwajum/general_video/seoik_vqa_result" \
    --window-size 15 \
    --batch-size 40 \
    --multi-gpu

#### a 6000 테스트 
###/mnt/nas_192tb/Project/hyundai_backhwajum/general_video/seoik_vqa_result/tmp/falldown/keyframe_25
## --input-root "/volume1/AI_data/Project/hyundai_backhwajum/general_video/seoik_vqa_result/tmp/g15_fraem/" \
PYTHONPATH="$(pwd)" python src/evaluation/evaluate_qualitative_video_threshold_image.py \
    --checkpoint ckpts/InternVL3-2B_hyundai_5_20 \
    --input-root "/home/piawsa6000/nas192/Project/hyundai_backhwajum/general_video/seoik_vqa_result/tmp/g15" \
    --output-root "/home/piawsa6000/nas192/Project/hyundai_backhwajum/general_video/seoik_vqa_result/gen_ai" \
    --window-size 15 \
    --batch-size 10 \
    --threshold 2 \
    --multi-gpu

PYTHONPATH="$(pwd)" python src/evaluation/evaluate_qualitative_video.py \
    --checkpoint ckpts/InternVL3-2B_hyundai_5_20 \
    --input-root "/home/piawsa6000/nas192/Project/hyundai_backhwajum/general_video/seoik_vqa_result/tmp/tmp_test" \
    --output-root "/home/piawsa6000/nas192/Project/hyundai_backhwajum/general_video/seoik_vqa_result/tmp" \
    --window-size 30 \
    --batch-size 1 \
    --multi-gpu

# h100 
PYTHONPATH="$(pwd)" python src/evaluation/evaluate_qualitative_video_threshold_image.py \
    --checkpoint ckpts/InternVL3-2B_hyundai_5_20 \
    --input-root "data/processed/hyundai_backhwajum/hyundai_video_macs_test/01_27" \
    --output-root "results/eval_quality/eva_quality_hyundai/InternVL3-2B_hyundai_5_20/falldown_poc_01_27" \
    --window-size 15 \
    --batch-size 40 \
    --threshold 1 \
    --multi-gpu