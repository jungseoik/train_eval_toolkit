
###### 강남 역삼2동_v2 test
PYTHONPATH="$(pwd)" python src/evaluation/evaluate_image_classfication.py \
    --checkpoint ckpts/InternVL3-2B_gangnam_rwf2000_gj_cctv_scvdALL_NOweapon_no_split_v2 \
    --annotation data/instruction/evaluation/test_gangnam_yeoksam2_v2_image_falldown.jsonl \
    --image-root data \
    --out-dir results/eval_result_image/gangnam/InternVL3-2B_gangnam_rwf2000_gj_cctv_scvdALL_NOweapon_no_split_v2 \
    --batch-size 20 \
    --multi-gpu
    
###### 강남 개포1동_v2 test
PYTHONPATH="$(pwd)" python src/evaluation/evaluate_image_classfication.py \
    --checkpoint ckpts/InternVL3-2B_gangnam_rwf2000_gj_cctv_scvdALL_NOweapon_no_split_v2 \
    --annotation data/instruction/evaluation/test_gangnam_gaepo1_v2_image_falldown.jsonl \
    --image-root data \
    --out-dir results/eval_result_image/gangnam/InternVL3-2B_gangnam_rwf2000_gj_cctv_scvdALL_NOweapon_no_split_v2 \
    --batch-size 10 \
    --multi-gpu

###### 강남 개포4동 test
PYTHONPATH="$(pwd)" python src/evaluation/evaluate_image_classfication.py \
    --checkpoint ckpts/InternVL3-2B_gangnam_rwf2000_gj_cctv_scvdALL_NOweapon_no_split_v2 \
    --annotation data/instruction/evaluation/test_gangnam_gaepo4_image_falldown.jsonl \
    --image-root data \
    --out-dir results/eval_result_image/gangnam/InternVL3-2B_gangnam_rwf2000_gj_cctv_scvdALL_NOweapon_no_split_v2 \
    --batch-size 10 \
    --multi-gpu

###### 강남 삼성동 test
PYTHONPATH="$(pwd)" python src/evaluation/evaluate_image_classfication.py \
    --checkpoint ckpts/InternVL3-2B_gangnam_rwf2000_gj_cctv_scvdALL_NOweapon_no_split_v2 \
    --annotation data/instruction/evaluation/test_gangnam_samsung_image_falldown.jsonl \
    --image-root data \
    --out-dir results/eval_result_image/gangnam/InternVL3-2B_gangnam_rwf2000_gj_cctv_scvdALL_NOweapon_no_split_v2 \
    --batch-size 20 \
    --multi-gpu

##############################################################################

###### 강남 역삼2동_v2 test
PYTHONPATH="$(pwd)" python src/evaluation/evaluate_image_classfication.py \
    --checkpoint ckpts/InternVL3-2B_gangnam_rwf2000_gj_cctv_scvdALL_NOweapon_no_split_v2 \
    --annotation data/instruction/evaluation/test_gangnam_yeoksam2_v2_image_falldown.jsonl \
    --image-root data \
    --out-dir results/eval_result_image/gangnam/InternVL3-2B_gangnam_rwf2000_gj_cctv_scvdALL_NOweapon_no_split_v2 \
    --batch-size 20 \
    --multi-gpu
    
###### 강남 개포1동_v2 test
PYTHONPATH="$(pwd)" python src/evaluation/evaluate_image_classfication.py \
    --checkpoint ckpts/InternVL3-2B_gangnam_rwf2000_gj_cctv_scvdALL_NOweapon_no_split_v2 \
    --annotation data/instruction/evaluation/test_gangnam_gaepo1_v2_image_falldown.jsonl \
    --image-root data \
    --out-dir results/eval_result_image/gangnam/InternVL3-2B_gangnam_rwf2000_gj_cctv_scvdALL_NOweapon_no_split_v2 \
    --batch-size 20 \
    --multi-gpu

###### 강남 개포4동 test
PYTHONPATH="$(pwd)" python src/evaluation/evaluate_image_classfication.py \
    --checkpoint ckpts/InternVL3-2B_gangnam_rwf2000_gj_cctv_scvdALL_NOweapon_no_split_v2 \
    --annotation data/instruction/evaluation/test_gangnam_gaepo4_image_falldown.jsonl \
    --image-root data \
    --out-dir results/eval_result_image/gangnam/InternVL3-2B_gangnam_rwf2000_gj_cctv_scvdALL_NOweapon_no_split_v2 \
    --batch-size 20 \
    --multi-gpu

###### 강남 삼성동 test
PYTHONPATH="$(pwd)" python src/evaluation/evaluate_image_classfication.py \
    --checkpoint ckpts/InternVL3-2B_gangnam_rwf2000_gj_cctv_scvdALL_NOweapon_no_split_v2 \
    --annotation data/instruction/evaluation/test_gangnam_samsung_image_falldown.jsonl \
    --image-root data \
    --out-dir results/eval_result_image/gangnam/InternVL3-2B_gangnam_rwf2000_gj_cctv_scvdALL_NOweapon_no_split_v2 \
    --batch-size 20 \
    --multi-gpu