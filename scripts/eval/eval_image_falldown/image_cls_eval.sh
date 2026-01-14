###### ABB VQA test
PYTHONPATH="$(pwd)" python src/evaluation/evaluate_image_classfication.py \
    --checkpoint ckpts/InternVL3-2B \
    --annotation data/instruction/evaluation/test_abb_vqa.jsonl \
    --image-root data \
    --out-dir results/eval_result_image \
    --batch-size 40 \
    --multi-gpu
###### ABB PE test
PYTHONPATH="$(pwd)" python src/evaluation/evaluate_image_classfication.py \
    --checkpoint ckpts/InternVL3-2B \
    --annotation data/instruction/evaluation/test_abb_pe.jsonl \
    --image-root data \
    --out-dir results/eval_result_image \
    --batch-size 40 \
    --multi-gpu

###### ABB VQA train test
PYTHONPATH="$(pwd)" python src/evaluation/evaluate_image_classfication.py \
    --checkpoint ckpts/InternVL3-2B \
    --annotation data/instruction/evaluation/test_train_abb_vqa.jsonl \
    --image-root data \
    --out-dir results/eval_result_image \
    --batch-size 10 \
    --multi-gpu
###### ABB PE train test
PYTHONPATH="$(pwd)" python src/evaluation/evaluate_image_classfication.py \
    --checkpoint ckpts/InternVL3-2B \
    --annotation data/instruction/evaluation/test_train_abb_pe.jsonl \
    --image-root data \
    --out-dir results/eval_result_image \
    --batch-size 10 \
    --multi-gpu




###### 강남 역삼2동 test
PYTHONPATH="$(pwd)" python src/evaluation/evaluate_image_classfication.py \
    --checkpoint ckpts/InternVL3-2B \
    --annotation data/instruction/evaluation/test_gangnam_gaepo1_falldown.jsonl \
    --image-root data \
    --out-dir results/eval_result_image/gangnam \
    --batch-size 40 \
    --multi-gpu
    
###### 강남 개포4동 test
PYTHONPATH="$(pwd)" python src/evaluation/evaluate_image_classfication.py \
    --checkpoint ckpts/InternVL3-2B_gangnam_gj_yeoksam_v2_gaepo4_1v2_samsung_rwf2000_fallown_violence \
    --annotation data/instruction/evaluation/test_gangnam_yeoksam2st_falldown.jsonl \
    --image-root data \
    --out-dir results/eval_result_image/gangnam \
    --batch-size 40 \
    --multi-gpu


###### 강남 역삼2동_v2 test
PYTHONPATH="$(pwd)" python src/evaluation/evaluate_image_classfication.py \
    --checkpoint ckpts/InternVL3-2B_gangnam_gj_yeoksam_v2_gaepo4_1v2_samsung_rwf2000_fallown_violence \
    --annotation data/instruction/evaluation/test_gangnam_yeoksam2_v2_image_falldown.jsonl \
    --image-root data \
    --out-dir results/eval_result_image/gangnam \
    --batch-size 40 \
    --multi-gpu
    
###### 강남 개포1동_v2 test
PYTHONPATH="$(pwd)" python src/evaluation/evaluate_image_classfication.py \
    --checkpoint ckpts/InternVL3-2B_gangnam_gj_yeoksam_v2_gaepo4_1v2_samsung_rwf2000_fallown_violence \
    --annotation data/instruction/evaluation/test_gangnam_gaepo1_v2_image_falldown.jsonl \
    --image-root data \
    --out-dir results/eval_result_image/gangnam \
    --batch-size 20 \
    --multi-gpu

###### 강남 개포4동 test
PYTHONPATH="$(pwd)" python src/evaluation/evaluate_image_classfication.py \
    --checkpoint ckpts/InternVL3-2B_gangnam_gj_yeoksam_v2_gaepo4_1v2_samsung_rwf2000_fallown_violence \
    --annotation data/instruction/evaluation/test_gangnam_gaepo4_image_falldown.jsonl \
    --image-root data \
    --out-dir results/eval_result_image/gangnam \
    --batch-size 40 \
    --multi-gpu

###### 강남 삼성동 test
PYTHONPATH="$(pwd)" python src/evaluation/evaluate_image_classfication.py \
    --checkpoint ckpts/InternVL3-2B_gangnam_gj_yeoksam_v2_gaepo4_1v2_samsung_rwf2000_fallown_violence \
    --annotation data/instruction/evaluation/test_gangnam_samsung_image_falldown.jsonl \
    --image-root data \
    --out-dir results/eval_result_image/gangnam \
    --batch-size 40 \
    --multi-gpu

