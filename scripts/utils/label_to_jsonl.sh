## 기존 라벨링을 jsonl로 변환코드

python main.py label2jsonl -i "data/processed/RWF-2000/train" -o "data/instruction/train/train_rwf2000.jsonl"
python main.py label2jsonl -i "data/processed/RWF-2000/val" -o "data/instruction/evaluation/test_rwf2000.jsonl"

# python main.py label2jsonl -i "data/processed/ai_hub_indoor_store_violence/violence" -o "data/instruction/evaluation/test_aihub_space_vio.jsonl"


python main.py label2jsonl -i "data/processed/ai_hub_spaces" -o "data/instruction/train/test_aihub_space_no_split.jsonl"

python main.py label2jsonl -i "data/processed/gj" -o "data/instruction/train/train_gj_no_split.jsonl"

python main.py label2jsonl -i "data/processed/RWF-2000/val" -o "data/instruction/evaluation/test_rwf2000_test.jsonl"



python main.py label2jsonl -i "data/processed/RWF-2000/val" -o "data/instruction/evaluation/test_rwf2000_test.jsonl"
python main.py label2jsonl -i "data/raw/ai_hub_cctv" -o "data/instruction/evaluation/total_cctv.jsonl"


### SCVD data split
python main.py label2jsonl -i "data/raw/smartcity_cctv_violence" -o "data/instruction/train/scvdALL_no_split.jsonl"
python main.py label2jsonl -i "data/raw/smartcity_cctv_violence" -o "data/instruction/train/scvdALL_no_split.jsonl" -opt "test"


python main.py label2jsonl -i "data/raw/smartcity_cctv_violence/SCVD/SCVD_converted/Test" -o "data/instruction/evaluation/test_scvd.jsonl" -opt "test"
python main.py label2jsonl -i "data/raw/smartcity_cctv_violence/SCVD/SCVD_converted_sec_split/Test" -o "data/instruction/evaluation/test_scvd_sec.jsonl" -opt "test"
python main.py label2jsonl -i "data/raw/smartcity_cctv_violence/SCVD/SCVD_converted/Test" -o "data/instruction/evaluation/test_scvd_NOweapon.jsonl" -opt "test"
python main.py label2jsonl -i "data/raw/smartcity_cctv_violence/SCVD/SCVD_converted_sec_split/Test" -o "data/instruction/evaluation/test_scvd_sec_NOweapon.jsonl" -opt "test"


python main.py label2jsonl -i "data/raw/smartcity_cctv_violence/SCVD/SCVD_converted/Train" -o "data/instruction/train/train_scvd_NOweapon.jsonl" 
python main.py label2jsonl -i "data/raw/smartcity_cctv_violence/SCVD/SCVD_converted_sec_split/Train" -o "data/instruction/train/train_scvd_sec_NOweapon.jsonl" 

python main.py label2jsonl -i "data/raw/smartcity_cctv_violence/SCVD/SCVD_converted/Train" -o "data/instruction/train/train_scvd.jsonl" 
python main.py label2jsonl -i "data/raw/smartcity_cctv_violence/SCVD/SCVD_converted_sec_split/Train" -o "data/instruction/train/train_scvd_sec.jsonl" 

### CCTV
python main.py label2jsonl -i "data/processed/ai_hub_cctv_encoding" -o "data/instruction/train/train_cctv_no_split.jsonl" 

### ABB falldown
python main.py label2jsonl -i "data/processed/ABB_banwoldang/PE/Train" -o "data/instruction/train/train_abb_pe.jsonl" -dt "image"
python main.py label2jsonl -i "data/processed/ABB_banwoldang/VQA/Train" -o "data/instruction/train/train_abb_vqa.jsonl" -dt "image" 
python main.py label2jsonl -i "data/processed/ABB_banwoldang/PE/Test" -o "data/instruction/train/train_abb_pe_test.jsonl" -dt "image" -opt "test"
python main.py label2jsonl -i "data/processed/ABB_banwoldang/VQA/Test" -o "data/instruction/train/train_abb_vqa_test.jsonl" -dt "image" -opt "test"
### 강남 개포1동 
python main.py label2jsonl -i "data/processed/gangnam/gaepo1/Test_dataset_gaepo1st" -o "data/instruction/evaluation/test_gangnam_gaepo1_falldown.jsonl" -dt "image" -opt "test"
python main.py label2jsonl -i "data/processed/gangnam/gaepo1/Train_dataset_gaepo1st" -o "data/instruction/train/train_gangnam_gaepo1_falldown.jsonl" -dt "image" 
python main.py label2jsonl -i "data/processed/gangnam/gaepo1/Test_dataset_gaepo1st" -o "data/instruction/evaluation/test_gangnam_gaepo1_violence.jsonl" -opt "test"
python main.py label2jsonl -i "data/processed/gangnam/gaepo1/Train_dataset_gaepo1st" -o "data/instruction/train/train_gangnam_gaepo1_violence.jsonl" 

### 강남 역삼2동 
python main.py label2jsonl -i "data/processed/gangnam/yeoksam2/Test_dataset_yeoksam2st" -o "data/instruction/evaluation/test_gangnam_yeoksam2st_falldown.jsonl" -dt "image" -opt "test"
python main.py label2jsonl -i "data/processed/gangnam/yeoksam2/Train_dataset_yeoksam2st" -o "data/instruction/train/train_gangnam_yeoksam2st_falldown.jsonl" -dt "image" 
python main.py label2jsonl -i "data/processed/gangnam/yeoksam2/Test_dataset_yeoksam2st" -o "data/instruction/evaluation/test_gangnam_yeoksam2st_violence.jsonl" -opt "test"
python main.py label2jsonl -i "data/processed/gangnam/yeoksam2/Train_dataset_yeoksam2st" -o "data/instruction/train/train_gangnam_yeoksam2st_violence.jsonl" 

#### 최신버전 
#####강남 삼성동 ### ### ### ### ### ### ### ### ### ### ### 
python main.py label2jsonl -i "data/processed/gangnam/samsung/Train/clean/image/falldown" -o "data/instruction/train/train_gangnam_samsung_image_falldown.jsonl" -dt "image" -opt "train" -ity "capture_frame" -itk "caption" -tn "falldown"
python main.py label2jsonl -i "data/processed/gangnam/samsung/Train/clean/image/violence" -o "data/instruction/train/train_gangnam_samsung_image_violence.jsonl" -dt "image" -opt "train" -ity "capture_frame" -itk "caption" -tn "violence"
python main.py label2jsonl -i "data/processed/gangnam/samsung/Train/clean/video/falldown" -o "data/instruction/train/train_gangnam_samsung_video_falldown.jsonl" -dt "video" -opt "train" -ity "clip" -itk "caption" -tn "falldown"
python main.py label2jsonl -i "data/processed/gangnam/samsung/Train/clean/video/violence" -o "data/instruction/train/train_gangnam_samsung_video_violence.jsonl" -dt "video" -opt "train" -ity "clip" -itk "caption" -tn "violence"

python main.py label2jsonl -i "data/processed/gangnam/samsung/Test/clean/image/falldown" -o "data/instruction/evaluation/test_gangnam_samsung_image_falldown.jsonl" -dt "image" -opt "train" -ity "capture_frame" -itk "caption" -tn "falldown" -opt "test"
python main.py label2jsonl -i "data/processed/gangnam/samsung/Test/clean/image/violence" -o "data/instruction/evaluation/test_gangnam_samsung_image_violence.jsonl" -dt "image" -opt "train" -ity "capture_frame" -itk "caption" -tn "violence" -opt "test"
python main.py label2jsonl -i "data/processed/gangnam/samsung/Test/clean/video/falldown" -o "data/instruction/evaluation/test_gangnam_samsung_video_falldown.jsonl" -dt "video" -opt "train" -ity "clip" -itk "caption" -tn "falldown" -opt "test"
python main.py label2jsonl -i "data/processed/gangnam/samsung/Test/clean/video/violence" -o "data/instruction/evaluation/test_gangnam_samsung_video_violence.jsonl" -dt "video" -opt "train" -ity "clip" -itk "caption" -tn "violence" -opt "test"


#####강남 개포 4동  ### ### ### ### ### ### ### ### ### ### ### 
python main.py label2jsonl -i "data/processed/gangnam/gaepo4/Train/clean/image/falldown" -o "data/instruction/train/train_gangnam_gaepo4_image_falldown.jsonl" -dt "image" -opt "train" -ity "capture_frame" -itk "caption" -tn "falldown"
python main.py label2jsonl -i "data/processed/gangnam/gaepo4/Train/clean/image/violence" -o "data/instruction/train/train_gangnam_gaepo4_image_violence.jsonl" -dt "image" -opt "train" -ity "capture_frame" -itk "caption" -tn "violence"
python main.py label2jsonl -i "data/processed/gangnam/gaepo4/Train/clean/video/falldown" -o "data/instruction/train/train_gangnam_gaepo4_video_falldown.jsonl" -dt "video" -opt "train" -ity "clip" -itk "caption" -tn "falldown"
python main.py label2jsonl -i "data/processed/gangnam/gaepo4/Train/clean/video/violence" -o "data/instruction/train/train_gangnam_gaepo4_video_violence.jsonl" -dt "video" -opt "train" -ity "clip" -itk "caption" -tn "violence"


python main.py label2jsonl -i "data/processed/gangnam/gaepo4/Test/clean/image/falldown" -o "data/instruction/evaluation/test_gangnam_gaepo4_image_falldown.jsonl" -dt "image" -opt "train" -ity "capture_frame" -itk "caption" -tn "falldown" -opt "test"
python main.py label2jsonl -i "data/processed/gangnam/gaepo4/Test/clean/image/violence" -o "data/instruction/evaluation/test_gangnam_gaepo4_image_violence.jsonl" -dt "image" -opt "train" -ity "capture_frame" -itk "caption" -tn "violence" -opt "test"
python main.py label2jsonl -i "data/processed/gangnam/gaepo4/Test/clean/video/falldown" -o "data/instruction/evaluation/test_gangnam_gaepo4_video_falldown.jsonl" -dt "video" -opt "train" -ity "clip" -itk "caption" -tn "falldown" -opt "test"
python main.py label2jsonl -i "data/processed/gangnam/gaepo4/Test/clean/video/violence" -o "data/instruction/evaluation/test_gangnam_gaepo4_video_violence.jsonl" -dt "video" -opt "train" -ity "clip" -itk "caption" -tn "violence" -opt "test"

#####강남 개포 1동 v2  ### ### ### ### ### ### ### ### ### ### ### 
python main.py label2jsonl -i "data/processed/gangnam/gaepo1_v2/Train/image/falldown" -o "data/instruction/train/train_gangnam_gaepo1_v2_image_falldown.jsonl" -dt "image" -opt "train" -ity "capture_frame" -itk "caption" -tn "falldown"
python main.py label2jsonl -i "data/processed/gangnam/gaepo1_v2/Train/image/violence" -o "data/instruction/train/train_gangnam_gaepo1_v2_image_violence.jsonl" -dt "image" -opt "train" -ity "capture_frame" -itk "caption" -tn "violence"
python main.py label2jsonl -i "data/processed/gangnam/gaepo1_v2/Train/video/falldown" -o "data/instruction/train/train_gangnam_gaepo1_v2_video_falldown.jsonl" -dt "video" -opt "train" -ity "clip" -itk "caption" -tn "falldown"
python main.py label2jsonl -i "data/processed/gangnam/gaepo1_v2/Train/video/violence" -o "data/instruction/train/train_gangnam_gaepo1_v2_video_violence.jsonl" -dt "video" -opt "train" -ity "clip" -itk "caption" -tn "violence"


python main.py label2jsonl -i "data/processed/gangnam/gaepo1_v2/Test/image/falldown" -o "data/instruction/evaluation/test_gangnam_gaepo1_v2_image_falldown.jsonl" -dt "image" -opt "train" -ity "capture_frame" -itk "caption" -tn "falldown" -opt "test"
python main.py label2jsonl -i "data/processed/gangnam/gaepo1_v2/Test/image/violence" -o "data/instruction/evaluation/test_gangnam_gaepo1_v2_image_violence.jsonl" -dt "image" -opt "train" -ity "capture_frame" -itk "caption" -tn "violence" -opt "test"
python main.py label2jsonl -i "data/processed/gangnam/gaepo1_v2/Test/video/falldown" -o "data/instruction/evaluation/test_gangnam_gaepo1_v2_video_falldown.jsonl" -dt "video" -opt "train" -ity "clip" -itk "caption" -tn "falldown" -opt "test"
python main.py label2jsonl -i "data/processed/gangnam/gaepo1_v2/Test/video/violence" -o "data/instruction/evaluation/test_gangnam_gaepo1_v2_video_violence.jsonl" -dt "video" -opt "train" -ity "clip" -itk "caption" -tn "violence" -opt "test"

#####se1 , sec2 없는 버전 아직 없음


#####강남 역삼 2동 v2 ### ### ### ### ### ### ### ### ### ### ### 
python main.py label2jsonl -i "data/processed/gangnam/yeoksam2_v2/Train/image/falldown" -o "data/instruction/train/train_gangnam_yeoksam2_v2_image_falldown.jsonl" -dt "image" -opt "train" -ity "capture_frame" -itk "caption" -tn "falldown"
python main.py label2jsonl -i "data/processed/gangnam/yeoksam2_v2/Train/image/violence" -o "data/instruction/train/train_gangnam_yeoksam2_v2_image_violence.jsonl" -dt "image" -opt "train" -ity "capture_frame" -itk "caption" -tn "violence"
python main.py label2jsonl -i "data/processed/gangnam/yeoksam2_v2/Train/video/falldown" -o "data/instruction/train/train_gangnam_yeoksam2_v2_video_falldown.jsonl" -dt "video" -opt "train" -ity "clip" -itk "caption" -tn "falldown"
python main.py label2jsonl -i "data/processed/gangnam/yeoksam2_v2/Train/video/violence" -o "data/instruction/train/train_gangnam_yeoksam2_v2_video_violence.jsonl" -dt "video" -opt "train" -ity "clip" -itk "caption" -tn "violence"


python main.py label2jsonl -i "data/processed/gangnam/yeoksam2_v2/Test/image/falldown" -o "data/instruction/evaluation/test_gangnam_yeoksam2_v2_image_falldown.jsonl" -dt "image" -opt "test" -ity "capture_frame" -itk "caption" -tn "falldown" 
python main.py label2jsonl -i "data/processed/gangnam/yeoksam2_v2/Test/image/violence" -o "data/instruction/evaluation/test_gangnam_yeoksam2_v2_image_violence.jsonl" -dt "image" -opt "test" -ity "capture_frame" -itk "caption" -tn "violence" 
python main.py label2jsonl -i "data/processed/gangnam/yeoksam2_v2/Test/video/falldown" -o "data/instruction/evaluation/test_gangnam_yeoksam2_v2_video_falldown.jsonl" -dt "video" -opt "test" -ity "clip" -itk "caption" -tn "falldown" 
python main.py label2jsonl -i "data/processed/gangnam/yeoksam2_v2/Test/video/violence" -o "data/instruction/evaluation/test_gangnam_yeoksam2_v2_video_violence.jsonl" -dt "video" -opt "test" -ity "clip" -itk "caption" -tn "violence" 

#####se1 , sec2 없는 버전 아직 없음
### ABB falldown
python main.py label2jsonl -i "data/processed/ABB_banwoldang/PE/Train" -o "data/instruction/evaluation/test_train_abb_pe.jsonl" -dt "image" -opt "test" -ity "capture_frame" -itk "caption" -tn "falldown" 
python main.py label2jsonl -i "data/processed/ABB_banwoldang/VQA/Train" -o "data/instruction/evaluation/test_train_abb_vqa.jsonl" -dt "image" -opt "test" -ity "capture_frame" -itk "caption" -tn "falldown" 
python main.py label2jsonl -i "data/processed/ABB_banwoldang/PE/Test" -o "data/instruction/train/train_abb_pe_test.jsonl" -dt "image" -opt "test"
python main.py label2jsonl -i "data/processed/ABB_banwoldang/VQA/Test" -o "data/instruction/train/train_abb_vqa_test.jsonl" -dt "image" -opt "test"

### 현대백화점
python main.py label2jsonl -i "data/processed/hyundai_backhwajum/dtro_hyundai/train" -o "data/instruction/train/train_dtro_hyundai_image_falldown.jsonl" -dt "image" -opt "train" -ity "capture_frame" -itk "caption" -tn "falldown"
python main.py label2jsonl -i "data/processed/hyundai_backhwajum/abb_hyundai/train" -o "data/instruction/train/train_abb_hyundai_image_falldown.jsonl" -dt "image" -opt "train" -ity "capture_frame" -itk "caption" -tn "falldown"
python main.py label2jsonl -i "data/processed/hyundai_backhwajum/hyundai_ai/train" -o "data/instruction/train/train_hyundai_ai_image_falldown.jsonl" -dt "image" -opt "train" -ity "capture_frame" -itk "caption" -tn "falldown"

python main.py label2jsonl -i "data/processed/hyundai_backhwajum/dtro_hyundai/test" -o "data/instruction/evaluation/test_dtro_hyundai_image_falldown.jsonl" -dt "image" -opt "test" -ity "capture_frame" -itk "caption" -tn "falldown" 
python main.py label2jsonl -i "data/processed/hyundai_backhwajum/abb_hyundai/test" -o "data/instruction/evaluation/test_abb_hyundai_image_falldown.jsonl" -dt "image" -opt "test" -ity "capture_frame" -itk "caption" -tn "falldown" 
python main.py label2jsonl -i "data/processed/hyundai_backhwajum/hyundai_ai/test" -o "data/instruction/evaluation/test_hyundai_ai_image_falldown.jsonl" -dt "image" -opt "test" -ity "capture_frame" -itk "caption" -tn "falldown" 

### 현대백화점 2번쨰
python main.py label2jsonl -i "data/processed/hyundai_backhwajum/hyundai_image_gen_ai_1st/train" -o "data/instruction/train/train_hyundai_image_gen_ai_1st_image_falldown.jsonl" -dt "image" -opt "train" -ity "capture_frame" -itk "caption" -tn "falldown"
python main.py label2jsonl -i "data/processed/hyundai_backhwajum/hyundai_PoC_5camera_gen_ai/train" -o "data/instruction/train/train_hyundai_PoC_5camera_gen_ai_image_falldown.jsonl" -dt "image" -opt "train" -ity "capture_frame" -itk "caption" -tn "falldown"
python main.py label2jsonl -i "data/processed/hyundai_backhwajum/hyundai_PoC_25camera_capture/train" -o "data/instruction/train/train_hyundai_PoC_25camera_capture_image_falldown.jsonl" -dt "image" -opt "train" -ity "capture_frame" -itk "caption" -tn "falldown"

python main.py label2jsonl -i "data/processed/hyundai_backhwajum/hyundai_image_gen_ai_1st/test" -o "data/instruction/evaluation/test_hyundai_image_gen_ai_1st_image_falldown.jsonl" -dt "image" -opt "test" -ity "capture_frame" -itk "caption" -tn "falldown" 
python main.py label2jsonl -i "data/processed/hyundai_backhwajum/hyundai_PoC_5camera_gen_ai/test" -o "data/instruction/evaluation/test_hyundai_PoC_5camera_gen_ai_image_falldown.jsonl" -dt "image" -opt "test" -ity "capture_frame" -itk "caption" -tn "falldown"
python main.py label2jsonl -i "data/processed/hyundai_backhwajum/hyundai_PoC_25camera_capture/test" -o "data/instruction/evaluation/test_hyundai_PoC_25camera_capture_image_falldown.jsonl" -dt "image" -opt "test" -ity "capture_frame" -itk "caption" -tn "falldown"

### 테스트셋 학습 구축
python main.py label2jsonl -i "data/processed/hyundai_backhwajum/hyundai_image_gen_ai_1st/test" -o "data/instruction/train/test_hyundai_image_gen_ai_1st_image_falldown.jsonl" -dt "image" -opt "train" -ity "capture_frame" -itk "caption" -tn "falldown" 
python main.py label2jsonl -i "data/processed/hyundai_backhwajum/hyundai_PoC_5camera_gen_ai/test" -o "data/instruction/train/test_hyundai_PoC_5camera_gen_ai_image_falldown.jsonl" -dt "image" -opt "train" -ity "capture_frame" -itk "caption" -tn "falldown"
python main.py label2jsonl -i "data/processed/hyundai_backhwajum/hyundai_PoC_25camera_capture/test" -o "data/instruction/train/test_hyundai_PoC_25camera_capture_image_falldown.jsonl" -dt "image" -opt "train" -ity "capture_frame" -itk "caption" -tn "falldown"


### 현대백화점 3번쨰
python main.py label2jsonl -i "data/processed/hyundai_backhwajum/hyundai_hard_negative_2st/train" -o "data/instruction/train/train_hyundai_hard_negative_2st_image_falldown.jsonl" -dt "image" -opt "train" -ity "capture_frame" -itk "caption" -tn "falldown"
python main.py label2jsonl -i "data/processed/hyundai_backhwajum/hyundai_image_gen_ai_only_sangrak/train" -o "data/instruction/train/train_hyundai_image_gen_ai_only_sangrak_image_falldown.jsonl" -dt "image" -opt "train" -ity "capture_frame" -itk "caption" -tn "falldown"

python main.py label2jsonl -i "data/processed/hyundai_backhwajum/hyundai_hard_negative_2st/test" -o "data/instruction/evaluation/test_hyundai_hard_negative_2st_image_falldown.jsonl" -dt "image" -opt "test" -ity "capture_frame" -itk "caption" -tn "falldown" 
python main.py label2jsonl -i "data/processed/hyundai_backhwajum/hyundai_image_gen_ai_only_sangrak/test" -o "data/instruction/evaluation/test_hyundai_image_gen_ai_only_sangrak_image_falldown.jsonl" -dt "image" -opt "test" -ity "capture_frame" -itk "caption" -tn "falldown"
## 테스트셋 학습 구축
python main.py label2jsonl -i "data/processed/hyundai_backhwajum/hyundai_hard_negative_2st/test" -o "data/instruction/train/test_hyundai_hard_negative_2st_image_falldown.jsonl" -dt "image" -opt "train" -ity "capture_frame" -itk "caption" -tn "falldown"
python main.py label2jsonl -i "data/processed/hyundai_backhwajum/hyundai_image_gen_ai_only_sangrak/test" -o "data/instruction/train/test_hyundai_image_gen_ai_only_sangrak_image_falldown.jsonl" -dt "image" -opt "train" -ity "capture_frame" -itk "caption" -tn "falldown"


### 현대백화점 4번쨰 하드네거티브 박스 추가
python main.py label2jsonl -i "data/processed/hyundai_backhwajum/hyundai_hard_negative_2st_box/train" -o "data/instruction/train/train_hyundai_hard_negative_2st_box_image_falldown.jsonl" -dt "image" -opt "train" -ity "capture_frame" -itk "caption" -tn "falldown"
python main.py label2jsonl -i "data/processed/hyundai_backhwajum/hyundai_hard_negative_2st_box/test" -o "data/instruction/evaluation/test_hyundai_hard_negative_2st_box_image_falldown.jsonl" -dt "image" -opt "test" -ity "capture_frame" -itk "caption" -tn "falldown"
## 테스트셋 학습 구축
python main.py label2jsonl -i "data/processed/hyundai_backhwajum/hyundai_hard_negative_2st_box/test" -o "data/instruction/train/test_hyundai_hard_negative_2st_box_image_falldown.jsonl" -dt "image" -opt "train" -ity "capture_frame" -itk "caption" -tn "falldown"



### 현대백화점 5번쨰 01.16 이미지 추가
# data/processed/hyundai_backhwajum/hyundai_01_16_QA/test
python main.py label2jsonl -i "data/processed/hyundai_backhwajum/hyundai_01_16_QA/train" -o "data/instruction/train/train_hyundai_01_16_QA_image_falldown.jsonl" -dt "image" -opt "train" -ity "capture_frame" -itk "caption" -tn "falldown"
python main.py label2jsonl -i "data/processed/hyundai_backhwajum/hyundai_01_16_QA/test" -o "data/instruction/evaluation/test_hyundai_01_16_QA_image_falldown.jsonl" -dt "image" -opt "test" -ity "capture_frame" -itk "caption" -tn "falldown"
## 테스트셋 학습 구축
python main.py label2jsonl -i "data/processed/hyundai_backhwajum/hyundai_01_16_QA/test" -o "data/instruction/train/test_hyundai_01_16_QA_image_falldown.jsonl" -dt "image" -opt "train" -ity "capture_frame" -itk "caption" -tn "falldown"


### 현대백화점 6번쨰 01.27 이미지 추가
# data/processed/hyundai_backhwajum/hyundai_01_27_QA/test
python main.py label2jsonl -i "data/processed/hyundai_backhwajum/hyundai_01_27_QA/train" -o "data/instruction/train/train_hyundai_01_27_QA_image_falldown.jsonl" -dt "image" -opt "train" -ity "capture_frame" -itk "caption" -tn "falldown"
python main.py label2jsonl -i "data/processed/hyundai_backhwajum/hyundai_01_27_QA/test" -o "data/instruction/evaluation/test_hyundai_01_27_QA_image_falldown.jsonl" -dt "image" -opt "test" -ity "capture_frame" -itk "caption" -tn "falldown"
## 테스트셋 학습 구축
python main.py label2jsonl -i "data/processed/hyundai_backhwajum/hyundai_01_27_QA/test" -o "data/instruction/train/test_hyundai_01_27_QA_image_falldown.jsonl" -dt "image" -opt "train" -ity "capture_frame" -itk "caption" -tn "falldown"
