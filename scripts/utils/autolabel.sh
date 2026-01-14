## example
python main.py autolabel -i "data/raw/rwf2000/RWF-2000/train/NonFight" -opt "normal"
python main.py autolabel -i "data/raw/rwf2000/RWF-2000/train/Fight" -opt "vio"
python main.py autolabel -i "data/processed/RWF-2000/val/NonFight" -opt "normal"


python main.py autolabel -i "data/raw/ai_hub_indoor_store_violence" -opt "vio_timestamp" -n 1
python main.py autolabel -i "data/processed/ai_hub_spaces" -opt "aihub_space" -n 32

python main.py autolabel -i "data/processed/gj_normal" -opt "gj_normal" -n 16
python main.py autolabel -i "data/processed/gj_violence" -opt "gj_violence" -n 16


python main.py autolabel -i "data/raw/ai_hub_cctv/noraml" -opt "cctv_normal" -n 16
python main.py autolabel -i "data/raw/ai_hub_cctv/violence/ai_hub_cctv_violence" -opt "cctv_violence" -n 16
python main.py autolabel -i "data/raw/ai_hub_cctv/violence/ai_hub_cctv_violence_assault" -opt "cctv_violence" -n 32
python main.py autolabel -i "data/raw/ai_hub_cctv/violence/ai_hub_cctv_violence_datafight" -opt "cctv_violence" -n 16


python main.py autolabel -i "data/raw/ai_hub_cctv/violence/ai_hub_cctv_violence_datafight" -opt "cctv_violence" -n 16
python main.py autolabel -i "data/raw/ai_hub_cctv/violence/ai_hub_cctv_violence_datafight" -opt "cctv_violence" -n 16


## 강남
python main.py autolabel -i "data/processed/gangnam/gaepo4/clean/image/falldown" -opt "gangnam" -n 16
python main.py autolabel -i "data/processed/gangnam/gaepo4/clean/image/normal" -opt "gangnam" -n 16
python main.py autolabel -i "data/processed/gangnam/gaepo4/clean/image/violence" -opt "gangnam" -n 16

python main.py autolabel -i "data/processed/gangnam/samsung/clean/image/falldown/falldown" -opt "gangnam" -n 16
python main.py autolabel -i "data/processed/gangnam/samsung/clean/image/violence/normal" -opt "gangnam" -n 16
python main.py autolabel -i "data/processed/gangnam/samsung/clean/image/falldown/normal" -opt "gangnam" -n 16
python main.py autolabel -i "data/processed/gangnam/samsung/clean/image/violence/violence" -opt "gangnam" -n 16
### 강남 비디오
## 강남 개포4
python main.py autolabel -i "data/processed/gangnam/gaepo4/clean/video/clip/falldown" -opt "gangnam" -n 16
python main.py autolabel -i "data/processed/gangnam/gaepo4/clean/video/raw/elevator_falldown" -opt "gangnam" -n 16
python main.py autolabel -i "data/processed/gangnam/gaepo4/clean/video/raw/elevator_normal" -opt "gangnam" -n 16
python main.py autolabel -i "data/processed/gangnam/gaepo4/clean/video/raw/elevator_violence" -opt "gangnam" -n 16
## 강남 삼성
python main.py autolabel -i "data/processed/gangnam/samsung/clean/video/falldown/falldown" -opt "gangnam" -n 16
python main.py autolabel -i "data/processed/gangnam/samsung/clean/video/falldown/normal" -opt "gangnam" -n 16
python main.py autolabel -i "data/processed/gangnam/samsung/clean/video/violence/normal" -opt "gangnam" -n 16
python main.py autolabel -i "data/processed/gangnam/samsung/clean/video/violence/violence" -opt "gangnam" -n 16
## 강남 개포1_v2
python main.py autolabel -i "data/processed/gangnam/gaepo1_v2/Train/video/violence/violence/clip" -opt "gangnam" -n 16 -m video
python main.py autolabel -i "data/processed/gangnam/gaepo1_v2/Train/image/violence/violence" -opt "gangnam" -n 16 -m image


python main.py autolabel -i "data/processed/gangnam/gaepo1_v2/Train/video/violence/normal" -opt "gangnam" -n 16 -m video
python main.py autolabel -i "data/processed/gangnam/gaepo1_v2/Train/video/falldown/normal" -opt "gangnam" -n 16 -m video

python main.py autolabel -i "data/processed/gangnam/gaepo1_v2/Train/image/falldown/normal" -opt "gangnam" -n 16 -m image
python main.py autolabel -i "data/processed/gangnam/gaepo1_v2/Train/image/violence/normal" -opt "gangnam" -n 16 -m image

python main.py autolabel -i "data/processed/gangnam/gaepo1_v2/Train/video/falldown/falldown" -opt "gangnam" -n 16 -m video
python main.py autolabel -i "data/processed/gangnam/gaepo1_v2/Train/image/falldown/falldown" -opt "gangnam" -n 16 -m image


## 강남 역삼2동_v2
python main.py autolabel -i "data/processed/gangnam/yeoksam2_v2/Train/video/violence/violence/clip" -opt "gangnam" -n 16 -m video
python main.py autolabel -i "data/processed/gangnam/yeoksam2_v2/Train/image/violence/violence" -opt "gangnam" -n 16 -m image


python main.py autolabel -i "data/processed/gangnam/yeoksam2_v2/Train/video/violence/normal"  -opt "gangnam" -n 16 -m video
python main.py autolabel -i "data/processed/gangnam/yeoksam2_v2/Train/video/falldown/normal" -opt "gangnam" -n 16 -m video

python main.py autolabel -i "data/processed/gangnam/yeoksam2_v2/Train/image/falldown/normal" -opt "gangnam" -n 16 -m image
python main.py autolabel -i "data/processed/gangnam/yeoksam2_v2/Train/image/violence/normal" -opt "gangnam" -n 16 -m image


python main.py autolabel -i "data/processed/gangnam/yeoksam2_v2/Train/video/falldown/falldown" -opt "gangnam" -n 16 -m video
python main.py autolabel -i "data/processed/gangnam/yeoksam2_v2/Train/image/falldown/falldown" -opt "gangnam" -n 16 -m image



#### SCVD  오토라벨링
python main.py autolabel -i "data/raw/smartcity_cctv_violence/SCVD/SCVD_converted/Train/Violence" -opt "scvd_violence" -n 16
python main.py autolabel -i "data/raw/smartcity_cctv_violence/SCVD/SCVD_converted/Train/Weaponized" -opt "scvd_violence" -n 16
python main.py autolabel -i "data/raw/smartcity_cctv_violence/SCVD/SCVD_converted/Train/Normal" -opt "scvd_normal" -n 16

python main.py autolabel -i "data/raw/smartcity_cctv_violence/SCVD/SCVD_converted/Test/Violence" -opt "scvd_violence" -n 16
python main.py autolabel -i "data/raw/smartcity_cctv_violence/SCVD/SCVD_converted/Test/Weaponized" -opt "scvd_violence" -n 16
python main.py autolabel -i "data/raw/smartcity_cctv_violence/SCVD/SCVD_converted/Test/Normal" -opt "scvd_normal" -n 16

python main.py autolabel -i "data/raw/smartcity_cctv_violence/SCVD/SCVD_converted_sec_split/Train/Violence" -opt "scvd_violence" -n 16
python main.py autolabel -i "data/raw/smartcity_cctv_violence/SCVD/SCVD_converted_sec_split/Train/Weaponized" -opt "scvd_violence" -n 16
python main.py autolabel -i "data/raw/smartcity_cctv_violence/SCVD/SCVD_converted_sec_split/Train/Normal" -opt "scvd_normal" -n 16

python main.py autolabel -i "data/raw/smartcity_cctv_violence/SCVD/SCVD_converted_sec_split/Test/Violence" -opt "scvd_violence" -n 16
python main.py autolabel -i "data/raw/smartcity_cctv_violence/SCVD/SCVD_converted_sec_split/Test/Weaponized" -opt "scvd_violence" -n 16
python main.py autolabel -i "data/raw/smartcity_cctv_violence/SCVD/SCVD_converted_sec_split/Test/Normal" -opt "scvd_normal" -n 16

## 현대백화점 에스컬레이터
python main.py autolabel -i "data/processed/hyundai_backhwajum/abb_hyundai/test/normal" -opt "hyundai_normal" -n 128 -m image
python main.py autolabel -i "data/processed/hyundai_backhwajum/abb_hyundai/train/normal" -opt "hyundai_normal" -n 128 -m image
python main.py autolabel -i "data/processed/hyundai_backhwajum/dtro_hyundai/normal" -opt "hyundai_normal" -n 128 -m image
python main.py autolabel -i "data/processed/hyundai_backhwajum/hyundai_ai/normal" -opt "hyundai_normal" -n 128 -m image

python main.py autolabel -i "data/processed/hyundai_backhwajum/abb_hyundai/test/falldown" -opt "hyundai_falldown" -n 128 -m image
python main.py autolabel -i "data/processed/hyundai_backhwajum/abb_hyundai/train/falldown" -opt "hyundai_falldown" -n 128 -m image
python main.py autolabel -i "data/processed/hyundai_backhwajum/dtro_hyundai/falldown" -opt "hyundai_falldown" -n 128 -m image
python main.py autolabel -i "data/processed/hyundai_backhwajum/hyundai_ai/falldown" -opt "hyundai_falldown" -n 128 -m image

## 현대백화점 에스컬레이터 1차
python main.py autolabel -i "data/processed/hyundai_backhwajum/hyundai_image_gen_ai_1st/normal" -opt "hyundai_normal" -n 128 -m image
python main.py autolabel -i "data/processed/hyundai_backhwajum/hyundai_PoC_5camera_gen_ai/gen/normal" -opt "hyundai_normal" -n 128 -m image
python main.py autolabel -i "data/processed/hyundai_backhwajum/hyundai_PoC_5camera_gen_ai/raw/normal" -opt "hyundai_normal" -n 128 -m image
python main.py autolabel -i "data/processed/hyundai_backhwajum/hyundai_PoC_25camera_capture/normal" -opt "hyundai_normal" -n 128 -m image

python main.py autolabel -i "data/processed/hyundai_backhwajum/hyundai_PoC_5camera_gen_ai/raw/falldown" -opt "hyundai_falldown" -n 128 -m image
python main.py autolabel -i "data/processed/hyundai_backhwajum/hyundai_image_gen_ai_1st/falldown" -opt "hyundai_falldown" -n 128 -m image
python main.py autolabel -i "data/processed/hyundai_backhwajum/hyundai_PoC_5camera_gen_ai/gen/falldown" -opt "hyundai_falldown" -n 128 -m image
python main.py autolabel -i "data/processed/hyundai_backhwajum/hyundai_PoC_25camera_capture/falldown" -opt "hyundai_falldown" -n 128 -m image
