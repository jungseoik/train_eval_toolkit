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
