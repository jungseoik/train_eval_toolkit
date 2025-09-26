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