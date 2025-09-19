## example
python main.py autolabel -i "data/raw/rwf2000/RWF-2000/train/NonFight" -opt "normal"
python main.py autolabel -i "data/raw/rwf2000/RWF-2000/train/Fight" -opt "vio"
python main.py autolabel -i "data/processed/RWF-2000/val/NonFight" -opt "normal"


python main.py autolabel -i "data/raw/ai_hub_indoor_store_violence" -opt "vio_timestamp" -n 1
python main.py autolabel -i "data/processed/ai_hub_spaces" -opt "aihub_space" -n 32

python main.py autolabel -i "data/processed/gj_normal" -opt "gj_normal" -n 16
python main.py autolabel -i "data/processed/gj_violence" -opt "gj_violence" -n 16


