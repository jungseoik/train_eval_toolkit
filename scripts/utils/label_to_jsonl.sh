## 기존 라벨링을 jsonl로 변환코드

python main.py label2jsonl -i "data/processed/RWF-2000/train" -o "data/instruction/train/train_rwf2000.jsonl"
python main.py label2jsonl -i "data/processed/RWF-2000/val" -o "data/instruction/evaluation/test_rwf2000.jsonl"

# python main.py label2jsonl -i "data/processed/ai_hub_indoor_store_violence/violence" -o "data/instruction/evaluation/test_aihub_space_vio.jsonl"


python main.py label2jsonl -i "data/processed/ai_hub_spaces" -o "data/instruction/train/test_aihub_space_no_split.jsonl"

python main.py label2jsonl -i "data/processed/gj" -o "data/instruction/train/train_gj_no_split.jsonl"


python main.py label2jsonl -i "data/processed/RWF-2000/val" -o "data/instruction/evaluation/test_rwf2000_test.jsonl"



