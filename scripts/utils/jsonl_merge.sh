## example
python main.py merge_jsonl -1 "test_gangnam.jsonl" -2 "result.jsonl" -o res.jsonl

## example
python main.py merge_jsonl -1 "data/instruction/train/train_rwf2000_no_split.jsonl" -2 "data/instruction/train/train_gangnam_vietnam_no_split.jsonl" -o data/instruction/train/train_gangnam_vietnam_rwf2000_no_split.jsonl

python main.py merge_jsonl -1 "data/instruction/train/train_aihub_space_no_split.jsonl" -2 "data/instruction/train/train_gangnam_vietnam_rwf2000_aihubstore_gj_no_split.jsonl" -o data/instruction/train/train_gangnam_vietnam_rwf2000_aihubstore_gj_space_no_split.jsonl