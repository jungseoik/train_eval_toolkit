## example
python main.py merge_jsonl -1 "test_gangnam.jsonl" -2 "result.jsonl" -o res.jsonl

## example
python main.py merge_jsonl -1 "data/instruction/train/train_rwf2000_no_split.jsonl" -2 "data/instruction/train/train_gangnam_vietnam_no_split.jsonl" -o data/instruction/train/train_gangnam_vietnam_rwf2000_no_split.jsonl

python main.py merge_jsonl -1 "data/instruction/train/train_aihub_space_no_split.jsonl" -2 "data/instruction/train/train_gangnam_vietnam_rwf2000_aihubstore_gj_no_split.jsonl" -o data/instruction/train/train_gangnam_vietnam_rwf2000_aihubstore_gj_space_no_split.jsonl

python main.py merge_jsonl -1 "data/instruction/train/train_rwf2000.jsonl" -2 "data/instruction/train/train_gangnam.jsonl" -o data/instruction/train/train_gangnam_rwf2000.jsonl

python main.py merge_jsonl -1 "data/instruction/evaluation/test_scvd.jsonl" -2 "data/instruction/evaluation/test_scvd_sec.jsonl" -o data/instruction/evaluation/test_scvdALL.jsonl

python main.py merge_jsonl -1 "data/instruction/train/train_scvd.jsonl" -2 "data/instruction/train/train_scvd_sec.jsonl" -o data/instruction/train/train_scvdALL.jsonl
python main.py merge_jsonl -1 "data/instruction/train/train_scvd_NOweapon.jsonl" -2 "data/instruction/train/train_scvd_sec_NOweapon.jsonl" -o data/instruction/train/train_scvdALL_NOweapon.jsonl


python main.py merge_jsonl -1 "data/instruction/train/train_scvdALL_NOweapon.jsonl" -2 "data/instruction/evaluation/test_scvdALL_NOweapon.jsonl" -o data/instruction/train/train_scvdALL_NOweapon_no_split.jsonl
