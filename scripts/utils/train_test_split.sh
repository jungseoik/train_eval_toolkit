## jsonl 파일에서 비율맞춰서 데이터 스플릿 하는 함수 폭력전용이므로 범용성있게 추후 수정해야함

python main.py train_test_split -i "data/instruction/train/train_total.jsonl" -r 0.1 -o "data/instruction"


## AI hub store
python main.py train_test_split -i "data/instruction/train/train_aihub_store_no_split.jsonl" -r 0.1 -o "data/instruction"
## gj
python main.py train_test_split -i "data/instruction/train/train_gj_no_split.jsonl" -r 0.1 -o "data/instruction"
## AI hub space
python main.py train_test_split -i "data/instruction/train/train_aihub_space_no_split.jsonl" -r 0.1 -o "data/instruction"

## AI hub CCTV
python main.py train_test_split -i "data/instruction/train/train_cctv_no_split.jsonl" -r 0.1 -o "data/instruction"
