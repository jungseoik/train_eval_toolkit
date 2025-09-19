### lora merge 하는거 필수~
mkdir ckpts/merge_result
PYTHONPATH="$(pwd)" python src/training/tools/merge_lora.py ckpts/lora ckpts/merge_result
cp ckpts/InternVL3-2B/*.py ckpts/merge_result/
cp ckpts/InternVL3-2B/config.json ckpts/merge_result/
