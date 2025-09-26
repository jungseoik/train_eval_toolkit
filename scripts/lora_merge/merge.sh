### lora merge 하는거 필수~
# mkdir ckpts/merge_result
# PYTHONPATH="$(pwd)" python src/training/tools/merge_lora.py ckpts/lora ckpts/merge_result
# cp ckpts/InternVL3-2B/*.py ckpts/merge_result/
# cp ckpts/InternVL3-2B/config.json ckpts/merge_result/
# rm -rf ckpts/lora


MERGE_DIR="InternVL3-2B_rwf2000"

mkdir ckpts/$MERGE_DIR
PYTHONPATH="$(pwd)" python src/training/tools/merge_lora.py ckpts/lora ckpts/$MERGE_DIR
cp ckpts/InternVL3-2B/*.py ckpts/$MERGE_DIR/
cp ckpts/InternVL3-2B/config.json ckpts/$MERGE_DIR/
rm -rf ckpts/lora



MERGE_DIR="InternVL3-2B_gangnam_rwf2000"

mkdir ckpts/$MERGE_DIR
PYTHONPATH="$(pwd)" python src/training/tools/merge_lora.py ckpts/lora ckpts/$MERGE_DIR
cp ckpts/InternVL3-2B/*.py ckpts/$MERGE_DIR/
cp ckpts/InternVL3-2B/config.json ckpts/$MERGE_DIR/
rm -rf ckpts/lora


MERGE_DIR="InternVL3-2B_cctv"
mkdir ckpts/$MERGE_DIR
PYTHONPATH="$(pwd)" python src/training/tools/merge_lora.py ckpts/lora ckpts/$MERGE_DIR
cp ckpts/InternVL3-2B/*.py ckpts/$MERGE_DIR/
cp ckpts/InternVL3-2B/config.json ckpts/$MERGE_DIR/
rm -rf ckpts/lora


MERGE_DIR="InternVL3-2B_gj"
mkdir ckpts/$MERGE_DIR
PYTHONPATH="$(pwd)" python src/training/tools/merge_lora.py ckpts/lora ckpts/$MERGE_DIR
cp ckpts/InternVL3-2B/*.py ckpts/$MERGE_DIR/
cp ckpts/InternVL3-2B/config.json ckpts/$MERGE_DIR/
rm -rf ckpts/lora

MERGE_DIR="InternVL3-2B_aihub_space"
mkdir ckpts/$MERGE_DIR
PYTHONPATH="$(pwd)" python src/training/tools/merge_lora.py ckpts/lora ckpts/$MERGE_DIR
cp ckpts/InternVL3-2B/*.py ckpts/$MERGE_DIR/
cp ckpts/InternVL3-2B/config.json ckpts/$MERGE_DIR/
rm -rf ckpts/lora

MERGE_DIR="InternVL3-2B_aihub_store"
mkdir ckpts/$MERGE_DIR
PYTHONPATH="$(pwd)" python src/training/tools/merge_lora.py ckpts/lora ckpts/$MERGE_DIR
cp ckpts/InternVL3-2B/*.py ckpts/$MERGE_DIR/
cp ckpts/InternVL3-2B/config.json ckpts/$MERGE_DIR/
rm -rf ckpts/lora


MERGE_DIR="InternVL3-2B_gangnam_rwf2000_gj"
mkdir ckpts/$MERGE_DIR
PYTHONPATH="$(pwd)" python src/training/tools/merge_lora.py ckpts/lora ckpts/$MERGE_DIR
cp ckpts/InternVL3-2B/*.py ckpts/$MERGE_DIR/
cp ckpts/InternVL3-2B/config.json ckpts/$MERGE_DIR/
rm -rf ckpts/lora



MERGE_DIR="InternVL3-2B_gangnam_rwf2000_gj_space_store"
mkdir ckpts/$MERGE_DIR
PYTHONPATH="$(pwd)" python src/training/tools/merge_lora.py ckpts/lora ckpts/$MERGE_DIR
cp ckpts/InternVL3-2B/*.py ckpts/$MERGE_DIR/
cp ckpts/InternVL3-2B/config.json ckpts/$MERGE_DIR/
rm -rf ckpts/lora



MERGE_DIR="InternVL3-2B_cctv"
mkdir ckpts/$MERGE_DIR
PYTHONPATH="$(pwd)" python src/training/tools/merge_lora.py ckpts/lora ckpts/$MERGE_DIR
cp ckpts/InternVL3-2B/*.py ckpts/$MERGE_DIR/
cp ckpts/InternVL3-2B/config.json ckpts/$MERGE_DIR/
rm -rf ckpts/lora

MERGE_DIR="InternVL3-2B_gangnam_rwf2000_gj_cctv"
mkdir ckpts/$MERGE_DIR
PYTHONPATH="$(pwd)" python src/training/tools/merge_lora.py ckpts/lora ckpts/$MERGE_DIR
cp ckpts/InternVL3-2B/*.py ckpts/$MERGE_DIR/
cp ckpts/InternVL3-2B/config.json ckpts/$MERGE_DIR/
rm -rf ckpts/lora


MERGE_DIR="InternVL3-2B_scvdALL_NOweapon"
mkdir ckpts/$MERGE_DIR
PYTHONPATH="$(pwd)" python src/training/tools/merge_lora.py ckpts/lora ckpts/$MERGE_DIR
cp ckpts/InternVL3-2B/*.py ckpts/$MERGE_DIR/
cp ckpts/InternVL3-2B/config.json ckpts/$MERGE_DIR/
rm -rf ckpts/lora