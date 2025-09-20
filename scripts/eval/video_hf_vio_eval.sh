export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
CUDA_VISIBLE_DEVICES=0,1 \
PYTHONPATH="$(pwd)" torchrun --nproc_per_node=2 src/evaluation/evaluate_pia_hf_bench_eff.py \
    --checkpoint ckpts/InternVL3-2B_gangnam_vietnam_aihubstore_gj_space_no_split \
    --video-root /home/piawsa6000/nas192/datasets/projects/huggingface_benchmarks_dataset/Leaderboard_bench/PIA_Violence/dataset/violence \
    --out-dir results/eval_hf_result/InternVL3-2B_gangnam_vietnam_aihubstore_gj_space_no_split \
    --window-size 15 \
    --workers-per-gpu 0 \
    --procs-per-gpu 4