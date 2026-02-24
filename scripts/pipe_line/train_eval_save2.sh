#!/usr/bin/env bash
set -euo pipefail

bash scripts/pipe_line/train_eval_save_hyundai_4_5.sh
bash scripts/pipe_line/train_eval_save_hyundai_4_10.sh
bash scripts/pipe_line/train_eval_save_hyundai_4_15.sh
bash scripts/pipe_line/train_eval_save_hyundai_4_20.sh
bash scripts/pipe_line/train_eval_save_hyundai_5_5.sh
bash scripts/pipe_line/train_eval_save_hyundai_5_10.sh
bash scripts/pipe_line/train_eval_save_hyundai_5_15.sh
bash scripts/pipe_line/train_eval_save_hyundai_5_20.sh
