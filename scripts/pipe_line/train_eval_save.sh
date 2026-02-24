#!/usr/bin/env bash
set -euo pipefail

bash scripts/pipe_line/train_eval_save_hyundai_3_10.sh
bash scripts/pipe_line/train_eval_save_hyundai_3_15.sh
bash scripts/pipe_line/train_eval_save_hyundai_3_20.sh
