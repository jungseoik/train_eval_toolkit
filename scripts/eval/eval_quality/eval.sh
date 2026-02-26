#!/usr/bin/env bash
# ============================================================
# [단일 평가 — 가장 기본 사용 예시]
# ============================================================

PYTHONPATH="$(pwd)" python src/evaluation/evaluate_qualitative_video_threshold_image.py \
    --checkpoint ckpts/InternVL3-2B_hyundai_5_20 \
    --input-root "data/processed/hyundai_backhwajum/hyundai_video_macs_test/01_27" \
    --output-root "results/eval_quality/eva_quality_hyundai/InternVL3-2B_hyundai_5_20/falldown_poc_01_27" \
    --window-size 15 \
    --batch-size 40 \
    --threshold 1 \
    --multi-gpu


# ============================================================
# Qualitative Video Evaluation (Threshold + Image overlay)
# evaluate_qualitative_video_threshold_image.py 옵션:
#   --checkpoint   : 모델 체크포인트 경로 (required)
#   --input-root   : 재귀적으로 탐색할 입력 비디오 루트 경로 (required)
#   --output-root  : 동일 폴더 구조로 저장할 출력 루트 경로 (required)
#   --window-size  : 슬라이딩 윈도우 크기 — 윈도우 마지막 프레임이 대표 (default: 15)
#   --batch-size   : 한 번에 추론할 대표 프레임 수 (default: 20)
#   --threshold    : falldown 연속 카운트 임계값 — 높을수록 보수적 판정 (default: 1)
#   --multi-gpu    : 멀티 GPU 분산 로딩 사용 (flag)
# ============================================================


# ============================================================
# [복수 경로 배치 평가 예시]
# MODEL_TAG와 INPUT_DIRS만 수정해 여러 폴더를 순차 실행
# ============================================================

MODEL_TAG="InternVL3-2B_hyundai_5_20"
CHECKPOINT="ckpts/${MODEL_TAG}"
WINDOW_SIZE=15
BATCH_SIZE=40
THRESHOLD=1

INPUT_DIRS=(
  "data/processed/hyundai_backhwajum/hyundai_video_macs_test/01_27"
  "data/processed/hyundai_backhwajum/hyundai_video_macs_test/falldown"
)

for INPUT_DIR in "${INPUT_DIRS[@]}"; do
  # input-root 마지막 경로 컴포넌트를 출력 폴더명으로 사용
  LEAF=$(basename "${INPUT_DIR}")
  OUTPUT_DIR="results/eval_quality/eva_quality_hyundai/${MODEL_TAG}/${LEAF}"

  PYTHONPATH="$(pwd)" python src/evaluation/evaluate_qualitative_video_threshold_image.py \
      --checkpoint "${CHECKPOINT}" \
      --input-root "${INPUT_DIR}" \
      --output-root "${OUTPUT_DIR}" \
      --window-size "${WINDOW_SIZE}" \
      --batch-size "${BATCH_SIZE}" \
      --threshold "${THRESHOLD}" \
      --multi-gpu
done
