# 정성 평가 가이드 (비디오 Threshold + 이미지 오버레이)

슬라이딩 윈도우 방식으로 비디오를 추론하고, falldown 판정 결과를 프레임에 오버레이한 영상을 저장하는 정성 평가 방법을 설명합니다.

## 사용 스크립트

| 항목 | 경로 |
|---|---|
| 실행 스크립트 | `scripts/eval/eval_quality/eval.sh` |
| 평가 모듈 | `src/evaluation/evaluate_qualitative_video_threshold_image.py` |
| 결과 저장 | `results/eval_quality/` |

## 동작 원리

1. `--input-root` 하위 비디오 파일을 재귀 탐색합니다.
2. 각 비디오에 슬라이딩 윈도우(`--window-size`)를 적용해 마지막 프레임을 대표 프레임으로 선택합니다.
3. 대표 프레임을 `--batch-size` 단위로 모델에 일괄 추론합니다.
4. falldown 판정이 `--threshold` 횟수 연속으로 나오면 해당 구간을 falldown으로 최종 판정합니다.
5. 각 프레임에 판정 결과를 텍스트로 오버레이한 비디오를 `--output-root`에 원본과 동일한 폴더 구조로 저장합니다.

## 기본 실행

```bash
PYTHONPATH="$(pwd)" python src/evaluation/evaluate_qualitative_video_threshold_image.py \
    --checkpoint ckpts/InternVL3-2B_hyundai_5_20 \
    --input-root "data/processed/hyundai_backhwajum/hyundai_video_macs_test/01_27" \
    --output-root "results/eval_quality/eva_quality_hyundai/InternVL3-2B_hyundai_5_20/falldown_poc_01_27" \
    --window-size 15 \
    --batch-size 40 \
    --threshold 1 \
    --multi-gpu
```

## 옵션

| 옵션 | 설명 | 기본값 |
|---|---|---|
| `--checkpoint` | 모델 체크포인트 경로 | (필수) |
| `--input-root` | 재귀 탐색할 입력 비디오 루트 경로 | (필수) |
| `--output-root` | 결과 비디오 저장 루트 경로 | (필수) |
| `--window-size` | 슬라이딩 윈도우 크기 — 윈도우 마지막 프레임이 대표 | `15` |
| `--batch-size` | 한 번에 추론할 대표 프레임 수 | `20` |
| `--threshold` | falldown 연속 카운트 임계값 — 높을수록 보수적 판정 | `1` |
| `--multi-gpu` | 멀티 GPU 분산 로딩 | (flag) |

### threshold 설정 기준

| 값 | 판정 성향 | 권장 상황 |
|---|---|---|
| `1` | 민감 — 1회 연속만으로 falldown 판정 | PoC 데모, 놓침 최소화 |
| `2` | 중간 — 2회 연속 필요 | 일반적인 운영 환경 |
| `3+` | 보수적 — 오탐 억제 | FP를 엄격히 줄여야 할 때 |

## 여러 경로 배치 평가

```bash
MODEL_TAG="InternVL3-2B_hyundai_5_20"
CHECKPOINT="ckpts/${MODEL_TAG}"

INPUT_DIRS=(
  "data/processed/hyundai_backhwajum/hyundai_video_macs_test/01_27"
  "data/processed/hyundai_backhwajum/hyundai_video_macs_test/falldown"
)

for INPUT_DIR in "${INPUT_DIRS[@]}"; do
  LEAF=$(basename "${INPUT_DIR}")
  OUTPUT_DIR="results/eval_quality/eva_quality_hyundai/${MODEL_TAG}/${LEAF}"

  PYTHONPATH="$(pwd)" python src/evaluation/evaluate_qualitative_video_threshold_image.py \
      --checkpoint "${CHECKPOINT}" \
      --input-root "${INPUT_DIR}" \
      --output-root "${OUTPUT_DIR}" \
      --window-size 15 \
      --batch-size 40 \
      --threshold 1 \
      --multi-gpu
done
```

자세한 실행 예시는 `scripts/eval/eval_quality/eval.sh`를 참조하세요.

## 결과

- `--output-root` 하위에 입력 폴더 구조를 그대로 유지하며 결과 비디오가 저장됩니다.
- 각 프레임에 `falldown` / `normal` 판정 텍스트가 오버레이됩니다.
