# 퀵스타트

데이터 다운로드 및 압축 해제까지 완료된 상태에서 바로 실행 가능한 가이드입니다.

> **전제 조건**
> - 환경 설치 완료 ([빠른 시작 2단계](../../README.md#2-환경-설치))
> - Gangnam 데이터 다운로드 & 압축 해제 완료 ([Gangnam 다운로드 가이드](../data/download/guideline_gangnam.md))
> - `ckpts/InternVL3-2B/` 베이스 모델 다운로드 완료

---

## 사용 데이터

`gangnam_yeoksam2_v2_image_falldown` — 역삼2 구역 이미지 기반 낙상 감지 데이터셋

JSONL 변환이 이미 완료된 샘플 파일로 구성됩니다.

| 파일 | 설명 |
|---|---|
| `data/instruction/train/sample_train.jsonl` | 학습용 JSONL (544개) |
| `data/instruction/evaluation/sample_test.jsonl` | 평가용 JSONL |
| `scripts/shell/data/sample.json` | 학습 메타 JSON |
| `scripts/shell/internvl3.0/train_sample_scripts.sh` | 학습 + LoRA 병합 스크립트 |

---

## Step 1 — 학습 실행

```bash
# 기본 실행 (GPU 8개)
bash scripts/shell/internvl3.0/train_sample_scripts.sh

# GPU 수 / 배치 크기 조정
## a5000 기준 batch size 1로 설정하세요(세부 배치사이즈는 다릅니다)
GPUS=4 PER_DEVICE_BATCH_SIZE=1 bash scripts/shell/internvl3.0/train_sample_scripts.sh
```

학습 완료 후 LoRA 병합까지 자동으로 수행되며, 최종 체크포인트는 `ckpts/InternVL3-2B_sample/`에 저장됩니다.

---

## Step 2 — 베이스 모델 평가 (학습 전)

> **반드시 베이스 모델을 먼저 측정하세요.** 학습 전후 성능 비교의 기준이 됩니다.

```bash
PYTHONPATH="$(pwd)" python src/evaluation/evaluate_image_classfication.py \
  --checkpoint ckpts/InternVL3-2B \
  --annotation data/instruction/evaluation/sample_test.jsonl \
  --image-root data \
  --out-dir results/eval_result_image/sample_base \
  --batch-size 5 \
  --multi-gpu
```

---

## Step 3 — 학습 후 모델 평가

```bash
PYTHONPATH="$(pwd)" python src/evaluation/evaluate_image_classfication.py \
  --checkpoint ckpts/InternVL3-2B_sample \
  --annotation data/instruction/evaluation/sample_test.jsonl \
  --image-root data \
  --out-dir results/eval_result_image/sample \
  --batch-size 5 \
  --multi-gpu
```

두 평가 결과를 비교해 파인튜닝 효과를 확인합니다.

| 결과 경로 | 설명 |
|---|---|
| `results/eval_result_image/sample_base/` | 베이스 모델 평가 결과 |
| `results/eval_result_image/sample/` | 학습 후 모델 평가 결과 |

- `result_*.json` — 샘플별 예측 결과
- `summary_*.txt` — Precision / Recall / F1 / Accuracy 요약

---

## 참고 문서

- 학습 파라미터 상세 설명: [docs/train/training.md](training.md)
- 학습 전 체크리스트: [docs/train/pre_training_checklist.md](pre_training_checklist.md)
- 평가 상세 가이드: [docs/eval/eval_image_falldown.md](../eval/eval_image_falldown.md)
