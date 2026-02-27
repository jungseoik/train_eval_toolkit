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
GPUS=4 PER_DEVICE_BATCH_SIZE=2 bash scripts/shell/internvl3.0/train_sample_scripts.sh
```

학습 완료 후 LoRA 병합까지 자동으로 수행되며, 최종 체크포인트는 `ckpts/InternVL3-2B_sample/`에 저장됩니다.

---

## Step 2 — 평가 실행

```bash
PYTHONPATH="$(pwd)" python src/evaluation/evaluate_image_classfication.py \
  --checkpoint ckpts/InternVL3-2B_sample \
  --annotation data/instruction/evaluation/sample_test.jsonl \
  --image-root data \
  --out-dir results/eval_result_image/sample \
  --batch-size 20 \
  --multi-gpu
```

평가 결과는 `results/eval_result_image/sample/` 에 저장됩니다.

- `result_*.json` — 샘플별 예측 결과
- `summary_*.txt` — Precision / Recall / F1 / Accuracy 요약

---

## 참고 문서

- 학습 파라미터 상세 설명: [docs/train/training.md](training.md)
- 학습 전 체크리스트: [docs/train/pre_training_checklist.md](pre_training_checklist.md)
- 평가 상세 가이드: [docs/eval/eval_image_falldown.md](../eval/eval_image_falldown.md)
