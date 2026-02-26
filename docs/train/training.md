# 학습 가이드 (InternVL3 LoRA 파인튜닝)

이 문서는 InternVL3 계열 VLM의 LoRA 파인튜닝 학습 스크립트 사용법을 설명합니다.

---

## 스크립트 종류

| 스크립트 | 설명 |
|---|---|
| `scripts/shell/internvl3.0/train_sample_scripts.sh` | 학습 + LoRA 병합만 수행 |
| `scripts/pipe_line/train_eval_save_sample_scripts.sh` | 학습 → LoRA 병합 → 이미지/영상 평가 → ffmpeg 인코딩 전 과정 파이프라인 |

---

## 빠른 시작

### 단독 학습 스크립트

```bash
# 기본 실행 (GPU 8개, PER_DEVICE_BATCH_SIZE=4)
bash scripts/shell/internvl3.0/train_sample_scripts.sh

# GPU 수 / 배치 크기 조정
GPUS=4 PER_DEVICE_BATCH_SIZE=2 bash scripts/shell/internvl3.0/train_sample_scripts.sh
```

### 전체 파이프라인 (학습 + 평가)

```bash
# 기본 실행 (epoch=20)
bash scripts/pipe_line/train_eval_save_sample_scripts.sh

# epoch 수 조정
EPOCHS=10 GPUS=4 PER_DEVICE_BATCH_SIZE=2 bash scripts/pipe_line/train_eval_save_sample_scripts.sh
```

---

## 주요 환경변수

스크립트 실행 시 환경변수로 오버라이드할 수 있는 값들입니다.

| 변수 | 기본값 | 설명 |
|---|---|---|
| `GPUS` | `8` | 사용할 GPU 수 |
| `TRAIN_BATCH_SIZE` | `512` | 전체 배치 사이즈 (gradient accumulation 자동 계산) |
| `PER_DEVICE_BATCH_SIZE` | `4` | GPU 당 배치 사이즈 (GPU 메모리에 맞게 조정) |
| `EPOCHS` | `20` | 학습 에폭 수 (파이프라인 스크립트에서만 적용) |

> `GRADIENT_ACCUMULATION = TRAIN_BATCH_SIZE / PER_DEVICE_BATCH_SIZE / GPUS`
> 예) GPUS=4, PER_DEVICE_BATCH_SIZE=2, TRAIN_BATCH_SIZE=512 → gradient_acc=64

---

## 스크립트 내부 설정 변수

스크립트 상단 설정 블록에서 직접 수정합니다.

```bash
BASE_MODEL="ckpts/InternVL3-2B"          # 베이스 모델 경로
META_PATH="scripts/shell/data/hyundai_8.json"  # 학습 데이터 메타 경로
DEEPSPEED_CONFIG="configs/deepspeed/zero_stage1_config.json"
LORA_OUTPUT_DIR="ckpts/lora"             # LoRA 임시 저장 경로
MERGE_DIR="ckpts/InternVL3-2B_sample"   # LoRA 병합 후 저장 경로

EPOCHS=5
LEARNING_RATE=4e-5
MAX_SEQ_LENGTH=16384
LORA_RANK=64
SAVE_STEPS=200
```

---

## 학습 파라미터 요약

| 파라미터 | 값 | 설명 |
|---|---|---|
| `--freeze_llm` | `True` | LLM 가중치 동결 |
| `--freeze_mlp` | `True` | MLP 커넥터 동결 |
| `--freeze_backbone` | `True` | 비전 인코더 동결 |
| `--use_llm_lora` | `64` | LoRA rank (LLM에만 적용) |
| `--learning_rate` | `4e-5` | 학습률 |
| `--lr_scheduler_type` | `cosine` | 스케줄러 |
| `--warmup_ratio` | `0.03` | 웜업 비율 |
| `--bf16` | `True` | BF16 혼합정밀도 학습 |
| `--grad_checkpoint` | `True` | Gradient Checkpointing (메모리 절약) |
| `--max_seq_length` | `16384` | 최대 시퀀스 길이 |
| `--max_dynamic_patch` | `32` | 이미지 동적 패치 최대 수 |
| `--deepspeed` | `zero_stage1` | DeepSpeed ZeRO Stage 1 |

---

## 메타데이터 파일 (--meta_path)

학습 데이터 구성을 JSON으로 정의합니다. 경로는 `scripts/shell/data/` 아래에 위치합니다.

```json
{
  "hyundai_train": {
    "root": "data",
    "annotation": "data/instruction/train/train_hyundai.jsonl",
    "data_augment": false,
    "repeat_time": 1,
    "length": 10000
  }
}
```

---

## 실행 흐름

```
torchrun (학습)
    └─> ckpts/lora/          ← LoRA 체크포인트 저장
        └─> merge_lora.py    ← LoRA + 베이스 모델 병합
            └─> ckpts/<MERGE_DIR>/   ← 추론 가능한 최종 체크포인트
                └─> (파이프라인만) 이미지/영상 평가 스크립트 순차 실행
```

---

## 출력 경로

| 경로 | 내용 |
|---|---|
| `ckpts/lora/` | 학습 중 저장되는 LoRA 체크포인트 (병합 후 자동 삭제) |
| `ckpts/lora/training_log.txt` | 학습 로그 |
| `ckpts/<MERGE_DIR>/` | LoRA 병합된 최종 모델 |
| `results/eval_result_image/` | 이미지 분류 평가 결과 |
| `results/eval_quality/` | 영상 퀄리티 평가 결과 |

---

## GPU 메모리 절약 팁

- `PER_DEVICE_BATCH_SIZE`를 줄이면 메모리 사용량 감소 (gradient accumulation 자동 증가)
- `--max_dynamic_patch 32` → 메모리 부족 시 줄여서 사용 (예: `12`, `6`)
- `--grad_checkpoint True` — 기본 활성화, 메모리 절약 대신 학습 속도 소폭 감소
