# 학습 전 준비사항 체크리스트

학습 스크립트를 실행하기 전 반드시 완료해야 할 3단계 준비 과정을 설명합니다.

```
[1] 라벨 → JSONL 변환
        ↓
[2] 메타 JSON 파일 생성 (scripts/shell/data/*.json)
        ↓
[3] 학습 스크립트에 메타 JSON 경로 지정 (META_PATH)
        ↓
학습 실행
```

---

## 1단계 — 라벨 → JSONL 변환

오토라벨링으로 생성된 `*.json` 파일들을 학습/평가용 JSONL로 변환합니다.

> 상세 옵션 설명: [docs/cleaning/label_to_jsonl.md](../cleaning/label_to_jsonl.md)

**Gangnam 비디오 예시 (학습용 · 평가용)**

```bash
# 학습용 JSONL 생성 (violence)
python main.py label2jsonl \
  -i data/processed/gangnam/yeoksam2_v2/Train/video/violence \
  -o data/instruction/train/train_gangnam_yeoksam2_v2_video_violence.jsonl \
  -dt video -opt train -ity clip -itk caption -tn violence

# 평가용 JSONL 생성 (falldown)
python main.py label2jsonl \
  -i data/processed/gangnam/yeoksam2_v2/Test/video/falldown \
  -o data/instruction/evaluation/test_gangnam_yeoksam2_v2_video_falldown.jsonl \
  -dt video -opt test -ity clip -itk caption -tn falldown
```

**변환 결과 확인**

```bash
python main.py jsonl_inform_check \
  -i data/instruction/train/train_gangnam_yeoksam2_v2_video_violence.jsonl
```

변환된 JSONL은 `data/instruction/train/` (학습용) 또는 `data/instruction/evaluation/` (평가용)에 저장합니다.

---

## 2단계 — 메타 JSON 파일 생성

학습 스크립트는 어떤 JSONL 파일을 학습에 사용할지 **메타 JSON**으로 지정받습니다.
메타 JSON은 `scripts/shell/data/` 디렉토리에 위치합니다.

### 포맷

```json
{
  "데이터셋_이름": {
    "root": "data/",
    "annotation": "data/instruction/train/학습용.jsonl",
    "data_augment": false,
    "repeat_time": 1,
    "length": 1000
  }
}
```

| 필드 | 설명 |
|---|---|
| `데이터셋_이름` | 식별용 키 (임의로 지정, 대문자 권장) |
| `root` | 미디어 파일의 기준 루트 디렉토리. JSONL 내 상대 경로의 prefix가 됩니다 |
| `annotation` | 1단계에서 생성한 학습용 JSONL 경로 |
| `data_augment` | 데이터 증강 여부 (현재 `false` 고정) |
| `repeat_time` | 해당 데이터셋 반복 횟수 (기본 `1`) |
| `length` | 데이터셋 샘플 수 (JSONL 줄 수와 일치시킵니다) |

### 단일 데이터셋 예시

```json
{
  "GANGNAM_VIDEO_VIOLENCE": {
    "root": "data/",
    "annotation": "data/instruction/train/train_gangnam_yeoksam2_v2_video_violence.jsonl",
    "data_augment": false,
    "repeat_time": 1,
    "length": 800
  }
}
```

저장 경로 예시: `scripts/shell/data/gangnam_yeoksam2_v2.json`

### 여러 데이터셋 혼합 예시

서로 다른 데이터셋을 하나의 메타 JSON에 묶으면 학습 시 자동으로 합쳐집니다.

```json
{
  "GANGNAM_YEOKSAM2_V2_VIOLENCE": {
    "root": "data/",
    "annotation": "data/instruction/train/train_gangnam_yeoksam2_v2_video_violence.jsonl",
    "data_augment": false,
    "repeat_time": 1,
    "length": 800
  },
  "GANGNAM_GAEPO1_V2_FALLDOWN": {
    "root": "data/",
    "annotation": "data/instruction/train/train_gangnam_gaepo1_v2_image_falldown.jsonl",
    "data_augment": false,
    "repeat_time": 1,
    "length": 910
  }
}
```

> `length`는 JSONL 줄 수와 정확히 일치해야 합니다. 다음 명령으로 확인합니다.
>
> ```bash
> wc -l data/instruction/train/train_gangnam_yeoksam2_v2_video_violence.jsonl
> ```

---

## 3단계 — 학습 스크립트에 메타 JSON 경로 지정

`scripts/shell/internvl3.0/train_sample_scripts.sh` (또는 사용할 `.sh` 파일) 상단의 `META_PATH` 변수를 2단계에서 만든 파일 경로로 수정합니다.

```bash
# ─── 사용자 설정 변수 ─────────────────────────────────────────────
BASE_MODEL="ckpts/InternVL3-2B"
META_PATH="scripts/shell/data/gangnam_yeoksam2_v2.json"   # ← 여기를 수정
DEEPSPEED_CONFIG="configs/deepspeed/zero_stage1_config.json"
LORA_OUTPUT_DIR="ckpts/lora"
MERGE_DIR="ckpts/InternVL3-2B_gangnam_sample"              # ← 출력 디렉토리명도 구분되게 설정
# ─────────────────────────────────────────────────────────────────
```

> **`MERGE_DIR`도 실험마다 다르게 설정**하세요. 같은 이름으로 두 번 실행하면 이전 체크포인트를 덮어씁니다.

---

## 체크리스트 요약

학습 실행 전 아래 항목을 순서대로 확인하세요.

- [ ] `data/instruction/train/*.jsonl` — 학습용 JSONL 생성 완료
- [ ] `wc -l` 로 JSONL 샘플 수 확인
- [ ] `scripts/shell/data/*.json` — 메타 JSON 파일 생성 및 `length` 값 일치 확인
- [ ] `scripts/shell/internvl3.0/*.sh` — `META_PATH` 수정 완료
- [ ] `scripts/shell/internvl3.0/*.sh` — `MERGE_DIR` 이름 중복 여부 확인
- [ ] `ckpts/InternVL3-2B/` — 베이스 모델 다운로드 완료 (빠른 시작 1단계)

모든 항목 확인 후 학습을 실행합니다.

```bash
GPUS=4 PER_DEVICE_BATCH_SIZE=2 bash scripts/shell/internvl3.0/train_sample_scripts.sh
```

상세 학습 파라미터 및 환경변수 설명은 [docs/train/training.md](training.md)를 참조하세요.
