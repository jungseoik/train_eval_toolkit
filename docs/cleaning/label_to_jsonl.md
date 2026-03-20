# label2jsonl — JSON 라벨 → 학습/평가용 JSONL 변환 가이드

오토라벨링으로 생성된 `*.json` 파일들을 InternVL 학습·평가에 사용할 수 있는 JSONL 포맷으로 변환합니다.

---

## 개요

```
python main.py label2jsonl [옵션]
```

입력 디렉토리를 **재귀적으로 탐색**하여 `*.json` 파일을 찾고, 동일 디렉토리에 있는 같은 이름의 미디어 파일(영상/이미지)과 쌍을 맞춰 JSONL 한 줄씩 출력합니다.

---

## 옵션 레퍼런스

| 옵션 | 축약 | 기본값 | 필수 여부 | 설명 |
|---|---|---|---|---|
| `--input_dir` | `-i` | — | **필수** | JSON 라벨과 미디어 파일이 위치한 루트 디렉토리 |
| `--output_file` | `-o` | — | **필수** | 결과를 저장할 JSONL 파일 경로 |
| `--mode` | `-opt` | `train` | 선택 | 변환 모드 (`train` / `test`) |
| `--data_type` | `-dt` | `video` | 선택 | 미디어 타입 (`video` / `image`) |
| `--item_type` | `-ity` | `clip` | 선택 | JSONL `type` 필드값 — 학습 프레임워크에서 데이터 유형 식별용 |
| `--item_task` | `-itk` | `caption` | 선택 | JSONL `task` 필드값 — 학습 프레임워크에서 태스크 유형 식별용 |
| `--task_name` | `-tn` | `violence` | 선택 | 분류 작업명 — 프롬프트 자동 선택에 사용 |
| `--base-dir` | — | `data/` | 선택 | 미디어 상대 경로 계산 기준 디렉토리 |

---

## 옵션 상세 설명

### `-opt` / `--mode` — 변환 모드

| 값 | 동작 |
|---|---|
| `train` | `human`(질문 프롬프트) + `gpt`(정답) 쌍으로 `conversations` 구성 → **학습용** |
| `test` | `gpt`(정답) 만 포함, `human` 파트 제외 → **평가용** |

> `train` 모드에서는 `--data_type`과 `--task_name` 조합으로 적합한 프롬프트가 자동으로 삽입됩니다.

---

### `-dt` / `--data_type` — 미디어 타입

| 값 | 탐색 확장자 | JSONL 키 |
|---|---|---|
| `video` | `.mp4`, `.avi`, `.mov`, `.mkv` | `"video"` |
| `image` | `.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp` | `"image"` |

---

### `-ity` / `--item_type` — 아이템 타입

JSONL 각 항목의 `"type"` 필드에 들어가는 값입니다. 학습 프레임워크에서 데이터 유형을 식별하는 키로 사용됩니다.

| 값 | 의미 | 권장 data_type |
|---|---|---|
| `clip` | 짧은 영상 클립 단위 | video |
| `capture_frame` | 캡처된 이미지 프레임 단위 | image |

자유 입력이 가능하므로, 새로운 데이터 유형에 맞게 지정할 수 있습니다.

---

### `-tn` / `--task_name` — 작업명 (프롬프트 선택 키)

`train` 모드에서 `--data_type`과 함께 조합되어 **삽입할 human 프롬프트를 결정**합니다.

프롬프트는 `configs/prompts/label2jsonl.yaml`에서 관리됩니다.

| `--data_type` | `--task_name` | YAML 키 |
|---|---|---|
| `video` | `violence` | `violence__video` |
| `video` | `falldown` | `falldown__video` |
| `image` | `violence` | `violence__image` |
| `image` | `falldown` | `falldown__image` |

> **새 작업 추가**: `configs/prompts/label2jsonl.yaml`에 `{task_name}__{data_type}` 형식으로 프롬프트를 추가하면 코드 변경 없이 바로 사용할 수 있습니다.

---

## 출력 JSONL 구조

### train 모드 (비디오 예시)

```json
{
  "id": 0,
  "type": "clip",
  "task": "caption",
  "video": "processed/gangnam/samsung/Train/clean/video/violence/clip_001.mp4",
  "conversations": [
    {
      "from": "human",
      "value": "<video>\nWatch this short video clip and respond with exactly one JSON object.\n..."
    },
    {
      "from": "gpt",
      "value": "{\"category\": \"violence\", \"description\": \"A man punches another man.\"}"
    }
  ]
}
```

### test 모드 (이미지 예시)

```json
{
  "id": 0,
  "type": "capture_frame",
  "task": "caption",
  "image": "processed/hyundai_backhwajum/hyundai_01_27_QA/test/falldown/frame_001.jpg",
  "conversations": [
    {
      "from": "gpt",
      "value": "{\"category\": \"falldown\", \"description\": \"Person lying on the floor.\"}"
    }
  ]
}
```

> 미디어 파일 경로는 `--base-dir` (기본 `data/`) 기준 상대 경로로 저장됩니다.

---

## 입력 JSON 포맷 요건

변환 대상 JSON 파일은 아래 두 구조 중 하나를 따라야 합니다.

**단일 dict 형식 (오토라벨링 기본 출력)**

```json
{
  "category": "violence",
  "description": "A man punches another man."
}
```

**list 형식 (레거시 포맷)**

```json
[
  {
    "category": "normal",
    "eng_caption": "Two people walking in a hallway."
  }
]
```

> `list` 형식에서는 첫 번째 항목의 `eng_caption` 또는 `en_caption` 키를 `description`으로 사용합니다.

---

## 사용 예시

### 비디오 — 학습용 JSONL 생성 (violence)

```bash
python main.py label2jsonl \
  -i data/processed/gangnam/yeoksam2_v2/Train/video/violence \
  -o data/instruction/train/train_gangnam_yeoksam2_v2_video_violence.jsonl \
  -dt video -opt train -ity clip -itk caption -tn violence
```

### 비디오 — 평가용 JSONL 생성 (falldown)

```bash
python main.py label2jsonl \
  -i data/processed/gangnam/yeoksam2_v2/Test/video/falldown \
  -o data/instruction/evaluation/test_gangnam_yeoksam2_v2_video_falldown.jsonl \
  -dt video -opt test -ity clip -itk caption -tn falldown
```

### 이미지 — 학습용 JSONL 생성 (falldown)

```bash
python main.py label2jsonl \
  -i data/processed/gangnam/yeoksam2_v2/Train/image/falldown \
  -o data/instruction/train/train_gangnam_yeoksam2_v2_image_falldown.jsonl \
  -dt image -opt train -ity capture_frame -itk caption -tn falldown
```

### 이미지 — 평가용 JSONL 생성 (falldown)

```bash
python main.py label2jsonl \
  -i data/processed/gangnam/yeoksam2_v2/Test/image/falldown \
  -o data/instruction/evaluation/test_gangnam_yeoksam2_v2_image_falldown.jsonl \
  -dt image -opt test -ity capture_frame -itk caption -tn falldown
```

---

## 변환 후 품질 점검

JSONL 생성 후 아래 명령으로 분포와 유효성을 확인하세요.

```bash
python main.py jsonl_inform_check \
  -i data/instruction/train/train_gangnam_samsung_video_violence.jsonl
```

---

## 관련 파일

| 파일 | 역할 |
|---|---|
| `main.py` | CLI 서브커맨드 `label2jsonl` 등록 |
| `src/preprocess/label2jsonl.py` | 핵심 변환 로직 |
| `configs/prompts/label2jsonl.yaml` | 프롬프트 템플릿 정의 (task_name × data_type) |
| `scripts/utils/label_to_jsonl.sh` | 프로젝트별 실제 사용 예시 모음 |
