# LMDeploy YAML 설정 작성 가이드

LMDeploy 벤치마크 파이프라인의 YAML 설정 파일을 처음부터 작성하기 위한 가이드입니다.

- 파이프라인 사용법 및 실행 방법: [lmdeploy_pipeline.md](lmdeploy_pipeline.md)
- 벤치마크 폴더 구조 상세: [../../.docs/bench_folder.md](../../.docs/bench_folder.md)

---

## 빠른 시작

기준 템플릿을 복사해서 `# <-- 변경` 필드만 수정하면 바로 사용할 수 있습니다.

```bash
cp configs/lmdeploy_pipeline/template/template.yaml configs/lmdeploy_pipeline/{모델명}_{카테고리}.yaml
```

> 기준 템플릿: [`configs/lmdeploy_pipeline/template/template.yaml`](../../configs/lmdeploy_pipeline/template/template.yaml)

---

## 전체 YAML 템플릿

아래는 `configs/lmdeploy_pipeline/template/template.yaml`의 내용입니다. `# <-- 변경` 표시된 필드만 수정하세요.
`# <-- 고정값` 표시된 필드는 서버 환경에 종속된 값이므로 변경하지 마세요.

```yaml
# ============================================================
# LMDeploy 벤치마크 평가 파이프라인 설정
#
# 사용법:
#   python -m src.lmdeploy_pipeline -c configs/lmdeploy_pipeline/<이 파일명>.yaml
# ============================================================

pipeline:
  name: "PIA_AI2team_VQA_falldown Benchmark (LMDeploy)"   # <-- 변경: 파이프라인 식별 이름
  steps:
    docker: true
    evaluate: true
    submit: true
  cleanup_docker: true

retry:
  max_attempts: 3
  wait_seconds: 30

# -- Docker 설정 --
docker:
  container_name: "pia_ai2team_vqa_falldown_lmdeploy"      # <-- 변경: 고유한 컨테이너 이름
  image: "openmmlab/lmdeploy:latest-cu12"                   # <-- 고정값: LMDeploy Docker 이미지
  model_path: "ckpts/PIA_AI2team_VQA_falldown"              # <-- 변경: 로컬 모델 경로
  hf_repo_id: "PIA-SPACE-LAB/PIA_AI2team_VQA_falldown"     # <-- 변경: (선택) HuggingFace repo ID
  container_model_path: "/model"                             # <-- 고정값
  gpus: "all"
  port: 23333
  ipc: "host"
  volumes: []
  lmdeploy_args:
    tp: 1                    # GPU 텐서 병렬 수
    session-len: 8192        # KV cache 최대 시퀀스 길이
    backend: "pytorch"       # 추론 백엔드
    cache-max-entry-count: 0.8   # KV cache GPU 메모리 비율 (0.0~1.0)
  startup:
    timeout_seconds: 300     # 서버 준비 대기 최대 시간 (초)
    poll_interval_seconds: 5
    stream_logs: true

# -- 평가 설정 --
evaluate:
  model: "/model"                                            # <-- 고정값: 컨테이너 내부 모델 경로
  run_name: "PIA_AI2team_VQA_falldown"                       # <-- 변경: 결과 디렉토리명
  api_base: "http://127.0.0.1:23333/v1"                     # <-- 고정값: LMDeploy API 엔드포인트
  bench_base_path: "/mnt/PoC_benchmark/huggingface_benchmarks_dataset/Leaderboard_bench"  # <-- 고정값: 서버측 벤치마크 경로
  output_path: "results/lmdeploy_eval"
  overwrite_results: false   # true=항상 덮어쓰기, false=기존 CSV 존재+row수 일치 시 스킵

  benchmarks:
    - "ABB_Falldown"
    - "Coupang_Falldown"
    - "DTRO_Falldown"
    - "GangNam_Falldown"
    - "Hyundai_Falldown"
    - "Innodep_Falldown"
    - "KhonKaen_Falldown"
    - "KISA_Falldown"
    - "Kumho_Falldown"
    - "PIA_Falldown"
    - "Soil_Falldown"
    - "UGO_Falldown"
    - "Yonsei_Falldown"

  # 프레임 샘플링
  window_size: 30            # Falldown 권장값: 30 (Fire는 15)
  interpolation: "forward"

  # 추론 파라미터
  concurrency: 10
  jpeg_quality: 95
  max_tokens: 15
  temperature: 0.0
  seed: 0
  negative_label: "normal"

  # 카테고리별 프롬프트
  prompt_templates:
    falldown: |
      Analyze this image carefully. Determine if a person has fallen down.

      Important classification rules:

      - The "falldown" category applies to any person who is lying down, regardless of:
        - the surface (e.g., floor, mattress, bed)
        - the posture (natural or unnatural)
        - the cause (e.g., sleeping, collapsing, lying intentionally)
      - This includes:
        - A person lying flat on the ground or other surfaces
        - A person collapsed or sprawled in any lying position
      - The "normal" category applies only if the person is:
        - sitting
        - standing
        - kneeling
        - or otherwise upright (not lying down)

      Answer in JSON format with BOTH of the following fields:
      - "category": either "falldown" or "normal"
      - "description": a brief reason why this classification was made (e.g., "person lying on a mattress", "person sitting on sofa")

      Example:
      {
        "category": "falldown",
        "description": "person lying on a mattress in natural posture"
      }

    default: "Is {category} visible in this image? Answer only with 0 (no) or 1 (yes)."

# -- 제출 설정 --
submit:
  gradio_url: "http://172.168.47.39:7860/"                   # <-- 고정값: 내부망 벤치마크 리더보드 서버
  model_name: "PIA_AI2team_VQA_falldown"                     # <-- 변경: 리더보드 표시 모델명
  task_name: "🌠 VLM 🖥️"
  datasets_used: "Finetuned"
  config_file: "config.json"
  results_base_dir: "results/lmdeploy_eval"
  interval_seconds: 60
```

---

## 섹션별 상세 설명

### pipeline

파이프라인의 이름과 실행할 단계를 설정합니다.

| 필드 | 설명 |
|------|------|
| `name` | 파이프라인 식별 이름. 실행 로그에 표시됨 |
| `steps.docker` | Docker 컨테이너 기동 단계 실행 여부 |
| `steps.evaluate` | 벤치마크 평가 단계 실행 여부 |
| `steps.submit` | 결과 제출 단계 실행 여부 |
| `cleanup_docker` | 파이프라인 종료 후 Docker 컨테이너 자동 제거 여부 |

### retry

벤치마크별 평가/제출 실패 시 재시도 설정입니다. 일반적으로 기본값을 그대로 사용합니다.

| 필드 | 기본값 | 설명 |
|------|--------|------|
| `max_attempts` | `3` | 최대 재시도 횟수 |
| `wait_seconds` | `30` | 재시도 대기 시간 (초) |

### docker

LMDeploy 서빙 컨테이너 설정입니다.

| 필드 | 변경 여부 | 설명 |
|------|-----------|------|
| `container_name` | **변경** | Docker 컨테이너 이름. 모델별로 고유하게 지정 |
| `image` | 고정 | `openmmlab/lmdeploy:latest-cu12` 사용 |
| `model_path` | **변경** | 로컬 모델 디렉토리 경로 (예: `ckpts/모델명`) |
| `hf_repo_id` | **변경** (선택) | HuggingFace 모델 repo ID. `model_path`에 모델이 없으면 자동 다운로드. 로컬에 모델이 이미 있으면 생략 가능 |
| `container_model_path` | 고정 | 항상 `"/model"`. 컨테이너 내부 마운트 위치 |
| `gpus` | 선택 | `"all"` 또는 `"\"device=0,1\""` 등 GPU 지정 |
| `port` | 선택 | 기본 `23333`. 다른 컨테이너와 포트 충돌 시에만 변경 |
| `ipc` | 고정 | `"host"` |
| `volumes` | 선택 | 추가 볼륨 마운트 리스트 (빈 리스트 `[]` 기본) |

#### lmdeploy_args

LMDeploy 서버에 전달되는 인자입니다.

| 인자 | 기본값 | 설명 |
|------|--------|------|
| `tp` | `1` | 텐서 병렬 수. GPU가 여러 장이면 증가 가능 |
| `session-len` | `8192` | KV cache 최대 시퀀스 길이 |
| `backend` | `"pytorch"` | 추론 백엔드 |
| `cache-max-entry-count` | `0.8` | KV cache에 할당할 GPU 메모리 비율 (0.0~1.0). OOM 발생 시 낮추기 |

#### startup

컨테이너 기동 대기 설정입니다.

| 인자 | 기본값 | 설명 |
|------|--------|------|
| `timeout_seconds` | `300` | 서버 준비 대기 최대 시간 (초). 대형 모델은 증가 필요 |
| `poll_interval_seconds` | `5` | health check 폴링 간격 (초) |
| `stream_logs` | `true` | 대기 중 Docker 로그 실시간 출력 |

### evaluate

벤치마크 평가 설정입니다.

#### 고정값 필드

아래 필드는 서버 환경에 종속된 고정값입니다. 변경하지 마세요.

| 필드 | 고정값 | 이유 |
|------|--------|------|
| `model` | `"/model"` | `docker.container_model_path`와 일치해야 함. LMDeploy가 이 경로로 모델을 등록함 |
| `api_base` | `"http://127.0.0.1:23333/v1"` | `docker.port`와 일치해야 함. 로컬 LMDeploy 서버 주소 |
| `bench_base_path` | `"/mnt/PoC_benchmark/..."` | 서버에 마운트된 벤치마크 데이터셋 경로. 모든 벤치마크 데이터가 이 경로 하위에 존재 |

> **주의**: `bench_base_path`는 서버에 실제로 마운트된 경로여야 합니다.
> API 요청 시 경로가 올바르지 않거나 벤치마크 폴더가 존재하지 않으면 즉시 **HTTP 400 에러**를 반환합니다.
> 올바른 경로: `/mnt/PoC_benchmark/huggingface_benchmarks_dataset/Leaderboard_bench`

#### 변경 가능 필드

| 필드 | 설명 |
|------|------|
| `run_name` | 결과 저장 디렉토리명. `output_path/run_name/` 하위에 벤치마크별 CSV 저장 |
| `output_path` | 결과 기본 경로. 기본값 `"results/lmdeploy_eval"` |
| `overwrite_results` | `true`: 항상 덮어쓰기 (기본값). `false`: 기존 CSV가 있고 GT와 row 수 일치하면 스킵. 중단된 평가 재개 시 `false` 권장 |
| `benchmarks` | 평가할 벤치마크 이름 목록. [벤치마크 전체 목록](#벤치마크-전체-목록) 참조 |

#### window_size (프레임 샘플링 간격)

영상에서 몇 프레임 간격으로 샘플링할지 결정합니다. 샘플링되지 않은 프레임은 `interpolation` 방식으로 채워집니다.

| 카테고리 | 권장값 | 의미 |
|----------|--------|------|
| **Fire** | `15` | 15프레임마다 1프레임 추론 |
| **Falldown** | `30` | 30프레임마다 1프레임 추론 |

- **값이 작을수록**: 더 많은 프레임을 추론 → 평가 시간 증가, 시간 해상도 향상
- **값이 클수록**: 더 적은 프레임을 추론 → 평가 시간 단축, 시간 해상도 감소

`interpolation`은 기본값 `"forward"`를 사용합니다. 샘플 프레임의 예측값을 다음 샘플까지 그대로 유지합니다.

#### 추론 파라미터

일반적으로 기본값을 그대로 사용합니다.

| 필드 | 기본값 | 설명 |
|------|--------|------|
| `concurrency` | `10` | 프레임 추출 + 추론 동시 처리 수 |
| `jpeg_quality` | `95` | 프레임 JPEG 인코딩 품질 |
| `max_tokens` | `15` | 모델 최대 생성 토큰 수 |
| `temperature` | `0.0` | 샘플링 온도. 0이면 결정적 출력 |
| `seed` | `0` | 재현성을 위한 시드 |
| `negative_label` | `"normal"` | 음성 클래스 레이블. 프롬프트의 "normal"과 일치해야 함 |

### submit

Gradio 리더보드에 결과를 제출하는 설정입니다.

#### 고정값 필드

| 필드 | 고정값 | 이유 |
|------|--------|------|
| `gradio_url` | `"http://172.168.47.39:7860/"` | 내부망 벤치마크 리더보드 서버 주소. 인프라팀에서 운영하는 고정 주소 |
| `task_name` | `"🌠 VLM 🖥️"` | 리더보드 태스크 구분자 |
| `config_file` | `"config.json"` | 제출 시 첨부할 설정 파일명 |
| `results_base_dir` | `"results/lmdeploy_eval"` | `evaluate.output_path`와 동일하게 유지 |

#### 변경 가능 필드

| 필드 | 설명 |
|------|------|
| `model_name` | 리더보드에 표시할 모델 이름 |
| `datasets_used` | 데이터셋 구분 (`"Finetuned"`, `"Pretrained"` 등) |
| `interval_seconds` | 벤치마크 간 제출 간격 (초). 서버 부하 방지용 |

---

## 벤치마크 전체 목록

벤치마크 이름은 `{출처}_{카테고리}` 형식입니다. 코드에서 `_` 기준 마지막 부분을 소문자로 변환하여 카테고리를 추출합니다 (예: `ABB_Falldown` → `falldown`).

### Fire 벤치마크 (6개)

| 벤치마크 이름 | 추출 카테고리 |
|---------------|---------------|
| `PIA_Fire` | fire |
| `SamsungCNT_Fire` | fire |
| `Soil_Fire` | fire |
| `Hyundai_Fire` | fire |
| `Coupang_Fire` | fire |
| `Kumho_Fire` | fire |

### Falldown 벤치마크 (13개)

| 벤치마크 이름 | 추출 카테고리 |
|---------------|---------------|
| `ABB_Falldown` | falldown |
| `Coupang_Falldown` | falldown |
| `DTRO_Falldown` | falldown |
| `GangNam_Falldown` | falldown |
| `Hyundai_Falldown` | falldown |
| `Innodep_Falldown` | falldown |
| `KhonKaen_Falldown` | falldown |
| `KISA_Falldown` | falldown |
| `Kumho_Falldown` | falldown |
| `PIA_Falldown` | falldown |
| `Soil_Falldown` | falldown |
| `UGO_Falldown` | falldown |
| `Yonsei_Falldown` | falldown |

> 벤치마크 이름은 정확히 위 표의 값을 사용해야 합니다. 대소문자가 다르면 데이터셋 경로를 찾지 못합니다.

---

## prompt_templates 작성 가이드

`prompt_templates`는 카테고리별로 모델에 전달할 프롬프트를 정의합니다.

### 작성 규칙

1. **키는 소문자 카테고리명**: `fire`, `falldown`, `smoke` 등. 코드가 벤치마크 이름에서 카테고리를 추출할 때 소문자로 변환하므로, 키도 소문자여야 매칭됨
2. **`default` 키는 필수**: 명시적으로 정의되지 않은 카테고리의 폴백으로 사용됨. 누락 시 에러 발생
3. **`{category}` 플레이스홀더**: 템플릿 문자열에 `{category}`를 넣으면 실행 시 실제 카테고리명으로 치환됨
4. **JSON 응답 포맷 권장**: 프롬프트에서 `"category"` 필드를 포함한 JSON 응답을 요청하면, 파서가 자동으로 값을 추출함. 유효한 값은 `[카테고리명, negative_label]` (예: `["falldown", "normal"]`)
5. **멀티라인은 YAML `|` 문법**: 긴 프롬프트는 `|` (리터럴 블록 스칼러)로 작성하여 줄바꿈을 유지

### 상세 프롬프트 템플릿

정확도가 중요한 카테고리에는 아래 구조를 따르는 상세 프롬프트를 권장합니다.

```text
# 프롬프트 템플릿 구조 (의사코드)
<카테고리명>: |
  Analyze this image carefully. [태스크 설명].

  Important classification rules:

  - The "<카테고리명>" category applies to:
    - [양성 조건 1]
    - [양성 조건 2]
  - The "normal" category applies to:
    - [음성 조건 1]
    - [음성 조건 2]

  Answer in JSON format with BOTH of the following fields:
  - "category": either "<카테고리명>" or "normal"
  - "description": a brief reason why this classification was made

  Example:
  {
    "category": "<카테고리명>",
    "description": "[예시 설명]"
  }
```

### 실제 사용 중인 프롬프트 예시

**Falldown 프롬프트** (상세):

```yaml
falldown: |
  Analyze this image carefully. Determine if a person has fallen down.

  Important classification rules:

  - The "falldown" category applies to any person who is lying down, regardless of:
    - the surface (e.g., floor, mattress, bed)
    - the posture (natural or unnatural)
    - the cause (e.g., sleeping, collapsing, lying intentionally)
  - This includes:
    - A person lying flat on the ground or other surfaces
    - A person collapsed or sprawled in any lying position
  - The "normal" category applies only if the person is:
    - sitting
    - standing
    - kneeling
    - or otherwise upright (not lying down)

  Answer in JSON format with BOTH of the following fields:
  - "category": either "falldown" or "normal"
  - "description": a brief reason why this classification was made

  Example:
  {
    "category": "falldown",
    "description": "person lying on a mattress in natural posture"
  }
```

**Fire 프롬프트** (상세):

```yaml
fire: |
  Analyze this image carefully. Only classify it as "fire" if there are ACTUAL FLAMES or ACTIVE FIRE visible in the image.

  Important classification rules:

  - The "fire" category applies ONLY to images with:
   - visible flames or burning fire
   - active combustion or blazing materials
   - actual fire sources (e.g., campfire, house fire, candle flame, burning car, torch flame)
  - The "normal" category applies to:
   - Fire-related objects WITHOUT actual flames (fire trucks, fire extinguishers, fire hydrants)
   - Red colors, sunset glows, red lights, or the sun
   - welding sparks, grinding sparks, cutting sparks, or metal sparks without visible flames
   - Any image without visible flames or active fire

  Answer in JSON format with BOTH of the following fields:
  - "category": either "fire" or "normal"
  - "description": a brief reason why this classification was made

  Example:
  {
    "category": "fire",
    "description": "visible flames from burning building"
  }
```

**간단 프롬프트** (기타 카테고리용):

```yaml
smoke: "Is {category} visible in this image? Answer only with 0 (no) or 1 (yes)."
violence: "Is {category} visible in this image? Answer only with 0 (no) or 1 (yes)."
default: "Is {category} visible in this image? Answer only with 0 (no) or 1 (yes)."
```

> 간단 프롬프트는 `{category}`가 실행 시 실제 카테고리명으로 치환됩니다. 상세 프롬프트에 비해 정확도가 낮을 수 있습니다.

---

## Fire 평가 시 변경 사항

Falldown 템플릿을 복사한 뒤 아래 필드만 변경하면 Fire 평가용 YAML이 됩니다.

```yaml
# pipeline.name 변경
pipeline:
  name: "InternVL3-2B Fire Benchmark (LMDeploy)"

# docker 섹션: 모델 관련 필드 변경
docker:
  container_name: "internvl3_2b_fire_lmdeploy"
  model_path: "ckpts/InternVL3-2B_hyundai_8_20"
  # hf_repo_id: "..."  # 필요 시 설정

# evaluate 섹션: 벤치마크 목록, window_size, 프롬프트 변경
evaluate:
  run_name: "InternVL3-2B_hyundai_8_20"

  benchmarks:
    - "PIA_Fire"
    - "SamsungCNT_Fire"
    - "Soil_Fire"
    - "Hyundai_Fire"
    - "Coupang_Fire"
    - "Kumho_Fire"

  window_size: 15    # Fire 권장값: 15 (Falldown은 30)

  prompt_templates:
    fire: |
      Analyze this image carefully. Only classify it as "fire" if there are ACTUAL FLAMES ...
      (위 Fire 프롬프트 전문 사용)

    default: "Is {category} visible in this image? Answer only with 0 (no) or 1 (yes)."

# submit 섹션: 모델명 변경
submit:
  model_name: "InternVL3-2B_hyundai_8_20"
```

---

## 모델 교체 시 변경 필드 요약

새 모델로 교체할 때는 아래 필드만 수정하면 됩니다.

| 필드 | 예시 | 설명 |
|------|------|------|
| `pipeline.name` | `"MyModel Falldown Benchmark (LMDeploy)"` | 식별 이름 |
| `docker.container_name` | `"mymodel_lmdeploy"` | 고유한 컨테이너명 |
| `docker.model_path` | `"ckpts/MyModel"` | 로컬 모델 경로 |
| `docker.hf_repo_id` | `"org/MyModel"` | (선택) HF repo ID |
| `evaluate.run_name` | `"MyModel"` | 결과 디렉토리명 |
| `submit.model_name` | `"MyModel"` | 리더보드 표시명 |
