# _autolabeling

Gemini API를 활용하여 이미지/비디오에 자동으로 JSON 라벨을 생성하는 모듈.

## 모듈 구조

```
src/_autolabeling/
├── labeler.py          - 메인 라벨링 오케스트레이터
├── prompt_loader.py    - YAML 프롬프트 로더
├── translator.py       - JSON description 영→한 번역
└── gemini/
    ├── client.py       - GeminiClient (analyze_image / analyze_video)
    └── translate_client.py - 번역 전용 Gemini 클라이언트

configs/
└── prompts/            - 도메인별 프롬프트 YAML 파일
    ├── base.yaml       - basic, normal, vio, vio_timestamp
    ├── aihub_space.yaml
    ├── gangnam.yaml
    ├── hyundai.yaml
    ├── cctv.yaml
    ├── scvd.yaml
    └── gj.yaml
```

## 프롬프트 관리

프롬프트는 `configs/prompts/*.yaml` 파일로 관리됩니다. 코드 변경 없이 YAML 파일만 수정하면 프롬프트를 변경할 수 있습니다.

### YAML 키 컨벤션

```yaml
prompts:
  {option}__{mode}: |
    프롬프트 텍스트...
```

`{option}__{mode}` 형태로, 더블 언더스코어(`__`)로 구분합니다.

### 새 프롬프트 추가

1. `configs/prompts/` 에 YAML 파일 생성 또는 기존 파일에 키 추가
2. CLI에서 `--options` 인자에 해당 option 이름 사용
3. 코드 변경 불필요

## 사전 요구사항

환경변수 설정이 필요합니다:

```bash
export GEMINI_API_KEY="your-api-key-here"
```

## 빠른 시작

Gangnam 데이터 기준 예시입니다. 데이터 다운로드 및 폴더 구조는 [docs/data/download/guideline_gangnam.md](../data/download/guideline_gangnam.md)를 참조하세요.

```bash
# 비디오 자동 라벨링 (Gangnam 기준)
python main.py autolabel \
  -i data/processed/gangnam/gaepo1_v2/Train/video/violence/violence/clip \
  -opt vio -n 16 -m video

# 이미지 자동 라벨링 (Gangnam 기준)
python main.py autolabel \
  -i data/processed/gangnam/yeoksam2_v2/Train/video/falldown \
  -opt gangnam -n 8 -m image

# 모델 지정 (기본: gemini-3-pro-preview)
python main.py autolabel \
  -i data/processed/gangnam/gaepo1_v2/Train/video/violence/violence/clip \
  -opt vio -n 16 -m video --model gemini-3-flash-preview

# 기존 라벨 강제 덮어쓰기 (기본: 스킵)
python main.py autolabel \
  -i data/processed/gangnam/gaepo1_v2/Train/video/violence/violence/clip \
  -opt vio -n 16 -m video --overwrite

# JSON description 번역 (영→한, 기본: gemini-2.5-flash)
python main.py translate \
  -i data/processed/gangnam/gaepo1_v2/Train/video/violence/violence/clip \
  -n 16

# 번역 모델 지정
python main.py translate \
  -i data/processed/gangnam/gaepo1_v2/Train/video/violence/violence/clip \
  -n 16 --model gemini-3-flash-preview
```

## 지원 options 목록

| options          | mode        | 설명                       |
|------------------|-------------|----------------------------|
| basic            | video       | 기본 폭력/정상 분류        |
| vio              | video       | 폭력 영상 라벨링           |
| normal           | video       | 정상 영상 라벨링           |
| vio_timestamp    | video       | 폭력 타임스탬프 추출       |
| aihub_space      | video       | AI Hub 공간 특화           |
| gj_normal        | video       | GJ 정상                    |
| gj_violence      | video       | GJ 폭력                    |
| cctv_normal      | video       | CCTV 정상                  |
| cctv_violence    | video       | CCTV 폭력                  |
| scvd_normal      | video       | SCVD 정상                  |
| scvd_violence    | video       | SCVD 폭력                  |
| gangnam          | video/image | 강남 특화                  |
| hyundai_normal   | image       | 현대 에스컬레이터 정상     |
| hyundai_falldown | image       | 현대 낙상 감지             |

## 모델 설정

CLI `--model` 옵션으로 Gemini 모델을 지정할 수 있습니다. 생략 시 기본값을 사용합니다.

| 커맨드 | 기본 모델 | CLI 오버라이드 |
|--------|-----------|----------------|
| `autolabel` | `gemini-3-pro-preview` | `--model <모델명>` |
| `translate` | `gemini-2.5-flash` | `--model <모델명>` |

## 덮어쓰기 동작

기본적으로 이미 유효한 JSON 라벨이 존재하면 스킵합니다. `--overwrite` 옵션으로 강제 재라벨링할 수 있습니다.

| 기존 JSON 상태 | 기본 동작 | `--overwrite` |
|----------------|-----------|---------------|
| description이 정상 | 스킵 | 덮어쓰기 |
| description이 비어있음 | 재처리 | 덮어쓰기 |
| JSON 파싱 실패 (깨진 파일) | 재처리 | 덮어쓰기 |
| JSON 파일 없음 | 새로 생성 | 새로 생성 |

## 동작 방식

1. API 연결 사전 검증 (API 키, 모델명, 네트워크 확인 — 실패 시 즉시 중단)
2. 입력 폴더 재귀 탐색 → 대상 파일 수집
3. `ProcessPoolExecutor` 병렬 처리
4. 각 파일마다 Gemini API 호출 → JSON 파싱 → `description` 검증 (최대 3회 재시도)
5. JSON 응답 → 원본 파일 위치에 `.json` 저장
6. 실패 목록 → `assets/logs/failed_files_{timestamp}.txt`에 저장

> 에러 처리, 재시도, JSON 파싱 등 동작 상세는 [autolabeling_internals.md](autolabeling_internals.md)를 참조하세요.

## 출력 JSON 형식

```json
{"category": "Violence", "description": "...설명..."}
```

## 번역 기능

```bash
python main.py translate -i <json_dir>
```

→ 각 JSON 파일에 `description_kor` 키를 추가로 저장합니다.

```json
{
  "category": "Violence",
  "description": "A person punches another person.",
  "description_kor": "한 사람이 다른 사람을 주먹으로 가격합니다."
}
```
