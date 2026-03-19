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
├── config_gemini.py    - Gemini 모델 설정 (model_name, project, location)
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

# JSON description 번역 (영→한)
python main.py translate \
  -i data/processed/gangnam/gaepo1_v2/Train/video/violence/violence/clip \
  -n 16
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

## 동작 방식

1. 입력 폴더 재귀 탐색 → 대상 파일 수집
2. `ProcessPoolExecutor` 병렬 처리
3. Gemini API 호출 (최대 3회 재시도)
4. JSON 응답 → 원본 파일 위치에 `.json` 저장
5. 실패 목록 → `failure_log_dir`에 타임스탬프 파일 저장

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
