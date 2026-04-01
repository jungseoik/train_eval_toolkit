# Task 19: HuggingFace 모델 다운로드 유효성 검증 강화

## 배경

외부에서 API로 평가 요청 시, HuggingFace에 불완전하게 업로드된 모델(config.json 또는 가중치 파일 누락)을 다운로드한 후에도 파이프라인이 계속 진행되어 Docker 컨테이너 기동 → 300초 타임아웃까지 낭비되는 문제 발생.

기존 코드는 `config.json` 누락을 감지하고도 경고(`print`)만 출력하고 진행시켜, 상대방은 5분 뒤에 모호한 타임아웃 에러만 받게 됨.

## 수정 내용

**파일**: `src/lmdeploy_pipeline/model_downloader.py`

### 1. `_is_valid_model_dir()` 검증 강화
- 기존: `config.json` 존재 여부만 확인
- 변경: `config.json` + 가중치 파일(`.safetensors`, `.bin`) 존재 여부 모두 확인

### 2. 다운로드 후 경고 → 에러 차단
- 기존: `config.json` 없으면 `print` 경고 후 계속 진행
- 변경: `_validate_model_dir()` 함수로 분리, 누락 시 `RuntimeError` 발생하여 Docker 기동 전 차단

### 3. SSE 에러 메시지 구체화
- 누락 파일 종류(config.json / 가중치), HuggingFace 저장소 URL을 에러 메시지에 포함
- `pipeline_worker.py`의 기존 `try/except (FileNotFoundError, RuntimeError)` 에서 자동으로 SSE 에러 이벤트로 전달됨

## 영향 범위

- API 서버(`src/api/pipeline_worker.py`): 기존 에러 핸들링 구조에서 자연스럽게 처리됨 (수정 불필요)
- CLI(`python -m src.lmdeploy_pipeline`): 동일하게 `ensure_model()` 호출하므로 동일하게 차단됨
- 기존 정상 모델: config.json + 가중치 모두 있으면 기존과 동일하게 통과

## 작성자

jungseoik
