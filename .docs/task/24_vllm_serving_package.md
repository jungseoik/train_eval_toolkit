# TASK 24: vLLM Serving 독립 패키지 구성

## 개요
vLLM 기반 멀티모달 추론 서버를 다른 레포에서 바로 사용할 수 있도록 독립된 `vllm_serving/` 폴더로 패키징.
Docker pull 후 바로 서버를 올리고, Python 비동기 클라이언트로 이미지+프롬프트 요청을 보내는 구조.

## 배경
- 기존 `train_eval_toolkit`의 vLLM 파이프라인은 평가 전용으로 결합되어 있어 다른 프로젝트에서 재사용이 어려움
- 다른 팀/레포에서 vLLM 서버를 빠르게 배포하고 비동기 추론을 수행할 수 있는 독립 패키지 필요

## 작업 내용

### 생성된 파일

| 파일 | 역할 |
|---|---|
| `vllm_serving/Dockerfile` | 커스텀 빌드용. Blackwell(`cu130-nightly`)과 Hopper/Ampere(`latest`) 모두 ARG로 지원 |
| `vllm_serving/docker-compose.yml` | 원클릭 서버 실행. `.env`로 모든 설정 주입. healthcheck 포함 |
| `vllm_serving/.env.example` | 환경변수 템플릿. 모델명, GPU, 포트, vLLM 인자 등 |
| `vllm_serving/client.py` | httpx 기반 비동기 클라이언트. Semaphore(3)으로 동시 요청 제한. JSON/yes-no 파싱 유틸리티 포함 |
| `vllm_serving/example.py` | 단일 추론, 배치 추론, bytes 전달 3가지 사용 예시 |
| `vllm_serving/requirements.txt` | 클라이언트 의존성 (httpx) |
| `vllm_serving/README.md` | 전체 사용 가이드: 빠른 시작, GPU 환경별 세팅, 클라이언트 API, 응답 파싱, 트러블슈팅 |

### 설계 결정

1. **Dockerfile 포함**: 공식 이미지를 그대로 쓸 수 있지만, 사내 레지스트리 푸시나 커스텀 설정이 필요할 때를 대비하여 포함
2. **GPU 환경 분기**: `VLLM_IMAGE` 환경변수로 Blackwell(`cu130-nightly`)과 일반 환경(`latest`)을 선택
3. **모델 자동 다운로드**: vLLM이 HuggingFace에서 자동 다운로드. `~/.cache/huggingface` 볼륨 마운트로 캐시 유지
4. **동시 요청 제한**: `asyncio.Semaphore(3)`으로 최대 배치 3 보장
5. **파싱 로직**: `client.py`에 `parse_json_response()`와 `parse_yes_no()` 유틸리티 내장

## 사용 흐름

```
.env 설정 → docker compose up -d → 모델 자동 다운로드 → 서버 Ready
→ Python client.py로 비동기 요청 (이미지+프롬프트) → 결과 수신 + 파싱
```

## 작성자
jungseoik
