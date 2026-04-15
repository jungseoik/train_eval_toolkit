# vLLM Serving

vLLM 기반 멀티모달 추론 서버를 Docker로 배포하고, Python 비동기 클라이언트로 요청을 보내기 위한 패키지.

---

## 목차

1. [빠른 시작](#빠른-시작)
2. [폴더 구조](#폴더-구조)
3. [Docker 서버 설정](#docker-서버-설정)
4. [GPU 환경별 세팅](#gpu-환경별-세팅)
5. [Python 클라이언트 사용법](#python-클라이언트-사용법)
6. [응답 파싱](#응답-파싱)
7. [설정 레퍼런스](#설정-레퍼런스)
8. [트러블슈팅](#트러블슈팅)

---

## 빠른 시작

```bash
# 1. 환경 설정
cp .env.example .env
vi .env  # MODEL_NAME, TENSOR_PARALLEL_SIZE 등 수정

# 2. 서버 실행
docker compose up -d

# 3. 서버 준비 확인 (모델 다운로드 + 로딩, 수 분 소요)
docker compose logs -f
# "Uvicorn running on ..." 메시지가 뜨면 준비 완료

# 4. 헬스 체크
curl http://localhost:8000/v1/models

# 5. 클라이언트로 추론
pip install httpx
python example.py
```

---

## 폴더 구조

```
vllm_serving/
├── README.md              # 이 문서
├── Dockerfile             # 커스텀 빌드가 필요할 때 사용
├── docker-compose.yml     # 서버 실행 (기본 방식)
├── .env.example           # 환경변수 템플릿
├── client.py              # 비동기 Python 클라이언트
├── example.py             # 사용 예시
└── requirements.txt       # 클라이언트 의존성
```

---

## Docker 서버 설정

### 방법 1: docker-compose (권장)

가장 간단한 방법. `.env` 파일로 모든 설정을 관리한다.

```bash
cp .env.example .env
# .env에서 MODEL_NAME, TENSOR_PARALLEL_SIZE 등 수정
docker compose up -d
```

서버 중지:
```bash
docker compose down
```

### 방법 2: docker run (직접 실행)

docker-compose 없이 직접 실행할 때:

```bash
docker run -d \
  --name vllm-server \
  --gpus all \
  --ipc host \
  -p 8000:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  vllm/vllm-openai:latest \
  --model Qwen/Qwen2.5-VL-7B-Instruct \
  --tensor-parallel-size 2 \
  --max-model-len 8192 \
  --max-num-seqs 128 \
  --limit-mm-per-prompt '{"image": 1}' \
  --seed 0
```

### 방법 3: Dockerfile (커스텀 빌드)

기본 이미지에 추가 설정이 필요하거나, 사내 레지스트리에 푸시할 때:

```bash
# Hopper / Ampere GPU
docker build -t my-vllm-server .

# Blackwell GPU
docker build -t my-vllm-server --build-arg BASE_IMAGE=vllm/vllm-openai:cu130-nightly .

# 실행
docker run -d --name vllm-server --gpus all --ipc host \
  -p 8000:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -e MODEL_NAME=Qwen/Qwen2.5-VL-7B-Instruct \
  -e TENSOR_PARALLEL_SIZE=2 \
  my-vllm-server
```

### 모델 자동 다운로드

vLLM은 컨테이너 시작 시 HuggingFace에서 모델을 자동 다운로드한다.
`~/.cache/huggingface` 볼륨을 마운트하면 호스트에 캐시가 저장되므로 재시작 시 다시 다운로드하지 않는다.

```
# 비공개 모델의 경우 HuggingFace 토큰 설정
docker compose exec vllm-server bash -c "huggingface-cli login --token <YOUR_TOKEN>"

# 또는 환경변수로 전달
# docker-compose.yml의 environment에 추가:
#   - HF_TOKEN=hf_xxxxx
```

---

## GPU 환경별 세팅

### Hopper / Ampere (H100, A100, L40S 등)

```env
VLLM_IMAGE=vllm/vllm-openai:latest
TENSOR_PARALLEL_SIZE=2    # GPU 수에 맞게 조정
MAX_MODEL_LEN=8192
GPU_MEMORY_UTILIZATION=0.9
```

### Blackwell (B200 등)

```env
VLLM_IMAGE=vllm/vllm-openai:cu130-nightly
TENSOR_PARALLEL_SIZE=4
MAX_MODEL_LEN=16384
GPU_MEMORY_UTILIZATION=0.9
```

### 단일 GPU (개발/테스트)

```env
VLLM_IMAGE=vllm/vllm-openai:latest
TENSOR_PARALLEL_SIZE=1
MAX_MODEL_LEN=4096
GPU_MEMORY_UTILIZATION=0.85
MAX_NUM_SEQS=32
```

### GPU 지정 (특정 GPU만 사용)

```env
# GPU 0, 1번만 사용
NVIDIA_VISIBLE_DEVICES=0,1
TENSOR_PARALLEL_SIZE=2
```

### Thinking 모드 비활성화 (Qwen3.5 등)

Qwen3.5 모델에서 thinking 모드를 끄려면 `EXTRA_VLLM_ARGS`에 추가:

```env
EXTRA_VLLM_ARGS=--default-chat-template-kwargs '{"enable_thinking": false}'
```

---

## Python 클라이언트 사용법

### 설치

```bash
pip install httpx
```

### 기본 사용법

```python
import asyncio
from client import VLLMClient

async def main():
    async with VLLMClient(
        api_base="http://서버IP:8000/v1",
        model="Qwen/Qwen2.5-VL-7B-Instruct",
        max_concurrency=3,  # 최대 동시 요청 3개
    ) as client:
        # 서버 준비 확인
        ready = await client.wait_until_ready(timeout=300)
        if not ready:
            raise RuntimeError("서버가 준비되지 않음")

        # 단일 추론
        response = await client.infer("image.jpg", "이 이미지에 무엇이 있나요?")
        print(response)

asyncio.run(main())
```

### 배치 추론

```python
async def batch_example():
    async with VLLMClient(max_concurrency=3) as client:
        items = [
            {"image": "img1.jpg"},
            {"image": "img2.jpg"},
            {"image": "img3.jpg"},
            {"image": "img4.jpg"},
            {"image": "img5.jpg"},
        ]
        # 최대 3개씩 동시 실행 (Semaphore 제한)
        results = await client.infer_batch(items, prompt="불이 보이나요? yes/no")

        for r in results:
            print(f"[{r['index']}] {r['response']} (error={r['error']})")
```

### bytes로 이미지 전달

파일 경로 대신 메모리에 있는 bytes를 직접 전달할 수 있다:

```python
image_bytes = open("image.jpg", "rb").read()
response = await client.infer(image=image_bytes, prompt="분석하세요")
```

### VLLMClient 파라미터

| 파라미터 | 기본값 | 설명 |
|---|---|---|
| `api_base` | `http://localhost:8000/v1` | vLLM 서버 주소 |
| `model` | `Qwen/Qwen2.5-VL-7B-Instruct` | 모델명 |
| `max_concurrency` | `3` | 최대 동시 요청 수 |
| `max_tokens` | `256` | 응답 최대 토큰 |
| `temperature` | `0.0` | 샘플링 온도 |
| `timeout` | `120.0` | 요청 타임아웃(초) |

---

## 응답 파싱

모델 응답은 raw 텍스트로 반환된다. 프롬프트에 따라 파싱 방식이 달라진다.

### JSON 파싱

프롬프트에서 JSON 형식 응답을 요청한 경우:

```python
# 프롬프트 예시
prompt = """이 이미지를 분석하세요.
JSON 형식으로 답하세요: {"category": "fire" 또는 "normal", "description": "이유"}"""

response = await client.infer("image.jpg", prompt)
# response = '{"category": "fire", "description": "visible flames"}'

# 파싱
parsed = VLLMClient.parse_json_response(response, valid_values=["fire", "normal"])
if parsed:
    category = parsed["category"]     # "fire"
    description = parsed.get("description", "")
```

`parse_json_response`는 아래 형태를 모두 처리한다:
- 순수 JSON: `{"category": "fire", "description": "..."}`
- 마크다운 코드블록: `` ```json {...} ``` ``
- JSON 앞뒤에 텍스트가 붙은 경우

### yes/no 파싱

간단한 이진 분류:

```python
prompt = "이 이미지에 연기가 보이나요? yes 또는 no로만 답하세요."

response = await client.infer("image.jpg", prompt)
detected = VLLMClient.parse_yes_no(response)  # True 또는 False
```

### 직접 파싱

위 유틸리티가 맞지 않으면 raw 텍스트를 직접 처리:

```python
response = await client.infer("image.jpg", "이 이미지를 설명하세요.")
# response는 str이므로 원하는 대로 가공
```

---

## 설정 레퍼런스

### .env 환경변수 전체 목록

| 변수명 | 기본값 | 설명 |
|---|---|---|
| `MODEL_NAME` | `Qwen/Qwen2.5-VL-7B-Instruct` | HuggingFace 모델 ID |
| `VLLM_IMAGE` | `vllm/vllm-openai:latest` | Docker 이미지 |
| `NVIDIA_VISIBLE_DEVICES` | `all` | 사용할 GPU |
| `TENSOR_PARALLEL_SIZE` | `1` | 텐서 병렬 GPU 수 |
| `HOST_PORT` | `8000` | 호스트 포트 |
| `MAX_MODEL_LEN` | `8192` | 최대 컨텍스트 길이 |
| `MAX_NUM_SEQS` | `128` | 최대 동시 시퀀스 수 |
| `MAX_NUM_BATCHED_TOKENS` | `131072` | 배치 토큰 상한 |
| `GPU_MEMORY_UTILIZATION` | `0.9` | GPU 메모리 사용률 |
| `EXTRA_VLLM_ARGS` | (없음) | 추가 vLLM CLI 인자 |

### 모델별 권장 설정

| 모델 | TENSOR_PARALLEL_SIZE | MAX_MODEL_LEN | 비고 |
|---|---|---|---|
| Qwen2.5-VL-7B-Instruct | 1 | 8192 | 단일 GPU 가능 |
| Qwen3.5-0.8B | 1 | 8192 | 경량 모델 |
| Qwen3.5-2B | 1 | 8192 | - |
| Qwen3.5-9B | 2 | 8192 | - |
| Qwen3.5-397B-A17B | 8 | 16384 | MoE, 최소 8x H100 |

---

## 트러블슈팅

### 서버가 시작되지 않음

```bash
# 로그 확인
docker compose logs -f

# 컨테이너 상태 확인
docker compose ps
```

### CUDA out of memory

GPU 메모리가 부족한 경우:
- `GPU_MEMORY_UTILIZATION`을 낮추기 (예: 0.85)
- `MAX_MODEL_LEN` 줄이기
- `TENSOR_PARALLEL_SIZE` 늘리기 (더 많은 GPU 사용)
- `MAX_NUM_SEQS` 줄이기

### 포트 충돌

```bash
# 사용 중인 포트 확인
lsof -i :8000

# .env에서 HOST_PORT 변경
HOST_PORT=8001
```

### 모델 다운로드 실패

```bash
# HuggingFace 캐시 권한 확인
ls -la ~/.cache/huggingface/

# 수동 다운로드
pip install huggingface_hub
huggingface-cli download Qwen/Qwen2.5-VL-7B-Instruct
```

### 클라이언트 연결 실패

```python
# 서버 주소가 맞는지 확인
# docker compose 실행 호스트의 IP를 사용
VLLMClient(api_base="http://192.168.1.100:8000/v1")

# localhost가 아닌 다른 머신에서 접근할 때는 서버 IP 사용
```

### 컨테이너 GPU 인식 안됨

```bash
# NVIDIA 드라이버 확인
nvidia-smi

# Docker NVIDIA runtime 확인
docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi
```
