# vLLM Serving

vLLM 기반 멀티모달 추론 서버를 Docker로 배포하고, Python 비동기 클라이언트로 요청을 보내기 위한 패키지.

**vLLM이란?** — 대규모 언어 모델(LLM)을 고속으로 서빙하는 오픈소스 추론 엔진. OpenAI API와 호환되는 HTTP 서버를 제공하여, 이미지+텍스트를 보내면 모델이 분석 결과를 반환한다. 별도 ML 프레임워크 설치 없이 Docker만으로 구동 가능.

---

## 목차

1. [사전 요구사항](#사전-요구사항)
2. [빠른 시작](#빠른-시작)
3. [구조 개요](#구조-개요)
4. [Docker 서버 설정](#docker-서버-설정)
5. [GPU 환경별 세팅](#gpu-환경별-세팅)
6. [API 구조 (HTTP 통신)](#api-구조-http-통신)
7. [Python 클라이언트 사용법](#python-클라이언트-사용법)
8. [다른 프로젝트에서 사용하기](#다른-프로젝트에서-사용하기)
9. [응답 파싱](#응답-파싱)
10. [에러 핸들링](#에러-핸들링)
11. [설정 레퍼런스](#설정-레퍼런스)
12. [트러블슈팅](#트러블슈팅)

---

## 사전 요구사항

서버를 올릴 머신에 아래가 설치되어 있어야 한다:

| 항목 | 확인 명령 | 설치 안내 |
|---|---|---|
| Docker Engine | `docker --version` | https://docs.docker.com/engine/install/ |
| Docker Compose V2 | `docker compose version` | Docker Engine에 포함 |
| NVIDIA 드라이버 | `nvidia-smi` | GPU에 맞는 드라이버 설치 |
| NVIDIA Container Toolkit | `nvidia-ctk --version` | https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html |
| 디스크 여유 공간 | `df -h` | 모델 크기 + Docker 이미지 (~10GB) 이상 필요 (아래 참고) |

**모델별 디스크 사용량 (HuggingFace 다운로드 기준):**

| 모델 | 대략적 크기 |
|---|---|
| Qwen3.5-0.8B | ~2 GB |
| Qwen3.5-2B | ~5 GB |
| Qwen2.5-VL-7B-Instruct | ~15 GB |
| Qwen3.5-9B | ~20 GB |

모델은 컨테이너 최초 시작 시 HuggingFace에서 **자동 다운로드**된다. `~/.cache/huggingface/`에 캐시되므로 재시작해도 다시 받지 않는다.

클라이언트 측 (요청을 보내는 쪽):
- Python 3.10 이상
- `httpx` 라이브러리 (`pip install httpx`)

---

## 빠른 시작

```bash
# 1. 환경 설정
cp .env.example .env
vi .env  # MODEL_NAME, TENSOR_PARALLEL_SIZE 등 수정

# 2. 서버 실행
docker compose up -d

# 3. 서버 준비 확인
#    - 최초 실행: 모델 다운로드(수~수십 분) + 로딩(1~5분)
#    - 재시작: 로딩만(1~5분)
docker compose logs -f
# "Uvicorn running on ..." 메시지가 뜨면 준비 완료

# 4. 헬스 체크
curl http://localhost:8000/v1/models

# 5. 클라이언트로 추론
pip install httpx
python example.py
```

---

## 구조 개요

```
┌─────────────────────────────────────────────────────┐
│  서버 머신 (GPU 장착)                                │
│  ┌───────────────────────────────────┐              │
│  │  Docker Container (vLLM)          │              │
│  │  - 모델 자동 다운로드/로딩          │              │
│  │  - OpenAI 호환 HTTP API 제공       │              │
│  │  - POST /v1/chat/completions      │              │
│  └──────────────┬────────────────────┘              │
│                 │ :8000                              │
└─────────────────┼───────────────────────────────────┘
                  │ HTTP (JSON)
┌─────────────────┼───────────────────────────────────┐
│  클라이언트 머신 │ (GPU 불필요)                       │
│  ┌──────────────┴────────────────────┐              │
│  │  Python 코드 (client.py)          │              │
│  │  - 이미지 + 프롬프트 전송           │              │
│  │  - 비동기 요청 (httpx)             │              │
│  │  - 최대 3개 동시 요청              │              │
│  └───────────────────────────────────┘              │
└─────────────────────────────────────────────────────┘
```

**서버와 클라이언트는 같은 머신에 있어도 되고, 다른 머신에 있어도 된다.**
다른 머신일 경우 클라이언트의 `api_base`에 서버 IP를 넣으면 된다:

```python
# 같은 머신
VLLMClient(api_base="http://localhost:8000/v1")

# 다른 머신 (서버 IP가 192.168.1.100인 경우)
VLLMClient(api_base="http://192.168.1.100:8000/v1")
```

### 폴더 구조

```
vllm_serving/
├── README.md              # 이 문서
├── Dockerfile             # 커스텀 빌드가 필요할 때 사용
├── docker-compose.yml     # 서버 실행 (기본 방식)
├── .env.example           # 환경변수 템플릿
├── client.py              # 비동기 Python 클라이언트 (다른 프로젝트에 복사하여 사용)
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

서버 상태 확인 / 중지:
```bash
docker compose logs -f    # 실시간 로그
docker compose ps         # 컨테이너 상태
docker compose down       # 서버 중지
docker compose restart    # 서버 재시작
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

사내 레지스트리에 푸시하거나, 추가 패키지를 설치할 때:

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

비공개(gated) 모델을 사용하려면 `.env`에 HuggingFace 토큰을 설정:

```env
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

---

## GPU 환경별 세팅

### Hopper / Ampere (H100, A100, L40S 등)

```env
VLLM_IMAGE=vllm/vllm-openai:latest
TENSOR_PARALLEL_SIZE=2    # GPU 수에 맞게 조정
MAX_MODEL_LEN=8192
GPU_MEMORY_UTILIZATION=0.9
KV_CACHE_MEMORY_BYTES=3G  # 이미지 1장 + 짧은 응답이면 2~3G로 충분
```

### Blackwell (B200 등)

```env
VLLM_IMAGE=vllm/vllm-openai:cu130-nightly
TENSOR_PARALLEL_SIZE=4
MAX_MODEL_LEN=16384
GPU_MEMORY_UTILIZATION=0.9
KV_CACHE_MEMORY_BYTES=3G
```

### 단일 GPU (개발/테스트)

```env
VLLM_IMAGE=vllm/vllm-openai:latest
TENSOR_PARALLEL_SIZE=1
MAX_MODEL_LEN=4096
GPU_MEMORY_UTILIZATION=0.85
KV_CACHE_MEMORY_BYTES=2G
MAX_NUM_SEQS=32
```

### KV Cache 메모리 설정 가이드

`KV_CACHE_MEMORY_BYTES`는 vLLM이 KV Cache에 할당할 GPU 메모리 상한이다.

- **미설정 (기본)**: vLLM이 `GPU_MEMORY_UTILIZATION` 기반으로 자동 계산 (GPU 여유 메모리를 최대한 사용)
- **명시적 설정**: 모델 로딩 후 남는 메모리에서 KV Cache에 사용할 양을 직접 지정

이 패키지의 기본 용도(이미지 1장 + 짧은 응답)에서는 KV Cache가 크게 필요하지 않다.
**2~3G면 충분**하고, GPU를 다른 프로세스와 공유하는 환경이면 명시적으로 제한하는 것이 안전하다.

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

## API 구조 (HTTP 통신)

vLLM 서버는 **OpenAI API 호환** 형식을 사용한다. 별도 SDK 없이 HTTP POST 요청만으로 통신 가능.
Python `client.py`를 쓰지 않아도, 어떤 언어든 HTTP 요청을 보낼 수 있다.

### 엔드포인트

| 메서드 | 경로 | 용도 |
|---|---|---|
| GET | `/v1/models` | 로드된 모델 목록 확인 (헬스 체크용) |
| POST | `/v1/chat/completions` | 채팅 추론 요청 (이미지+텍스트) |

### 요청 형식

`POST http://{서버IP}:{포트}/v1/chat/completions` 에 JSON body를 전송한다:

```json
{
  "model": "Qwen/Qwen2.5-VL-7B-Instruct",
  "messages": [
    {
      "role": "user",
      "content": [
        {
          "type": "image_url",
          "image_url": {
            "url": "data:image/jpeg;base64,{BASE64_ENCODED_IMAGE}"
          }
        },
        {
          "type": "text",
          "text": "이 이미지에 불이 있나요?"
        }
      ]
    }
  ],
  "max_tokens": 256,
  "temperature": 0.0,
  "seed": 0
}
```

**중요: `model` 필드는 `.env`의 `MODEL_NAME`과 정확히 같아야 한다.** 다르면 404 에러가 발생한다.

### 응답 형식

```json
{
  "id": "cmpl-xxx",
  "object": "chat.completion",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "{\"category\": \"fire\", \"description\": \"visible flames\"}"
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 1234,
    "completion_tokens": 15,
    "total_tokens": 1249
  }
}
```

실제 모델 응답은 `choices[0].message.content` 에 문자열로 들어온다.

### curl로 직접 요청 보내기

```bash
# 1. 이미지를 base64로 인코딩
IMAGE_B64=$(base64 -w 0 test_image.jpg)

# 2. 추론 요청
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-VL-7B-Instruct",
    "messages": [
      {
        "role": "user",
        "content": [
          {
            "type": "image_url",
            "image_url": {"url": "data:image/jpeg;base64,'${IMAGE_B64}'"}
          },
          {
            "type": "text",
            "text": "이 이미지를 설명하세요."
          }
        ]
      }
    ],
    "max_tokens": 256,
    "temperature": 0.0
  }'
```

### 텍스트만 보내기 (이미지 없이)

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-VL-7B-Instruct",
    "messages": [
      {"role": "user", "content": "안녕하세요?"}
    ],
    "max_tokens": 128
  }'
```

### Thinking 모드 제어 (Qwen3.5)

요청 단위로 thinking 모드를 끄려면 `chat_template_kwargs`를 추가:

```json
{
  "model": "Qwen/Qwen3.5-2B",
  "messages": [...],
  "chat_template_kwargs": {"enable_thinking": false}
}
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
        model="Qwen/Qwen2.5-VL-7B-Instruct",  # .env의 MODEL_NAME과 동일해야 함
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
| `model` | `Qwen/Qwen2.5-VL-7B-Instruct` | 모델명 (**서버의 MODEL_NAME과 일치 필수**) |
| `max_concurrency` | `3` | 최대 동시 요청 수 |
| `max_tokens` | `256` | 응답 최대 토큰 |
| `temperature` | `0.0` | 샘플링 온도 (0.0 = 동일 입력에 항상 같은 결과) |
| `timeout` | `120.0` | 요청 타임아웃(초) |

---

## 다른 프로젝트에서 사용하기

이 패키지의 `client.py`는 단일 파일로, 다른 프로젝트에 **복사해서 사용**하는 방식이다.

### 설정 방법

```bash
# 1. 본인 프로젝트에 client.py 복사
cp vllm_serving/client.py  /path/to/your/project/

# 2. 의존성 설치
pip install httpx
```

### 프로젝트에서 import

```python
# your_project/main.py
from client import VLLMClient   # client.py가 같은 디렉토리에 있을 때

# 또는 하위 디렉토리에 넣었을 때
from utils.client import VLLMClient
```

### 전체 통합 예시

실제 프로젝트에서 사용하는 전형적인 패턴:

```python
import asyncio
from pathlib import Path
from client import VLLMClient


VLLM_SERVER = "http://192.168.1.100:8000/v1"  # vLLM 서버 주소
MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"         # 서버의 MODEL_NAME과 동일


async def analyze_images(image_paths: list[Path]) -> list[dict]:
    """이미지 리스트를 분석하여 결과를 반환."""
    async with VLLMClient(
        api_base=VLLM_SERVER,
        model=MODEL,
        max_concurrency=3,
    ) as client:
        # 서버 상태 확인
        if not await client.health_check():
            raise ConnectionError(f"vLLM 서버에 연결할 수 없음: {VLLM_SERVER}")

        items = [{"image": str(p)} for p in image_paths]
        prompt = '{"category": "fire" 또는 "normal"}로 답하세요.'
        results = await client.infer_batch(items, prompt=prompt)

        # 결과 파싱
        analyzed = []
        for r in results:
            if r["error"]:
                analyzed.append({"file": str(image_paths[r["index"]]), "error": r["error"]})
            else:
                parsed = VLLMClient.parse_json_response(r["response"], ["fire", "normal"])
                analyzed.append({
                    "file": str(image_paths[r["index"]]),
                    "category": parsed["category"] if parsed else "unknown",
                    "raw": r["response"],
                })
        return analyzed


# 사용
if __name__ == "__main__":
    images = list(Path("./data").glob("*.jpg"))
    results = asyncio.run(analyze_images(images))
    for r in results:
        print(r)
```

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

## 에러 핸들링

### 요청 실패 시

`client.infer()`는 서버 에러 시 `httpx.HTTPStatusError`를 발생시킨다:

```python
import httpx

try:
    response = await client.infer("image.jpg", "분석하세요")
except httpx.HTTPStatusError as e:
    print(f"서버 에러: {e.response.status_code}")
    print(f"상세: {e.response.text}")
except httpx.ConnectError:
    print("서버에 연결할 수 없음 — 서버가 실행 중인지 확인")
except httpx.ReadTimeout:
    print("응답 타임아웃 — timeout 값을 늘리거나 max_tokens를 줄이기")
```

### 배치에서의 에러

`infer_batch()`는 개별 요청 에러를 잡아서 결과에 포함한다 (전체가 실패하지 않음):

```python
results = await client.infer_batch(items, prompt="...")
for r in results:
    if r["error"]:
        print(f"[{r['index']}] 실패: {r['error']}")
    else:
        print(f"[{r['index']}] 성공: {r['response']}")
```

### 자주 발생하는 에러

| 에러 | 원인 | 해결 |
|---|---|---|
| `404 Not Found` | model 파라미터가 서버의 MODEL_NAME과 불일치 | 클라이언트의 `model=`을 `.env`의 `MODEL_NAME`과 동일하게 맞추기 |
| `ConnectError` | 서버가 아직 준비 안 됨 / IP 잘못됨 | `wait_until_ready()` 사용, 또는 서버 IP 확인 |
| `ReadTimeout` | 모델 응답이 느림 (큰 max_tokens 등) | `timeout=` 값 늘리기 |
| `500 Internal Server Error` | 서버 측 GPU OOM 등 | `docker compose logs`로 원인 확인 |

---

## 설정 레퍼런스

### .env 환경변수 전체 목록

| 변수명 | 기본값 | 설명 |
|---|---|---|
| `MODEL_NAME` | `Qwen/Qwen2.5-VL-7B-Instruct` | HuggingFace 모델 ID |
| `VLLM_IMAGE` | `vllm/vllm-openai:latest` | Docker 이미지 (GPU 아키텍처에 맞게) |
| `NVIDIA_VISIBLE_DEVICES` | `all` | 사용할 GPU (`0,1` 또는 `all`) |
| `TENSOR_PARALLEL_SIZE` | `1` | 텐서 병렬 GPU 수 (사용할 GPU 장수) |
| `HOST_PORT` | `8000` | 호스트에서 접근할 포트 |
| `MAX_MODEL_LEN` | `8192` | 최대 컨텍스트 길이 (토큰) |
| `MAX_NUM_SEQS` | `128` | 서버가 동시에 처리할 수 있는 최대 시퀀스 수 |
| `MAX_NUM_BATCHED_TOKENS` | `131072` | 한 번에 배치 처리할 최대 토큰 수 |
| `GPU_MEMORY_UTILIZATION` | `0.9` | GPU 메모리 중 vLLM이 사용할 비율 (0.0~1.0) |
| `KV_CACHE_MEMORY_BYTES` | (자동) | KV Cache 메모리 상한. 미설정 시 자동 계산. 이미지 1장 용도면 `2G`~`3G` 권장 |
| `HF_TOKEN` | (없음) | HuggingFace 토큰 (비공개 모델 다운로드 시 필요) |
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
# 로그 확인 (에러 메시지가 여기에 나옴)
docker compose logs -f

# 컨테이너 상태 확인
docker compose ps
```

### CUDA out of memory

GPU 메모리가 부족한 경우 (우선순위 순):
1. `KV_CACHE_MEMORY_BYTES`를 명시적으로 제한 (예: `2G`)
2. `GPU_MEMORY_UTILIZATION`을 낮추기 (예: `0.85`)
3. `MAX_MODEL_LEN` 줄이기
4. `TENSOR_PARALLEL_SIZE` 늘리기 (더 많은 GPU 사용)
5. `MAX_NUM_SEQS` 줄이기

### 포트 충돌

```bash
# 사용 중인 포트 확인
lsof -i :8000

# .env에서 HOST_PORT 변경
HOST_PORT=8001
```

### 모델 다운로드 실패 / 느림

```bash
# HuggingFace 캐시 권한 확인
ls -la ~/.cache/huggingface/

# 수동 다운로드 (네트워크 문제 시)
pip install huggingface_hub
huggingface-cli download Qwen/Qwen2.5-VL-7B-Instruct

# 비공개 모델이면 토큰 필요
# .env에 HF_TOKEN=hf_xxx 추가
```

### 404 Not Found 에러

model 파라미터가 서버에 로드된 모델명과 다를 때 발생:

```bash
# 서버에 로드된 모델명 확인
curl http://localhost:8000/v1/models
# 응답의 "id" 필드가 클라이언트에서 써야 할 model 값
```

### 다른 머신에서 연결이 안 됨

```bash
# 서버 머신에서 방화벽 확인
sudo ufw status
# 포트 개방 (필요 시)
sudo ufw allow 8000/tcp

# 서버 IP 확인
hostname -I
```

### 컨테이너 GPU 인식 안됨

```bash
# NVIDIA 드라이버 확인
nvidia-smi

# Docker NVIDIA runtime 확인
docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi

# 위 명령이 실패하면 NVIDIA Container Toolkit 설치 필요
# https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html
```
