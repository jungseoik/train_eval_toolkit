# 9. vLLM Docker Red Image 단일 응답 테스트

## 개요
vLLM Docker 컨테이너를 기동하여 Qwen/Qwen3.5-2B 모델에 224x224 빨간색 이미지를 전송하고 응답을 확인하는 단일 추론 테스트.

## 테스트 환경
- **GPU**: NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition (97,887 MiB)
- **Docker 이미지**: `vllm/vllm-openai:cu130-nightly`
- **모델**: `Qwen/Qwen3.5-2B` (멀티모달 VLM, vision_config 포함)
- **서버 포트**: 8000
- **vLLM 인자**: `--max-model-len 4096 --trust-remote-code`

## 테스트 절차

### 1. 기존 Docker 확인
- 기존 vLLM 컨테이너: 없음 (이전 컨테이너 전부 Exited 상태)
- 포트 8000: 사용 가능

### 2. Docker 기동
```bash
docker run -d \
  --name vllm-red-test \
  --gpus all \
  --ipc host \
  -p 8000:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  vllm/vllm-openai:cu130-nightly \
  --model Qwen/Qwen3.5-2B \
  --max-model-len 4096 \
  --trust-remote-code
```
- 서버 준비 완료까지 약 **105초** 소요

### 3. 테스트 입력
- **이미지**: 224x224 PNG, 전체 빨간색 (RGB: 255, 0, 0), 674 bytes
- **프롬프트**: `"What is this image? Describe it."`
- **max_tokens**: 100
- **temperature**: 0.0 (결정적 생성)

### 4. API 요청
- **엔드포인트**: `POST http://127.0.0.1:8000/v1/chat/completions`
- **페이로드 형식**: OpenAI Chat Completions API 호환 (image_url + text)

## 테스트 결과

### 모델 응답
> This image is a solid, uniform field of pure red. It contains no other elements, shapes, textures, or variations in color — it is entirely filled with the same bright, saturated red throughout the entire frame.
>
> In terms of technical description:
> - **Color**: Pure red (RGB: 255, 0, 0)
> - **Shape**: Square or rectangular (aspect ratio appears to be 1:1)
> - **Texture**: Smooth and flat — no

### 토큰 사용량
| 항목 | 값 |
|---|---|
| prompt_tokens | 87 |
| completion_tokens | 100 |
| total_tokens | 187 |
| finish_reason | length (max_tokens=100 도달) |

### 판정
- **성공**: 모델이 빨간 이미지를 정확히 인식 (색상, RGB 값, 형태, 텍스처까지 정확히 기술)
- vLLM OpenAI-compatible API를 통한 이미지 추론 정상 동작 확인

## Docker 정리
- `docker rm -f vllm-red-test` 실행
- GPU 메모리 정상 반환 확인: 499 MiB (테스트 전과 동일)

## 참고
- `test.py` 코드를 참고하여 OpenAI Chat Completions API 형식으로 요청
- `src/vllm_pipeline/docker_manager.py`의 Docker 기동 패턴 참조
- 모델 캐시는 `~/.cache/huggingface`에서 마운트하여 재다운로드 방지

---
작성자: jungseoik
작성일: 2026-03-23
