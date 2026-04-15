# =============================================================
# vLLM OpenAI-Compatible Serving Dockerfile
#
# GPU 아키텍처별 베이스 이미지 선택:
#   - Blackwell (B200 등): vllm/vllm-openai:cu130-nightly
#   - Hopper / Ampere (H100, A100 등): vllm/vllm-openai:latest
#
# 빌드 예시:
#   # Blackwell GPU
#   docker build -t vllm-server --build-arg BASE_IMAGE=vllm/vllm-openai:cu130-nightly .
#
#   # Hopper / Ampere GPU (기본값)
#   docker build -t vllm-server .
# =============================================================

ARG BASE_IMAGE=vllm/vllm-openai:latest
FROM ${BASE_IMAGE}

# 기본 환경변수 (docker-compose 또는 docker run -e 로 오버라이드 가능)
ENV MODEL_NAME="Qwen/Qwen3.5-2B"
ENV TENSOR_PARALLEL_SIZE=1
ENV MAX_MODEL_LEN=8192
ENV MAX_NUM_SEQS=128
ENV MAX_NUM_BATCHED_TOKENS=131072
ENV GPU_MEMORY_UTILIZATION=0.9
ENV KV_CACHE_MEMORY_BYTES=""
ENV PORT=8000

# HuggingFace 캐시 디렉토리
ENV HF_HOME=/root/.cache/huggingface

EXPOSE ${PORT}

# 엔트리포인트: 환경변수를 vLLM serve 인자로 변환
# KV_CACHE_MEMORY_BYTES가 설정되어 있으면 --kv-cache-memory-bytes 인자를 추가
ENTRYPOINT ["sh", "-c", "\
    KV_CACHE_ARG=\"\"; \
    if [ -n \"${KV_CACHE_MEMORY_BYTES}\" ]; then \
        KV_CACHE_ARG=\"--kv-cache-memory-bytes ${KV_CACHE_MEMORY_BYTES}\"; \
    fi; \
    exec python -m vllm.entrypoints.openai.api_server \
    --model ${MODEL_NAME} \
    --tensor-parallel-size ${TENSOR_PARALLEL_SIZE} \
    --max-model-len ${MAX_MODEL_LEN} \
    --max-num-seqs ${MAX_NUM_SEQS} \
    --max-num-batched-tokens ${MAX_NUM_BATCHED_TOKENS} \
    --gpu-memory-utilization ${GPU_MEMORY_UTILIZATION} \
    --port ${PORT} \
    --limit-mm-per-prompt '{\"image\": 1}' \
    --seed 0 \
    ${KV_CACHE_ARG} \
    ${EXTRA_VLLM_ARGS} \
"]
