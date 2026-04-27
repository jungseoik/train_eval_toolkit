# Cosmos-Reason2 모델 평가 가이드

## 모델 개요

NVIDIA Cosmos-Reason2는 Qwen3-VL 기반의 reasoning VLM(Vision Language Model)이다.
기존 vLLM 파이프라인과 호환되며, 프레임 레벨 이미지 추론 시 base64 전송 방식을 그대로 사용할 수 있다.

- 모델: `nvidia/Cosmos-Reason2-2B` (24GB VRAM), `nvidia/Cosmos-Reason2-8B` (32GB VRAM)
- Gated model: HuggingFace에서 라이선스 동의 + `huggingface-cli login` 필요
- vLLM >= 0.11.0 지원

## Reasoning vs Non-Reasoning 모드

Cosmos-Reason2는 두 가지 모드로 동작한다.

### Non-Reasoning (단순 분류 작업에 권장)

- `reasoning-parser` vLLM 플래그 **제거**
- 모델이 `<think>` 토큰 없이 바로 답변 출력
- 속도가 빠르고 max_tokens만큼만 생성

```yaml
vllm_args:
  # reasoning-parser 없음
  seed: 0
```

### Reasoning (복잡한 추론 작업용)

- `reasoning-parser: qwen3` 활성화
- 매 요청마다 `<think>...</think>` 토큰을 생성한 후 최종 답변 출력
- think 토큰은 max_tokens 제한에 포함되지 않아 **생성 시간이 대폭 증가**
- API 응답에서 `choices[0].message.reasoning_content`에 think 내용 분리

```yaml
vllm_args:
  reasoning-parser: qwen3
  seed: 1234
```

### 주의: 단순 분류에 reasoning-parser를 켜면 안 되는 이유

fire/falldown 같은 이진 분류는 JSON 응답 하나면 충분하다.
reasoning-parser가 켜져 있으면 프레임마다 수십~수백 토큰의 think 과정을 거치므로
**동일한 평가에 수 배의 시간이 소요**된다. 반드시 non-reasoning 모드를 사용할 것.

## vLLM 서버 설정 주의사항

### kv-cache-memory-bytes

| 설정 | 결과 |
|------|------|
| 30G | 안정적 동작 (권장) |
| 70G 이상 | 장시간 구동 시 OOM 크래시 위험 |

Cosmos-Reason2-2B는 모델 자체가 5.7GB 정도지만,
CUDA graph + encoder cache + KV cache가 VRAM을 추가로 소비한다.
kv-cache를 너무 크게 잡으면 여유 메모리가 부족해 장시간 평가 중 컨테이너가 죽을 수 있다.

### max-model-len

권장 범위: 8192 ~ 16384. 프레임 단위 이미지 분류는 8192로도 충분하다.

### 모델 기본 샘플링 파라미터

모델의 `generation_config.json`에 아래 값이 내장되어 있다:

```
temperature: 0.7, top_k: 20, top_p: 0.8
```

API 호출 시 `temperature: 0.0`을 명시하면 greedy decoding으로 오버라이드된다.

## YAML 설정 예시 (Non-Reasoning, 단순 분류)

```yaml
docker:
  model: "nvidia/Cosmos-Reason2-2B"
  vllm_args:
    tensor-parallel-size: 1
    max-model-len: 16384
    kv-cache-memory-bytes: "30G"
    max-num-seqs: 128
    max-num-batched-tokens: 131072
    limit-mm-per-prompt: '{"image": 1}'
    # reasoning-parser: qwen3  # 단순 분류에는 불필요
    seed: 0

evaluate:
  max_tokens: 15
  temperature: 0.0
  seed: 0
```

## 장애 대응

### Docker 크래시 발생 시

파이프라인에 `InferenceAbortError`가 구현되어 있다.
API 요청 실패 시 1회 재시도 후에도 실패하면 즉시 평가를 중단하고
중단 지점(벤치마크/영상/프레임)을 로그에 출력한다.

```
[EVAL] *** ABORTED: PIA_Falldown / video_name.mp4 / frame=150
[EVAL] *** Cause: All connection attempts failed
[EVAL] *** overwrite_results=false로 재실행하면 이 지점부터 재개됩니다
```

`overwrite_results: false`로 동일 YAML을 재실행하면 완료된 CSV는 건너뛰고
중단된 영상부터 자동 재개된다.
