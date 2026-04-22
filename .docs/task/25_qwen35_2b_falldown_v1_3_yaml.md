# Task 25: PIA_AI2team_VQA_falldown_v1.3 vLLM 평가 YAML 추가

## 개요

`PIA-SPACE-LAB/PIA_AI2team_VQA_falldown_v1.3` (Qwen3.5-2B 베이스 파인튜닝 falldown 모델) 을
vLLM 벤치마크 평가 파이프라인에서 실행하기 위한 YAML 설정 파일을 추가했다.

기존 vllm_pipeline의 `qwen35_2b_fire.yaml` (CLS 모드, fire) 과
`cosmos_reason2_2b_falldown.yaml` (JSON 모드, falldown) 패턴을 조합하여
"CLS 모드 + Falldown" 조합의 첫 vllm_pipeline 설정을 제공한다.

## 배경

- 기존 `configs/lmdeploy_pipeline/PIA_AI2team_VQA_falldown_v1.0.yaml` 은 LMDeploy 기반
- 동일 계열 모델(v1.3) 을 vLLM 서버로 서빙하여 벤치마크를 돌리고자 함
- 모델은 `<CLS_FALLDOWN>` 특수 토큰으로 `yes` / `no` 분류 응답을 반환 → CLS 모드 필요

## 변경 사항

### 신규 파일

- `configs/vllm_pipeline/qwen35_2b_falldown_v1_3.yaml`

### 핵심 필드 값

| 섹션 | 필드 | 값 | 근거 |
|------|------|-----|------|
| pipeline | name | `PIA_AI2team_VQA_falldown_v1.3 Benchmark (vLLM)` | 식별용 |
| docker | container_name | `pia_ai2team_vqa_falldown_v1_3_vllm` | 고유 컨테이너명 |
| docker | image | `vllm/vllm-openai:cu130-nightly` | vllm_pipeline 표준 이미지 |
| docker | model | `PIA-SPACE-LAB/PIA_AI2team_VQA_falldown_v1.3` | HF 레포 |
| docker | vllm_args.max-model-len | `8192` | Qwen3.5-2B 베이스 기준 |
| evaluate | model | `PIA-SPACE-LAB/PIA_AI2team_VQA_falldown_v1.3` | docker.model 과 동일 |
| evaluate | run_name | `PIA_AI2team_VQA_falldown_v1.3` | 결과 디렉토리명 |
| evaluate | eval_mode | `cls` | 특수 토큰 분류 모드 |
| evaluate | max_tokens | `1` | yes/no 단일 토큰 응답 |
| evaluate | window_size | `30` | Falldown 기준 프레임 간격 |
| evaluate | benchmarks | Falldown 13개 | ABB, Coupang, DTRO, GangNam, Hyundai, Innodep, KhonKaen, KISA, Kumho, PIA, Soil, UGO, Yonsei |
| evaluate | prompt_templates.falldown | `<CLS_FALLDOWN>` | CLS 모드 특수 토큰 |
| submit | model_name | `PIA_AI2team_VQA_falldown_v1.3` | 리더보드 표시명 |
| submit | datasets_used | `Finetuned` | 파인튜닝 모델 표기 |

## 검증

`src/vllm_pipeline/config.py::load_pipeline_config` 로 YAML 파싱이 정상 완료됨을 확인.

```
name: PIA_AI2team_VQA_falldown_v1.3 Benchmark (vLLM)
docker.model: PIA-SPACE-LAB/PIA_AI2team_VQA_falldown_v1.3
evaluate.eval_mode: cls
evaluate.max_tokens: 1
evaluate.window_size: 30
benchmarks count: 13
falldown prompt: '<CLS_FALLDOWN>'
submit.datasets_used: Finetuned
OK
```

## 실행 방법

```bash
conda activate llm
python -m src.vllm_pipeline -c configs/vllm_pipeline/qwen35_2b_falldown_v1_3.yaml

# 특정 단계만
python -m src.vllm_pipeline -c configs/vllm_pipeline/qwen35_2b_falldown_v1_3.yaml --steps docker
python -m src.vllm_pipeline -c configs/vllm_pipeline/qwen35_2b_falldown_v1_3.yaml --steps evaluate submit
```

## 참고 및 주의사항

- `vllm_pipeline` 은 `lmdeploy_pipeline` 과 달리 `hf_repo_id` 기반 사전 다운로드 필드가 없다.
  vLLM 컨테이너가 기동 시 HF 에서 직접 로드하므로 호스트의 `~/.cache/huggingface` 볼륨 마운트를
  통해 캐시가 공유된다. 모델이 gated repo 라면 호스트에서 `hf auth login` 선행 필요.
- CLS 모드 프롬프트는 `<CLS_FALLDOWN>` 특수 토큰만 전달하므로 `default` 프롬프트에도
  동일 토큰을 지정하여 폴백 시 엉뚱한 프롬프트가 들어가지 않도록 했다.

## 관련 파일

- `configs/vllm_pipeline/qwen35_2b_fire.yaml` - CLS 모드 패턴 참고
- `configs/vllm_pipeline/cosmos_reason2_2b_falldown.yaml` - Falldown 벤치마크 구성 참고
- `configs/lmdeploy_pipeline/PIA_AI2team_VQA_falldown_v1.0.yaml` - 동일 계열 LMDeploy 설정 참고
- `docs/eval/vllm_pipeline.md` - vLLM 파이프라인 사용 가이드
