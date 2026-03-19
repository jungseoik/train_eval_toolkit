"""
vLLM 평가 파이프라인 CLI.

사용법:
    python -m src.vllm_pipeline.cli -c configs/vllm_pipeline/qwen35_2b_fire.yaml
    python -m src.vllm_pipeline.cli -c configs/vllm_pipeline/qwen35_2b_fire.yaml --steps evaluate submit
"""

import argparse

from src.vllm_pipeline.runner import run_pipeline


def main():
    parser = argparse.ArgumentParser(description="vLLM 평가 파이프라인")
    parser.add_argument("-c", "--config", required=True, help="YAML 설정 파일 경로")
    parser.add_argument(
        "--steps", nargs="*", choices=["docker", "evaluate", "submit"],
        help="실행할 단계 (미지정 시 YAML의 steps 사용)",
    )
    args = parser.parse_args()
    run_pipeline(args.config, override_steps=args.steps)


if __name__ == "__main__":
    main()
