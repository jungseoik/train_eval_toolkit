"""
LMDeploy 평가 파이프라인 CLI.

파인튜닝 완료된 InternVL3 계열 로컬 모델의 최종 벤치마크 평가 파이프라인.
(테스트셋 평가와는 별개 프로세스)

사용법:
    python -m src.lmdeploy_pipeline -c configs/lmdeploy_pipeline/internvl3_2b_fire.yaml
    python -m src.lmdeploy_pipeline -c configs/lmdeploy_pipeline/internvl3_2b_fire.yaml --steps evaluate submit
"""

import argparse

from src.lmdeploy_pipeline.runner import run_pipeline


def main():
    parser = argparse.ArgumentParser(description="LMDeploy 벤치마크 평가 파이프라인")
    parser.add_argument("-c", "--config", required=True, help="YAML 설정 파일 경로")
    parser.add_argument(
        "--steps", nargs="*", choices=["docker", "evaluate", "submit"],
        help="실행할 단계 (미지정 시 YAML의 steps 사용)",
    )
    args = parser.parse_args()
    run_pipeline(args.config, override_steps=args.steps)


if __name__ == "__main__":
    main()
