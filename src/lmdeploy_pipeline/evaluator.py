"""
평가 래퍼 모듈.

lmdeploy_bench_eval.py의 evaluate_benchmark() 함수를 호출.
YAML 설정에서 SimpleNamespace를 생성하여 config 모듈 대신 전달.
"""

from __future__ import annotations

import time
from types import SimpleNamespace

from src.evaluation.lmdeploy_bench_eval import evaluate_benchmark

from .config import EvalConfig


def _build_cfg_namespace(eval_cfg: EvalConfig) -> SimpleNamespace:
    """EvalConfig에서 evaluate_benchmark()가 기대하는 cfg 객체 생성."""
    return SimpleNamespace(
        BENCHMARKS=eval_cfg.benchmarks,
        MODEL=eval_cfg.model,
        RUN_NAME=eval_cfg.run_name,
        API_BASE=eval_cfg.api_base,
        BENCH_BASE_PATH=eval_cfg.bench_base_path,
        OUTPUT_PATH=eval_cfg.output_path,
        WINDOW_SIZE=eval_cfg.window_size,
        CONCURRENCY=eval_cfg.concurrency,
        INTERPOLATION=eval_cfg.interpolation,
        JPEG_QUALITY=eval_cfg.jpeg_quality,
        MAX_TOKENS=eval_cfg.max_tokens,
        TEMPERATURE=eval_cfg.temperature,
        SEED=eval_cfg.seed,
        NEGATIVE_LABEL=eval_cfg.negative_label,
        PROMPT_TEMPLATES=eval_cfg.prompt_templates,
    )


def run_evaluation(eval_cfg: EvalConfig, retry_max: int, retry_wait: int) -> dict:
    """
    벤치마크 목록을 순회하며 평가 실행.

    YAML에서 로드한 EvalConfig를 SimpleNamespace로 변환하여
    evaluate_benchmark()에 전달.

    반환: {"success": [...], "failed": [...]}
    """
    cfg = _build_cfg_namespace(eval_cfg)
    benchmarks = eval_cfg.benchmarks

    print(f"\n[EVAL] Model     : {cfg.MODEL}")
    print(f"[EVAL] API       : {cfg.API_BASE}")
    print(f"[EVAL] Benchmarks: {benchmarks}")

    success = []
    failed = []

    for idx, bench_name in enumerate(benchmarks, 1):
        print(f"\n[EVAL] [{idx}/{len(benchmarks)}] {bench_name}")

        succeeded = False
        last_error = None

        for attempt in range(1, retry_max + 1):
            try:
                if attempt > 1:
                    print(f"[EVAL] Retry {attempt}/{retry_max} for {bench_name}")
                evaluate_benchmark(bench_name, cfg)
                succeeded = True
                break
            except Exception as e:
                last_error = str(e)
                print(f"[EVAL] Error: {e}")
                if attempt < retry_max:
                    print(f"[EVAL] Waiting {retry_wait}s before retry...")
                    time.sleep(retry_wait)

        if succeeded:
            success.append(bench_name)
        else:
            failed.append({"benchmark": bench_name, "error": last_error})
            print(f"[EVAL] FAILED after {retry_max} attempts: {bench_name}")

    return {"success": success, "failed": failed}
