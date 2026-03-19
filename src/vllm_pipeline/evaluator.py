"""
평가 래퍼 모듈.

기존 vllm_bench_eval.py의 함수를 import해서 호출.
기존 코드는 일체 수정하지 않음.
"""

from __future__ import annotations

import time
from types import ModuleType

from src.evaluation.vllm_bench_eval import evaluate_benchmark, load_config

from .config import EvalConfig


def run_evaluation(eval_cfg: EvalConfig, retry_max: int, retry_wait: int) -> dict:
    """
    벤치마크 목록을 순회하며 평가 실행.

    1. load_config()로 기존 config.py 로드
    2. overrides 적용
    3. 벤치마크별 evaluate_benchmark() 호출
    4. 실패 시 retry (벤치마크 단위)

    반환: {"success": [...], "failed": [...]}
    """
    cfg_module = load_config(eval_cfg.eval_config_path)
    _apply_overrides(cfg_module, eval_cfg.overrides)

    benchmarks = eval_cfg.benchmarks or cfg_module.BENCHMARKS

    print(f"\n[EVAL] Config    : {eval_cfg.eval_config_path}")
    print(f"[EVAL] Model     : {getattr(cfg_module, 'MODEL', 'N/A')}")
    print(f"[EVAL] API       : {getattr(cfg_module, 'API_BASE', 'N/A')}")
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
                evaluate_benchmark(bench_name, cfg_module)
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


def _apply_overrides(cfg_module: ModuleType, overrides: dict) -> None:
    """config 모듈의 속성을 런타임에 덮어씀."""
    for key, value in overrides.items():
        setattr(cfg_module, key, value)
