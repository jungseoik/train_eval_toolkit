"""
Gradio 제출 모듈.

평가 결과 CSV를 Gradio API로 제출.
기존 test_submit.py 로직을 함수로 독립 구현.
"""

from __future__ import annotations

import time
from pathlib import Path

from gradio_client import Client, handle_file

from .config import SubmitConfig


def submit_results(
    submit_cfg: SubmitConfig,
    benchmarks: list[str],
    retry_max: int,
    retry_wait: int,
) -> dict:
    """
    평가 결과 CSV를 Gradio API로 제출.

    반환: {"success": [...], "failed": [...]}
    """
    print(f"\n[SUBMIT] Gradio URL  : {submit_cfg.gradio_url}")
    print(f"[SUBMIT] Model       : {submit_cfg.model_name}")
    print(f"[SUBMIT] Benchmarks  : {benchmarks}")

    client = Client(submit_cfg.gradio_url)
    results_base = Path(submit_cfg.results_base_dir)

    success = []
    failed = []

    for idx, bench_name in enumerate(benchmarks, 1):
        csv_folder = results_base / submit_cfg.model_name / bench_name

        print(f"\n[SUBMIT] [{idx}/{len(benchmarks)}] {bench_name}")
        print(f"[SUBMIT] CSV folder: {csv_folder}")

        if not csv_folder.exists():
            print(f"[SUBMIT] SKIP - folder not found: {csv_folder}")
            failed.append({"benchmark": bench_name, "error": "CSV folder not found"})
            continue

        ok = _submit_single_benchmark(
            client, submit_cfg, bench_name, csv_folder, retry_max, retry_wait,
        )

        if ok:
            success.append(bench_name)
        else:
            failed.append({"benchmark": bench_name, "error": "submit failed after retries"})

        if idx < len(benchmarks):
            print(f"[SUBMIT] Waiting {submit_cfg.interval_seconds}s before next submit...")
            time.sleep(submit_cfg.interval_seconds)

    return {"success": success, "failed": failed}


def _submit_single_benchmark(
    client: Client,
    submit_cfg: SubmitConfig,
    benchmark_name: str,
    csv_folder: Path,
    max_attempts: int,
    wait_seconds: int,
) -> bool:
    """단일 벤치마크 제출. retry 포함."""
    csv_paths = sorted(csv_folder.glob("*.csv"))
    if not csv_paths:
        print(f"[SUBMIT] No CSV files in {csv_folder}")
        return False

    print(f"[SUBMIT] CSV count: {len(csv_paths)}")

    for attempt in range(1, max_attempts + 1):
        try:
            if attempt > 1:
                print(f"[SUBMIT] Retry {attempt}/{max_attempts}")

            result = client.predict(
                model_name=submit_cfg.model_name,
                benchmark_name=benchmark_name,
                task_name=submit_cfg.task_name,
                datasets_used=submit_cfg.datasets_used,
                config_file=handle_file(str(submit_cfg.config_file)),
                csv_files=[handle_file(str(p)) for p in csv_paths],
                api_name="/model_submit",
            )

            print(f"[SUBMIT] Result: {result}")
            return True

        except Exception as e:
            print(f"[SUBMIT] Error: {e}")
            if attempt < max_attempts:
                print(f"[SUBMIT] Waiting {wait_seconds}s before retry...")
                time.sleep(wait_seconds)

    print(f"[SUBMIT] FAILED after {max_attempts} attempts: {benchmark_name}")
    return False
