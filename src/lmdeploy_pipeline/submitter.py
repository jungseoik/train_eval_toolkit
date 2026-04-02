"""
Gradio 제출 모듈.

평가 결과 CSV를 Gradio API로 제출.
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


_FAIL_MARKERS = ("실패", "Failed", "failed")


def _is_benchmark_failed(result: str) -> bool:
    """응답 문자열에서 벤치마크 실행 실패 여부를 판별."""
    result_str = str(result)
    return any(marker in result_str for marker in _FAIL_MARKERS)


def _try_api_call(
    client: Client,
    submit_cfg: SubmitConfig,
    benchmark_name: str,
    csv_paths: list[Path],
    max_attempts: int,
    wait_seconds: int,
) -> str | None:
    """API 호출 (네트워크 에러 시 retry). 성공하면 응답 문자열, 실패하면 None."""
    for attempt in range(1, max_attempts + 1):
        try:
            if attempt > 1:
                print(f"[SUBMIT] API retry {attempt}/{max_attempts}")

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
            return str(result)

        except Exception as e:
            print(f"[SUBMIT] Error: {e}")
            if attempt < max_attempts:
                print(f"[SUBMIT] Waiting {wait_seconds}s before retry...")
                time.sleep(wait_seconds)

    return None


def _submit_single_benchmark(
    client: Client,
    submit_cfg: SubmitConfig,
    benchmark_name: str,
    csv_folder: Path,
    max_attempts: int,
    wait_seconds: int,
) -> bool:
    """단일 벤치마크 제출. API retry + 벤치마크 실패 retry 포함."""
    csv_paths = sorted(csv_folder.glob("*.csv"))
    if not csv_paths:
        print(f"[SUBMIT] No CSV files in {csv_folder}")
        return False

    print(f"[SUBMIT] CSV count: {len(csv_paths)}")

    bench_retry = submit_cfg.benchmark_fail_retry
    bench_wait = submit_cfg.benchmark_fail_wait

    for bench_attempt in range(1, bench_retry + 1):
        if bench_attempt > 1:
            print(f"[SUBMIT] ⏳ Benchmark retry {bench_attempt}/{bench_retry} (waiting {bench_wait}s)...")
            time.sleep(bench_wait)

        result = _try_api_call(
            client, submit_cfg, benchmark_name, csv_paths, max_attempts, wait_seconds,
        )

        if result is None:
            print(f"[SUBMIT] FAILED - API call failed after {max_attempts} attempts: {benchmark_name}")
            return False

        if not _is_benchmark_failed(result):
            return True

        print(f"[SUBMIT] ⚠️ Benchmark execution failed in server response")

    print(f"[SUBMIT] FAILED after {bench_retry} benchmark attempts: {benchmark_name}")
    return False
