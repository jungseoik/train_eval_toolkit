"""
평가 래퍼 모듈.

vllm_bench_eval.py의 evaluate_benchmark() 함수를 호출.
YAML 설정에서 SimpleNamespace를 생성하여 config 모듈 대신 전달.

각 벤치마크 평가는 별도 프로세스(subprocess)에서 실행하여
Python 힙 메모리 누적(fragmentation)을 방지한다.
프로세스 종료 시 OS가 메모리를 100% 회수하므로
장시간 평가에서도 메모리가 안정적으로 유지된다.
"""

from __future__ import annotations

import json
import multiprocessing
import os
import tempfile
import threading
import time
from pathlib import Path
from types import SimpleNamespace

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
        OVERWRITE_RESULTS=eval_cfg.overwrite_results,
        EVAL_MODE=eval_cfg.eval_mode,
        ENABLE_THINKING=eval_cfg.enable_thinking,
    )


def _cfg_to_dict(cfg: SimpleNamespace) -> dict:
    """SimpleNamespace를 pickle-safe dict로 변환."""
    return vars(cfg)


def _update_bench_progress(
    state: dict | None, idx: int, status: str, **extra,
) -> None:
    """progress_state의 벤치마크 항목을 업데이트한다."""
    if state is None or state.get("progress") is None:
        return
    progress = state["progress"]
    bench_entry = progress["benchmarks"][idx]
    bench_entry["status"] = status
    bench_entry.update(extra)
    if status == "in_progress":
        progress["current"] = bench_entry["name"]
    elif status in ("completed", "failed", "skipped"):
        progress["completed"] = sum(
            1 for b in progress["benchmarks"] if b["status"] in ("completed", "failed", "skipped")
        )
        if all(b["status"] != "in_progress" for b in progress["benchmarks"]):
            progress["current"] = None


# ============================================================
# Subprocess 기반 벤치마크 실행
# ============================================================

def _benchmark_worker(
    bench_name: str,
    cfg_dict: dict,
    progress_file: str,
    result_file: str,
    retry_max: int,
    retry_wait: int,
) -> None:
    """별도 프로세스에서 실행되는 벤치마크 평가 워커.

    - evaluate_benchmark()를 직접 호출
    - 진행도를 progress_file(JSON)에 기록
    - 결과를 result_file(JSON)에 기록
    - 프로세스 종료 시 OS가 메모리를 전부 회수
    """
    from types import SimpleNamespace
    from src.evaluation.vllm_bench_eval import BenchmarkSkipError, evaluate_benchmark

    cfg = SimpleNamespace(**cfg_dict)

    # 파일 기반 progress_state 구성
    progress_state = {
        "_progress_file": progress_file,
        "progress": {
            "benchmarks": [{"name": bench_name, "status": "in_progress"}],
        },
    }

    last_error = None
    succeeded = False

    for attempt in range(1, retry_max + 1):
        try:
            if attempt > 1:
                print(f"[EVAL] Retry {attempt}/{retry_max} for {bench_name}")
            evaluate_benchmark(
                bench_name, cfg,
                progress_state=progress_state,
                bench_idx=0,
            )
            succeeded = True
            break
        except BenchmarkSkipError as e:
            last_error = str(e)
            print(f"[EVAL] SKIP (failure): {e}")
            break
        except Exception as e:
            last_error = str(e)
            print(f"[EVAL] Error: {e}")
            if attempt < retry_max:
                print(f"[EVAL] Waiting {retry_wait}s before retry...")
                time.sleep(retry_wait)

    # 결과를 파일에 기록
    result = {
        "succeeded": succeeded,
        "error": last_error,
    }
    try:
        with open(result_file, "w", encoding="utf-8") as f:
            json.dump(result, f)
    except Exception:
        pass


def _poll_progress_file(
    progress_file: str,
    progress_state: dict | None,
    bench_idx: int,
    stop_event: threading.Event,
    poll_interval: float = 1.0,
) -> None:
    """백그라운드에서 progress_file을 폴링하여 progress_state를 업데이트."""
    if progress_state is None or progress_state.get("progress") is None:
        return

    while not stop_event.is_set():
        try:
            if os.path.exists(progress_file):
                with open(progress_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                child_entry = data.get("progress", {}).get("benchmarks", [{}])[0]
                parent_entry = progress_state["progress"]["benchmarks"][bench_idx]
                if "video" in child_entry:
                    parent_entry["video"] = child_entry["video"]
                if "frame" in child_entry:
                    parent_entry["frame"] = child_entry["frame"]
                elif "frame" in parent_entry:
                    del parent_entry["frame"]
        except (json.JSONDecodeError, OSError, IndexError, KeyError):
            pass
        stop_event.wait(poll_interval)


def _restart_docker(docker_cfg, bench_idx: int, total: int) -> None:
    """벤치마크 사이에 Docker 컨테이너를 재시작하여 서버 측 메모리를 해소한다."""
    from .docker_manager import stop_container, start_container, wait_for_ready

    print(f"\n[EVAL] Docker 재시작 ({bench_idx}/{total} 벤치마크 완료 후)")
    print(f"[EVAL] 컨테이너 정리 중: {docker_cfg.container_name}")
    stop_container(docker_cfg.container_name)

    print(f"[EVAL] 컨테이너 재시작 중...")
    start_container(docker_cfg)

    print(f"[EVAL] 서버 준비 대기 중...")
    ready = wait_for_ready(docker_cfg)
    if not ready:
        print(f"[EVAL] WARNING: Docker 서버 준비 실패. 평가를 계속 시도합니다.")
    else:
        print(f"[EVAL] Docker 재시작 완료. 메모리 초기화됨.")


def run_evaluation(
    eval_cfg: EvalConfig,
    retry_max: int,
    retry_wait: int,
    progress_state: dict | None = None,
    docker_cfg=None,
    docker_restart_interval: int = 0,
) -> dict:
    """
    벤치마크 목록을 순회하며 평가 실행.

    각 벤치마크를 별도 프로세스에서 실행하여
    메모리 누적을 방지한다. 벤치마크 내부의
    병렬 처리(concurrency)는 그대로 유지된다.

    docker_restart_interval > 0이면 N개 벤치마크마다
    Docker 컨테이너를 재시작하여 서버 측 메모리 누적도 해소한다.

    반환: {"success": [...], "failed": [...]}
    """
    cfg = _build_cfg_namespace(eval_cfg)
    cfg_dict = _cfg_to_dict(cfg)
    benchmarks = eval_cfg.benchmarks

    print(f"\n[EVAL] Model     : {cfg.MODEL}")
    print(f"[EVAL] API       : {cfg.API_BASE}")
    print(f"[EVAL] Benchmarks: {benchmarks}")
    print(f"[EVAL] Mode      : subprocess isolation (memory-safe)")
    if docker_restart_interval > 0 and docker_cfg is not None:
        print(f"[EVAL] Docker restart: every {docker_restart_interval} benchmark(s)")

    # spawn 컨텍스트 사용: 스레드 안에서 fork하면 deadlock 위험
    ctx = multiprocessing.get_context("spawn")

    success = []
    failed = []

    for idx, bench_name in enumerate(benchmarks):
        # Docker 재시작 (첫 번째 벤치마크는 이미 떠있으므로 스킵)
        if (
            docker_restart_interval > 0
            and docker_cfg is not None
            and idx > 0
            and idx % docker_restart_interval == 0
        ):
            _restart_docker(docker_cfg, idx, len(benchmarks))

        print(f"\n[EVAL] [{idx+1}/{len(benchmarks)}] {bench_name}")
        _update_bench_progress(progress_state, idx, "in_progress")

        # 임시 파일: 진행도 + 결과 전달용
        progress_fd, progress_file = tempfile.mkstemp(suffix=".json", prefix="eval_progress_")
        os.close(progress_fd)
        result_fd, result_file = tempfile.mkstemp(suffix=".json", prefix="eval_result_")
        os.close(result_fd)

        # 진행도 폴링 스레드 시작
        stop_event = threading.Event()
        poll_thread = threading.Thread(
            target=_poll_progress_file,
            args=(progress_file, progress_state, idx, stop_event),
            daemon=True,
        )
        poll_thread.start()

        try:
            # 별도 프로세스에서 벤치마크 실행
            proc = ctx.Process(
                target=_benchmark_worker,
                args=(bench_name, cfg_dict, progress_file, result_file, retry_max, retry_wait),
            )
            proc.start()
            proc.join()

            # 결과 읽기
            try:
                with open(result_file, "r", encoding="utf-8") as f:
                    result = json.load(f)
            except (json.JSONDecodeError, OSError):
                result = {"succeeded": False, "error": "프로세스 결과 파일 읽기 실패"}

            # 프로세스 비정상 종료 (OOM 등)
            if proc.exitcode != 0 and not result.get("succeeded"):
                if proc.exitcode is not None and proc.exitcode < 0:
                    sig_name = f"signal {-proc.exitcode}"
                    result = {"succeeded": False, "error": f"프로세스 비정상 종료: {sig_name}"}
                elif not result.get("error"):
                    result = {"succeeded": False, "error": f"exit code: {proc.exitcode}"}

        finally:
            stop_event.set()
            poll_thread.join(timeout=3)

            for f in (progress_file, result_file):
                try:
                    os.unlink(f)
                except OSError:
                    pass

        if result.get("succeeded"):
            success.append(bench_name)
            _update_bench_progress(progress_state, idx, "completed")
        else:
            error = result.get("error", "unknown error")
            failed.append({"benchmark": bench_name, "error": error})
            _update_bench_progress(progress_state, idx, "failed")
            print(f"[EVAL] FAILED after {retry_max} attempts: {bench_name}")

    return {"success": success, "failed": failed}
