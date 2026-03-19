"""
파이프라인 오케스트레이터.

Docker → 평가 → 제출 → 정리 전체 파이프라인을 순차 실행.
"""

from __future__ import annotations

import time

from .config import PipelineConfig, load_pipeline_config
from .docker_manager import check_existing_container, start_container, stop_container, wait_for_ready
from .evaluator import run_evaluation
from .submitter import submit_results


def run_pipeline(yaml_path: str, override_steps: list[str] | None = None) -> None:
    """
    전체 파이프라인 실행.

    1. PipelineConfig 로드
    2. [DOCKER] 단계
    3. [EVAL] 단계
    4. [SUBMIT] 단계
    5. [CLEANUP] Docker 컨테이너 정리
    6. 최종 리포트
    """
    cfg = load_pipeline_config(yaml_path)

    steps = dict(cfg.steps)
    if override_steps is not None:
        steps = {k: (k in override_steps) for k in ["docker", "evaluate", "submit"]}

    print("=" * 60)
    print(f"Pipeline: {cfg.name}")
    print("=" * 60)
    print(f"Steps    : docker={steps.get('docker', False)}, "
          f"evaluate={steps.get('evaluate', False)}, "
          f"submit={steps.get('submit', False)}")
    print(f"Cleanup  : {cfg.cleanup_docker}")
    print(f"Retry    : max={cfg.retry_max_attempts}, wait={cfg.retry_wait_seconds}s")

    report = {
        "docker": None,
        "eval": None,
        "submit": None,
        "cleanup": None,
    }

    pipeline_start = time.time()
    docker_started = False

    # ── DOCKER ──
    if steps.get("docker", False):
        docker_started = True
        docker_ok, docker_msg = _run_docker_step(cfg)
        report["docker"] = docker_msg
        if not docker_ok:
            print(f"\n[PIPELINE] Docker failed - aborting pipeline")
            _print_report(cfg.name, report, time.time() - pipeline_start)
            return
    else:
        print("\n[PIPELINE] Docker step skipped")

    # ── EVALUATE ──
    eval_success_benchmarks = []
    if steps.get("evaluate", False):
        eval_result = run_evaluation(
            cfg.evaluate, cfg.retry_max_attempts, cfg.retry_wait_seconds,
        )
        eval_success_benchmarks = eval_result["success"]
        n_success = len(eval_result["success"])
        n_failed = len(eval_result["failed"])
        n_total = n_success + n_failed

        if eval_result["failed"]:
            failed_names = [f["benchmark"] for f in eval_result["failed"]]
            report["eval"] = f"{n_success}/{n_total} benchmarks succeeded\n          X {', '.join(failed_names)}"
        else:
            report["eval"] = f"{n_success}/{n_total} benchmarks succeeded"
    else:
        print("\n[PIPELINE] Evaluate step skipped")

    # ── SUBMIT ──
    if steps.get("submit", False):
        submit_benchmarks = eval_success_benchmarks if eval_success_benchmarks else (
            cfg.evaluate.benchmarks or []
        )
        if not submit_benchmarks:
            print("\n[PIPELINE] No benchmarks to submit")
            report["submit"] = "No benchmarks to submit"
        else:
            submit_result = submit_results(
                cfg.submit, submit_benchmarks,
                cfg.retry_max_attempts, cfg.retry_wait_seconds,
            )
            n_success = len(submit_result["success"])
            n_total = n_success + len(submit_result["failed"])
            if submit_result["failed"]:
                failed_names = [f["benchmark"] for f in submit_result["failed"]]
                report["submit"] = f"{n_success}/{n_total} submitted\n          X {', '.join(failed_names)}"
            else:
                report["submit"] = f"{n_success}/{n_total} submitted"
    else:
        print("\n[PIPELINE] Submit step skipped")

    # ── CLEANUP ──
    if docker_started and cfg.cleanup_docker:
        print(f"\n[CLEANUP] Removing container: {cfg.docker.container_name}")
        stop_container(cfg.docker.container_name)
        report["cleanup"] = f"Container {cfg.docker.container_name} removed (GPU freed)"
    elif docker_started:
        report["cleanup"] = f"Container {cfg.docker.container_name} kept running"

    total_elapsed = time.time() - pipeline_start
    _print_report(cfg.name, report, total_elapsed)


def _run_docker_step(cfg: PipelineConfig) -> tuple[bool, str]:
    """Docker 단계 실행. (성공여부, 메시지) 반환."""
    docker_cfg = cfg.docker
    start_time = time.time()

    status = check_existing_container(docker_cfg.container_name)
    if status == "running":
        print(f"[DOCKER] Existing container '{docker_cfg.container_name}' is running - removing...")
        stop_container(docker_cfg.container_name)
    elif status == "stopped":
        print(f"[DOCKER] Existing container '{docker_cfg.container_name}' is stopped - removing...")
        stop_container(docker_cfg.container_name)

    try:
        start_container(docker_cfg)
    except RuntimeError as e:
        return False, f"Container start failed: {e}"

    if wait_for_ready(docker_cfg):
        elapsed = time.time() - start_time
        return True, f"Container {docker_cfg.container_name} ready ({elapsed:.1f}s)"
    else:
        return False, f"Container {docker_cfg.container_name} failed to start"


def _print_report(name: str, report: dict, total_seconds: float) -> None:
    """최종 리포트 출력."""
    print(f"\n{'=' * 60}")
    print(f"Pipeline Complete: {name}")
    print(f"{'=' * 60}")

    if report["docker"]:
        ok = "V" if "ready" in report["docker"] else "X"
        print(f"[DOCKER]  {ok} {report['docker']}")

    if report["eval"]:
        ok = "V" if "X" not in report["eval"] else "!"
        print(f"[EVAL]    {ok} {report['eval']}")

    if report["submit"]:
        ok = "V" if "X" not in report["submit"] else "!"
        print(f"[SUBMIT]  {ok} {report['submit']}")

    if report["cleanup"]:
        print(f"[CLEANUP] V {report['cleanup']}")

    minutes = total_seconds / 60
    print(f"\nTotal elapsed: {total_seconds:.1f}s ({minutes:.1f}min)")
    print("=" * 60)
