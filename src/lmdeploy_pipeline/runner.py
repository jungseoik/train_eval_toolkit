"""
파이프라인 오케스트레이터.

Docker -> 평가 -> 제출 -> 정리 전체 파이프라인을 순차 실행.
비정상 종료(예외, SIGTERM, SIGHUP 등) 시에도 Docker 컨테이너 cleanup을 보장.
"""

from __future__ import annotations

import atexit
import signal
import subprocess
import sys
import time

from .config import PipelineConfig, load_pipeline_config
from .docker_manager import check_existing_container, start_container, stop_container, wait_for_ready
from .evaluator import run_evaluation
from .model_downloader import ensure_model
from .submitter import submit_results

# 프로세스 수준 cleanup 상태 (signal handler/atexit에서 참조)
_cleanup_state: dict = {
    "container_name": None,
    "cleanup_enabled": False,
    "cleaned": False,
}


def _emergency_cleanup() -> None:
    """비정상 종료 시 컨테이너 정리. 멱등 — 여러 번 호출해도 안전."""
    if _cleanup_state["cleaned"] or not _cleanup_state["cleanup_enabled"]:
        return
    container = _cleanup_state["container_name"]
    if not container:
        return
    _cleanup_state["cleaned"] = True
    try:
        _dump_container_logs(container)
        print(f"\n[EMERGENCY CLEANUP] Removing container: {container}")
        subprocess.run(["docker", "rm", "-f", container], capture_output=True, text=True)
    except Exception:
        pass


def _signal_handler(signum: int, _frame) -> None:
    """SIGTERM/SIGHUP 수신 시 cleanup 후 종료."""
    sig_name = signal.Signals(signum).name
    print(f"\n[SIGNAL] {sig_name} received - cleaning up...")
    _emergency_cleanup()
    sys.exit(128 + signum)


def _dump_container_logs(container_name: str) -> None:
    """컨테이너 삭제 전 최근 로그를 콘솔에 출력 (사후 디버깅용)."""
    try:
        result = subprocess.run(
            ["docker", "logs", "--tail", "50", container_name],
            capture_output=True, text=True, timeout=10,
        )
        if result.stdout.strip():
            print(f"\n[CLEANUP] Last 50 lines of container logs ({container_name}):")
            for line in result.stdout.strip().splitlines():
                print(f"  {line}")
    except Exception:
        pass


def run_pipeline(yaml_path: str, override_steps: list[str] | None = None) -> None:
    """
    전체 파이프라인 실행.

    1. PipelineConfig 로드
    2. [MODEL] 모델 존재 확인 / HuggingFace 다운로드
    3. [DOCKER] 단계
    4. [EVAL] 단계
    5. [SUBMIT] 단계
    6. [CLEANUP] Docker 컨테이너 정리
    7. 최종 리포트
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
    print(f"Model    : {cfg.docker.model_path}")
    print(f"Cleanup  : {cfg.cleanup_docker}")
    print(f"Retry    : max={cfg.retry_max_attempts}, wait={cfg.retry_wait_seconds}s")

    report = {
        "model": None,
        "docker": None,
        "eval": None,
        "submit": None,
        "cleanup": None,
    }

    pipeline_start = time.time()
    docker_started = False

    # -- MODEL: 존재 확인 / HuggingFace 다운로드 --
    try:
        actual_model_path = ensure_model(cfg.docker)
        if actual_model_path != cfg.docker.model_path:
            cfg.docker.model_path = actual_model_path
        report["model"] = f"Model ready: {cfg.docker.model_path}"
    except (FileNotFoundError, RuntimeError) as e:
        report["model"] = f"Model failed: {e}"
        print(f"\n[PIPELINE] Model check failed - aborting pipeline")
        total_elapsed = time.time() - pipeline_start
        _print_report(cfg.name, report, total_elapsed)
        return

    # signal handler / atexit 등록
    _cleanup_state["container_name"] = cfg.docker.container_name
    _cleanup_state["cleanup_enabled"] = cfg.cleanup_docker
    _cleanup_state["cleaned"] = False
    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGHUP, _signal_handler)
    atexit.register(_emergency_cleanup)

    try:
        # -- DOCKER --
        if steps.get("docker", False):
            docker_started = True
            docker_ok, docker_msg = _run_docker_step(cfg)
            report["docker"] = docker_msg
            if not docker_ok:
                print(f"\n[PIPELINE] Docker failed - aborting pipeline")
                return
        else:
            print("\n[PIPELINE] Docker step skipped")

        # -- EVALUATE --
        eval_success_benchmarks = []
        if steps.get("evaluate", False):
            eval_result = run_evaluation(
                cfg.evaluate, cfg.retry_max_attempts, cfg.retry_wait_seconds,
                docker_cfg=cfg.docker,
                docker_restart_interval=cfg.docker_restart_interval,
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

        # -- SUBMIT --
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
    finally:
        # -- CLEANUP (정상/비정상 모두 도달) --
        if docker_started and cfg.cleanup_docker and not _cleanup_state["cleaned"]:
            _dump_container_logs(cfg.docker.container_name)
            print(f"\n[CLEANUP] Removing container: {cfg.docker.container_name}")
            stop_container(cfg.docker.container_name)
            _cleanup_state["cleaned"] = True
            report["cleanup"] = f"Container {cfg.docker.container_name} removed (GPU freed)"
        elif docker_started and not cfg.cleanup_docker:
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

    if report["model"]:
        ok = "V" if "ready" in report["model"] else "X"
        print(f"[MODEL]   {ok} {report['model']}")

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
