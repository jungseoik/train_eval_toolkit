"""
Docker 컨테이너 관리 모듈.

vLLM Docker 컨테이너의 생명주기(시작, 상태 확인, 대기, 종료)를 관리.
모델 로딩 중 실시간 로그 스트리밍과 치명적 에러 감지를 지원.
"""

from __future__ import annotations

import os
import subprocess
import threading
import time
from typing import Optional

import httpx

from .config import DockerConfig

FATAL_PATTERNS = [
    "CUDA out of memory",
    "torch.cuda.OutOfMemoryError",
    "RuntimeError",
    "Address already in use",
    "No such file or directory",
    "permission denied",
]


def check_existing_container(name: str) -> str:
    """
    동일 이름의 Docker 컨테이너 상태 확인.

    반환: "running", "stopped", "none"
    """
    result = subprocess.run(
        ["docker", "inspect", "--format", "{{.State.Status}}", name],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        return "none"
    status = result.stdout.strip()
    if status == "running":
        return "running"
    return "stopped"


def stop_container(name: str) -> None:
    """docker rm -f로 컨테이너 제거. 미존재 시 무시."""
    subprocess.run(
        ["docker", "rm", "-f", name],
        capture_output=True, text=True,
    )


def start_container(cfg: DockerConfig) -> str:
    """docker run -d 명령을 조립하여 실행. container_id 반환."""
    cmd = [
        "docker", "run", "-d",
        "--name", cfg.container_name,
        "--gpus", cfg.gpus,
        "--ipc", cfg.ipc,
        "-p", f"{cfg.port}:8000",
    ]

    for vol in cfg.volumes:
        cmd.extend(["-v", os.path.expanduser(vol)])

    cmd.append(cfg.image)

    cmd.extend(["--model", cfg.model])

    for key, value in cfg.vllm_args.items():
        cmd.append(f"--{key}")
        cmd.append(str(value))

    print(f"[DOCKER] Starting container: {cfg.container_name}")
    print(f"[DOCKER] Command: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Docker run failed:\n{result.stderr}")

    container_id = result.stdout.strip()
    print(f"[DOCKER] Container started: {container_id[:12]}")
    return container_id


def wait_for_ready(cfg: DockerConfig) -> bool:
    """
    vLLM 서버가 요청을 받을 준비가 될 때까지 대기.

    1. 백그라운드 스레드에서 docker logs 실시간 스트리밍
    2. 메인 스레드에서 /v1/models HTTP 폴링
    3. 치명적 에러 감지 시 즉시 False 반환
    4. timeout 초과 시 False 반환
    """
    stop_event = threading.Event()
    fatal_error: dict[str, Optional[str]] = {"message": None}

    log_thread = None
    if cfg.stream_logs:
        log_thread = threading.Thread(
            target=_stream_docker_logs,
            args=(cfg.container_name, stop_event, fatal_error),
            daemon=True,
        )
        log_thread.start()

    start_time = time.time()
    ready = False

    print(f"[DOCKER] Waiting for server ready (timeout: {cfg.timeout_seconds}s)...")

    try:
        while time.time() - start_time < cfg.timeout_seconds:
            if fatal_error["message"]:
                print(f"\n[DOCKER] Fatal error detected: {fatal_error['message']}")
                return False

            if _check_health(cfg.port):
                elapsed = time.time() - start_time
                print(f"\n[DOCKER] Server ready! ({elapsed:.1f}s)")
                ready = True
                return True

            time.sleep(cfg.poll_interval_seconds)

        if not ready:
            print(f"\n[DOCKER] Timeout after {cfg.timeout_seconds}s")
            return False
    finally:
        stop_event.set()
        if log_thread and log_thread.is_alive():
            log_thread.join(timeout=5)

    return False


def _stream_docker_logs(
    container_name: str,
    stop_event: threading.Event,
    fatal_error: dict[str, Optional[str]],
) -> None:
    """docker logs -f를 서브프로세스로 실행하여 실시간 출력."""
    try:
        proc = subprocess.Popen(
            ["docker", "logs", "-f", container_name],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        for line in iter(proc.stdout.readline, ""):
            if stop_event.is_set():
                break
            line = line.rstrip()
            if line:
                print(f"  [LOG] {line}")
                error_msg = _detect_fatal_error(line)
                if error_msg:
                    fatal_error["message"] = error_msg
                    break
        proc.terminate()
        proc.wait(timeout=5)
    except Exception:
        pass


def _check_health(port: int) -> bool:
    """GET /v1/models → 200이면 True."""
    try:
        resp = httpx.get(f"http://127.0.0.1:{port}/v1/models", timeout=5.0)
        return resp.status_code == 200
    except Exception:
        return False


def _detect_fatal_error(log_line: str) -> Optional[str]:
    """로그 라인에서 치명적 에러 패턴 감지. 에러 메시지 반환 또는 None."""
    for pattern in FATAL_PATTERNS:
        if pattern.lower() in log_line.lower():
            return log_line
    return None
