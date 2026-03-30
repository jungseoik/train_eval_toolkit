"""
파이프라인 단계별 실행 + SSE 이벤트 생성.

기존 lmdeploy_pipeline 내부 모듈을 단계별로 호출하여
각 단계 완료 시 SSE 이벤트를 yield한다.
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import tempfile
import threading
import time
import os
from typing import AsyncGenerator

import yaml

from src.lmdeploy_pipeline.config import load_pipeline_config
from src.lmdeploy_pipeline.docker_manager import (
    check_existing_container,
    start_container,
    stop_container,
    wait_for_ready,
)
from src.lmdeploy_pipeline.evaluator import run_evaluation
from src.lmdeploy_pipeline.model_downloader import ensure_model
from src.lmdeploy_pipeline.submitter import submit_results


def sse_event(step: str, message: str, **kwargs) -> str:
    """SSE 형식의 data 라인 생성."""
    payload = {"step": step, "message": message}
    payload.update(kwargs)
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def validate_yaml(yaml_content: str) -> tuple[bool, str | dict]:
    """YAML 문자열을 파싱하고 필수 키를 검증한다.

    Returns:
        (True, parsed_dict) 또는 (False, error_message)
    """
    try:
        config = yaml.safe_load(yaml_content)
    except yaml.YAMLError as e:
        return False, f"YAML 파싱 에러: {e}"

    if not isinstance(config, dict):
        return False, "YAML 최상위가 dict가 아닙니다."

    required_keys = ["pipeline", "docker", "evaluate"]
    missing = [k for k in required_keys if k not in config]
    if missing:
        return False, f"필수 섹션 누락: {', '.join(missing)}"

    return True, config


def save_yaml_to_temp(yaml_content: str) -> str:
    """YAML 문자열을 임시 파일로 저장하고 경로를 반환한다."""
    fd, path = tempfile.mkstemp(suffix=".yaml", prefix="pipeline_")
    with os.fdopen(fd, "w") as f:
        f.write(yaml_content)
    return path


def _get_error_hint(error: Exception) -> str:
    """에러 유형에 따른 사용자 힌트 반환."""
    msg = str(error).lower()
    if "cuda out of memory" in msg or "oom" in msg:
        return "lmdeploy_args.tp를 늘리거나 cache-max-entry-count를 줄여보세요."
    if "address already in use" in msg or "port" in msg:
        return "docker.port를 변경하거나 기존 컨테이너를 정리해주세요."
    if "model" in msg and ("not found" in msg or "no such file" in msg):
        return "docker.model_path를 확인하거나 hf_repo_id를 설정해주세요."
    if "huggingface" in msg or "hf_repo_id" in msg:
        return "HuggingFace 로그인(hf auth login)을 확인하거나 hf_repo_id를 확인해주세요."
    return "YAML 설정을 확인하고 다시 제출해주세요."


def _background_eval_submit_cleanup(
    cfg,
    state: dict,
    lock: threading.Lock,
    yaml_path: str,
) -> None:
    """백그라운드에서 평가 + 제출 + cleanup을 실행한다."""
    try:
        # 평가
        eval_result = run_evaluation(
            cfg.evaluate, cfg.retry_max_attempts, cfg.retry_wait_seconds,
        )
        eval_success = eval_result["success"]

        # 제출
        steps = dict(cfg.steps)
        if steps.get("submit", False):
            submit_benchmarks = eval_success if eval_success else (cfg.evaluate.benchmarks or [])
            if submit_benchmarks:
                submit_results(
                    cfg.submit, submit_benchmarks,
                    cfg.retry_max_attempts, cfg.retry_wait_seconds,
                )

        n_success = len(eval_result["success"])
        n_failed = len(eval_result["failed"])
        state["status"] = "completed"
        state["result"] = f"평가 완료: {n_success}/{n_success + n_failed} 벤치마크 성공"
        if eval_result["failed"]:
            failed_names = [f["benchmark"] for f in eval_result["failed"]]
            state["result"] += f", 실패: {', '.join(failed_names)}"

    except Exception as e:
        state["status"] = "failed"
        state["error"] = str(e)
        state["hint"] = _get_error_hint(e)
    finally:
        # cleanup
        if cfg.cleanup_docker:
            try:
                subprocess.run(
                    ["docker", "logs", "--tail", "50", cfg.docker.container_name],
                    capture_output=True, text=True, timeout=10,
                )
                stop_container(cfg.docker.container_name)
            except Exception:
                pass

        # 임시 파일 삭제
        try:
            os.remove(yaml_path)
        except OSError:
            pass

        state["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        lock.release()


async def run_pipeline_sse(
    yaml_content: str,
    state: dict,
    lock: threading.Lock,
) -> AsyncGenerator[str, None]:
    """파이프라인을 단계별로 실행하며 SSE 이벤트를 yield한다.

    Docker 준비 → 벤치마크 시작 확인까지 SSE 스트리밍.
    이후 평가/제출/cleanup은 백그라운드 스레드에서 진행.
    """
    loop = asyncio.get_event_loop()
    yaml_path = None

    try:
        # 1. YAML 검증
        yield sse_event("received", "YAML 수신 완료")
        valid, result = validate_yaml(yaml_content)
        if not valid:
            yield sse_event("error", result, hint="YAML 형식을 확인해주세요. 작성 가이드: docs/eval/lmdeploy_yaml_guide.md")
            lock.release()
            return

        yield sse_event("yaml_validated", "YAML 검증 완료")

        # 2. 임시 파일 저장 + config 로드
        yaml_path = save_yaml_to_temp(yaml_content)
        try:
            cfg = await loop.run_in_executor(None, load_pipeline_config, yaml_path)
        except Exception as e:
            yield sse_event("error", f"설정 로드 실패: {e}", hint="YAML 필드를 확인해주세요.")
            lock.release()
            _safe_remove(yaml_path)
            return

        pipeline_name = cfg.name
        state["pipeline_name"] = pipeline_name
        state["started_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        yield sse_event("config_loaded", f"파이프라인: {pipeline_name}")

        # 3. 모델 확인
        yield sse_event("model_check", "모델 확인 중...")
        try:
            actual_path = await loop.run_in_executor(None, ensure_model, cfg.docker)
            if actual_path != cfg.docker.model_path:
                cfg.docker.model_path = actual_path
            yield sse_event("model_ready", f"모델 준비 완료: {cfg.docker.model_path}")
        except (FileNotFoundError, RuntimeError) as e:
            yield sse_event("error", f"모델 에러: {e}", hint=_get_error_hint(e))
            lock.release()
            _safe_remove(yaml_path)
            return

        # 4. Docker 기동
        yield sse_event("docker_starting", "Docker 컨테이너 기동 중...")
        try:
            existing = await loop.run_in_executor(
                None, check_existing_container, cfg.docker.container_name,
            )
            if existing in ("running", "stopped"):
                await loop.run_in_executor(None, stop_container, cfg.docker.container_name)

            await loop.run_in_executor(None, start_container, cfg.docker)
        except RuntimeError as e:
            yield sse_event("error", f"Docker 기동 실패: {e}", hint=_get_error_hint(e))
            lock.release()
            _safe_remove(yaml_path)
            return

        # 5. 서버 준비 대기
        yield sse_event("docker_waiting", "LMDeploy 서버 준비 대기 중...")
        docker_start = time.time()
        try:
            ready = await loop.run_in_executor(None, wait_for_ready, cfg.docker)
        except Exception as e:
            yield sse_event("error", f"서버 준비 실패: {e}", hint=_get_error_hint(e))
            # cleanup container
            try:
                await loop.run_in_executor(None, stop_container, cfg.docker.container_name)
            except Exception:
                pass
            lock.release()
            _safe_remove(yaml_path)
            return

        if not ready:
            yield sse_event("error", "서버 준비 시간 초과 (timeout)", hint="startup.timeout_seconds를 늘리거나 Docker 로그를 확인해주세요.")
            try:
                await loop.run_in_executor(None, stop_container, cfg.docker.container_name)
            except Exception:
                pass
            lock.release()
            _safe_remove(yaml_path)
            return

        docker_elapsed = time.time() - docker_start
        yield sse_event("docker_ready", f"컨테이너 준비 완료 ({docker_elapsed:.1f}s)")

        # 6. 벤치마크 시작 알림
        benchmarks = cfg.evaluate.benchmarks or []
        yield sse_event("eval_started", f"벤치마크 평가 시작 ({len(benchmarks)}개). 리더보드에서 결과를 확인하세요.")
        yield sse_event("done", "평가가 시작되었습니다. 추후 벤치마크 결과를 리더보드에서 확인해보세요.")

        # 7. 백그라운드 스레드에서 평가 + 제출 + cleanup
        state["status"] = "evaluating"
        thread = threading.Thread(
            target=_background_eval_submit_cleanup,
            args=(cfg, state, lock, yaml_path),
            daemon=True,
        )
        thread.start()
        # yaml_path는 백그라운드에서 삭제하므로 여기서는 건드리지 않음

    except Exception as e:
        yield sse_event("error", f"예기치 않은 에러: {e}", hint=_get_error_hint(e))
        if lock.locked():
            lock.release()
        if yaml_path:
            _safe_remove(yaml_path)


def _safe_remove(path: str) -> None:
    try:
        os.remove(path)
    except OSError:
        pass
