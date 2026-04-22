"""
파이프라인 단계별 실행 + SSE 이벤트 생성.

`pipeline.mode` 필드에 따라 vLLM / LMDeploy 백엔드 중 하나를 선택하여
단계별로 호출하고 각 단계 완료 시 SSE 이벤트를 yield한다.
"""

from __future__ import annotations

import asyncio
import importlib
import json
import os
import re
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, AsyncGenerator

import yaml


SUPPORTED_MODES = ("vllm", "lmdeploy")


def _load_backend(mode: str) -> dict[str, Any]:
    """mode에 해당하는 파이프라인 백엔드 함수들을 lazy import하여 dict로 반환.

    Two PipelineConfig dataclasses have identical names but diverging fields,
    so we avoid top-level imports and resolve them per-request.
    """
    if mode not in SUPPORTED_MODES:
        raise ValueError(
            f"Unsupported pipeline.mode='{mode}'. Must be one of {SUPPORTED_MODES}."
        )

    pkg = importlib.import_module(f"src.{mode}_pipeline")
    config_mod = importlib.import_module(f"src.{mode}_pipeline.config")
    docker_mod = importlib.import_module(f"src.{mode}_pipeline.docker_manager")
    evaluator_mod = importlib.import_module(f"src.{mode}_pipeline.evaluator")
    submitter_mod = importlib.import_module(f"src.{mode}_pipeline.submitter")

    backend: dict[str, Any] = {
        "mode": mode,
        "load_pipeline_config": config_mod.load_pipeline_config,
        "check_existing_container": docker_mod.check_existing_container,
        "start_container": docker_mod.start_container,
        "stop_container": docker_mod.stop_container,
        "wait_for_ready": docker_mod.wait_for_ready,
        "run_evaluation": evaluator_mod.run_evaluation,
        "submit_results": submitter_mod.submit_results,
    }

    # mode-specific: model step 래퍼 + 서버 라벨
    if mode == "vllm":
        md_mod = importlib.import_module("src.vllm_pipeline.model_downloader")
        tp_mod = importlib.import_module("src.vllm_pipeline.tokenizer_patcher")

        def _model_step(cfg) -> dict[str, Any]:
            download = md_mod.ensure_model(cfg.docker)
            try:
                patch = tp_mod.patch_tokenizer_config(cfg.docker.hf_repo_id or cfg.docker.model)
            except Exception as e:
                patch = {
                    "patched": False, "reason": f"error:{e}", "file": None,
                    "model_type": None, "tokenizer_class": None,
                }
            return {
                "downloaded": download["downloaded"],
                "repo_id": download["repo_id"],
                "tokenizer_patch": patch["reason"],
                "model_type": patch.get("model_type"),
                "tokenizer_class": patch.get("tokenizer_class"),
                "summary": (
                    f"repo={download['repo_id']}, "
                    f"downloaded={download['downloaded']}, "
                    f"tokenizer_patch={patch['reason']}"
                    + (f", model_type={patch.get('model_type')}" if patch.get("model_type") else "")
                ),
            }

        backend["server_label"] = "vLLM"
    else:  # lmdeploy
        md_mod = importlib.import_module("src.lmdeploy_pipeline.model_downloader")

        def _model_step(cfg) -> dict[str, Any]:
            actual_path = md_mod.ensure_model(cfg.docker)
            if actual_path != cfg.docker.model_path:
                cfg.docker.model_path = actual_path
            return {
                "model_path": actual_path,
                "summary": f"model_path={actual_path}",
            }

        backend["server_label"] = "LMDeploy"

    backend["model_step"] = _model_step
    return backend


def sse_event(step: str, message: str, **kwargs) -> str:
    """SSE 형식의 data 라인 생성."""
    payload = {"step": step, "message": message}
    payload.update(kwargs)
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def validate_yaml(yaml_content: str) -> tuple[bool, str | dict]:
    """YAML 문자열을 파싱하고 필수 키를 검증한다.

    mode별 차등 검증:
        공통: `pipeline`, `docker`, `evaluate` 섹션 필수 + `pipeline.mode` 필수.
        vLLM: `docker.model` 필수.
        LMDeploy: `docker.model_path` 필수.

    Returns:
        (True, parsed_dict) 또는 (False, error_message)
    """
    try:
        config = yaml.safe_load(yaml_content)
    except yaml.YAMLError as e:
        return False, f"YAML 파싱 에러: {e}"

    if not isinstance(config, dict):
        return False, "YAML 최상위가 dict가 아닙니다."

    required_sections = ["pipeline", "docker", "evaluate"]
    missing = [k for k in required_sections if k not in config]
    if missing:
        return False, f"필수 섹션 누락: {', '.join(missing)}"

    mode = config.get("pipeline", {}).get("mode")
    if not mode:
        return False, (
            "pipeline.mode 필드가 필요합니다. "
            f"지원 모드: {', '.join(SUPPORTED_MODES)}"
        )
    if mode not in SUPPORTED_MODES:
        return False, (
            f"pipeline.mode='{mode}'는 지원되지 않습니다. "
            f"지원 모드: {', '.join(SUPPORTED_MODES)}"
        )

    docker = config.get("docker", {})
    if mode == "vllm":
        if "model" not in docker:
            return False, "docker.model 필드가 필요합니다. (vLLM 모드)"
    else:  # lmdeploy
        if "model_path" not in docker:
            return False, "docker.model_path 필드가 필요합니다. (LMDeploy 모드)"

    return True, config


UPLOADED_YAML_DIR = Path(__file__).resolve().parent.parent.parent / "results" / "api" / "uploaded_yamls"


def save_yaml_permanent(yaml_content: str, pipeline_name: str = "unknown") -> str:
    """YAML을 영구 저장 디렉토리에 저장하고 경로를 반환한다."""
    UPLOADED_YAML_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    safe_name = re.sub(r"[^\w\-]", "_", pipeline_name)[:80]
    filename = f"{timestamp}_{safe_name}.yaml"
    path = UPLOADED_YAML_DIR / filename
    path.write_text(yaml_content, encoding="utf-8")
    return str(path)


def validate_paths(config: dict) -> tuple[bool, str | None]:
    """YAML 설정의 파일시스템 경로를 검증한다.

    Returns:
        (True, None) 또는 (False, error_message)
    """
    errors: list[str] = []
    evaluate = config.get("evaluate", {})

    bench_base = evaluate.get("bench_base_path", "")
    if not bench_base:
        errors.append("evaluate.bench_base_path가 지정되지 않았습니다.")
    elif not Path(bench_base).exists():
        errors.append(f"bench_base_path가 존재하지 않습니다: {bench_base}")

    benchmarks = evaluate.get("benchmarks", [])
    if not benchmarks:
        errors.append("evaluate.benchmarks가 비어 있습니다.")

    if bench_base and Path(bench_base).exists() and benchmarks:
        missing = [b for b in benchmarks if not (Path(bench_base) / b).exists()]
        if missing:
            errors.append(
                f"벤치마크 폴더를 찾을 수 없습니다 ({len(missing)}/{len(benchmarks)}개): "
                f"{', '.join(missing)}"
            )

    output_path = evaluate.get("output_path", "")
    if output_path:
        out = Path(output_path)
        if out.exists() and not os.access(str(out), os.W_OK):
            errors.append(f"output_path에 쓰기 권한이 없습니다: {output_path}")

    if errors:
        return False, "\n".join(errors)
    return True, None


def _get_error_hint(error: Exception, mode: str | None = None) -> str:
    """에러 유형 + mode에 따른 사용자 힌트 반환."""
    msg = str(error).lower()
    if "cuda out of memory" in msg or "oom" in msg:
        if mode == "vllm":
            return "docker.vllm_args의 kv-cache-memory-bytes를 줄이거나 tensor-parallel-size를 늘려보세요."
        return "docker.lmdeploy_args.tp를 늘리거나 cache-max-entry-count를 줄여보세요."
    if "address already in use" in msg or "port" in msg:
        return "docker.port를 변경하거나 기존 컨테이너를 정리해주세요."
    if "tokenizer" in msg and ("tokenizersbackend" in msg or "not exist" in msg):
        return (
            "Unsloth 저장 모델의 tokenizer_class 불일치 가능성. "
            "vLLM 모드에서는 자동 패치가 수행되며, 실패 시 docker 데몬 상태를 확인하세요."
        )
    if "model" in msg and ("not found" in msg or "no such file" in msg):
        if mode == "vllm":
            return "docker.model(HF repo ID)을 확인하거나 docker.hf_repo_id로 사전 다운로드를 설정하세요."
        return "docker.model_path를 확인하거나 hf_repo_id를 설정해주세요."
    if "huggingface" in msg or "hf_repo_id" in msg:
        return "HuggingFace 로그인(hf auth login)을 확인하거나 repo ID를 확인해주세요."
    if mode == "vllm":
        return "YAML 설정을 확인하고 docs/eval/vllm_pipeline.md 가이드를 참고하세요."
    return "YAML 설정을 확인하고 docs/eval/lmdeploy_yaml_guide.md 가이드를 참고하세요."


def _background_eval_submit_cleanup(
    cfg,
    state: dict,
    lock: threading.Lock,
    yaml_path: str,
    backend: dict[str, Any],
) -> None:
    """백그라운드에서 평가 + 제출 + cleanup을 실행한다."""
    mode = backend["mode"]
    run_evaluation = backend["run_evaluation"]
    submit_results = backend["submit_results"]
    stop_container = backend["stop_container"]

    try:
        eval_result = run_evaluation(
            cfg.evaluate, cfg.retry_max_attempts, cfg.retry_wait_seconds,
            progress_state=state,
            docker_cfg=cfg.docker,
            docker_restart_interval=cfg.docker_restart_interval,
        )
        eval_success = eval_result["success"]

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
        n_total = n_success + n_failed

        if n_success == 0 and n_total > 0:
            state["status"] = "failed"
            failed_names = [f["benchmark"] for f in eval_result["failed"]]
            first_error = eval_result["failed"][0].get("error", "unknown")
            state["error"] = f"모든 벤치마크({n_total}개)가 실패했습니다: {', '.join(failed_names)}"
            state["hint"] = f"첫 번째 에러: {first_error}"
        elif n_failed > 0:
            state["status"] = "completed"
            failed_names = [f["benchmark"] for f in eval_result["failed"]]
            state["result"] = (
                f"평가 완료: {n_success}/{n_total} 벤치마크 성공, "
                f"실패: {', '.join(failed_names)}"
            )
        else:
            state["status"] = "completed"
            state["result"] = f"평가 완료: {n_success}/{n_total} 벤치마크 성공"

    except Exception as e:
        state["status"] = "failed"
        state["error"] = str(e)
        state["hint"] = _get_error_hint(e, mode=mode)
    finally:
        if cfg.cleanup_docker:
            try:
                subprocess.run(
                    ["docker", "logs", "--tail", "50", cfg.docker.container_name],
                    capture_output=True, text=True, timeout=10,
                )
                stop_container(cfg.docker.container_name)
            except Exception:
                pass

        state["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")

        if state.get("started_at"):
            try:
                t0 = time.strptime(state["started_at"], "%Y-%m-%dT%H:%M:%S")
                t1 = time.strptime(state["finished_at"], "%Y-%m-%dT%H:%M:%S")
                elapsed = int(time.mktime(t1) - time.mktime(t0))
                m, s = divmod(elapsed, 60)
                h, m = divmod(m, 60)
                if h > 0:
                    state["elapsed"] = f"{h}시간 {m}분 {s}초"
                elif m > 0:
                    state["elapsed"] = f"{m}분 {s}초"
                else:
                    state["elapsed"] = f"{s}초"
            except Exception:
                pass

        lock.release()


async def run_pipeline_sse(
    yaml_content: str,
    state: dict,
    lock: threading.Lock,
) -> AsyncGenerator[str, None]:
    """파이프라인을 단계별로 실행하며 SSE 이벤트를 yield한다.

    mode 추출 후 해당 백엔드를 로드하여 model_check → docker 기동 → 벤치마크 시작까지
    SSE 스트리밍. 평가/제출/cleanup은 백그라운드 스레드에서 진행한다.
    """
    loop = asyncio.get_event_loop()
    yaml_path = None
    background_started = False
    mode: str | None = None

    try:
        # 1. YAML 검증
        yield sse_event("received", "YAML 수신 완료")
        valid, result = validate_yaml(yaml_content)
        if not valid:
            yield sse_event(
                "error", result,
                hint="YAML 형식을 확인해주세요. docs/eval/pipeline_api.md 참조.",
            )
            return

        mode = result.get("pipeline", {}).get("mode")
        yield sse_event("yaml_validated", "YAML 검증 완료", mode=mode)

        # 2. 백엔드 로드
        try:
            backend = _load_backend(mode)
        except Exception as e:
            yield sse_event("error", f"백엔드 로드 실패: {e}")
            return

        # 3. YAML 영구 저장 + config 로드
        pipeline_name_raw = result.get("pipeline", {}).get("name", "unknown")
        yaml_path = save_yaml_permanent(yaml_content, pipeline_name_raw)
        try:
            cfg = await loop.run_in_executor(
                None, backend["load_pipeline_config"], yaml_path, mode,
            )
        except Exception as e:
            yield sse_event("error", f"설정 로드 실패: {e}", hint=_get_error_hint(e, mode=mode))
            return

        pipeline_name = cfg.name
        state["pipeline_name"] = pipeline_name
        state["mode"] = mode
        state["started_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        yield sse_event("config_loaded", f"파이프라인: {pipeline_name}", mode=mode)

        # 4. 모델 확인 (mode별로 동작 다름)
        yield sse_event("model_check", "모델 준비 중...")
        try:
            model_info = await loop.run_in_executor(None, backend["model_step"], cfg)
            yield sse_event("model_ready", f"모델 준비 완료: {model_info['summary']}", **model_info)
        except (FileNotFoundError, RuntimeError) as e:
            yield sse_event("error", f"모델 에러: {e}", hint=_get_error_hint(e, mode=mode))
            return

        # 5. Docker 기동
        yield sse_event("docker_starting", "Docker 컨테이너 기동 중...")
        try:
            existing = await loop.run_in_executor(
                None, backend["check_existing_container"], cfg.docker.container_name,
            )
            if existing in ("running", "stopped"):
                await loop.run_in_executor(None, backend["stop_container"], cfg.docker.container_name)

            await loop.run_in_executor(None, backend["start_container"], cfg.docker)
        except RuntimeError as e:
            yield sse_event("error", f"Docker 기동 실패: {e}", hint=_get_error_hint(e, mode=mode))
            return

        # 6. 서버 준비 대기
        yield sse_event("docker_waiting", f"{backend['server_label']} 서버 준비 대기 중...")
        docker_start = time.time()
        try:
            ready = await loop.run_in_executor(None, backend["wait_for_ready"], cfg.docker)
        except Exception as e:
            yield sse_event("error", f"서버 준비 실패: {e}", hint=_get_error_hint(e, mode=mode))
            try:
                await loop.run_in_executor(None, backend["stop_container"], cfg.docker.container_name)
            except Exception:
                pass
            return

        if not ready:
            yield sse_event(
                "error", "서버 준비 시간 초과 (timeout)",
                hint="startup.timeout_seconds를 늘리거나 Docker 로그를 확인해주세요.",
            )
            try:
                await loop.run_in_executor(None, backend["stop_container"], cfg.docker.container_name)
            except Exception:
                pass
            return

        docker_elapsed = time.time() - docker_start
        yield sse_event("docker_ready", f"컨테이너 준비 완료 ({docker_elapsed:.1f}s)")

        # 7. 벤치마크 시작 알림 + progress 초기화
        benchmarks = cfg.evaluate.benchmarks or []
        state["progress"] = {
            "total": len(benchmarks),
            "completed": 0,
            "current": None,
            "benchmarks": [
                {"name": b, "status": "pending"} for b in benchmarks
            ],
        }
        yield sse_event(
            "eval_started",
            f"벤치마크 평가 시작 ({len(benchmarks)}개). 리더보드에서 결과를 확인하세요.",
        )
        yield sse_event("done", "평가가 시작되었습니다. 추후 벤치마크 결과를 리더보드에서 확인해보세요.")

        # 8. 백그라운드 평가 + 제출 + cleanup
        state["status"] = "evaluating"
        background_started = True
        thread = threading.Thread(
            target=_background_eval_submit_cleanup,
            args=(cfg, state, lock, yaml_path, backend),
            daemon=False,
        )
        thread.start()

    except GeneratorExit:
        pass
    except Exception as e:
        yield sse_event("error", f"예기치 않은 에러: {e}", hint=_get_error_hint(e, mode=mode))
    finally:
        if not background_started:
            state["status"] = "idle"
            state["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
            if lock.locked():
                lock.release()
