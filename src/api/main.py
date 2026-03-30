"""
LMDeploy 벤치마크 평가 파이프라인 API 서버.

SSE(Server-Sent Events)로 실시간 진행 상황을 스트리밍.
한 번에 하나의 평가만 처리 (동시 요청 거절).

실행:
    uvicorn src.api.main:app --host 0.0.0.0 --port 9000
"""

from __future__ import annotations

import subprocess
import threading
import time

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from .pipeline_worker import run_pipeline_sse, validate_yaml

GPU_VRAM_THRESHOLD_MB = 50_000  # 50GB


def check_gpu_vram() -> tuple[bool, str]:
    """GPU VRAM 사용량을 확인한다.

    Returns:
        (True, 정보 문자열) 사용 가능 시,
        (False, 에러 메시지) VRAM 초과 시.
    """
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.used,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            return True, "nvidia-smi 실행 실패 (체크 건너뜀)"

        total_used_mb = 0
        gpu_details = []
        for line in result.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            idx, used, total = parts[0], int(parts[1]), int(parts[2])
            total_used_mb += used
            gpu_details.append(f"GPU {idx}: {used}MB / {total}MB")

        if total_used_mb > GPU_VRAM_THRESHOLD_MB:
            detail = "\n".join(gpu_details)
            return False, (
                f"GPU가 현재 사용 중입니다 (VRAM 사용량: {total_used_mb / 1000:.1f}GB / 임계값: {GPU_VRAM_THRESHOLD_MB / 1000:.0f}GB).\n"
                f"{detail}\n"
                f"담당자에게 GPU 리소스에 대해 문의해주세요."
            )

        return True, f"VRAM 사용량: {total_used_mb / 1000:.1f}GB (여유 있음)"

    except Exception as e:
        return True, f"GPU 체크 실패: {e} (체크 건너뜀)"

app = FastAPI(
    title="LMDeploy Pipeline API",
    description="LMDeploy 벤치마크 평가 파이프라인 요청 서버",
    version="1.0.0",
)

# 전역 상태
_lock = threading.Lock()
_state: dict = {
    "status": "idle",       # idle / running / evaluating / completed / failed
    "pipeline_name": None,
    "started_at": None,
    "finished_at": None,
    "error": None,
    "hint": None,
    "result": None,
}


def _reset_state() -> None:
    _state.update({
        "status": "running",
        "pipeline_name": None,
        "started_at": None,
        "finished_at": None,
        "error": None,
        "hint": None,
        "result": None,
    })


class YamlRequest(BaseModel):
    yaml_content: str


@app.get("/health")
async def health():
    return {"status": "ok", "server": "LMDeploy Pipeline API", "port": 9000}


@app.get("/pipeline/status")
async def pipeline_status():
    return dict(_state)


@app.post("/pipeline/run/file")
async def run_from_file(file: UploadFile = File(...)):
    """YAML 파일 업로드로 파이프라인 실행 (SSE 스트리밍)."""
    if not _lock.acquire(blocking=False):
        return JSONResponse(
            status_code=409,
            content={
                "status": "busy",
                "message": "이미 평가가 진행 중입니다. 완료 후 다시 요청해주세요.",
                "current_pipeline": _state.get("pipeline_name"),
                "started_at": _state.get("started_at"),
            },
        )

    try:
        content = await file.read()
        yaml_content = content.decode("utf-8")
    except Exception as e:
        _lock.release()
        raise HTTPException(status_code=400, detail=f"파일 읽기 실패: {e}")

    # 빠른 검증 (SSE 전에 명백한 에러 차단)
    valid, result = validate_yaml(yaml_content)
    if not valid:
        _lock.release()
        raise HTTPException(status_code=400, detail={
            "status": "error",
            "message": result,
            "hint": "YAML 형식을 확인해주세요. 작성 가이드: docs/eval/lmdeploy_yaml_guide.md",
        })

    # GPU VRAM 체크
    gpu_ok, gpu_msg = check_gpu_vram()
    if not gpu_ok:
        _lock.release()
        return JSONResponse(
            status_code=503,
            content={"status": "gpu_busy", "message": gpu_msg},
        )

    _reset_state()

    return EventSourceResponse(
        run_pipeline_sse(yaml_content, _state, _lock),
        media_type="text/event-stream",
    )


@app.post("/pipeline/run/yaml")
async def run_from_yaml(body: YamlRequest):
    """JSON body의 YAML 문자열로 파이프라인 실행 (SSE 스트리밍)."""
    if not _lock.acquire(blocking=False):
        return JSONResponse(
            status_code=409,
            content={
                "status": "busy",
                "message": "이미 평가가 진행 중입니다. 완료 후 다시 요청해주세요.",
                "current_pipeline": _state.get("pipeline_name"),
                "started_at": _state.get("started_at"),
            },
        )

    yaml_content = body.yaml_content

    valid, result = validate_yaml(yaml_content)
    if not valid:
        _lock.release()
        raise HTTPException(status_code=400, detail={
            "status": "error",
            "message": result,
            "hint": "YAML 형식을 확인해주세요. 작성 가이드: docs/eval/lmdeploy_yaml_guide.md",
        })

    # GPU VRAM 체크
    gpu_ok, gpu_msg = check_gpu_vram()
    if not gpu_ok:
        _lock.release()
        return JSONResponse(
            status_code=503,
            content={"status": "gpu_busy", "message": gpu_msg},
        )

    _reset_state()

    return EventSourceResponse(
        run_pipeline_sse(yaml_content, _state, _lock),
        media_type="text/event-stream",
    )
