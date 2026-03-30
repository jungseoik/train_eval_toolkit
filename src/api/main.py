"""
LMDeploy 벤치마크 평가 파이프라인 API 서버.

SSE(Server-Sent Events)로 실시간 진행 상황을 스트리밍.
한 번에 하나의 평가만 처리 (동시 요청 거절).

실행:
    uvicorn src.api.main:app --host 0.0.0.0 --port 9000
"""

from __future__ import annotations

import threading
import time

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from .pipeline_worker import run_pipeline_sse, validate_yaml

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

    _reset_state()

    return EventSourceResponse(
        run_pipeline_sse(yaml_content, _state, _lock),
        media_type="text/event-stream",
    )
