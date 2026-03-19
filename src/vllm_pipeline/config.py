"""
vLLM 파이프라인 설정 모듈.

YAML 파일을 로드하여 타입 안전한 dataclass 객체로 변환.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DockerConfig:
    container_name: str
    image: str
    model: str
    gpus: str = "all"
    port: int = 8000
    ipc: str = "host"
    volumes: list[str] = field(default_factory=list)
    vllm_args: dict[str, Any] = field(default_factory=dict)
    timeout_seconds: int = 300
    poll_interval_seconds: int = 5
    stream_logs: bool = True


@dataclass
class EvalConfig:
    eval_config_path: str
    benchmarks: list[str] = field(default_factory=list)
    overrides: dict[str, Any] = field(default_factory=dict)


@dataclass
class SubmitConfig:
    gradio_url: str
    model_name: str
    task_name: str
    datasets_used: str
    results_base_dir: str
    config_file: str = "config.json"
    interval_seconds: int = 60


@dataclass
class PipelineConfig:
    name: str
    steps: dict[str, bool]
    cleanup_docker: bool
    retry_max_attempts: int
    retry_wait_seconds: int
    docker: DockerConfig
    evaluate: EvalConfig
    submit: SubmitConfig


def load_pipeline_config(yaml_path: str) -> PipelineConfig:
    """YAML 파일을 로드하여 PipelineConfig 데이터클래스로 변환."""
    path = Path(yaml_path)
    if not path.exists():
        raise FileNotFoundError(f"Pipeline config not found: {path}")

    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    pipeline = raw.get("pipeline", {})
    retry = raw.get("retry", {})
    docker_raw = raw.get("docker", {})
    eval_raw = raw.get("evaluate", {})
    submit_raw = raw.get("submit", {})

    startup = docker_raw.pop("startup", {})

    docker_cfg = DockerConfig(
        container_name=docker_raw["container_name"],
        image=docker_raw["image"],
        model=docker_raw["model"],
        gpus=docker_raw.get("gpus", "all"),
        port=docker_raw.get("port", 8000),
        ipc=docker_raw.get("ipc", "host"),
        volumes=docker_raw.get("volumes", []),
        vllm_args=docker_raw.get("vllm_args", {}),
        timeout_seconds=startup.get("timeout_seconds", 300),
        poll_interval_seconds=startup.get("poll_interval_seconds", 5),
        stream_logs=startup.get("stream_logs", True),
    )

    eval_cfg = EvalConfig(
        eval_config_path=eval_raw.get("eval_config_path", ""),
        benchmarks=eval_raw.get("benchmarks", []),
        overrides=eval_raw.get("overrides", {}),
    )

    submit_cfg = SubmitConfig(
        gradio_url=submit_raw.get("gradio_url", ""),
        model_name=submit_raw.get("model_name", ""),
        task_name=submit_raw.get("task_name", ""),
        datasets_used=submit_raw.get("datasets_used", ""),
        config_file=submit_raw.get("config_file", "config.json"),
        results_base_dir=submit_raw.get("results_base_dir", ""),
        interval_seconds=submit_raw.get("interval_seconds", 60),
    )

    return PipelineConfig(
        name=pipeline.get("name", "Unnamed Pipeline"),
        steps=pipeline.get("steps", {"docker": True, "evaluate": True, "submit": True}),
        cleanup_docker=pipeline.get("cleanup_docker", True),
        retry_max_attempts=retry.get("max_attempts", 3),
        retry_wait_seconds=retry.get("wait_seconds", 30),
        docker=docker_cfg,
        evaluate=eval_cfg,
        submit=submit_cfg,
    )
