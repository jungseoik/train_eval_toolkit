"""vLLM/LMDeploy config 로더의 expected_mode guard 검증."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from src.vllm_pipeline.config import load_pipeline_config as vllm_load
from src.lmdeploy_pipeline.config import load_pipeline_config as lmdeploy_load


def _minimal_vllm_yaml(tmp_path: Path, mode: str = "vllm") -> Path:
    cfg = {
        "pipeline": {"name": "t", "mode": mode},
        "docker": {
            "container_name": "t",
            "image": "vllm/vllm-openai:cu130-nightly",
            "model": "Qwen/Qwen3.5-2B",
        },
        "evaluate": {
            "benchmarks": [],
            "model": "Qwen/Qwen3.5-2B",
            "run_name": "t",
            "bench_base_path": "/tmp",
            "output_path": "/tmp",
        },
        "submit": {
            "gradio_url": "http://x",
            "model_name": "t",
            "task_name": "t",
            "datasets_used": "Pretrained",
            "results_base_dir": "results",
        },
    }
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return path


def _minimal_lmdeploy_yaml(tmp_path: Path, mode: str = "lmdeploy") -> Path:
    cfg = {
        "pipeline": {"name": "t", "mode": mode},
        "docker": {
            "container_name": "t",
            "image": "openmmlab/lmdeploy:latest-cu12",
            "model_path": "/tmp/m",
        },
        "evaluate": {
            "benchmarks": [],
            "model": "/model",
            "run_name": "t",
            "bench_base_path": "/tmp",
            "output_path": "/tmp",
        },
        "submit": {
            "gradio_url": "http://x",
            "model_name": "t",
            "task_name": "t",
            "datasets_used": "Finetuned",
            "results_base_dir": "results",
        },
    }
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return path


class TestVllmLoader:
    def test_valid(self, tmp_path):
        p = _minimal_vllm_yaml(tmp_path, "vllm")
        cfg = vllm_load(str(p), expected_mode="vllm")
        assert cfg.mode == "vllm"

    def test_rejects_lmdeploy_yaml(self, tmp_path):
        p = _minimal_vllm_yaml(tmp_path, "lmdeploy")
        with pytest.raises(ValueError, match="expects 'vllm'"):
            vllm_load(str(p), expected_mode="vllm")

    def test_rejects_missing_mode(self, tmp_path):
        import yaml as y
        cfg = {
            "pipeline": {"name": "t"},
            "docker": {"container_name": "t", "image": "x", "model": "m"},
            "evaluate": {
                "benchmarks": [], "model": "m", "run_name": "t",
                "bench_base_path": "/tmp", "output_path": "/tmp",
            },
            "submit": {"gradio_url": "x", "model_name": "t", "task_name": "t",
                       "datasets_used": "x", "results_base_dir": "r"},
        }
        p = tmp_path / "cfg.yaml"
        p.write_text(y.safe_dump(cfg), encoding="utf-8")
        with pytest.raises(ValueError, match="pipeline.mode is required"):
            vllm_load(str(p), expected_mode="vllm")

    def test_rejects_unsupported_mode(self, tmp_path):
        p = _minimal_vllm_yaml(tmp_path, "onnxruntime")
        with pytest.raises(ValueError, match="Unsupported pipeline.mode"):
            vllm_load(str(p), expected_mode="vllm")


class TestLmdeployLoader:
    def test_valid(self, tmp_path):
        p = _minimal_lmdeploy_yaml(tmp_path, "lmdeploy")
        cfg = lmdeploy_load(str(p), expected_mode="lmdeploy")
        assert cfg.mode == "lmdeploy"

    def test_rejects_vllm_yaml(self, tmp_path):
        p = _minimal_lmdeploy_yaml(tmp_path, "vllm")
        with pytest.raises(ValueError, match="expects 'lmdeploy'"):
            lmdeploy_load(str(p), expected_mode="lmdeploy")

    def test_no_guard_still_validates_mode(self, tmp_path):
        """expected_mode 미지정이어도 mode 필드 자체는 필수."""
        import yaml as y
        cfg = {
            "pipeline": {"name": "t"},
            "docker": {"container_name": "t", "image": "x", "model_path": "/m"},
            "evaluate": {
                "benchmarks": [], "model": "m", "run_name": "t",
                "bench_base_path": "/tmp", "output_path": "/tmp",
            },
            "submit": {"gradio_url": "x", "model_name": "t", "task_name": "t",
                       "datasets_used": "x", "results_base_dir": "r"},
        }
        p = tmp_path / "cfg.yaml"
        p.write_text(y.safe_dump(cfg), encoding="utf-8")
        with pytest.raises(ValueError, match="pipeline.mode is required"):
            lmdeploy_load(str(p))
