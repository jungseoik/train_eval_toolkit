"""
LMDeploy 파이프라인 단위 테스트.

실제 Docker/API 서버 없이 모듈 구조, config 로드, Docker 명령어 조립,
namespace 변환, parser 로직을 검증.
"""

import os
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml


# ============================================================
# Import 테스트
# ============================================================

class TestImports:
    """모듈 import 오류 없음 확인."""

    def test_import_config(self):
        from src.lmdeploy_pipeline.config import (
            DockerConfig, EvalConfig, PipelineConfig, SubmitConfig, load_pipeline_config,
        )

    def test_import_docker_manager(self):
        from src.lmdeploy_pipeline.docker_manager import (
            check_existing_container, start_container, stop_container, wait_for_ready,
        )

    def test_import_evaluator(self):
        from src.lmdeploy_pipeline.evaluator import run_evaluation

    def test_import_model_downloader(self):
        from src.lmdeploy_pipeline.model_downloader import ensure_model

    def test_import_runner(self):
        from src.lmdeploy_pipeline.runner import run_pipeline

    def test_import_cli(self):
        from src.lmdeploy_pipeline.cli import main

    def test_import_bench_eval(self):
        from src.evaluation.lmdeploy_bench_eval import (
            classify, evaluate_benchmark, get_category_from_bench,
            parse_model_output,
        )


# ============================================================
# Config 로드 테스트
# ============================================================

class TestConfig:
    """YAML -> PipelineConfig 변환 검증."""

    @pytest.fixture
    def sample_yaml(self, tmp_path):
        """테스트용 YAML 파일 생성."""
        config = {
            "pipeline": {
                "name": "Test Pipeline",
                "steps": {"docker": True, "evaluate": True, "submit": False},
                "cleanup_docker": True,
            },
            "retry": {"max_attempts": 2, "wait_seconds": 10},
            "docker": {
                "container_name": "test-container",
                "image": "openmmlab/lmdeploy:latest-cu12",
                "model_path": "/tmp/test-model",
                "container_model_path": "/model",
                "gpus": "all",
                "port": 23333,
                "lmdeploy_args": {"tp": 1, "session-len": 4096},
                "startup": {
                    "timeout_seconds": 120,
                    "poll_interval_seconds": 3,
                    "stream_logs": False,
                },
            },
            "evaluate": {
                "model": "/model",
                "run_name": "test-run",
                "api_base": "http://127.0.0.1:23333/v1",
                "bench_base_path": "/tmp/bench",
                "output_path": "/tmp/output",
                "benchmarks": ["PIA_Fire"],
                "window_size": 15,
                "concurrency": 5,
                "interpolation": "forward",
                "jpeg_quality": 90,
                "max_tokens": 10,
                "temperature": 0.0,
                "seed": 42,
                "negative_label": "normal",
                "prompt_templates": {"default": "test prompt"},
            },
            "submit": {
                "gradio_url": "http://localhost:7860/",
                "model_name": "test-model",
                "task_name": "test-task",
                "datasets_used": "test",
                "results_base_dir": "/tmp/results",
            },
        }
        yaml_path = tmp_path / "test_config.yaml"
        yaml_path.write_text(yaml.dump(config), encoding="utf-8")
        return yaml_path

    def test_load_pipeline_config(self, sample_yaml):
        from src.lmdeploy_pipeline.config import load_pipeline_config

        cfg = load_pipeline_config(str(sample_yaml))

        assert cfg.name == "Test Pipeline"
        assert cfg.steps["docker"] is True
        assert cfg.steps["submit"] is False
        assert cfg.retry_max_attempts == 2
        assert cfg.retry_wait_seconds == 10

    def test_docker_config_fields(self, sample_yaml):
        from src.lmdeploy_pipeline.config import load_pipeline_config

        cfg = load_pipeline_config(str(sample_yaml))
        docker = cfg.docker

        assert docker.container_name == "test-container"
        assert docker.image == "openmmlab/lmdeploy:latest-cu12"
        assert docker.model_path == "/tmp/test-model"
        assert docker.container_model_path == "/model"
        assert docker.port == 23333
        assert docker.lmdeploy_args == {"tp": 1, "session-len": 4096}
        assert docker.timeout_seconds == 120
        assert docker.stream_logs is False

    def test_eval_config_fields(self, sample_yaml):
        from src.lmdeploy_pipeline.config import load_pipeline_config

        cfg = load_pipeline_config(str(sample_yaml))
        ev = cfg.evaluate

        assert ev.model == "/model"
        assert ev.api_base == "http://127.0.0.1:23333/v1"
        assert ev.benchmarks == ["PIA_Fire"]
        assert ev.window_size == 15
        assert ev.seed == 42

    def test_config_not_found(self):
        from src.lmdeploy_pipeline.config import load_pipeline_config

        with pytest.raises(FileNotFoundError):
            load_pipeline_config("/nonexistent/config.yaml")


# ============================================================
# Docker 명령어 조립 테스트
# ============================================================

class TestDockerManager:
    """start_container() 가 올바른 docker run 명령을 생성하는지 검증."""

    def test_start_container_command(self, monkeypatch):
        """Docker 실행 없이 명령어 문자열만 검증."""
        from src.lmdeploy_pipeline.config import DockerConfig

        cfg = DockerConfig(
            container_name="test-lmdeploy",
            image="openmmlab/lmdeploy:latest-cu12",
            model_path="/data/models/InternVL3-2B",
            container_model_path="/model",
            gpus="all",
            port=23333,
            ipc="host",
            volumes=["~/.cache/huggingface:/root/.cache/huggingface"],
            lmdeploy_args={"tp": 2, "session-len": 8192, "backend": "pytorch"},
        )

        captured_cmd = []

        def mock_run(cmd, **kwargs):
            captured_cmd.extend(cmd)
            result = type("Result", (), {"returncode": 0, "stdout": "abc123\n", "stderr": ""})()
            return result

        monkeypatch.setattr("subprocess.run", mock_run)

        from src.lmdeploy_pipeline.docker_manager import start_container
        container_id = start_container(cfg)

        assert container_id == "abc123"
        cmd_str = " ".join(captured_cmd)

        # 필수 요소 확인
        assert "docker run -d" in cmd_str
        assert "--name test-lmdeploy" in cmd_str
        assert "--gpus all" in cmd_str
        assert "-p 23333:23333" in cmd_str
        assert "/data/models/InternVL3-2B:/model:ro" in cmd_str
        assert "openmmlab/lmdeploy:latest-cu12" in cmd_str
        assert "lmdeploy serve api_server /model" in cmd_str
        assert "--server-port 23333" in cmd_str
        assert "--tp 2" in cmd_str
        assert "--session-len 8192" in cmd_str
        assert "--backend pytorch" in cmd_str


# ============================================================
# Evaluator Namespace 변환 테스트
# ============================================================

class TestEvaluator:
    """EvalConfig -> SimpleNamespace 필드 매핑 확인."""

    def test_build_cfg_namespace(self):
        from src.lmdeploy_pipeline.config import EvalConfig
        from src.lmdeploy_pipeline.evaluator import _build_cfg_namespace

        eval_cfg = EvalConfig(
            benchmarks=["PIA_Fire", "Soil_Fire"],
            model="/model",
            run_name="test-run",
            api_base="http://127.0.0.1:23333/v1",
            bench_base_path="/bench",
            output_path="/output",
            window_size=15,
            concurrency=10,
            interpolation="forward",
            jpeg_quality=95,
            max_tokens=15,
            temperature=0.0,
            seed=0,
            negative_label="normal",
            prompt_templates={"default": "test"},
        )

        ns = _build_cfg_namespace(eval_cfg)

        assert ns.BENCHMARKS == ["PIA_Fire", "Soil_Fire"]
        assert ns.MODEL == "/model"
        assert ns.API_BASE == "http://127.0.0.1:23333/v1"
        assert ns.WINDOW_SIZE == 15
        assert ns.CONCURRENCY == 10
        assert ns.INTERPOLATION == "forward"
        assert ns.TEMPERATURE == 0.0
        assert ns.NEGATIVE_LABEL == "normal"
        assert ns.PROMPT_TEMPLATES == {"default": "test"}


# ============================================================
# Parser 테스트
# ============================================================

class TestParser:
    """parse_model_output() 와 classify() 동작 확인."""

    def test_parse_json_output(self):
        from src.evaluation.lmdeploy_bench_eval import parse_model_output

        raw = '{"category": "falldown", "description": "person lying on floor"}'
        result = parse_model_output(raw, ["falldown", "normal"])
        assert result == "falldown"

    def test_parse_markdown_codeblock(self):
        from src.evaluation.lmdeploy_bench_eval import parse_model_output

        raw = '```json\n{"category": "fire", "description": "flames visible"}\n```'
        result = parse_model_output(raw, ["fire", "normal"])
        assert result == "fire"

    def test_parse_normal_output(self):
        from src.evaluation.lmdeploy_bench_eval import parse_model_output

        raw = '{"category": "normal", "description": "no fire detected"}'
        result = parse_model_output(raw, ["fire", "normal"])
        assert result == "normal"

    def test_parse_no_json(self):
        from src.evaluation.lmdeploy_bench_eval import parse_model_output

        raw = "This is just plain text without any JSON"
        result = parse_model_output(raw, ["fire", "normal"])
        assert result == "no_json_found"

    def test_parse_regex_fallback(self):
        from src.evaluation.lmdeploy_bench_eval import parse_model_output

        raw = 'Some text with "category": "falldown" embedded'
        result = parse_model_output(raw, ["falldown", "normal"])
        assert result == "falldown"

    def test_classify_positive(self):
        from src.evaluation.lmdeploy_bench_eval import classify

        assert classify("falldown", "falldown") == 1

    def test_classify_negative(self):
        from src.evaluation.lmdeploy_bench_eval import classify

        assert classify("normal", "falldown") == 0

    def test_classify_error(self):
        from src.evaluation.lmdeploy_bench_eval import classify

        assert classify("no_json_found", "falldown") == 0
        assert classify("parsing_failed", "falldown") == 0


# ============================================================
# 유틸리티 테스트
# ============================================================

class TestUtils:
    """벤치마크명에서 카테고리 추출 등 유틸 검증."""

    def test_get_category_from_bench(self):
        from src.evaluation.lmdeploy_bench_eval import get_category_from_bench

        assert get_category_from_bench("PIA_Fire") == "fire"
        assert get_category_from_bench("Innodep_Falldown") == "falldown"
        assert get_category_from_bench("KhonKaen_Smoke") == "smoke"
        assert get_category_from_bench("GangNam_Violence") == "violence"
        assert get_category_from_bench("ABB_Sittingdown") == "sittingdown"


# ============================================================
# 모델 다운로더 테스트
# ============================================================

class TestModelDownloader:
    """ensure_model() 로직 검증 (실제 다운로드 없이)."""

    def test_model_exists(self, tmp_path):
        """모델이 이미 존재하면 그대로 경로 반환."""
        from src.lmdeploy_pipeline.config import DockerConfig
        from src.lmdeploy_pipeline.model_downloader import ensure_model

        model_dir = tmp_path / "test_model"
        model_dir.mkdir()
        (model_dir / "config.json").write_text("{}")

        cfg = DockerConfig(
            container_name="test",
            image="test:latest",
            model_path=str(model_dir),
        )

        result = ensure_model(cfg)
        assert result == str(model_dir)

    def test_model_missing_no_hf_repo_id(self, tmp_path):
        """모델이 없고 hf_repo_id도 없으면 FileNotFoundError."""
        from src.lmdeploy_pipeline.config import DockerConfig
        from src.lmdeploy_pipeline.model_downloader import ensure_model

        cfg = DockerConfig(
            container_name="test",
            image="test:latest",
            model_path=str(tmp_path / "nonexistent"),
        )

        with pytest.raises(FileNotFoundError, match="hf_repo_id"):
            ensure_model(cfg)

    def test_model_missing_with_hf_repo_id(self, tmp_path, monkeypatch):
        """모델이 없고 hf_repo_id가 있으면 snapshot_download 호출."""
        from src.lmdeploy_pipeline.config import DockerConfig
        from src.lmdeploy_pipeline.model_downloader import ensure_model

        model_dir = tmp_path / "downloaded_model"
        download_calls = []

        def mock_snapshot_download(repo_id, repo_type, local_dir):
            download_calls.append({"repo_id": repo_id, "local_dir": local_dir})
            Path(local_dir).mkdir(parents=True, exist_ok=True)
            (Path(local_dir) / "config.json").write_text("{}")

        monkeypatch.setattr(
            "src.lmdeploy_pipeline.model_downloader.snapshot_download",
            mock_snapshot_download,
            raising=False,
        )
        # snapshot_download가 이미 import 되어있을 수 있으므로 모듈 레벨에서도 패치
        import src.lmdeploy_pipeline.model_downloader as mdl
        monkeypatch.setattr(mdl, "_download_from_hf", lambda repo_id, target_path: (
            mock_snapshot_download(repo_id, "model", str(target_path)) or str(target_path)
        ))

        cfg = DockerConfig(
            container_name="test",
            image="test:latest",
            model_path=str(model_dir),
            hf_repo_id="PIA-SPACE-LAB/PIA_AI2team_VQA_falldown",
        )

        result = ensure_model(cfg)
        assert result == str(model_dir)
        assert len(download_calls) == 1
        assert download_calls[0]["repo_id"] == "PIA-SPACE-LAB/PIA_AI2team_VQA_falldown"

    def test_is_valid_model_dir(self, tmp_path):
        """config.json이 있어야 유효한 모델 디렉토리."""
        from src.lmdeploy_pipeline.model_downloader import _is_valid_model_dir

        # 빈 디렉토리
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        assert _is_valid_model_dir(empty_dir) is False

        # config.json이 있는 디렉토리
        valid_dir = tmp_path / "valid"
        valid_dir.mkdir()
        (valid_dir / "config.json").write_text("{}")
        assert _is_valid_model_dir(valid_dir) is True

        # 존재하지 않는 디렉토리
        assert _is_valid_model_dir(tmp_path / "nonexistent") is False

    def test_config_hf_repo_id_default(self, tmp_path):
        """hf_repo_id 미지정 시 빈 문자열."""
        config = {
            "pipeline": {"name": "Test", "steps": {"docker": True, "evaluate": True, "submit": False}, "cleanup_docker": True},
            "retry": {"max_attempts": 1, "wait_seconds": 5},
            "docker": {
                "container_name": "test",
                "image": "test:latest",
                "model_path": "/tmp/model",
                "startup": {},
            },
            "evaluate": {
                "model": "/model", "run_name": "test", "api_base": "http://localhost:23333/v1",
                "bench_base_path": "/tmp", "output_path": "/tmp",
            },
            "submit": {},
        }
        yaml_path = tmp_path / "test.yaml"
        yaml_path.write_text(yaml.dump(config), encoding="utf-8")

        from src.lmdeploy_pipeline.config import load_pipeline_config
        cfg = load_pipeline_config(str(yaml_path))
        assert cfg.docker.hf_repo_id == ""

    def test_config_hf_repo_id_set(self, tmp_path):
        """YAML에서 hf_repo_id 설정 시 정상 로드."""
        config = {
            "pipeline": {"name": "Test", "steps": {"docker": True, "evaluate": True, "submit": False}, "cleanup_docker": True},
            "retry": {"max_attempts": 1, "wait_seconds": 5},
            "docker": {
                "container_name": "test",
                "image": "test:latest",
                "model_path": "ckpts/MyModel",
                "hf_repo_id": "org/my-model",
                "startup": {},
            },
            "evaluate": {
                "model": "/model", "run_name": "test", "api_base": "http://localhost:23333/v1",
                "bench_base_path": "/tmp", "output_path": "/tmp",
            },
            "submit": {},
        }
        yaml_path = tmp_path / "test.yaml"
        yaml_path.write_text(yaml.dump(config), encoding="utf-8")

        from src.lmdeploy_pipeline.config import load_pipeline_config
        cfg = load_pipeline_config(str(yaml_path))
        assert cfg.docker.hf_repo_id == "org/my-model"
