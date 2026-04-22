"""src/api/pipeline_worker.py 의 duals mode 디스패치 검증."""

from __future__ import annotations

import pytest

from src.api.pipeline_worker import (
    SUPPORTED_MODES,
    _get_error_hint,
    _load_backend,
    validate_yaml,
)


def _yaml_fragment(mode: str, docker_key: str = "model", docker_val: str = "Qwen/Qwen3.5-2B") -> str:
    return f"""
pipeline:
  name: test
  mode: {mode}
  steps:
    docker: true
    evaluate: true
    submit: false
docker:
  container_name: test
  image: vllm/vllm-openai:cu130-nightly
  {docker_key}: "{docker_val}"
evaluate:
  bench_base_path: /tmp
  benchmarks: []
  model: {docker_val}
  run_name: test
  output_path: /tmp
  window_size: 30
  concurrency: 1
  interpolation: forward
  jpeg_quality: 95
  max_tokens: 1
  temperature: 0.0
  seed: 0
  negative_label: normal
"""


class TestValidateYaml:
    def test_valid_vllm(self):
        ok, result = validate_yaml(_yaml_fragment("vllm", "model", "Qwen/Qwen3.5-2B"))
        assert ok
        assert result["pipeline"]["mode"] == "vllm"

    def test_valid_lmdeploy(self):
        ok, result = validate_yaml(_yaml_fragment("lmdeploy", "model_path", "/tmp/m"))
        assert ok
        assert result["pipeline"]["mode"] == "lmdeploy"

    def test_missing_mode(self):
        yaml = "pipeline:\n  name: t\ndocker:\n  model: x\nevaluate:\n  bench_base_path: /tmp\n"
        ok, msg = validate_yaml(yaml)
        assert not ok
        assert "pipeline.mode" in msg

    def test_unsupported_mode(self):
        ok, msg = validate_yaml(_yaml_fragment("triton"))
        assert not ok
        assert "지원되지 않습니다" in msg

    def test_vllm_requires_model_not_model_path(self):
        ok, msg = validate_yaml(_yaml_fragment("vllm", "model_path", "/tmp/m"))
        assert not ok
        assert "docker.model" in msg

    def test_lmdeploy_requires_model_path_not_model(self):
        ok, msg = validate_yaml(_yaml_fragment("lmdeploy", "model", "repo/x"))
        assert not ok
        assert "docker.model_path" in msg

    def test_missing_sections(self):
        ok, msg = validate_yaml("pipeline:\n  name: t\n  mode: vllm\n")
        assert not ok
        assert "필수 섹션 누락" in msg


class TestLoadBackend:
    @pytest.mark.parametrize("mode", SUPPORTED_MODES)
    def test_known_mode_returns_required_keys(self, mode):
        backend = _load_backend(mode)
        expected = {
            "mode", "load_pipeline_config", "check_existing_container",
            "start_container", "stop_container", "wait_for_ready",
            "run_evaluation", "submit_results", "model_step", "server_label",
        }
        assert expected.issubset(set(backend.keys()))
        assert backend["mode"] == mode

    def test_unknown_mode(self):
        with pytest.raises(ValueError, match="Unsupported pipeline.mode"):
            _load_backend("garbage")

    def test_server_label_differs(self):
        assert _load_backend("vllm")["server_label"] == "vLLM"
        assert _load_backend("lmdeploy")["server_label"] == "LMDeploy"


class TestErrorHint:
    def test_oom_hint_differs_by_mode(self):
        err = RuntimeError("CUDA out of memory")
        vllm_hint = _get_error_hint(err, mode="vllm")
        lmdeploy_hint = _get_error_hint(err, mode="lmdeploy")
        assert "vllm_args" in vllm_hint
        assert "lmdeploy_args" in lmdeploy_hint

    def test_tokenizer_hint(self):
        err = ValueError("Tokenizer class TokenizersBackend does not exist")
        hint = _get_error_hint(err, mode="vllm")
        assert "Unsloth" in hint or "tokenizer" in hint.lower()

    def test_fallback_hint(self):
        err = RuntimeError("totally generic error")
        hint = _get_error_hint(err, mode="vllm")
        assert "docs/eval" in hint
