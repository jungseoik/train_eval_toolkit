"""src/vllm_pipeline/tokenizer_patcher.py 단위 테스트."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from src.vllm_pipeline import tokenizer_patcher as tp


UNSLOTH_TOKENIZER_CONFIG = {
    "tokenizer_class": "TokenizersBackend",
    "backend": "tokenizers",
    "pad_token": "<|endoftext|>",
    "processor_class": "Qwen3VLProcessor",
}

OFFICIAL_TOKENIZER_CONFIG = {
    "tokenizer_class": "Qwen2Tokenizer",
    "pad_token": "<|endoftext|>",
}


def _write_json(path: Path, obj: dict) -> Path:
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")
    return path


class TestPatchTokenizerConfig:
    def test_not_in_cache(self):
        """모델이 캐시에 없을 때 not_in_cache 반환."""
        with patch.object(tp, "_resolve_blob", return_value=None):
            result = tp.patch_tokenizer_config("nonexistent/model")
        assert result["patched"] is False
        assert result["reason"] == "not_in_cache"
        assert result["file"] is None

    def test_already_ok(self, tmp_path):
        """공식 모델처럼 `tokenizer_class: Qwen2Tokenizer`인 경우 skip."""
        blob = _write_json(tmp_path / "tokenizer_config.json", OFFICIAL_TOKENIZER_CONFIG)
        with patch.object(tp, "_resolve_blob", return_value=blob):
            result = tp.patch_tokenizer_config("official/model")
        assert result["patched"] is False
        assert result["reason"] == "already_ok"

    def test_qwen3_5_patch_user_owned(self, tmp_path):
        """qwen3_5 model_type은 allowlist에 있으므로 Qwen2Tokenizer로 패치."""
        blob = _write_json(tmp_path / "tokenizer_config.json", UNSLOTH_TOKENIZER_CONFIG)
        with patch.object(tp, "_resolve_blob", return_value=blob), \
             patch.object(tp, "_resolve_model_type", return_value="qwen3_5"), \
             patch.object(tp, "_is_root_owned", return_value=False):
            result = tp.patch_tokenizer_config("unsloth/qwen3.5-finetune")
        assert result["patched"] is True
        assert result["reason"] == "patched"
        assert result["model_type"] == "qwen3_5"
        assert result["tokenizer_class"] == "Qwen2Tokenizer"
        text = blob.read_text(encoding="utf-8")
        assert '"tokenizer_class": "Qwen2Tokenizer"' in text
        assert "TokenizersBackend" not in text

    def test_idempotent(self, tmp_path):
        """두 번째 호출은 already_ok."""
        blob = _write_json(tmp_path / "tokenizer_config.json", UNSLOTH_TOKENIZER_CONFIG)
        with patch.object(tp, "_resolve_blob", return_value=blob), \
             patch.object(tp, "_resolve_model_type", return_value="qwen3_5"), \
             patch.object(tp, "_is_root_owned", return_value=False):
            first = tp.patch_tokenizer_config("unsloth/model")
            second = tp.patch_tokenizer_config("unsloth/model")
        assert first["reason"] == "patched"
        assert second["reason"] == "already_ok"

    def test_unsupported_model_type_does_not_patch(self, tmp_path):
        """allowlist에 없는 model_type이면 건드리지 않는다."""
        blob = _write_json(tmp_path / "tokenizer_config.json", UNSLOTH_TOKENIZER_CONFIG)
        with patch.object(tp, "_resolve_blob", return_value=blob), \
             patch.object(tp, "_resolve_model_type", return_value="llama3"), \
             patch.object(tp, "_is_root_owned", return_value=False):
            result = tp.patch_tokenizer_config("unsloth/some-llama-finetune")
        assert result["patched"] is False
        assert result["reason"] == "unsupported_model_type"
        assert result["model_type"] == "llama3"
        assert result["tokenizer_class"] is None
        text = blob.read_text(encoding="utf-8")
        assert "TokenizersBackend" in text

    def test_model_type_unknown_does_not_patch(self, tmp_path):
        """config.json 누락 등으로 model_type 판독 실패 시 건드리지 않는다."""
        blob = _write_json(tmp_path / "tokenizer_config.json", UNSLOTH_TOKENIZER_CONFIG)
        with patch.object(tp, "_resolve_blob", return_value=blob), \
             patch.object(tp, "_resolve_model_type", return_value=None), \
             patch.object(tp, "_is_root_owned", return_value=False):
            result = tp.patch_tokenizer_config("unknown/model")
        assert result["patched"] is False
        assert result["reason"] == "model_type_unknown"
        assert result["model_type"] is None
        text = blob.read_text(encoding="utf-8")
        assert "TokenizersBackend" in text

    def test_custom_mapping_override(self, tmp_path):
        """model_type_to_tokenizer 인자로 매핑을 덮어쓰는 긴급 대응."""
        blob = _write_json(tmp_path / "tokenizer_config.json", UNSLOTH_TOKENIZER_CONFIG)
        override = {"custom_family": "LlamaTokenizerFast"}
        with patch.object(tp, "_resolve_blob", return_value=blob), \
             patch.object(tp, "_resolve_model_type", return_value="custom_family"), \
             patch.object(tp, "_is_root_owned", return_value=False):
            result = tp.patch_tokenizer_config(
                "custom/model", model_type_to_tokenizer=override,
            )
        assert result["patched"] is True
        assert result["tokenizer_class"] == "LlamaTokenizerFast"
        assert '"tokenizer_class": "LlamaTokenizerFast"' in blob.read_text(encoding="utf-8")

    def test_default_mapping_contains_qwen3_5(self):
        """문서화된 기본 allowlist에 qwen3_5가 포함되어 있어야 한다."""
        assert tp.MODEL_TYPE_TO_TOKENIZER.get("qwen3_5") == "Qwen2Tokenizer"

    @pytest.mark.integration
    def test_real_qwen_cache_is_already_ok(self):
        """실제 환경에 Qwen/Qwen3.5-2B가 캐시돼 있으면 already_ok여야 함 (false positive 방지)."""
        pytest.importorskip("huggingface_hub")
        result = tp.patch_tokenizer_config("Qwen/Qwen3.5-2B")
        assert result["patched"] is False
        assert result["reason"] in ("already_ok", "not_in_cache")
