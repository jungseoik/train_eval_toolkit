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

    def test_unsloth_patch_user_owned(self, tmp_path):
        """유저 소유 blob은 Python sed 경로로 패치."""
        blob = _write_json(tmp_path / "tokenizer_config.json", UNSLOTH_TOKENIZER_CONFIG)
        with patch.object(tp, "_resolve_blob", return_value=blob), \
             patch.object(tp, "_is_root_owned", return_value=False):
            result = tp.patch_tokenizer_config("unsloth/model")
        assert result["patched"] is True
        assert result["reason"] == "patched"
        # 파일이 실제로 교체됐는지
        text = blob.read_text(encoding="utf-8")
        assert '"tokenizer_class": "Qwen2Tokenizer"' in text
        assert "TokenizersBackend" not in text

    def test_idempotent(self, tmp_path):
        """두 번째 호출은 already_ok."""
        blob = _write_json(tmp_path / "tokenizer_config.json", UNSLOTH_TOKENIZER_CONFIG)
        with patch.object(tp, "_resolve_blob", return_value=blob), \
             patch.object(tp, "_is_root_owned", return_value=False):
            first = tp.patch_tokenizer_config("unsloth/model")
            second = tp.patch_tokenizer_config("unsloth/model")
        assert first["reason"] == "patched"
        assert second["reason"] == "already_ok"

    def test_custom_replacement(self, tmp_path):
        """replacement 인자로 다른 tokenizer_class 지정 가능."""
        blob = _write_json(tmp_path / "tokenizer_config.json", UNSLOTH_TOKENIZER_CONFIG)
        custom = '"tokenizer_class": "LlamaTokenizerFast"'
        with patch.object(tp, "_resolve_blob", return_value=blob), \
             patch.object(tp, "_is_root_owned", return_value=False):
            result = tp.patch_tokenizer_config("unsloth/model", replacement=custom)
        assert result["patched"] is True
        assert "LlamaTokenizerFast" in blob.read_text(encoding="utf-8")

    @pytest.mark.integration
    def test_real_qwen_cache_is_already_ok(self):
        """실제 환경에 Qwen/Qwen3.5-2B가 캐시돼 있으면 already_ok여야 함 (false positive 방지)."""
        pytest.importorskip("huggingface_hub")
        result = tp.patch_tokenizer_config("Qwen/Qwen3.5-2B")
        # 캐시가 없을 수도 있으므로 둘 다 허용하되 patched는 절대 아님
        assert result["patched"] is False
        assert result["reason"] in ("already_ok", "not_in_cache")
