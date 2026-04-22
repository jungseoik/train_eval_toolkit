"""
Unsloth로 파인튜닝한 모델의 tokenizer_config.json 자동 패치.

배경:
    Unsloth의 save_pretrained는 `tokenizer_config.json` 의 `tokenizer_class` 필드에
    `"TokenizersBackend"` 를 기록한다. 공식 Qwen 계열은 `"Qwen2Tokenizer"` 로 저장되어
    있기 때문에 이 값이 남아 있으면 vLLM 컨테이너의 `AutoTokenizer.from_pretrained`
    호출이 `ValueError: Tokenizer class TokenizersBackend does not exist or is not
    currently imported.` 로 실패한다.

이 모듈은 vLLM 파이프라인의 MODEL 단계에서 다음을 수행한다:
    1. 대상 모델의 HF 캐시에서 `tokenizer_config.json` blob 파일을 찾는다.
    2. 파일에 `"tokenizer_class": "TokenizersBackend"` 가 존재하면, 같은 snapshot의
       `config.json` 에서 `model_type`을 읽어 `MODEL_TYPE_TO_TOKENIZER` allowlist와
       비교한다. 허용된 경우에만 해당 tokenizer_class로 교체.
    3. 허용되지 않은 model_type은 건드리지 않고 `unsupported_model_type`으로 보고 →
       운영자가 매핑을 확장하거나 수동 대응할 수 있게 한다.
    4. blob 파일이 root 소유(Docker가 생성한 경우)이면 `docker run --rm alpine sed`
       로 패치. 일반 유저 소유이면 Python이 직접 치환.

멱등하므로 여러 번 호출해도 안전하다. 파일이 캐시에 없으면 `not_in_cache`
반환하며, 실제 다운로드는 vLLM 컨테이너 기동 시 수행되고 그 이후 재호출 시
패치된다.

새 모델군(예: 다른 base의 Unsloth 파인튜닝) 확장 시:
    `MODEL_TYPE_TO_TOKENIZER`에 `"<model_type>": "<TokenizerClass>"` 한 줄 추가.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

TARGET_TOKEN = '"tokenizer_class": "TokenizersBackend"'

# model_type → 치환 대상 tokenizer_class 매핑 (allowlist).
# 허용되지 않은 model_type은 건드리지 않는다 (안전 기본값).
# 새 케이스 등장 시 여기에 한 줄 추가.
MODEL_TYPE_TO_TOKENIZER: dict[str, str] = {
    "qwen3_5": "Qwen2Tokenizer",
}


def _resolve_blob(model_id: str) -> Path | None:
    """HF 캐시의 tokenizer_config.json blob 경로를 반환. 없으면 None."""
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return None
    path = try_to_load_from_cache(repo_id=model_id, filename="tokenizer_config.json")
    if not path:
        return None
    p = Path(path).resolve()
    return p if p.exists() else None


def _resolve_model_type(model_id: str) -> str | None:
    """같은 snapshot의 config.json에서 model_type을 읽어 반환. 실패 시 None."""
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return None
    path = try_to_load_from_cache(repo_id=model_id, filename="config.json")
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    value = data.get("model_type")
    return value if isinstance(value, str) else None


def _is_root_owned(path: Path) -> bool:
    try:
        return path.stat().st_uid == 0 and os.getuid() != 0
    except FileNotFoundError:
        return False


def _patch_with_docker(blob: Path, replacement: str) -> None:
    """root 소유 blob 파일을 docker run --rm alpine sed로 치환."""
    blob_dir = blob.parent
    cmd = [
        "docker", "run", "--rm",
        "-v", f"{blob_dir}:/w",
        "alpine",
        "sh", "-c",
        f'sed -i \'s/"tokenizer_class": "TokenizersBackend"/{replacement.replace("/", r"\\/")}/\' /w/{blob.name}',
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=60)


def _patch_with_python(blob: Path, replacement: str) -> None:
    text = blob.read_text(encoding="utf-8")
    new_text = text.replace(TARGET_TOKEN, replacement)
    blob.write_text(new_text, encoding="utf-8")


def patch_tokenizer_config(
    model_id: str,
    model_type_to_tokenizer: dict[str, str] | None = None,
) -> dict[str, Any]:
    """
    Unsloth tokenizer_class 잘못 저장된 HF 캐시 blob을 패치(멱등).

    Args:
        model_id: HuggingFace repo ID.
        model_type_to_tokenizer: 선택 — `MODEL_TYPE_TO_TOKENIZER`를 덮어쓸 매핑.
            테스트/긴급 대응용. None이면 모듈 기본 매핑 사용.

    Returns:
        {"patched": bool, "reason": str, "file": str | None, "model_type": str | None,
         "tokenizer_class": str | None}
        reason: "not_in_cache" | "already_ok" | "patched" | "patch_failed"
              | "unsupported_model_type" | "model_type_unknown"
    """
    mapping = model_type_to_tokenizer if model_type_to_tokenizer is not None else MODEL_TYPE_TO_TOKENIZER

    blob = _resolve_blob(model_id)
    if blob is None:
        return {
            "patched": False, "reason": "not_in_cache", "file": None,
            "model_type": None, "tokenizer_class": None,
        }

    text = blob.read_text(encoding="utf-8", errors="replace")
    if TARGET_TOKEN not in text:
        return {
            "patched": False, "reason": "already_ok", "file": str(blob),
            "model_type": None, "tokenizer_class": None,
        }

    # TokenizersBackend 감지됨 → model_type 기반 allowlist 검증
    model_type = _resolve_model_type(model_id)
    if model_type is None:
        return {
            "patched": False, "reason": "model_type_unknown", "file": str(blob),
            "model_type": None, "tokenizer_class": None,
        }

    target_tokenizer = mapping.get(model_type)
    if target_tokenizer is None:
        return {
            "patched": False, "reason": "unsupported_model_type", "file": str(blob),
            "model_type": model_type, "tokenizer_class": None,
        }

    replacement = f'"tokenizer_class": "{target_tokenizer}"'

    if _is_root_owned(blob):
        _patch_with_docker(blob, replacement)
    else:
        _patch_with_python(blob, replacement)

    text_after = blob.read_text(encoding="utf-8", errors="replace")
    if TARGET_TOKEN in text_after:
        return {
            "patched": False, "reason": "patch_failed", "file": str(blob),
            "model_type": model_type, "tokenizer_class": target_tokenizer,
        }

    return {
        "patched": True, "reason": "patched", "file": str(blob),
        "model_type": model_type, "tokenizer_class": target_tokenizer,
    }
