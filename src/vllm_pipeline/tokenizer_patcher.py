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
    2. 파일에 `"tokenizer_class": "TokenizersBackend"` 가 존재하면 사전 매핑
       테이블에 따라 교체(기본 `"Qwen2Tokenizer"`). 이미 교체 상태이면 skip.
    3. blob 파일이 root 소유(Docker가 생성한 경우)이면 `docker run --rm alpine sed`
       로 패치. 일반 유저 소유이면 Python이 직접 치환.

멱등하므로 여러 번 호출해도 안전하다. 파일이 캐시에 없으면 `not_in_cache`
반환하며, 실제 다운로드는 vLLM 컨테이너 기동 시 수행되고 그 이후 재호출 시
패치된다.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any

TARGET_TOKEN = '"tokenizer_class": "TokenizersBackend"'
# 대상 모델군 확장 시 여기에 매핑 추가.
DEFAULT_REPLACEMENT = '"tokenizer_class": "Qwen2Tokenizer"'


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
    replacement: str = DEFAULT_REPLACEMENT,
) -> dict[str, Any]:
    """
    Unsloth tokenizer_class 잘못 저장된 HF 캐시 blob을 패치(멱등).

    Returns:
        {"patched": bool, "reason": str, "file": str | None}
        reason: "not_in_cache" | "already_ok" | "patched"
    """
    blob = _resolve_blob(model_id)
    if blob is None:
        return {"patched": False, "reason": "not_in_cache", "file": None}

    text = blob.read_text(encoding="utf-8", errors="replace")
    if TARGET_TOKEN not in text:
        return {"patched": False, "reason": "already_ok", "file": str(blob)}

    if _is_root_owned(blob):
        _patch_with_docker(blob, replacement)
    else:
        _patch_with_python(blob, replacement)

    # 패치 후 재검증
    text_after = blob.read_text(encoding="utf-8", errors="replace")
    if TARGET_TOKEN in text_after:
        return {"patched": False, "reason": "patch_failed", "file": str(blob)}

    return {"patched": True, "reason": "patched", "file": str(blob)}
