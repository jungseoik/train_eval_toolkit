"""공통 import 검증 헬퍼."""

import importlib
import pytest


def can_import(module: str) -> bool:
    """모듈 import 가능 여부 반환."""
    try:
        importlib.import_module(module)
        return True
    except ImportError:
        return False


def skip_if_missing(module: str):
    """해당 모듈이 없으면 pytest.skip."""
    return pytest.mark.skipif(
        not can_import(module),
        reason=f"{module} not installed (GPU build or optional dependency)",
    )
