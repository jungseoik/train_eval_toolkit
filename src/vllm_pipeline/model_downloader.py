"""
vLLM 파이프라인용 모델 사전 확보 모듈.

LMDeploy 쪽은 로컬 `model_path` 볼륨 마운트 방식인 반면, vLLM은
HuggingFace 캐시에서 직접 로드한다(`vllm serve <repo_id>`). 여기서는
다음 두 가지를 수행한다:

1. `docker.hf_repo_id`가 지정되어 있으면 `snapshot_download`로 HF 캐시에 선제 다운로드.
2. 지정되지 않았으면 `docker.model`이 HF 캐시에 이미 있는지 확인만 수행(없어도 실패하지 않음
   — 실제 다운로드는 vLLM 컨테이너 기동 시 수행되므로 정보 로그만 남긴다).

반환값은 {"downloaded": bool, "repo_id": str, "cache_dir": str | None}.
"""

from __future__ import annotations

from pathlib import Path

from .config import DockerConfig


def _repo_id_for_check(docker_cfg: DockerConfig) -> str:
    """캐시/다운로드에 사용할 HF repo ID 결정. hf_repo_id 우선, 없으면 docker.model."""
    return docker_cfg.hf_repo_id or docker_cfg.model


def _is_cached(repo_id: str) -> bool:
    """HF 캐시에 해당 repo의 config.json이 이미 존재하는지."""
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return False
    path = try_to_load_from_cache(repo_id=repo_id, filename="config.json")
    return bool(path) and Path(path).exists()


def ensure_model(docker_cfg: DockerConfig) -> dict:
    """
    vLLM 모델을 HuggingFace 캐시에 확보.

    Returns:
        {"downloaded": bool, "repo_id": str, "cache_dir": str | None}

    Raises:
        RuntimeError: hf_repo_id가 지정되어 있고 다운로드에 실패한 경우.
    """
    repo_id = _repo_id_for_check(docker_cfg)
    result = {"downloaded": False, "repo_id": repo_id, "cache_dir": None}

    # 로컬 경로처럼 보이면(슬래시 포함 외 절대/상대 경로) 스킵
    if repo_id.startswith("/") or repo_id.startswith("."):
        print(f"[MODEL] Local path detected, skipping HF download: {repo_id}")
        return result

    # 이미 캐시된 경우
    if _is_cached(repo_id):
        print(f"[MODEL] HF cache hit: {repo_id}")
        return result

    # hf_repo_id 미지정 → vLLM 컨테이너에 위임
    if not docker_cfg.hf_repo_id:
        print(
            f"[MODEL] '{repo_id}' not in HF cache and no hf_repo_id specified. "
            f"vLLM container will download on startup."
        )
        return result

    # 선제 다운로드
    print(f"[MODEL] Pre-downloading from HF: {repo_id}")
    cache_dir = _snapshot_download(repo_id)
    result["downloaded"] = True
    result["cache_dir"] = cache_dir
    print(f"[MODEL] Download complete: {cache_dir}")
    return result


def _snapshot_download(repo_id: str) -> str:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as e:
        raise RuntimeError(
            "huggingface_hub 패키지가 설치되어 있지 않습니다.\n"
            "설치: pip install huggingface_hub"
        ) from e
    try:
        return snapshot_download(repo_id=repo_id, repo_type="model")
    except Exception as e:
        raise RuntimeError(
            f"HuggingFace 모델 다운로드 실패: {repo_id}\n"
            f"원인: {e}\n"
            f"확인 사항:\n"
            f"  - HuggingFace 로그인 여부: hf auth login\n"
            f"  - repo ID가 올바른지 확인: {repo_id}\n"
            f"  - 네트워크 연결 상태 확인"
        ) from e
