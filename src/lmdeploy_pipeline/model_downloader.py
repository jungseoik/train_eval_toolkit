"""
HuggingFace 모델 다운로드 모듈.

파이프라인 실행 전 모델이 로컬에 존재하는지 확인하고,
없으면 HuggingFace Hub에서 다운로드.
"""

from __future__ import annotations

from pathlib import Path

from .config import DockerConfig


WEIGHT_EXTENSIONS = (".safetensors", ".bin")


def _is_valid_model_dir(model_path: Path) -> bool:
    """모델 디렉토리가 유효한지 확인 (config.json + 가중치 파일 존재 여부로 판단)."""
    if not model_path.is_dir():
        return False
    if not (model_path / "config.json").exists():
        return False
    # 가중치 파일이 최소 1개 이상 존재해야 함
    has_weights = any(
        f.suffix in WEIGHT_EXTENSIONS
        for f in model_path.iterdir()
        if f.is_file()
    )
    return has_weights


def ensure_model(docker_cfg: DockerConfig) -> str:
    """
    모델이 로컬에 존재하는지 확인하고, 없으면 HuggingFace에서 다운로드.

    Returns:
        실제 모델 경로 (str)

    Raises:
        FileNotFoundError: 모델이 없고 hf_repo_id도 미지정인 경우
        RuntimeError: 다운로드 실패
    """
    model_path = Path(docker_cfg.model_path)

    # 1. 모델이 이미 존재하면 그대로 사용
    if _is_valid_model_dir(model_path):
        print(f"[MODEL] 모델 확인 완료: {model_path}")
        return str(model_path)

    # 2. 모델이 없는데 hf_repo_id도 없으면 에러
    if not docker_cfg.hf_repo_id:
        raise FileNotFoundError(
            f"모델 경로에 유효한 모델이 없습니다: {model_path}\n"
            f"해결 방법:\n"
            f"  1. 모델을 직접 다운로드하여 '{model_path}'에 배치\n"
            f"  2. YAML에 docker.hf_repo_id 필드를 추가하여 자동 다운로드 활성화\n"
            f"     예: hf_repo_id: \"PIA-SPACE-LAB/PIA_AI2team_VQA_falldown\""
        )

    # 3. HuggingFace에서 다운로드
    print(f"[MODEL] 모델이 '{model_path}'에 없습니다. HuggingFace에서 다운로드합니다.")
    print(f"[MODEL] HuggingFace repo: {docker_cfg.hf_repo_id}")

    return _download_from_hf(docker_cfg.hf_repo_id, model_path)


def _download_from_hf(repo_id: str, target_path: Path) -> str:
    """
    HuggingFace Hub에서 모델을 다운로드.

    huggingface_hub.snapshot_download를 사용하여 cache 없이 직접 target_path에 저장.
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise RuntimeError(
            "huggingface_hub 패키지가 설치되어 있지 않습니다.\n"
            "설치: pip install huggingface_hub"
        )

    target_path.mkdir(parents=True, exist_ok=True)

    print(f"[MODEL] 다운로드 시작: {repo_id} -> {target_path}")

    try:
        snapshot_download(
            repo_id=repo_id,
            repo_type="model",
            local_dir=str(target_path),
        )
    except Exception as e:
        raise RuntimeError(
            f"HuggingFace 모델 다운로드 실패: {repo_id}\n"
            f"원인: {e}\n"
            f"확인 사항:\n"
            f"  - HuggingFace 로그인 여부: hf auth login\n"
            f"  - repo ID가 올바른지 확인: {repo_id}\n"
            f"  - 네트워크 연결 상태 확인"
        ) from e

    # 다운로드 후 유효성 검증
    _validate_model_dir(target_path, repo_id)
    print(f"[MODEL] 다운로드 완료: {target_path}")

    return str(target_path)


def _validate_model_dir(model_path: Path, repo_id: str) -> None:
    """다운로드된 모델 디렉토리의 필수 파일을 검증한다.

    Raises:
        RuntimeError: config.json 또는 가중치 파일이 없을 때
    """
    has_config = (model_path / "config.json").exists()
    has_weights = any(
        f.suffix in WEIGHT_EXTENSIONS
        for f in model_path.iterdir()
        if f.is_file()
    )

    missing = []
    if not has_config:
        missing.append("config.json")
    if not has_weights:
        missing.append(f"가중치 파일 ({', '.join(WEIGHT_EXTENSIONS)})")

    if missing:
        raise RuntimeError(
            f"HuggingFace에서 다운로드한 모델이 불완전합니다: {repo_id}\n"
            f"누락 파일: {', '.join(missing)}\n"
            f"모델 경로: {model_path}\n"
            f"확인 사항:\n"
            f"  - HuggingFace 저장소에 모델 파일이 모두 업로드되었는지 확인\n"
            f"  - 저장소 주소: https://huggingface.co/{repo_id}"
        )
