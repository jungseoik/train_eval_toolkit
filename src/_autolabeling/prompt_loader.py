import yaml
from pathlib import Path

PROMPTS_DIR = Path(__file__).resolve().parents[2] / "configs" / "prompts"


def load_all_prompts() -> dict[tuple[str, str], str]:
    """configs/prompts/*.yaml 파일을 모두 읽어 (option, mode) -> prompt 딕셔너리를 반환합니다."""
    prompt_map: dict[tuple[str, str], str] = {}
    for yaml_file in sorted(PROMPTS_DIR.glob("*.yaml")):
        with open(yaml_file, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        for key, prompt_text in data.get("prompts", {}).items():
            option, mode = key.split("__")
            prompt_map[(option, mode)] = prompt_text
    return prompt_map
