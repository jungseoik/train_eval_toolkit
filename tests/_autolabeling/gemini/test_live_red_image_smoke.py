import os
from pathlib import Path

import pytest
from PIL import Image


pytestmark = pytest.mark.integration


@pytest.mark.skipif(
    os.getenv("RUN_LIVE_GEMINI_TESTS") != "1",
    reason="RUN_LIVE_GEMINI_TESTS 없음 또는 '1' 아님 (라이브 테스트 비활성).",
)
def test_live_gemini_red_image_smoke(tmp_path: Path) -> None:
    if not os.getenv("GEMINI_API_KEY"):
        pytest.skip("GEMINI_API_KEY 없음 (라이브 Gemini 테스트 불가).")
    try:
        from src._autolabeling.gemini.client import GeminiClient
        from src.utils.json_parser import parse_json_from_response
    except ImportError as exc:
        pytest.skip(f"런타임 의존성 없음: {exc}")

    image_path = tmp_path / "red_image.png"
    Image.new("RGB", (256, 256), (255, 0, 0)).save(image_path)

    prompt = (
        "Look at this image and respond with exactly one JSON object.\n"
        '- Use only keys: "category" and "description".\n'
        '- Set "category" to "red" if the dominant color is red, otherwise "not_red".\n'
        '- Keep "description" short.\n'
    )

    client = GeminiClient(
        model_name=os.getenv("GEMINI_MODEL_NAME", "gemini-2.0-flash"),
    )

    response_text = client.analyze_image(str(image_path), custom_prompt=prompt)
    print("\n=== GEMINI RAW RESPONSE ===")
    print(response_text)

    assert response_text.strip(), "Gemini returned an empty response."
    assert not response_text.startswith("❌"), f"Gemini client error: {response_text}"

    parsed = parse_json_from_response(response_text)
    print("\n=== PARSED JSON ===")
    print(parsed)

    assert parsed is not None, "Could not parse JSON from Gemini response."
