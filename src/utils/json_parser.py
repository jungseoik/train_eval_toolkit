import json
import re
from typing import Optional, Dict, Any


def parse_json_from_response(response_text: str) -> Optional[Dict[str, Any]]:
    """API 응답 텍스트에서 JSON 객체를 파싱하여 딕셔너리로 반환합니다.

    다단계 파싱 전략을 사용합니다.
    1. 마크다운 코드 블록(```json ... ```)을 우선적으로 탐색합니다.
    2. 마크다운이 없을 경우, 텍스트에서 첫 '{'와 마지막 '}' 사이를 추출합니다.

    Args:
        response_text (str): Gemini API로부터 받은 원본 텍스트 응답.

    Returns:
        Optional[Dict[str, Any]]: 파싱에 성공하면 파이썬 딕셔너리를 반환하고,
                                   실패하면 None을 반환합니다.
    """
    json_str = None
    match = re.search(r"```json\s*(\{.*?\})\s*```", response_text, re.DOTALL)
    if match:
        json_str = match.group(1)
    else:
        start_index = response_text.find('{')
        end_index = response_text.rfind('}')
        if start_index != -1 and end_index != -1 and start_index < end_index:
            json_str = response_text[start_index : end_index + 1]

    if json_str:
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            print(f"Warning: JSON 파싱 실패. Raw: {json_str[:200]}")

    return None
