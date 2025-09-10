import json
import re
from typing import List, Optional, Dict, Any

def parse_json_from_response(response_text: str) -> Optional[Dict[str, Any]]:
    """API 응답 텍스트에서 JSON 객체를 안정적으로 파싱하여 딕셔너리로 반환합니다.

    LLM의 다양한 응답 형식을 처리하기 위해 다단계 파싱 전략을 사용합니다.
    1. 마크다운 코드 블록(```json ... ```)을 우선적으로 탐색합니다.
    2. 마크다운이 없을 경우, 텍스트에서 첫 '{'와 마지막 '}' 사이를 추출합니다.
    3. 위 방법들이 실패하면, "category"와 "description" 키의 값을 직접 추출합니다.

    Args:
        response_text (str): Gemini API로부터 받은 원본 텍스트 응답.

    Returns:
        Optional[Dict[str, Any]]: 파싱에 성공하면 파이썬 딕셔너리를 반환하고,
                                   실패하면 None을 반환합니다.
    """
    # 1단계 & 2단계 시도
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
            # 1, 2단계 파싱 실패 시 3단계로 넘어감
            print("Info: JSON object/block found, but failed to parse. Falling back to key-value extraction.")

    # 3단계: 핵심 키-값 직접 추출
    try:
        # "key": "value" 패턴을 찾되, value 부분에 따옴표가 이스케이프(\")된 경우도 고려
        category_match = re.search(r'["\']category["\']\s*:\s*["\'](.*?)["\']', response_text, re.IGNORECASE)
        description_match = re.search(r'["\']description["\']\s*:\s*["\'](.*?)["\']', response_text, re.IGNORECASE | re.DOTALL)

        if category_match and description_match:
            # 정규식으로 찾은 값을 사용하여 직접 딕셔너리 생성
            data = {
                "category": category_match.group(1).strip(),
                "description": description_match.group(1).strip()
            }
            print("Info: Successfully extracted key-values directly.")
            return data
    except Exception as e:
        print(f"An unexpected error occurred during key-value extraction: {e}")
        # 예외 발생 시에도 최종 실패 메시지를 출력하도록 함수를 계속 진행

    print("Error: All parsing attempts failed.")
    print(f"--- Raw Response ---\n{response_text}\n--------------------")
    return None

if __name__ == '__main__':
    case1 = 'Here is the result: ```json\n{"category": "violence", "description": "A man punches another."}\n```'
    case2 = 'Of course. {"category": "non-violence", "description": "Two people are hugging."}'
    case3 = '"category": "violence", "description": "A group fight."' # 중괄호 없는 케이스
    case4 = 'I could not determine the category.'
    case5 = 'Here is what I found. "category":"violence", "description" : "Pushing match." I hope this helps.' # 따옴표 종류가 다른 케이스

    print("--- Testing Case 1 (Markdown Block) ---")
    print(parse_json_from_response(case1))
    print("\n" + "="*30 + "\n")

    print("--- Testing Case 2 (JSON Object) ---")
    print(parse_json_from_response(case2))
    print("\n" + "="*30 + "\n")
    
    print("--- Testing Case 3 (Key-Value Only) ---")
    print(parse_json_from_response(case3))
    print("\n" + "="*30 + "\n")

    print("--- Testing Case 4 (No JSON) ---")
    print(parse_json_from_response(case4))
    print("\n" + "="*30 + "\n")
    
    print("--- Testing Case 5 (Messy Key-Value) ---")
    print(parse_json_from_response(case5))