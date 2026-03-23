from google import genai

DEFAULT_TRANSLATE_MODEL = "gemini-2.5-flash"


def translate_english_to_korean(english_sentence: str, model_name: str = DEFAULT_TRANSLATE_MODEL) -> str:
    """영어 문장을 한국어로 번역한 결과만 반환합니다.

    환경변수 GEMINI_API_KEY가 설정되어 있어야 합니다.
    """
    client = genai.Client()
    prompt = (
        "다음 영어 문장을 한국어로 자연스럽게 번역해 주세요. 번역 결과만 출력하세요.\n\n"
        f"영어 문장: {english_sentence}\n\n"
        "한글 번역:"
    )
    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
    )
    result = response.text.strip()
    print("번역 응답 :", result)
    return result


def translate_korean_to_english(korean_sentence: str, model_name: str = DEFAULT_TRANSLATE_MODEL) -> str:
    """한국어 문장을 영어로 번역한 결과만 반환합니다.

    환경변수 GEMINI_API_KEY가 설정되어 있어야 합니다.
    """
    client = genai.Client()
    prompt = (
        "다음 한글 문장을 영어로 자연스럽게 번역해 주세요. 번역 결과만 출력하세요.\n\n"
        f"한글 문장: {korean_sentence}\n\n"
        "영어 번역:"
    )
    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
    )
    return response.text.strip()
