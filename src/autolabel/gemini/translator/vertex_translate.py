# vertex_translate.py
import os
import vertexai
from vertexai.preview.generative_models import GenerativeModel

# 환경 및 모델 초기화
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = "configs/gmail-361002-cbcf95afec4a.json"
vertexai.init(project="gmail-361002", location="us-central1")
model = GenerativeModel(model_name='gemini-2.0-flash')

def translate_english_to_korean(english_sentence: str) -> str:
    """영어 문장을 한글로 번역한 결과만 반환하는 함수"""
    prompt = f"""다음 영어 문장을 한국어로 자연스럽게 번역해 주세요. 번역 결과만 출력하세요.

영어 문장: {english_sentence}

한글 번역:"""
    response = model.generate_content(prompt)
    print("번역 응답 : " , response.text.strip())
    return response.text.strip()


def translate_korean_to_english(korean_sentence: str) -> str:
    """한글 문장을 영어로 번역한 결과만 반환하는 함수"""
    prompt = f"""다음 한글 문장을 영어로 자연스럽게 번역해 주세요. 번역 결과만 출력하세요.

한글 문장: {korean_sentence}

영어 번역:"""
    response = model.generate_content(prompt)
    # print("번역 응답 : " , response.text.strip())
    return response.text.strip()

if __name__ == "__main__":
    print(translate_korean_to_english("2개의 에스컬레이터 중 오른쪽 에스컬레이터에서 남성이 머리를 위쪽 방향으로 하여 쓰러진 상태에 있습니다. 주변에 사람이 없습니다. 응급상황으로 판단됩니다."))
    