from configs.config_gemini import ENV_AUTH_V2, PROMPT_IMAGE , PROMPT_VIDEO
from google import genai
from google.genai import types
import os

class GeminiImageAnalyzer:
    def __init__(self, model_name: str = "gemini-3-flash-preview", project: str = "gmail-361002", location: str = "us-central1"):
        os.environ['GEMINI_API_KEY'] = ENV_AUTH_V2
        self.client = genai.Client()
        self.model_name = model_name
        self.whole_response = None

    def analyze_image(self, image_path: str, custom_prompt: str = PROMPT_IMAGE) -> str:
        """
        이미지 분석을 수행합니다.
        
        Args:
            image_path (str): 분석할 이미지 경로
            custom_prompt (str): 커스텀 프롬프트 (선택사항)
        
        Returns:
            str: 분석 결과
        """
        try:
            with open(image_path, "rb") as f:
                image_bytes = f.read()
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[
                    types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
                    custom_prompt
                ]
            )
            self.whole_response = response
            return response.text.strip()
        except Exception as e:
            return f"❌ gemini 오류 발생 ({self.model_name}): {str(e)}"
    def analyze_video(
        self,
        video_path: str,
        custom_prompt: str = PROMPT_VIDEO,
        mime_type: str  = "video/mp4",
    ) -> str:
        """
        비디오 분석
        
        Args:
            video_path (str): 분석할 비디오 경로
            custom_prompt (str): 커스텀 프롬프트 (선택, 기본: PROMPT_VIDEO)
            mime_type (str | None): 강제 MIME 타입 지정 (미지정 시 확장자로 추정, 기본값 video/mp4)
        Returns:
            str: 분석 결과 텍스트
        """
        try:
            with open(video_path, "rb") as f:
                video_bytes = f.read()

            response = self.client.models.generate_content(
                model=self.model_name,
                contents=types.Content(
                    parts=[
                        types.Part(
                            inline_data=types.Blob(data=video_bytes, mime_type=mime_type)
                        ),
                        types.Part(text=custom_prompt)
                    ]
                )
            )
            self.whole_response = response
            return response.text.strip()
        except Exception as e:
            return f"❌ gemini 오류 발생 ({self.model_name}): {str(e)}"

