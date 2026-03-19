from google import genai
from google.genai import types


class GeminiClient:
    """google-genai SDK 기반 Gemini 클라이언트.

    환경변수 GEMINI_API_KEY가 설정되어 있어야 합니다.
    """

    def __init__(self, model_name: str = "gemini-3-flash-preview"):
        self.client = genai.Client()
        self.model_name = model_name

    def analyze_image(self, image_path: str, custom_prompt: str = "") -> str:
        """이미지를 분석하고 결과 텍스트를 반환합니다."""
        try:
            with open(image_path, "rb") as f:
                image_bytes = f.read()
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[
                    types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
                    custom_prompt,
                ],
            )
            return response.text.strip()
        except Exception as e:
            return f"❌ gemini 오류 발생 ({self.model_name}): {str(e)}"

    def analyze_video(
        self,
        video_path: str,
        custom_prompt: str = "",
        mime_type: str = "video/mp4",
    ) -> str:
        """비디오를 분석하고 결과 텍스트를 반환합니다."""
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
                        types.Part(text=custom_prompt),
                    ]
                ),
            )
            return response.text.strip()
        except Exception as e:
            return f"❌ gemini 오류 발생 ({self.model_name}): {str(e)}"
