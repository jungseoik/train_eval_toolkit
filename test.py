
from src.autolabel.gemini.gemini_api import GeminiImageAnalyzer
from configs.config_gemini import PROMPT_VIDEO

analyzer = GeminiImageAnalyzer(model_name="gemini-2.5-pro" ,project="gmail-361002", location="us-central1")
result = analyzer.analyze_video(video_path="" , custom_prompt=PROMPT_VIDEO)
print(result)