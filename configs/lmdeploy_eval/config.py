# ============================================================
# LMDeploy 벤치마크 평가 설정
#
# 파인튜닝 완료된 InternVL3 계열 로컬 모델의 최종 벤치마크 평가 전용.
# (테스트셋 평가 Precision/Recall/F1 과는 별개 프로세스)
# ============================================================

# ------------------------------------------------------------
# 평가할 벤치마크 목록
# bench_folder.md 의 유효한 폴더명만 기입
#
# 전체 벤치마크 목록 (카테고리별)
#
# [falldown]
#   ABB_Falldown, Coupang_Falldown, DTRO_Falldown, GangNam_Falldown,
#   Hyundai_Falldown, Innodep_Falldown, KhonKaen_Falldown, KISA_Falldown,
#   Kumho_Falldown, PIA_Falldown, Soil_Falldown, UGO_Falldown,
#   Yonsei_Falldown
#
# [sittingdown]
#   ABB_Sittingdown, Coupang_Sittingdown, DTRO_Sittingdown, GangNam_Sittingdown,
#   Hyundai_Sittingdown, Innodep_Sittingdown, KhonKaen_Sittingdown,
#   Kumho_Sittingdown, Soil_Sittingdown, UGO_Sittingdown, Yonsei_Sittingdown
#
# [smoke]
#   Coupang_Smoke, Hyundai_Smoke, Innodep_Smoke, KhonKaen_Smoke,
#   Kumho_Smoke, SamsungCNT_Smoke, Soil_Smoke, Yonsei_Smoke
#
# [fire]
#   Coupang_Fire, Hyundai_Fire, Kumho_Fire, PIA_Fire,
#   SamsungCNT_Fire, Soil_Fire
#
# [violence]
#   GangNam_Violence, Innodep_Violence, KhonKaen_Violence,
#   KISA_Violence, PIA_Violence
#
# [smoking]
#   PIA_Smoking
# ------------------------------------------------------------
BENCHMARKS = [
    "PIA_Fire",
    "SamsungCNT_Fire",
    "Soil_Fire",
    "Hyundai_Fire",
    "Coupang_Fire",
    "Kumho_Fire",
]


# ------------------------------------------------------------
# 벤치마크 데이터 루트 경로
# ------------------------------------------------------------
BENCH_BASE_PATH = "/mnt/PoC_benchmark/huggingface_benchmarks_dataset/Leaderboard_bench"

# ------------------------------------------------------------
# 결과 CSV 저장 경로
# {OUTPUT_PATH}/{RUN_NAME}/{BenchmarkName}/{video_stem}.csv 형태로 저장됨
# ------------------------------------------------------------
OUTPUT_PATH = "/home/gpuadmin/Repo/seoik/train_eval_toolkit/results/lmdeploy_eval"
RUN_NAME    = "InternVL3-2B_hyundai_8_20"

# ------------------------------------------------------------
# LMDeploy 서버 설정
# LMDeploy 기본 포트: 23333
# ------------------------------------------------------------
API_BASE = "http://127.0.0.1:23333/v1"
MODEL = "/model"

# ------------------------------------------------------------
# 프레임 샘플링 설정
#
# WINDOW_SIZE: N 프레임마다 1개 프레임 추론
#   예) WINDOW_SIZE=30 이면 0, 30, 60, 90, ... 프레임만 추론
#
# INTERPOLATION: 샘플링되지 않은 프레임 채우기 방식
#   "forward"  : 샘플 프레임 k의 예측값으로 [k, k+WINDOW_SIZE) 구간 전체를 채움
#   "backward" : 샘플 프레임 k의 예측값으로 (k-WINDOW_SIZE, k] 구간을 채움
# ------------------------------------------------------------
WINDOW_SIZE = 15
INTERPOLATION = "forward"

# ------------------------------------------------------------
# 이미지 인코딩
# ------------------------------------------------------------
JPEG_QUALITY = 95

# ------------------------------------------------------------
# 동시 처리 수 (비디오 1개당 프레임 추출+추론 병렬 수)
# ------------------------------------------------------------
CONCURRENCY = 10

# ------------------------------------------------------------
# 추론 파라미터
# ------------------------------------------------------------
MAX_TOKENS = 15
TEMPERATURE = 0.0
SEED = 0

# ------------------------------------------------------------
# 파서 설정
# ------------------------------------------------------------
NEGATIVE_LABEL = "normal"

# ------------------------------------------------------------
# 카테고리별 프롬프트 템플릿
# ------------------------------------------------------------
PROMPT_TEMPLATES = {
    "falldown": """Analyze this image carefully. Determine if a person has fallen down.

Important classification rules:

- The "falldown" category applies to any person who is lying down, regardless of:
  - the surface (e.g., floor, mattress, bed)
  - the posture (natural or unnatural)
  - the cause (e.g., sleeping, collapsing, lying intentionally)
- This includes:
  - A person lying flat on the ground or other surfaces
  - A person collapsed or sprawled in any lying position
- The "normal" category applies only if the person is:
  - sitting
  - standing
  - kneeling
  - or otherwise upright (not lying down)

Answer in JSON format with BOTH of the following fields:
- "category": either "falldown" or "normal"
- "description": a brief reason why this classification was made (e.g., "person lying on a mattress", "person sitting on sofa")

Example:
{
  "category": "falldown",
  "description": "person lying on a mattress in natural posture"
}""",

    "smoke": """Is {category} visible in this image? Answer only with 0 (no) or 1 (yes).""",

    "violence": """Is {category} visible in this image? Answer only with 0 (no) or 1 (yes).""",

    "sittingdown": """Is {category} visible in this image? Answer only with 0 (no) or 1 (yes).""",

    "fire": """Analyze this image carefully. Only classify it as "fire" if there are ACTUAL FLAMES or ACTIVE FIRE visible in the image.

Important classification rules:

- The "fire" category applies ONLY to images with:
 - visible flames or burning fire
 - active combustion or blazing materials
 - actual fire sources (e.g., campfire, house fire, candle flame, burning car, torch flame)
- This includes:
 - Real flames from any source
 - Active burning or combustion
 - Visible fire with smoke and flames
 - Burning vehicles or cars on fire
 - Torches, flamethrowers, or similar devices with visible flames
- The "normal" category applies to:
 - Fire-related objects WITHOUT actual flames (fire trucks, fire extinguishers, fire hydrants)
 - Red colors, sunset glows, red lights, or the sun
 - Fire emojis or cartoon representations of fire
 - Text or words containing "fire" (signs, labels, written text)
 - Any image without visible flames or active fire
 - welding sparks, grinding sparks, cutting sparks, or metal sparks without visible flames
 - glowing metal fragments, hot particles, or red/orange sparks from construction tools
 - brief spark showers from machinery that do not show sustained flames or active burning


Answer in JSON format with BOTH of the following fields:
- "category": either "fire" or "normal"
- "description": a brief reason why this classification was made (e.g., "visible flames from campfire", "fire truck without flames")

Example:
{
 "category": "fire",
 "description": "visible flames from burning building"
}""",


    "smoking": """Is {category} visible in this image? Answer only with 0 (no) or 1 (yes).""",

    # 위에 정의되지 않은 카테고리는 이 템플릿으로 fallback
    "default": """Is {category} visible in this image? Answer only with 0 (no) or 1 (yes).""",
}
