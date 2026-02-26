import os
import json
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Optional
from functools import partial
from datetime import datetime

from src._autolabeling.gemini.client import GeminiClient
from configs.config_gemini import (
    PROMPT_VIDEO,
    GEMINI_MODEL_CONFIG,
    PROMPT_VIDEO_NORMAL_LABEL_ENHANCED,
    PROMPT_VIDEO_VIOLENCE_LABEL_ENHANCED,
)
from configs.config_gemini_violence_timestamp import PROMPT_VIDEO_VIOLENCE_TIMESTAMP_ENHANCED
from configs.config_gemini_aihub_space import PROMPT_VIDEO_VIOLENCE_LABEL_ENHANCED_AIHUB_SPACE
from configs.config_gemini_gj_vio import PROMPT_VIDEO_VIOLENCE_LABEL_GJ, PROMPT_VIDEO_NORMAL_LABEL_GJ
from configs.config_gemini_cctv import PROMPT_VIDEO_VIOLENCE_LABEL_CCTV, PROMPT_VIDEO_NORMAL_LABEL_CCTV
from configs.config_gemini_scvd import PROMPT_VIDEO_VIOLENCE_LABEL_SCVD, PROMPT_VIDEO_NORMAL_LABEL_SCVD
from configs.config_gemini_gangnam import (
    PROMPT_IMAGE_NORMAL_LABEL_GANGNAM,
    PROMPT_VIDEO_NORMAL_LABEL_GANGNAM,
)
from configs.config_gemini_hyundai import (
    PROMPT_IMAGE_FALLDOWN_LABEL_ESCALATOR,
    PROMPT_IMAGE_NORMAL_LABEL_ESCALATOR,
)
from src.utils.json_parser import parse_json_from_response

# (options, mode) → prompt 딕셔너리 룩업
_OPTION_PROMPT_MAP: dict[tuple[str, str], str] = {
    ("basic",           "video"): PROMPT_VIDEO,
    ("normal",          "video"): PROMPT_VIDEO_NORMAL_LABEL_ENHANCED,
    ("vio",             "video"): PROMPT_VIDEO_VIOLENCE_LABEL_ENHANCED,
    ("vio_timestamp",   "video"): PROMPT_VIDEO_VIOLENCE_TIMESTAMP_ENHANCED,
    ("aihub_space",     "video"): PROMPT_VIDEO_VIOLENCE_LABEL_ENHANCED_AIHUB_SPACE,
    ("gj_normal",       "video"): PROMPT_VIDEO_NORMAL_LABEL_GJ,
    ("gj_violence",     "video"): PROMPT_VIDEO_VIOLENCE_LABEL_GJ,
    ("cctv_normal",     "video"): PROMPT_VIDEO_NORMAL_LABEL_CCTV,
    ("cctv_violence",   "video"): PROMPT_VIDEO_VIOLENCE_LABEL_CCTV,
    ("scvd_normal",     "video"): PROMPT_VIDEO_NORMAL_LABEL_SCVD,
    ("scvd_violence",   "video"): PROMPT_VIDEO_VIOLENCE_LABEL_SCVD,
    ("gangnam",         "video"): PROMPT_VIDEO_NORMAL_LABEL_GANGNAM,
    ("gangnam",         "image"): PROMPT_IMAGE_NORMAL_LABEL_GANGNAM,
    ("hyundai_normal",  "image"): PROMPT_IMAGE_NORMAL_LABEL_ESCALATOR,
    ("hyundai_falldown","image"): PROMPT_IMAGE_FALLDOWN_LABEL_ESCALATOR,
}


def _label_single_file(
    file_path: str,
    prompt: str,
    model_name: str,
    project: str,
    location: str,
    media_type: str,
    max_retries: int = 3,
    overwrite: bool = False,
) -> Optional[str]:
    """단일 파일(이미지 또는 비디오)을 분석하고 결과를 JSON으로 저장합니다."""
    print(f"Processing: {file_path}...")
    try:
        output_folder = os.path.dirname(file_path)
        base_filename = os.path.splitext(os.path.basename(file_path))[0]
        output_filepath = os.path.join(output_folder, f"{base_filename}.json")

        if os.path.exists(output_filepath) and not overwrite:
            try:
                with open(output_filepath, "r", encoding="utf-8") as f:
                    existing_data = json.load(f)
                if "description" in existing_data and existing_data["description"]:
                    print(f"⏩ Skipping: Valid label file already exists for {os.path.basename(file_path)}")
                    return output_filepath
                else:
                    print(f"🔄 Reprocessing: Empty description in {os.path.basename(file_path)}")
            except (json.JSONDecodeError, KeyError) as e:
                print(f"🔄 Reprocessing: Invalid JSON file for {os.path.basename(file_path)}: {e}")

        if os.path.exists(output_filepath) and overwrite:
            print(f"🔄 Overwriting existing file: {os.path.basename(file_path)}")

        client = GeminiClient(model_name=model_name, project=project, location=location)

        api_response_text = None
        for attempt in range(max_retries):
            try:
                if media_type == "image":
                    api_response_text = client.analyze_image(image_path=file_path, custom_prompt=prompt)
                else:
                    api_response_text = client.analyze_video(video_path=file_path, custom_prompt=prompt)

                if api_response_text:
                    break
                else:
                    print(f"Attempt {attempt + 1}/{max_retries}: API returned empty response. Retrying...")
                    time.sleep(1)
            except Exception as e:
                print(f"Attempt {attempt + 1}/{max_retries}: API call failed: {e}. Retrying...")
                time.sleep(1)
        else:
            print(f"Error: All {max_retries} attempts failed for {file_path}. Skipping this file.")
            return None

        data = parse_json_from_response(api_response_text)
        if data is None:
            return None

        with open(output_filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

        print(f"✅ Successfully labeled and saved -> {output_filepath}")
        return output_filepath

    except Exception as e:
        print(f"An unexpected error occurred while processing {file_path}: {e}")
        return None


def autolabel_files_recursively(
    input_folder: str,
    failure_log_dir: str,
    num_workers: Optional[int] = None,
    options: str = "basic",
    mode: str = "video",
    overwrite: bool = False,
) -> None:
    """지정된 폴더와 모든 하위 폴더를 순회하며 파일을 찾아 병렬로 오토라벨링합니다.

    결과 JSON 파일은 각 원본 파일과 동일한 위치에 저장되며,
    실패한 파일 목록은 failure_log_dir에 타임스탬프 파일로 저장됩니다.

    Args:
        input_folder: 검색을 시작할 최상위 폴더 경로.
        failure_log_dir: 실패 로그 파일을 저장할 폴더 경로.
        num_workers: 사용할 CPU 코어 수 (None이면 기본값 사용).
        options: 라벨링 옵션 (예: "vio", "normal", "gangnam", ...).
        mode: 처리 대상 타입 ("video" 또는 "image").
        overwrite: True이면 기존 JSON 파일을 덮어씀.
    """
    if mode == "video":
        supported_extensions = (".mp4", ".avi", ".mov", ".mkv")
    else:
        supported_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

    files_to_process = []
    for root, _, files in os.walk(input_folder):
        for file in files:
            if file.lower().endswith(supported_extensions):
                files_to_process.append(os.path.join(root, file))

    if not files_to_process:
        media_label = "video" if mode == "video" else "image"
        print(f"No {media_label} files found in {input_folder} and its subdirectories.")
        return

    print(f"Found {len(files_to_process)} files to process.")
    print(f"Using {num_workers if num_workers else 'all available'} CPU cores.")

    prompt = _OPTION_PROMPT_MAP.get((options, mode))
    if prompt is None:
        raise ValueError(
            f"Unsupported options/mode combination: options='{options}', mode='{mode}'. "
            f"Available keys: {list(_OPTION_PROMPT_MAP.keys())}"
        )

    worker_func = partial(
        _label_single_file,
        prompt=prompt,
        model_name=GEMINI_MODEL_CONFIG["model_name"],
        project=GEMINI_MODEL_CONFIG["project"],
        location=GEMINI_MODEL_CONFIG["location"],
        media_type=mode,
        overwrite=overwrite,
    )

    failed_files = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = executor.map(worker_func, files_to_process)
        for file_path, result in zip(files_to_process, results):
            if result is None:
                failed_files.append(file_path)

    if failed_files:
        os.makedirs(failure_log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        failure_log_path = os.path.join(failure_log_dir, f"failed_files_{timestamp}.txt")
        with open(failure_log_path, "w", encoding="utf-8") as f:
            for file_path in failed_files:
                f.write(f"{file_path}\n")
        print(f"\n⚠️ Found {len(failed_files)} failed files. List saved to: {failure_log_path}")

    success_count = len(files_to_process) - len(failed_files)
    print(f"\n--- Autolabeling Complete ---")
    print(f"Successfully processed {success_count} out of {len(files_to_process)} files.")
