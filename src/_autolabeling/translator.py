import os
import json
from concurrent.futures import ProcessPoolExecutor
from typing import Optional
from datetime import datetime

from src._autolabeling.gemini.translate_client import translate_english_to_korean


def _translate_and_update_single_json(json_path: str) -> Optional[str]:
    """단일 JSON 파일을 읽어 'description' 값을 번역하고 'description_kor'로 추가합니다."""
    print(f"Processing JSON: {json_path}...")
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if "description" not in data:
            print(f"⏩ Skipping: 'description' key not found in {os.path.basename(json_path)}")
            return json_path

        english_description = data["description"]

        if "description_kor" in data and data["description_kor"]:
            print(f"⏩ Skipping: 'description_kor' already exists for {os.path.basename(json_path)}")
            return json_path

        if not english_description or not english_description.strip():
            print(f"⏩ Skipping: 'description' value is empty in {os.path.basename(json_path)}")
            data["description_kor"] = ""
            return json_path

        korean_description = translate_english_to_korean(english_description)
        data["description_kor"] = korean_description

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

        print(f"✅ Successfully translated and updated -> {json_path}")
        return json_path

    except json.JSONDecodeError as e:
        print(f"❌ Error: Invalid JSON format for {json_path}: {e}")
        return None
    except Exception as e:
        print(f"❌ An unexpected error occurred while processing {json_path}: {e}")
        return None


def translate_descriptions_recursively(
    input_folder: str,
    failure_log_dir: str,
    num_workers: Optional[int] = None,
) -> None:
    """지정된 폴더와 모든 하위 폴더를 순회하며 JSON 파일을 찾아 병렬로 번역합니다.

    각 JSON 파일의 'description' 값을 한국어로 번역하여
    'description_kor' 키로 추가 저장합니다.

    Args:
        input_folder: 검색을 시작할 최상위 폴더 경로.
        failure_log_dir: 실패 로그 파일을 저장할 폴더 경로.
        num_workers: 사용할 CPU 코어 수 (None이면 기본값 사용).
    """
    json_files_to_process = []
    for root, _, files in os.walk(input_folder):
        for file in files:
            if file.lower().endswith(".json"):
                json_files_to_process.append(os.path.join(root, file))

    if not json_files_to_process:
        print(f"No JSON files found in {input_folder} and its subdirectories.")
        return

    print(f"Found {len(json_files_to_process)} JSON files to process.")
    print(f"Using {num_workers if num_workers else 'default'} CPU cores for translation.")

    failed_files = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = executor.map(_translate_and_update_single_json, json_files_to_process)
        for json_path, result in zip(json_files_to_process, results):
            if result is None:
                failed_files.append(json_path)

    if failed_files:
        os.makedirs(failure_log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        failure_log_path = os.path.join(failure_log_dir, f"failed_translation_{timestamp}.txt")
        with open(failure_log_path, "w", encoding="utf-8") as f:
            for file_path in failed_files:
                f.write(f"{file_path}\n")
        print(f"\n⚠️ Found {len(failed_files)} failed files. List saved to: {failure_log_path}")

    success_count = len(json_files_to_process) - len(failed_files)
    print(f"\n--- Translation and Update Complete ---")
    print(f"Successfully processed {success_count} out of {len(json_files_to_process)} JSON files.")
