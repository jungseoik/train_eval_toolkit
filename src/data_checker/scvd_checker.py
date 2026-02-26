import json
import os

import os
import json
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Optional, List
from functools import partial
from datetime import datetime

from src._autolabeling.gemini.client import GeminiClient as GeminiImageAnalyzer
from configs.config_gemini import GEMINI_MODEL_CONFIG, PROMPT_VIDEO
from configs.config_gemini_scvd import PROMPT_VIDEO_CHECK_LABEL_SCVD
from src.utils.json_parser import parse_json_from_response


def extract_mismatch_paths(json_file_path):
    """
    주어진 JSON 파일을 읽어서, video 경로에서 추출한 카테고리와
    conversations의 category 값이 다른 경우,
    data/ prefix를 붙인 전체 경로를 리스트로 반환
    """

    results = []
    with open(json_file_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)

            # 1. video 경로에서 카테고리 추출
            video_path = data["video"]
            parts = video_path.lower().split("/")
            if "normal" in parts:
                path_category = "normal"
            elif "violence" in parts or "weaponized" in parts:
                path_category = "violence"
            else:
                path_category = "unknown"

            # 2. conversations 에서 category 추출
            conv_value = data["conversations"][0]["value"]
            conv_json = json.loads(conv_value)
            conv_category = conv_json["category"].lower()

            # 3. 불일치 확인
            if path_category != conv_category:
                full_path = os.path.join("data", video_path)
                results.append(full_path)

    return results


def _relabel_single_video(
    video_path: str,
    prompt: str,
    model_name: str,
    project: str,
    location: str,
    max_retries: int = 3
) -> Optional[str]:
    """불일치 비디오를 다시 라벨링"""
    print(f"🔄 Re-labeling: {video_path}...")
    try:
        output_folder = os.path.dirname(video_path)
        base_filename = os.path.splitext(os.path.basename(video_path))[0]
        output_filepath = os.path.join(output_folder, f"{base_filename}.json")

        analyzer = GeminiImageAnalyzer(model_name=model_name, project=project, location=location)

        api_response_text = None
        for attempt in range(max_retries):
            try:
                api_response_text = analyzer.analyze_video(video_path=video_path, custom_prompt=prompt)
                if api_response_text:
                    break
                else:
                    print(f"Attempt {attempt+1}/{max_retries}: Empty response. Retrying...")
                    time.sleep(1)
            except Exception as e:
                print(f"Attempt {attempt+1}/{max_retries}: API error {e}. Retrying...")
                time.sleep(1)
        else:
            print(f"❌ Failed: {video_path}")
            return None

        data = parse_json_from_response(api_response_text)
        if data is None:
            return None

        with open(output_filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

        print(f"✅ Re-labeled -> {output_filepath}")
        return output_filepath

    except Exception as e:
        print(f"Unexpected error on {video_path}: {e}")
        return None


def relabel_mismatched_videos(
    jsonl_file: str,
    failure_log_dir: str,
    num_workers: Optional[int] = None
):
    """JSONL에서 불일치 경로를 뽑아 그 비디오들만 다시 라벨링"""
    mismatched_paths = extract_mismatch_paths(jsonl_file)
    if not mismatched_paths:
        print("🎉 No mismatched videos found!")
        return

    print(f"Found {len(mismatched_paths)} mismatched videos to re-label.")
    print(f"Using {num_workers if num_workers else 'all available'} CPU cores.")

    worker_func = partial(
        _relabel_single_video,
        prompt=PROMPT_VIDEO_CHECK_LABEL_SCVD,
        model_name=GEMINI_MODEL_CONFIG['model_name'],
        project=GEMINI_MODEL_CONFIG['project'],
        location=GEMINI_MODEL_CONFIG['location']
    )

    failed_videos = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = executor.map(worker_func, mismatched_paths)
        for video_path, result in zip(mismatched_paths, results):
            if result is None:
                failed_videos.append(video_path)

    if failed_videos:
        os.makedirs(failure_log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        failure_log_path = os.path.join(failure_log_dir, f"failed_relabels_{timestamp}.txt")
        with open(failure_log_path, "w", encoding="utf-8") as f:
            for video_path in failed_videos:
                f.write(f"{video_path}\n")
        print(f"\n⚠️ Failed {len(failed_videos)} videos. List saved to {failure_log_path}")

    success_count = len(mismatched_paths) - len(failed_videos)
    print(f"\n--- Re-labeling Complete ---")
    print(f"✅ Successfully re-labeled {success_count} / {len(mismatched_paths)} mismatched videos.")


if __name__ == "__main__":
    JSONL_FILE = "/home/piawsa6000/nas192/datasets/projects/gangnam_innovation/violence_fintuning/TADO_Violence_GangNAM/data/instruction/train/scvdALL_no_split.jsonl"  # 여기에 불일치 검사할 JSONL 경로
    FAILURE_LOG_DIR = "assets/logs"
    NUM_CORES = 16

    print("--- Running Gemini Re-Labeler for mismatched videos ---")
    relabel_mismatched_videos(
        jsonl_file=JSONL_FILE,
        failure_log_dir=FAILURE_LOG_DIR,
        num_workers=NUM_CORES
    )


    result = extract_mismatch_paths(JSONL_FILE)
    print(len(result))

