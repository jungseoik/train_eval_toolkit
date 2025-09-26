import os
import glob
import json
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Optional, List, Tuple
from functools import partial
from datetime import datetime
from src.autolabel.gemini.gemini_api import GeminiImageAnalyzer
from configs.config_gemini import PROMPT_VIDEO, GEMINI_MODEL_CONFIG, PROMPT_VIDEO_NORMAL_LABEL ,PROMPT_VIDEO_VIOLENCE_LABEL, PROMPT_VIDEO_NORMAL_LABEL_ENHANCED ,PROMPT_VIDEO_VIOLENCE_LABEL_ENHANCED
from configs.config_gemini_violence_timestamp import PROMPT_VIDEO_VIOLENCE_TIMESTAMP_ENHANCED
from configs.config_gemini_aihub_space import PROMPT_VIDEO_VIOLENCE_LABEL_ENHANCED_AIHUB_SPACE 
from configs.config_gemini_gj_vio import PROMPT_VIDEO_VIOLENCE_LABEL_GJ , PROMPT_VIDEO_NORMAL_LABEL_GJ
from configs.config_gemini_cctv import PROMPT_VIDEO_VIOLENCE_LABEL_CCTV , PROMPT_VIDEO_NORMAL_LABEL_CCTV
from configs.config_gemini_scvd import PROMPT_VIDEO_VIOLENCE_LABEL_SCVD , PROMPT_VIDEO_NORMAL_LABEL_SCVD


from src.utils.json_parser import parse_json_from_response

def _label_single_video(
    video_path: str,
    prompt: str,
    model_name: str,
    project: str,
    location: str,
    max_retries: int = 3
) -> Optional[str]:
    """(내부 함수) 단일 비디오를 분석하고 결과를 JSON으로 저장합니다.

    이 함수는 비디오와 동일한 폴더에 JSON 파일을 생성합니다.
    """
    # ... (이 함수의 핵심 로직은 이전과 대부분 동일)
    print(f"Processing: {video_path}...")
    try:
        # JSON 파일을 비디오와 동일한 폴더에 저장
        output_folder = os.path.dirname(video_path)
        base_filename = os.path.splitext(os.path.basename(video_path))[0]
        output_filepath = os.path.join(output_folder, f"{base_filename}.json")
        
        # --- 이미 결과 파일이 존재하면 건너뛰는 로직 ---
        # if os.path.exists(output_filepath):
        #     print(f"⏩ Skipping: Label file already exists for {os.path.basename(video_path)}")
        #     return output_filepath # 이미 성공한 것으로 간주하고 경로 반환
        if os.path.exists(output_filepath):
            try:
                with open(output_filepath, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
                
                # clips가 존재하고 비어있지 않으면 건너뛰기
                if "clips" in existing_data and len(existing_data["clips"]) > 0:
                    print(f"⏩ Skipping: Valid label file already exists for {os.path.basename(video_path)}")
                    return output_filepath
                else:
                    print(f"🔄 Reprocessing: Empty clips found in {os.path.basename(video_path)}")
            except (json.JSONDecodeError, KeyError) as e:
                print(f"🔄 Reprocessing: Invalid JSON file for {os.path.basename(video_path)}: {e}")


        analyzer = GeminiImageAnalyzer(model_name=model_name, project=project, location=location)
        
        api_response_text = None
        for attempt in range(max_retries):
            try:
                api_response_text = analyzer.analyze_video(video_path=video_path, custom_prompt=prompt)
                if api_response_text:
                    break
                else:
                    print(f"Attempt {attempt + 1}/{max_retries}: API returned empty response for {video_path}. Retrying...")
                    time.sleep(1)
            except Exception as e:
                print(f"Attempt {attempt + 1}/{max_retries}: API call failed for {video_path}. Error: {e}. Retrying...")
                time.sleep(1)
        else:
            print(f"Error: All {max_retries} attempts failed for {video_path}. Skipping this file.")
            return None

        data = parse_json_from_response(api_response_text)
        if data is None:
            return None

        with open(output_filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        
        print(f"✅ Successfully labeled and saved -> {output_filepath}")
        return output_filepath

    except Exception as e:
        print(f"An unexpected error occurred while processing {video_path}: {e}")
        return None


def autolabel_videos_recursively(
    input_folder: str,
    failure_log_dir: str,
    num_workers: Optional[int] = None,
    options: str = "basic"
):
    """지정된 폴더와 모든 하위 폴더를 순회하며 비디오를 찾아 병렬로 오토라벨링합니다.

    결과 JSON 파일은 각 원본 비디오와 동일한 위치에 저장되며,
    실패한 비디오 목록은 별도의 로그 디렉토리에 저장됩니다.

    Args:
        input_folder (str): 검색을 시작할 최상위 폴더 경로.
        failure_log_dir (str): 실패 로그 파일을 저장할 폴더 경로.
        num_workers (Optional[int]): 사용할 CPU 코어 수.
    """
    supported_extensions = (".mp4", ".avi", ".mov", ".mkv")
    videos_to_process = []
    # os.walk를 사용하여 모든 하위 디렉토리 순회
    for root, _, files in os.walk(input_folder):
        for file in files:
            if file.lower().endswith(supported_extensions):
                videos_to_process.append(os.path.join(root, file))

    if not videos_to_process:
        print(f"No video files found in {input_folder} and its subdirectories.")
        return

    print(f"Found {len(videos_to_process)} videos to process.")
    print(f"Using {num_workers if num_workers else 'all available'} CPU cores.")
    
    if options == "basic":
        worker_func = partial(
            _label_single_video,
            prompt=PROMPT_VIDEO,
            model_name=GEMINI_MODEL_CONFIG['model_name'],
            project=GEMINI_MODEL_CONFIG['project'],
            location=GEMINI_MODEL_CONFIG['location']
        )
    elif options == "normal":
        worker_func = partial(
            _label_single_video,
            prompt=PROMPT_VIDEO_NORMAL_LABEL_ENHANCED,
            model_name=GEMINI_MODEL_CONFIG['model_name'],
            project=GEMINI_MODEL_CONFIG['project'],
            location=GEMINI_MODEL_CONFIG['location']
        )
    elif options == "vio":
        worker_func = partial(
            _label_single_video,
            prompt=PROMPT_VIDEO_VIOLENCE_LABEL_ENHANCED,
            model_name=GEMINI_MODEL_CONFIG['model_name'],
            project=GEMINI_MODEL_CONFIG['project'],
            location=GEMINI_MODEL_CONFIG['location']
        )
    elif options == "vio_timestamp":
        worker_func = partial(
            _label_single_video,
            prompt=PROMPT_VIDEO_VIOLENCE_TIMESTAMP_ENHANCED,
            model_name=GEMINI_MODEL_CONFIG['model_name'],
            project=GEMINI_MODEL_CONFIG['project'],
            location=GEMINI_MODEL_CONFIG['location']
        )
    elif options == "aihub_space":
        worker_func = partial(
            _label_single_video,
            prompt=PROMPT_VIDEO_VIOLENCE_LABEL_ENHANCED_AIHUB_SPACE,
            model_name=GEMINI_MODEL_CONFIG['model_name'],
            project=GEMINI_MODEL_CONFIG['project'],
            location=GEMINI_MODEL_CONFIG['location']
        )
    elif options == "gj_normal":
        worker_func = partial(
            _label_single_video,
            prompt= PROMPT_VIDEO_NORMAL_LABEL_GJ,
            model_name=GEMINI_MODEL_CONFIG['model_name'],
            project=GEMINI_MODEL_CONFIG['project'],
            location=GEMINI_MODEL_CONFIG['location']
        )
    elif options == "gj_violence":
        worker_func = partial(
            _label_single_video,
            prompt= PROMPT_VIDEO_VIOLENCE_LABEL_GJ,
            model_name=GEMINI_MODEL_CONFIG['model_name'],
            project=GEMINI_MODEL_CONFIG['project'],
            location=GEMINI_MODEL_CONFIG['location']
        )
    elif options == "cctv_normal":
        worker_func = partial(
            _label_single_video,
            prompt= PROMPT_VIDEO_NORMAL_LABEL_CCTV,
            model_name=GEMINI_MODEL_CONFIG['model_name'],
            project=GEMINI_MODEL_CONFIG['project'],
            location=GEMINI_MODEL_CONFIG['location']
        )
    elif options == "cctv_violence":
            worker_func = partial(
                _label_single_video,
                prompt= PROMPT_VIDEO_VIOLENCE_LABEL_CCTV,
                model_name=GEMINI_MODEL_CONFIG['model_name'],
                project=GEMINI_MODEL_CONFIG['project'],
                location=GEMINI_MODEL_CONFIG['location']
            )
    elif options == "scvd_normal":
            worker_func = partial(
                _label_single_video,
                prompt= PROMPT_VIDEO_NORMAL_LABEL_SCVD,
                model_name=GEMINI_MODEL_CONFIG['model_name'],
                project=GEMINI_MODEL_CONFIG['project'],
                location=GEMINI_MODEL_CONFIG['location']
            )
    elif options == "scvd_violence":
            worker_func = partial(
                _label_single_video,
                prompt= PROMPT_VIDEO_VIOLENCE_LABEL_SCVD,
                model_name=GEMINI_MODEL_CONFIG['model_name'],
                project=GEMINI_MODEL_CONFIG['project'],
                location=GEMINI_MODEL_CONFIG['location']
            )

    failed_videos = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = executor.map(worker_func, videos_to_process)
        for video_path, result in zip(videos_to_process, results):
            if result is None:
                failed_videos.append(video_path)

    # 실패 목록 파일 저장
    if failed_videos:
        os.makedirs(failure_log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        failure_log_path = os.path.join(failure_log_dir, f"failed_videos_{timestamp}.txt")
        
        with open(failure_log_path, 'w', encoding='utf-8') as f:
            for video_path in failed_videos:
                f.write(f"{video_path}\n")
        print(f"\n⚠️ Found {len(failed_videos)} failed videos. List saved to: {failure_log_path}")

    success_count = len(videos_to_process) - len(failed_videos)
    print(f"\n--- Autolabeling Complete ---")
    print(f"Successfully processed {success_count} out of {len(videos_to_process)} videos.")


if __name__ == '__main__':
    INPUT_ROOT_DIR = "data/processed/elevator_normal_clips_1sec"
    FAILURE_LOG_DIR = "assets/logs"
    NUM_CORES = 8

    print("--- Running Gemini Autolabeler (Recursive) ---")
    autolabel_videos_recursively(
        input_folder=INPUT_ROOT_DIR,
        failure_log_dir=FAILURE_LOG_DIR,
        num_workers=NUM_CORES
    )