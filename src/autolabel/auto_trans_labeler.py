import os
import json
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Optional
from functools import partial
from datetime import datetime
from src.autolabel.gemini.translator.vertex_translate import translate_english_to_korean

def _translate_and_update_single_json(
    json_path: str,
) -> Optional[str]:
    """
    (내부 함수) 단일 JSON 파일을 읽어 'description' 키의 값을 번역하고
    'description_kor' 키로 추가하여 저장합니다.
    """
    print(f"Processing JSON: {json_path}...")
    
    try:
        # 1. JSON 파일 로드
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 2. 'description' 키 확인 및 번역 대상 추출
        if "description" not in data:
            # 키가 없으면 에러 처리 대신 메시지를 출력하고 건너뜁니다.
            print(f"⏩ Skipping: 'description' key not found in {os.path.basename(json_path)}")
            return json_path
        
        english_description = data["description"]
        
        # 이미 번역된 내용이 있는지 확인 및 건너뛰기
        if "description_kor" in data and data["description_kor"]:
             print(f"⏩ Skipping: 'description_kor' already exists for {os.path.basename(json_path)}")
             return json_path
        
        # 번역할 내용이 비어있는지 확인
        if not english_description or not english_description.strip():
            print(f"⏩ Skipping: 'description' value is empty in {os.path.basename(json_path)}")
            data["description_kor"] = "" # 빈 값이라도 키는 추가할 수 있음
            return json_path

        # 3. 번역 작업 수행 (외부 함수 사용)
        korean_description = translate_english_to_korean(english_description)

        # 4. 'description_kor' 키 추가 및 데이터 업데이트
        data["description_kor"] = korean_description

        # 5. JSON 파일 저장
        with open(json_path, 'w', encoding='utf-8') as f:
            # ensure_ascii=False는 한글이 깨지지 않고 저장되도록 합니다.
            json.dump(data, f, ensure_ascii=False, indent=4)

        print(f"✅ Successfully translated and updated -> {json_path}")
        return json_path

    except json.JSONDecodeError as e:
        # JSON 파일 형식이 잘못된 경우
        print(f"❌ Error: Invalid JSON format for {json_path}: {e}")
        return None
    except Exception as e:
        # 그 외 예상치 못한 오류
        print(f"❌ An unexpected error occurred while processing {json_path}: {e}")
        return None


def translate_descriptions_recursively(
    input_folder: str,
    failure_log_dir: str,
    num_workers: Optional[int] = None,
):
    """
    지정된 폴더와 모든 하위 폴더를 순회하며 JSON 파일을 찾아 병렬로 번역 및 업데이트합니다.

    Args:
        input_folder (str): 검색을 시작할 최상위 폴더 경로.
        failure_log_dir (str): 실패 로그 파일을 저장할 폴더 경로.
        num_workers (Optional[int]): 사용할 CPU 코어 수.
    """
    
    supported_extensions = (".json",)
    json_files_to_process = []
    
    # os.walk를 사용하여 모든 하위 디렉토리 순회
    for root, _, files in os.walk(input_folder):
        for file in files:
            if file.lower().endswith(supported_extensions):
                json_files_to_process.append(os.path.join(root, file))

    if not json_files_to_process:
        print(f"No JSON files found in {input_folder} and its subdirectories.")
        return

    print(f"Found {len(json_files_to_process)} JSON files to process.")
    # num_workers가 None이면 ProcessPoolExecutor가 기본값(CPU 코어 수)을 사용합니다.
    print(f"Using {num_workers if num_workers else 'default'} CPU cores for translation.")

    worker_func = _translate_and_update_single_json

    failed_files = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # map 함수를 사용하여 JSON 파일 목록에 대해 병렬 처리
        results = executor.map(worker_func, json_files_to_process)
        for json_path, result in zip(json_files_to_process, results):
            if result is None:
                failed_files.append(json_path)

    # 실패 목록 파일 저장
    if failed_files:
        os.makedirs(failure_log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        failure_log_path = os.path.join(failure_log_dir, f"failed_translation_{timestamp}.txt")
        
        with open(failure_log_path, 'w', encoding='utf-8') as f:
            for file_path in failed_files:
                f.write(f"{file_path}\n")
        print(f"\n⚠️ Found {len(failed_files)} failed files. List saved to: {failure_log_path}")

    success_count = len(json_files_to_process) - len(failed_files)
    print(f"\n--- Translation and Update Complete ---")
    print(f"Successfully processed {success_count} out of {len(json_files_to_process)} JSON files.")

if __name__ == '__main__':
    # --- 실행 설정 --- PYTHONPATH="$(pwd)" python src/autolabel/auto_trans_labeler.py
    INPUT_ROOT_DIR = "data/processed/gangnam/gaepo1_v2/Train" 
    INPUT_ROOT_DIR = "data/processed/gangnam/yeoksam2_v2/Train"  
    FAILURE_LOG_DIR = "assets/logs"
    NUM_CORES = 16

    print("--- Running JSON Description Translator (Recursive) ---")
    translate_descriptions_recursively(
        input_folder=INPUT_ROOT_DIR,
        failure_log_dir=FAILURE_LOG_DIR,
        num_workers=NUM_CORES
    )