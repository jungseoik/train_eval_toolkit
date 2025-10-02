import pandas as pd
import os
from glob import glob
from src.autolabel.gemini.translator.vertex_translate import translate_korean_to_english
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import json

# IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".webp"]
IMAGE_EXTS  = [".mp4", ".mov" , ".avi"]


def merge_csv_files(file1_path = None, file2_path = None, folder = None, output="merged.csv") -> None:
    dataframes = []
    if file1_path and file2_path:
        dataframes.append(pd.read_csv(file1_path))
        dataframes.append(pd.read_csv(file2_path))
    elif folder:
        for f in glob(os.path.join(folder, "*.csv")):
            dataframes.append(pd.read_csv(f))
    else:
        raise ValueError("no path or folder path")
    
    merged = pd.concat(dataframes, ignore_index=True)
    merged.to_csv(output, index=False)
    print("--------megerd complete-----------")    
    return None

def translate_kor_to_eng_csv(input_file:str, target_col:str, output_file:str = "translated.csv",num_workers: int = 1):
    """
    CSV에서 target_col을 번역해 target_col_eng 컬럼으로 추가 후 저장
    
    Args:
        input_file (str): 입력 CSV 파일 경로
        target_col (str): 번역할 대상 컬럼 이름
        output_file (str): 출력 CSV 파일 경로 (기본값: translated.csv)
        num_workers (int): 프로세스 개수 (기본값: CPU 코어수 - 1)
    """
    df = pd.read_csv(input_file)
    if target_col not in df.columns:
        raise ValueError(f"{target_col}이 csv 칼럼에 존재하지 않습니다.")
    
    texts = df[target_col].astype(str).tolist()
    if num_workers is None:
        num_workers = max(1, cpu_count() -1)
    
    translated = []
    with Pool(processes=num_workers) as pool:
        for res in tqdm(pool.imap(translate_korean_to_english , texts) , total = len(texts)):
            translated.append(res)
        
    new_col = f"{target_col}_eng"
    df[new_col] = translated
    df.to_csv(output_file , index=False)
    return df


def find_file_recursive(root_dir: str, filename: str):
    """폴더 전체를 재귀적으로 탐색하여 filename.*(image) 파일 경로 반환"""
    for dirpath, _, files in os.walk(root_dir):
        for f in files:
            name, ext = os.path.splitext(f)
            if name == filename and ext.lower() in IMAGE_EXTS:
                return os.path.join(dirpath, f)
    return None

def process_row(args):
    """한 행(row)을 처리해서 JSON 저장"""
    row, file_col, cat_col, desc_col, search_dir = args
    filename = str(row[file_col])
    category = str(row[cat_col])
    description = str(row[desc_col])

    img_path = find_file_recursive(search_dir, filename)
    if not img_path:
        return f"⚠️ 이미지 못 찾음: {filename}"

    # JSON 경로
    json_path = os.path.splitext(img_path)[0] + ".json"

    data = {
        "category": category,
        "description": description
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    return f"✅ 완료: {json_path}"

def generate_label_json_from_csv(
    input_csv: str,
    file_col: str,
    category_col: str,
    desc_col: str,
    search_dir: str,
    num_workers: int = 1
):
    df = pd.read_csv(input_csv)

    if num_workers is None:
        num_workers = max(1, cpu_count() - 1)

    tasks = [
        (row, file_col, category_col, desc_col, search_dir)
        for _, row in df.iterrows()
    ]

    results = []
    with Pool(processes=num_workers) as pool:
        for res in tqdm(pool.imap(process_row, tasks), total=len(tasks)):
            results.append(res)

    return results

if __name__ == "__main__":
    # merge_csv_files(folder="assets/gangnam")
    
    
#     df = translate_kor_to_eng_csv(
#     input_file="assets/gangnam/falldown/역삼2동.csv",
#     target_col="Description",
#     output_file="assets/gangnam/falldown/역삼2동_eng.csv",
#     num_workers=16
# )

    results = generate_label_json_from_csv(
        input_csv="assets/gangnam/violence/개포1동_violence_eng.csv",
        file_col="클립명",      # CSV의 파일명 칼럼
        category_col="구분",  # CSV의 카테고리 칼럼
        desc_col="디스크립션_eng",   # CSV의 설명 칼럼
        search_dir="data/processed/gangnam/gaepo1",  # 이미지 탐색 폴더
        num_workers=16
    )
    
    # results = generate_label_json_from_csv(
    #     input_csv="assets/gangnam/violence/역삼2동_violence_eng.csv",
    #     file_col="클립명",      # CSV의 파일명 칼럼
    #     category_col="구분",  # CSV의 카테고리 칼럼
    #     desc_col="디스크립션_eng",   # CSV의 설명 칼럼
    #     search_dir="data/processed/gangnam/yeoksam2",  # 이미지 탐색 폴더
    #     num_workers=16
    # )