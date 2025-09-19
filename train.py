import json
import os
from pathlib import Path

def count_empty_clips(directory_path):
    """
    지정된 디렉토리의 모든 JSON 파일을 열어서 
    clips가 비어있는 파일 개수를 카운트하는 함수
    """
    directory = Path(directory_path)
    empty_count = 0
    total_count = 0
    empty_files = []
    
    # 디렉토리 내 모든 .json 파일 확인
    for json_file in directory.glob("*.json"):
        total_count += 1
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            # clips가 비어있는지 확인
            if "clips" in data and len(data["clips"]) == 0:
                empty_count += 1
                empty_files.append(json_file.name)
                
        except (json.JSONDecodeError, KeyError, FileNotFoundError) as e:
            print(f"Error reading {json_file.name}: {e}")
    
    print(f"총 JSON 파일 개수: {total_count}")
    print(f"clips가 비어있는 파일 개수: {empty_count}")
    print(f"비율: {empty_count/total_count*100:.1f}%" if total_count > 0 else "비율: 0%")
    
    if empty_files:
        print(f"\n비어있는 파일들:")
        for file in empty_files[:10]:  # 처음 10개만 출력
            print(f"  - {file}")
        if len(empty_files) > 10:
            print(f"  ... 외 {len(empty_files) - 10}개")
    
    return empty_count, total_count

if __name__ == "__main__":
    json_directory = "data/raw/ai_hub_indoor_store_violence"
    
    empty, total = count_empty_clips(json_directory)