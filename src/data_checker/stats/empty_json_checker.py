import json
import os
import sys
import argparse
from pathlib import Path

def count_empty_clips(directory_path):
    """
    지정된 디렉토리의 모든 JSON 파일을 열어서 
    clips가 비어있는 파일 개수를 카운트하고,
    파일명 + 전체 경로도 출력하는 함수
    """
    directory = Path(directory_path)

    if not directory.exists():
        print(f"❌ 오류: 지정한 경로가 존재하지 않습니다: {directory_path}")
        return None, None
    if not directory.is_dir():
        print(f"❌ 오류: '{directory_path}'는 디렉토리가 아닙니다.")
        return None, None

    empty_count = 0
    total_count = 0
    empty_files = []
    
    # 디렉토리 내 모든 .json 파일 확인
    for json_file in directory.glob("*.json"):
        total_count += 1
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            if "clips" in data and len(data["clips"]) == 0:
                empty_count += 1
                empty_files.append((json_file.name, str(json_file.resolve())))
                
        except (json.JSONDecodeError, KeyError, FileNotFoundError) as e:
            print(f"⚠️ Error reading {json_file.name}: {e}")
    
    print(f"\n📊 총 JSON 파일 개수: {total_count}")
    print(f"📉 clips가 비어있는 파일 개수: {empty_count}")
    print(f"📈 비율: {empty_count/total_count*100:.1f}%" if total_count > 0 else "비율 계산 불가")
    
    if empty_files:
        print(f"\n📌 clips가 비어있는 JSON 파일 목록:")
        for name, full_path in empty_files[:10]:
            print(f"  - {name}")
            print(f"    경로: {full_path}")
        if len(empty_files) > 10:
            print(f"  ... 외 {len(empty_files) - 10}개")
    
    return empty_count, total_count


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--json_dir",
        type=str,
        default="data/raw/ai_hub_indoor_store_violence",
        help="JSON 디렉토리 경로 (기본: data/raw/ai_hub_indoor_store_violence)"
    )

    args = parser.parse_args()
    json_directory = args.json_dir

    print(f"➡️ 분석 대상 디렉토리: {json_directory}")
    empty, total = count_empty_clips(json_directory)
