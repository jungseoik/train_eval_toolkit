import os
import json
from collections import Counter

def analyze_jsonl_folder(folder_path):
    for filename in os.listdir(folder_path):
        if not filename.endswith(".jsonl"):
            continue
        
        file_path = os.path.join(folder_path, filename)
        categories = []

        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    row = json.loads(line.strip())
                    # GPT 응답 부분 파싱
                    gpt_value = row["conversations"][-1]["value"]
                    gpt_obj = json.loads(gpt_value)
                    categories.append(gpt_obj["category"])
                except Exception as e:
                    print(f"⚠️ Error parsing line in {filename}: {e}")
        
        total = len(categories)
        counter = Counter(categories)
        
        print(f"\n📂 File: {filename}")
        print(f"총 row 개수: {total}")
        for cat, count in counter.items():
            ratio = count / total * 100 if total > 0 else 0
            print(f"- {cat}: {count}개 ({ratio:.2f}%)")

# 사용 예시
folder = "data/instruction/train"  # 👉 여기 원하는 경로로 바꾸세요
analyze_jsonl_folder(folder)
