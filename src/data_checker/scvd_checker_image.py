import json
import os

def extract_mismatch_paths(json_file_path):
    """
    JSONL 파일에서 video 경로와 conversations category 불일치한 것들을 찾는다.
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


def fix_json_labels(mismatched_paths: list):
    """
    주어진 비디오 경로 리스트에 대해 동일 경로의 .json 파일을 열어
    category/description을 교정 후 저장한다.
    """
    for video_path in mismatched_paths:
        json_path = os.path.splitext(video_path)[0] + ".json"
        if not os.path.exists(json_path):
            print(f"⚠️ JSON file not found for {video_path}")
            continue

        # video_path 기반 카테고리 결정
        if "normal" in video_path.lower():
            correct_category = "normal"
            correct_description = "The video depicts a normal, non-violent situation with peaceful behavior throughout."
        else:
            correct_category = "violence"
            correct_description = "The video shows a violent situation with aggressive physical behavior."

        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # 카테고리 & 설명 교정
            data["category"] = correct_category
            data["description"] = correct_description

            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=4)

            print(f"✅ Fixed JSON -> {json_path}")

        except Exception as e:
            print(f"❌ Failed to fix {json_path}: {e}")


if __name__ == "__main__":
    JSONL_FILE = "/home/piawsa6000/nas192/datasets/projects/gangnam_innovation/violence_fintuning/TADO_Violence_GangNAM/data/instruction/train/scvdALL_no_split.jsonl"

    mismatched = extract_mismatch_paths(JSONL_FILE)
    print(f"Found {len(mismatched)} mismatched files")

    fix_json_labels(mismatched)
