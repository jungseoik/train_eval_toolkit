import json
from pathlib import Path
from tqdm import tqdm

HUMAN_PROMPT_VALUE = """<video>
Watch this short video clip and respond with exactly one JSON object.

[Rules]
- The category must be either 'violence' or 'normal'.  
- Classify as violence if any of the following actions are present:  
  * Punching  
  * Kicking  
  * Weapon Threat
  * Weapon Attack
  * Falling/Takedown  
  * Pushing/Shoving  
  * Brawling/Group Fight  
- If none of the above are observed, classify as normal.  
- The following cases must always be classified as normal:  
  * Affection (hugging, holding hands, light touches)  
  * Helping (supporting, assisting)  
  * Accidental (unintentional bumping)  
  * Playful (non-aggressive playful contact)  

[Output Format]
- Output exactly one JSON object.  
- The object must contain only two keys: "category" and "description".  
- The description should briefly and objectively describe the scene.  

Example (violence):  
{"category":"violence","description":"A man in a black jacket punches another man, who stumbles backward."}

Example (normal):  
{"category":"normal","description":"Two people are hugging inside an elevator."}
"""


def create_final_dataset(root_dir: str, base_dir:str = "data/") -> list:
    """
    pathlib을 사용해 지정된 디렉토리와 모든 하위 디렉토리에서 JSON 파일을 재귀적으로 찾아
    요청된 최종 데이터셋 구조로 변환합니다.
    """
    final_dataset = []
    current_id = 0
    
    # 파일 검색 경로는 기존과 동일
    search_path = Path(root_dir)
    # 2. 상대 경로 계산을 위한 기준 경로를 새로 정의
    base_path = Path(base_dir)
    root_path = Path(root_dir)

    if not root_path.is_dir():
        print(f"오류: 디렉토리 '{root_dir}'를 찾을 수 없습니다.")
        return []

    print(f"'{root_dir}' 디렉토리에서 JSON 파일 탐색을 시작합니다...")
    
    json_files = list(root_path.rglob('*.json'))
    
    for json_path_obj in tqdm(json_files, desc="JSON 파일 처리 중"):
        try:
            with open(json_path_obj, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # --- 수정된 부분 시작 ---

            category = None
            description = None

            # 1. data가 딕셔너리 형태일 경우 (기존 구조)
            if isinstance(data, dict):
                category = data.get("category")
                description = data.get("description")
            
            # 2. data가 리스트 형태일 경우 (en_caption이 있는 구조)
            elif isinstance(data, list) and data:  # 리스트이고 비어있지 않은지 확인
                first_item = data[0]
                if isinstance(first_item, dict):
                    category = first_item.get("category")
                    description = first_item.get("eng_caption") or first_item.get("en_caption") # 'en_caption' 키에서 설명 추출
            
            # --- 수정된 부분 끝 ---

            if category is None or description is None:
                print(f"경고: '{json_path_obj}'에 필수 키가 없어 건너뜁니다.")
                continue

            # video_stem = json_path_obj.stem
            # video_filename = f"{video_stem}.mp4"
            # video_path = json_path_obj.with_name(video_filename)
            # --- 💡 수정된 부분 시작 💡 ---
            
            video_stem = json_path_obj.stem
            video_path = None
            
            # JSON 파일과 동일한 이름(stem)을 가진 모든 파일을 찾음
            # 예: 'video1.json' -> 'video1.*' (video1.mp4, video1.mov 등)
            possible_files = list(json_path_obj.parent.glob(f"{video_stem}.*"))

            for file in possible_files:
                # 찾은 파일 중, 확장자가 .json이 아닌 첫 번째 파일이 비디오 파일임
                if file.suffix.lower() != '.json':
                    video_path = file
                    break # 비디오 파일을 찾았으므로 반복 중단

            # 만약 해당하는 비디오 파일을 찾지 못했다면, 이 JSON 파일은 건너뜀
            if not video_path:
                print(f"경고: '{json_path_obj}'에 해당하는 비디오 파일을 찾을 수 없습니다. 건너뜁니다.")
                continue
            
            # --- 💡 수정된 부분 끝 💡 ---
            
            
            # video_relative_path = video_path.relative_to(root_path).as_posix()
            video_relative_path = video_path.relative_to(base_path).as_posix()
            gpt_value_dict = {
                "category": category,
                "description": description
            }
            gpt_value_string = json.dumps(gpt_value_dict, ensure_ascii=False)

            item = {
                "id": current_id,
                "type": "clip",
                "task": "caption",
                "video": video_relative_path,
                "conversations": [
                    {"from": "human", "value": HUMAN_PROMPT_VALUE},
                    {"from": "gpt", "value": gpt_value_string}
                ]
            }
            final_dataset.append(item)
            current_id += 1

        except json.JSONDecodeError:
            print(f"경고: '{json_path_obj}'는 올바른 JSON 파일이 아닙니다. 건너뜁니다.")
        except Exception as e:
            print(f"오류: '{json_path_obj}' 처리 중 예외 발생: {e}")

    return final_dataset
def label_to_jsonl_result_save(input_dir, output_file_path, base_dir = "data/"):
    my_dataset = create_final_dataset(input_dir, base_dir)
    if my_dataset:
            # ❗️ 생성된 데이터셋을 .jsonl 형식으로 저장
            try:
                with open(output_file_path, 'w', encoding='utf-8') as f:
                    for entry in tqdm(my_dataset, desc="JSONL 파일 저장 중"):
                        # 각 딕셔너리를 JSON 문자열로 변환하고 줄바꿈 문자를 추가하여 파일에 쓴다.
                        f.write(json.dumps(entry, ensure_ascii=False) + '\n')
                
                print(f"\n✅ 처리 완료! 총 {len(my_dataset)}개의 항목을 '{output_file_path}' 파일에 저장했습니다.")
            except Exception as e:
                print(f"\n❌ 오류: '{output_file_path}' 파일 저장 중 오류 발생: {e}")
    else:
        print(f"\n처리된 데이터가 없습니다. 입력 경로 '{input_dir}'를 확인해주세요.")

# --- 스크립트를 직접 실행할 때 사용되는 부분 ---
if __name__ == '__main__':
    # ❗️ 여기에 실제 데이터가 있는 상위 폴더 경로를 입력하세요.

    input_directory = "data"  
    output_file_path = "final_dataset.jsonl"
    label_to_jsonl_result_save(input_directory , output_file_path)

    