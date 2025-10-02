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
HUMAN_PROMPT_VALUE_FALLDOWN = """<image>
Analyze this image carefully. Determine if a person has fallen down.

Important classification rules:

- The "falldown" category applies to any person who is lying down, regardless of:
  - the surface (e.g., floor, mattress, bed)
  - the posture (natural or unnatural)
  - the cause (e.g., sleeping, collapsing, lying intentionally)
- This includes:
  - A person lying flat on the ground or other surfaces
  - A person collapsed or sprawled in any lying position
- The "normal" category applies only if the person is:
  - sitting
  - standing
  - kneeling
  - or otherwise upright (not lying down)

Answer in JSON format with BOTH of the following fields:
- "category": either "falldown" or "normal"
- "description": a brief reason why this classification was made (e.g., "person lying on a mattress", "person sitting on sofa")

Example:
{ 
  "category": "falldown", 
  "description": "person lying on a mattress in natural posture" 
}
"""



def create_final_dataset(root_dir: str, base_dir:str = "data/", mode: str = "train",
                         data_type: str = "video") -> list:
    """
    pathlib을 사용해 지정된 디렉토리와 모든 하위 디렉토리에서 JSON 파일을 재귀적으로 찾아
    요청된 최종 데이터셋 구조로 변환합니다.
    
    Args:
        root_dir (str): JSON과 비디오 파일이 있는 루트 디렉토리.
        base_dir (str): 상대 경로 계산을 위한 기준 디렉토리.
        mode (str): 처리 모드 ('train' 또는 'test'). 'test' 모드에서는 human 프롬프트를 제외합니다.
        data_type (str): 'video' 또는 'image' 선택
    """
    final_dataset = []
    current_id = 0
    
    search_path = Path(root_dir)
    base_path = Path(base_dir)
    root_path = Path(root_dir)

    if not root_path.is_dir():
        print(f"오류: 디렉토리 '{root_dir}'를 찾을 수 없습니다.")
        return []

    print(f"'{root_dir}' 디렉토리에서 JSON 파일 탐색을 시작합니다... (모드: {mode})")
    
    json_files = list(root_path.rglob('*.json'))
    
    for json_path_obj in tqdm(json_files, desc="JSON 파일 처리 중"):
        try:
            with open(json_path_obj, 'r', encoding='utf-8') as f:
                data = json.load(f)

            category = None
            description = None

            if isinstance(data, dict):
                category = data.get("category")
                description = data.get("description")
            
            elif isinstance(data, list) and data:
                first_item = data[0]
                if isinstance(first_item, dict):
                    category = first_item.get("category")
                    description = first_item.get("eng_caption") or first_item.get("en_caption")

            if category is None or description is None:
                print(f"경고: '{json_path_obj}'에 필수 키가 없어 건너뜁니다.")
                continue

            video_stem = json_path_obj.stem
            # video_path = None
            
            possible_files = list(json_path_obj.parent.glob(f"{video_stem}.*"))
            media_path = None
            
            for file in possible_files:
                if file.suffix.lower() != '.json':
                    if data_type == "video" and file.suffix.lower() in [".mp4", ".avi", ".mov", ".mkv"]:
                        media_path = file
                        break
                    elif data_type == "image" and file.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
                        media_path = file
                        break
                    # video_path = file
                    # break 

            if not media_path:
                print(f"경고: '{json_path_obj}'에 해당하는 비디오 파일을 찾을 수 없습니다. 건너뜁니다.")
                continue
            
            # video_relative_path = video_path.relative_to(base_path).as_posix()
            
            gpt_value_dict = {
                "category": category,
                "description": description
            }
            gpt_value_string = json.dumps(gpt_value_dict, ensure_ascii=False)

            # --- 💡 추가된 부분 시작: mode에 따라 conversations 구조를 다르게 설정 ---
            conversations = []
            if mode == "test":
                # 'test' 모드일 경우 'gpt' 부분만 추가
                conversations.append({"from": "gpt", "value": gpt_value_string})
            else:
                if data_type == "video":
                # 기본('train') 모드일 경우 기존과 동일하게 'human'과 'gpt' 모두 추가
                    conversations.append({"from": "human", "value": HUMAN_PROMPT_VALUE})
                    conversations.append({"from": "gpt", "value": gpt_value_string})
                elif data_type == "image":
                    conversations.append({"from": "human", "value": HUMAN_PROMPT_VALUE_FALLDOWN})
                    conversations.append({"from": "gpt", "value": gpt_value_string})
                    
            # --- 💡 추가된 부분 끝 ---
            media_relative_path = media_path.relative_to(base_path).as_posix()
            if data_type == "video":
                item = {
                    "id": current_id,
                    "type": "clip",
                    "task": "caption",
                    "video": media_relative_path,
                    "conversations": conversations
                }
            elif data_type == "image":
                item = {
                    "id": current_id,
                    "type": "capture_frame",
                    "task": "caption",
                    "image": media_relative_path,
                    "conversations": conversations
                }
            
            # item = {
            #     "id": current_id,
            #     "type": "clip",
            #     "task": "caption",
            #     "video": video_relative_path,
            #     # 'conversations' 키에 위에서 생성한 리스트를 할당
            #     "conversations": conversations
            # }
            final_dataset.append(item)
            current_id += 1

        except json.JSONDecodeError:
            print(f"경고: '{json_path_obj}'는 올바른 JSON 파일이 아닙니다. 건너뜁니다.")
        except Exception as e:
            print(f"오류: '{json_path_obj}' 처리 중 예외 발생: {e}")

    return final_dataset

# --- 💡 수정된 부분: mode 파라미터 추가 ---
def label_to_jsonl_result_save(input_dir, output_file_path, mode="train", data_type="video", base_dir="data/" ):
    # create_final_dataset 함수 호출 시 mode 인자 전달
    my_dataset = create_final_dataset(input_dir, base_dir, mode=mode, data_type=data_type)
    if my_dataset:
            try:
                with open(output_file_path, 'w', encoding='utf-8') as f:
                    for entry in tqdm(my_dataset, desc="JSONL 파일 저장 중"):
                        f.write(json.dumps(entry, ensure_ascii=False) + '\n')
                
                print(f"\n✅ 처리 완료! 총 {len(my_dataset)}개의 항목을 '{output_file_path}' 파일에 저장했습니다.")
            except Exception as e:
                print(f"\n❌ 오류: '{output_file_path}' 파일 저장 중 오류 발생: {e}")
    else:
        print(f"\n처리된 데이터가 없습니다. 입력 경로 '{input_dir}'를 확인해주세요.")

# --- 스크립트를 직접 실행할 때 사용되는 부분 ---
if __name__ == '__main__':
    input_directory = "data"  
    output_file_path = "final_dataset.jsonl"
    
    # --- 💡 추가된 부분: 처리 모드 설정 ---
    # 'test'로 설정하면 human 파트가 제외됩니다.
    # 기존처럼 human 파트를 포함하려면 'train'으로 설정하세요.
    processing_mode = "test" 
    
    # 함수 호출 시 설정한 mode를 전달
    label_to_jsonl_result_save(input_directory, output_file_path, mode=processing_mode)

    