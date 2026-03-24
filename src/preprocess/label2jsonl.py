import json
import yaml
from pathlib import Path
from tqdm import tqdm

LABEL2JSONL_YAML = Path(__file__).resolve().parents[2] / "configs" / "label_convert" / "prompts.yaml"


def _load_prompts() -> dict:
    """configs/label_convert/prompts.yaml에서 프롬프트를 로딩한다.

    Returns:
        {(data_type, task_name): prompt_text, ...}
    """
    with open(LABEL2JSONL_YAML, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    mapping = {}
    for task_name, types in data.get("prompts", {}).items():
        for data_type, prompt_text in types.items():
            mapping[(data_type, task_name)] = prompt_text
    return mapping


def create_final_dataset(root_dir: str, base_dir: str = "data/", mode: str = "train",
                         data_type: str = "video", item_type: str = "clip",
                         item_task: str = "caption", task_name: str = "violence"):
    """
    지정된 디렉토리의 JSON 라벨 파일을 재귀 탐색하여 JSONL 데이터셋 구조로 변환한다.

    Args:
        root_dir: JSON과 미디어 파일이 있는 루트 디렉토리.
        base_dir: 상대 경로 계산을 위한 기준 디렉토리.
        mode: 처리 모드 ('train' 또는 'test'). test에서는 human 프롬프트를 제외.
        data_type: 미디어 타입 ('video' 또는 'image').
        item_type: JSONL 'type' 필드값 (예: 'clip', 'capture_frame').
        item_task: JSONL 'task' 필드값 (예: 'caption').
        task_name: 분류 작업명 — 프롬프트 선택 키 (예: 'violence', 'falldown').

    Returns:
        (final_dataset, skip_counts) 튜플.
    """
    final_dataset = []
    current_id = 0
    skip_counts = {
        "missing_keys": 0,
        "no_media": 0,
        "no_prompt": 0,
        "json_error": 0,
        "other_error": 0,
    }

    base_path = Path(base_dir)
    root_path = Path(root_dir)

    if not root_path.is_dir():
        print(f"오류: 디렉토리 '{root_dir}'를 찾을 수 없습니다.")
        return [], skip_counts

    prompt_mapping = _load_prompts()

    # train 모드에서 프롬프트 존재 여부 사전 검증
    if mode != "test":
        prompt_key = (data_type, task_name)
        if prompt_key not in prompt_mapping:
            available = [f"{tn} ({dt})" for (dt, tn) in prompt_mapping.keys()]
            print(f"오류: '{data_type}/{task_name}' 조합에 대한 프롬프트가 없습니다.")
            print(f"  사용 가능: {', '.join(available)}")
            print(f"  configs/label_convert/prompts.yaml에 '{task_name}' 블록을 추가하세요.")
            return [], skip_counts

    print(f"'{root_dir}' 디렉토리에서 JSON 파일 탐색을 시작합니다... (모드: {mode})")

    json_files = list(root_path.rglob("*.json"))

    VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}
    IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    media_exts = VIDEO_EXTS if data_type == "video" else IMAGE_EXTS

    for json_path_obj in tqdm(json_files, desc="JSON 파일 처리 중"):
        try:
            with open(json_path_obj, "r", encoding="utf-8") as f:
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
                skip_counts["missing_keys"] += 1
                continue

            # 미디어 파일 매칭
            stem = json_path_obj.stem
            media_path = None
            for file in json_path_obj.parent.glob(f"{stem}.*"):
                if file.suffix.lower() in media_exts:
                    media_path = file
                    break

            if not media_path:
                skip_counts["no_media"] += 1
                continue

            gpt_value_string = json.dumps(
                {"category": category, "description": description},
                ensure_ascii=False,
            )

            conversations = []
            if mode == "test":
                conversations.append({"from": "gpt", "value": gpt_value_string})
            else:
                human_prompt = prompt_mapping[(data_type, task_name)]
                conversations.append({"from": "human", "value": human_prompt})
                conversations.append({"from": "gpt", "value": gpt_value_string})

            media_relative_path = media_path.relative_to(base_path).as_posix()
            media_key = "video" if data_type == "video" else "image"

            item = {
                "id": current_id,
                "type": item_type,
                "task": item_task,
                media_key: media_relative_path,
                "conversations": conversations,
            }

            final_dataset.append(item)
            current_id += 1

        except json.JSONDecodeError:
            skip_counts["json_error"] += 1
        except Exception as e:
            skip_counts["other_error"] += 1
            print(f"오류: '{json_path_obj}' 처리 중 예외 발생: {e}")

    return final_dataset, skip_counts


def label_to_jsonl_result_save(input_dir, output_file_path, mode="train",
                                data_type="video", base_dir="data/",
                                item_type="clip", item_task="caption",
                                task_name="violence"):
    my_dataset, skip_counts = create_final_dataset(
        input_dir, base_dir, mode=mode, data_type=data_type,
        item_type=item_type, item_task=item_task, task_name=task_name,
    )

    if my_dataset:
        try:
            Path(output_file_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_file_path, "w", encoding="utf-8") as f:
                for entry in tqdm(my_dataset, desc="JSONL 파일 저장 중"):
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")

            total_skipped = sum(skip_counts.values())
            print(f"\n✅ 처리 완료! 총 {len(my_dataset)}개 저장, {total_skipped}개 스킵")
            if total_skipped > 0:
                for reason, count in skip_counts.items():
                    if count > 0:
                        print(f"  - {reason}: {count}건")
            print(f"  → {output_file_path}")
        except Exception as e:
            print(f"\n❌ 오류: '{output_file_path}' 파일 저장 중 오류 발생: {e}")
    else:
        total_skipped = sum(skip_counts.values())
        print(f"\n처리된 데이터가 없습니다. (스킵: {total_skipped}건)")
        if total_skipped > 0:
            for reason, count in skip_counts.items():
                if count > 0:
                    print(f"  - {reason}: {count}건")
        print(f"입력 경로 '{input_dir}'를 확인해주세요.")
