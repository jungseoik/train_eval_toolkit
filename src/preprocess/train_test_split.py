import json
import random
from collections import defaultdict
import os

def print_category_distribution(dataset, title):
    """주어진 데이터셋의 카테고리 분포를 출력하는 헬퍼 함수"""
    print(f"\n--- {title} 카테고리 분포 ---")
    counts = defaultdict(int)
    total_items = len(dataset)

    if total_items == 0:
        print("  데이터가 없습니다.")
        return

    for item in dataset:
        try:
            # gpt 응답은 항상 마지막에 위치
            gpt_response_str = item['conversations'][-1]['value']
            gpt_response_json = json.loads(gpt_response_str)
            category = gpt_response_json.get('category')
            if category:
                counts[category] += 1
        except (json.JSONDecodeError, KeyError, IndexError):
            continue
    
    for category, count in sorted(counts.items()):
        percentage = (count / total_items) * 100
        print(f"  - '{category}': {count}개 ({percentage:.2f}%)")


def split_dataset_final(input_file='data.jsonl', test_ratio=0.1, output_dir='.'):
    """
    JSONL 데이터셋을 학습용과 평가용으로 분리합니다.

    Args:
        input_file (str): 원본 JSONL 파일 경로
        test_ratio (float): 평가용 데이터셋의 비율 (0.0 ~ 1.0)
        output_dir (str): train/test JSONL을 저장할 디렉토리 경로
    """
    # 1. 데이터 그룹화
    grouped_data = defaultdict(lambda: defaultdict(list))
    
    total_lines = 0
    processed_lines = 0

    print(f"'{input_file}' 파일을 읽는 중...")

    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                total_lines += 1
                try:
                    data = json.loads(line.strip())

                    if data.get("type") not in ["clip", "video"]:
                        continue
                    
                    video_path = data.get("video", "")
                    if not video_path: continue
                    video_group_key = os.path.dirname(video_path)

                    inner_json_str = data["conversations"][1]["value"]
                    inner_data = json.loads(inner_json_str)
                    category = inner_data.get("category")

                    if not category: continue

                    grouped_data[video_group_key][category].append(data)
                    processed_lines += 1

                except (json.JSONDecodeError, KeyError, IndexError) as e:
                    print(f"  경고: 라인 파싱 오류, 건너뜁니다. 오류: {e}, 라인: {line[:80]}...")
                    continue

    except FileNotFoundError:
        print(f"오류: '{input_file}' 파일을 찾을 수 없습니다.")
        return

    print(f"파일 읽기 완료. 총 {total_lines} 라인 중 {processed_lines} 개의 'clip'/'video' 데이터를 처리합니다.")

    # 3. 데이터 분리
    train_set = []
    test_set = []

    print("\n--- 데이터 분리 시작 ---")
    for group_key, categories in grouped_data.items():
        display_key = os.path.join(*group_key.split(os.sep)[-2:])
        print(f"\n> '{display_key}' 그룹 처리 중...")
        for category, items in categories.items():
            random.shuffle(items)
            
            num_test = round(len(items) * test_ratio)
            
            test_samples = items[:num_test]
            train_samples = items[num_test:]
            
            test_set.extend(test_samples)
            train_set.extend(train_samples)
            
            print(f"  - '{category}' 카테고리: 총 {len(items)}개 -> 학습용 {len(train_samples)}개 | 평가용 {len(test_samples)}개")

    # 4. ID 기준 임시 정렬
    print("\n--- ID 기준 임시 정렬 중 ---")
    train_set.sort(key=lambda x: x.get('id', 0))
    test_set.sort(key=lambda x: x.get('id', 0))
    print("정렬 완료.")
    
    # 5. ID 재인덱싱
    print("--- ID 재인덱싱 시작 ---")
    for i, item in enumerate(train_set):
        item['id'] = i
    for i, item in enumerate(test_set):
        item['id'] = i
    print("ID 재인덱싱 완료.")

    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)

    # 6. 파일 저장
    try:
        train_path = os.path.join(output_dir, 'train.jsonl')
        test_path = os.path.join(output_dir, 'test.jsonl')

        with open(train_path, 'w', encoding='utf-8') as f_train:
            for item in train_set:
                f_train.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        with open(test_path, 'w', encoding='utf-8') as f_test:
            for item in test_set:
                item_copy = item.copy()
                item_copy['conversations'] = [item_copy['conversations'][1]]
                f_test.write(json.dumps(item_copy, ensure_ascii=False) + '\n')

    except IOError as e:
        print(f"파일 쓰기 오류: {e}")
        return

    # --- 카테고리 분포 출력 ---
    print_category_distribution(train_set, "학습용 데이터셋(Train Set)")
    print_category_distribution(test_set, "평가용 데이터셋(Test Set)")

    print("\n---")
    print("✅ 데이터 분리 및 재인덱싱 완료!")
    print(f"총 학습용 데이터: {len(train_set)} 라인 ('{train_path}')")
    print(f"총 평가용 데이터: {len(test_set)} 라인 ('{test_path}')")
    print("---")

if __name__ == '__main__':
    source_jsonl_file = 'data/instruction/train/train_total.jsonl'
    output_dir = 'data/instruction/' 
    if not os.path.exists(source_jsonl_file):
        print(f"warning: '{source_jsonl_file}' no file.")
    split_dataset_final(input_file=source_jsonl_file, test_ratio=0.1, output_dir=output_dir)