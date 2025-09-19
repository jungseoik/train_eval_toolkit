import json
import os
from typing import List, Dict, Any
from pathlib import Path
class JSONLMerger:
    """JSONL 파일을 합치고 ID를 연속적으로 다시 매기는 클래스"""
    def __init__(self):
        pass
    def read_jsonl(self, file_path: str) -> List[Dict[Any, Any]]:
        """JSONL 파일을 읽어서 리스트로 반환"""
        data = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:  # 빈 줄 제외
                        data.append(json.loads(line))
            return data
        except FileNotFoundError:
            print(f"파일을 찾을 수 없습니다: {file_path}")
            return []
        except json.JSONDecodeError as e:
            print(f"JSON 파싱 에러: {e}")
            return []
    
    def write_jsonl(self, data: List[Dict[Any, Any]], output_path: str):
        """데이터를 JSONL 파일로 저장"""
        try:
            path = Path(output_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                for item in data:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
            print(f"파일이 성공적으로 저장되었습니다: {output_path}")
        except Exception as e:
            print(f"파일 저장 에러: {e}")
    
    def merge_jsonl_files(self, file1_path: str, file2_path: str, output_path: str):
        """두 JSONL 파일을 합치고 ID를 연속적으로 다시 매기기"""
        print(f"첫 번째 파일 읽는 중: {file1_path}")
        data1 = self.read_jsonl(file1_path)
        print(f"두 번째 파일 읽는 중: {file2_path}")
        data2 = self.read_jsonl(file2_path)
        
        if not data1 and not data2:
            print("읽을 데이터가 없습니다.")
            return
        
        merged_data = data1 + data2
        for i, item in enumerate(merged_data):
            item['id'] = i
        
        print(f"총 {len(merged_data)}개 항목이 합쳐졌습니다.")
        print(f"ID 범위: 0 ~ {len(merged_data) - 1}")
        
        self.write_jsonl(merged_data, output_path)
        
        return merged_data

if __name__ == "__main__":
    merger = JSONLMerger()
    
    file1 = "final_dataset_preprocess.jsonl"  # 첫 번째 JSONL 파일
    file2 = "final_dataset_raw.jsonl"  # 두 번째 JSONL 파일
    output = "merged.jsonl"  # 출력 파일
    
    # 파일 합치기
    merger.merge_jsonl_files(file1, file2, output)