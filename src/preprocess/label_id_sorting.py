import json
from typing import List, Dict
from pathlib import Path

def load_jsonl(path: str) -> List[Dict]:
    """JSONL 파일을 읽어 리스트로 반환"""
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows

def reindex_by_id(rows: List[Dict], key: str = "id") -> List[Dict]:
    """key 기준 정렬 후 0부터 다시 인덱스 부여"""
    rows_sorted = sorted(rows, key=lambda x: int(x.get(key, 0)))
    for new_id, obj in enumerate(rows_sorted):
        obj[key] = new_id
    return rows_sorted

def save_jsonl(rows: List[Dict], path: str):
    """리스트를 JSONL 형식으로 저장"""
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, "w", encoding="utf-8") as f:
        for obj in rows:
            f.write(json.dumps(obj, ensure_ascii=False))
            f.write("\n")
            
if __name__ == '__main__':
    rows = load_jsonl("test_gangnam_only.jsonl")
    rows_reindexed = reindex_by_id(rows)
    save_jsonl(rows_reindexed, "data_reindexed.jsonl")
