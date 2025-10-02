import argparse
import json
import sys
from collections import Counter
from typing import Any, Dict, Iterable, Optional, Union, List
from pathlib import Path

def remove_human_video_prompts(jsonl_path, output_path=None):
    """
    JSONL 파일에서 {"from": "human", "value": "<video>..."} 형태의 대화만 삭제하고 다시 저장
    
    Args:
        jsonl_path (str): 원본 JSONL 파일 경로
        output_path (str, optional): 수정된 JSONL 저장 경로 (None이면 원본 파일 덮어씀)
    """
    if output_path is None:
        output_path = jsonl_path  # 원본 덮어쓰기

    new_lines = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)

            if "conversations" in obj:
                # "human" + "<video>" 프롬프트 제거
                obj["conversations"] = [
                    conv for conv in obj["conversations"]
                    if not (conv.get("from") == "human" and conv.get("value", "").startswith("<video>"))
                ]
            new_lines.append(obj)

    # 다시 저장
    with open(output_path, "w", encoding="utf-8") as f:
        for obj in new_lines:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print(f"✅ 저장 완료: {output_path}")
# ============================================
# JSONL 분석 함수들
# ============================================

def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """JSONL 파일을 읽어서 dict 리스트로 반환"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"⚠️  Line {line_no}: JSON 파싱 실패 - {e}")
    return data


def parse_json_value(value: Any) -> Optional[Dict[str, Any]]:
    """문자열을 JSON으로 파싱, 이미 dict면 그대로 반환"""
    if isinstance(value, dict):
        return value
    
    if isinstance(value, str):
        try:
            parsed = json.loads(value.strip())
            return parsed if isinstance(parsed, dict) else None
        except (json.JSONDecodeError, AttributeError):
            return None
    
    return None


def extract_category(item: Dict[str, Any]) -> Optional[str]:
    """
    한 레코드에서 마지막 gpt 응답의 category 추출
    
    Returns:
        category 문자열 (소문자) 또는 None
    """
    conversations = item.get('conversations')
    if not isinstance(conversations, list):
        return None
    
    gpt_messages = [
        msg for msg in conversations 
        if isinstance(msg, dict) and msg.get('from') == 'gpt'
    ]
    
    if not gpt_messages:
        return None
    
    # 마지막 gpt 메시지부터 역순으로 탐색
    for msg in reversed(gpt_messages):
        parsed = parse_json_value(msg.get('value'))
        if parsed and 'category' in parsed:
            return str(parsed['category']).strip().lower()
    
    return None


def get_category_distribution(data: List[Dict[str, Any]]) -> Dict[str, int]:
    """데이터셋의 category 분포 집계"""
    categories = []
    
    for item in data:
        cat = extract_category(item)
        if cat:
            categories.append(cat)
    
    return dict(Counter(categories))


def _print_single_stats(file_path: str, stats: Dict[str, int], total: int) -> None:
    """단일 파일의 통계 출력 (내부 헬퍼 함수)"""
    valid_count = sum(stats.values())
    
    print(f"\n📄 {Path(file_path).name}")
    print(f"{'─'*60}")
    print(f"  총 샘플: {total:,}개  |  유효: {valid_count:,}개")
    print(f"{'─'*60}")
    
    # 모든 카테고리 출력 (빈도순)
    for cat, count in sorted(stats.items(), key=lambda x: x[1], reverse=True):
        pct = (count / valid_count * 100) if valid_count > 0 else 0
        print(f"    {cat:15s}: {count:7,}개  ({pct:6.2f}%)")
    
    # 누락된 샘플
    missing = total - valid_count
    if missing > 0:
        print(f"  ⚠️  Category 누락: {missing:,}개")


def print_dataset_info(file_paths: Union[str, List[str]]) -> None:
    """
    JSONL 데이터셋의 정보 출력
    
    Args:
        file_paths: JSONL 파일 경로 (단일 문자열 또는 리스트)
    """
    # 단일 파일을 리스트로 변환
    if isinstance(file_paths, str):
        file_paths = [file_paths]
    
    print(f"\n{'='*60}")
    print(f"📊 Dataset Analysis")
    print(f"{'='*60}")
    
    # 각 파일별 데이터 수집
    all_data = []
    file_stats = []
    
    for file_path in file_paths:
        data = load_jsonl(file_path)
        stats = get_category_distribution(data)
        
        all_data.extend(data)
        file_stats.append({
            'path': file_path,
            'data': data,
            'stats': stats,
            'total': len(data)
        })
    
    # 1. 개별 파일 정보 출력
    if len(file_paths) > 1:
        print(f"\n[ 개별 파일 정보 ]")
        for fs in file_stats:
            _print_single_stats(fs['path'], fs['stats'], fs['total'])
    
    # 2. 전체 통합 정보 출력
    if len(file_paths) > 1:
        print(f"\n{'='*60}")
        print(f"📊 전체 통합 통계")
        print(f"{'='*60}")
    
    total_samples = len(all_data)
    total_stats = get_category_distribution(all_data)
    valid_count = sum(total_stats.values())
    
    print(f"\n총 파일 수: {len(file_paths)}개")
    print(f"총 샘플 수: {total_samples:,}개")
    print(f"유효 샘플: {valid_count:,}개")
    
    print(f"\n{'─'*60}")
    print(f"Category 분포")
    print(f"{'─'*60}")
    
    # 모든 카테고리 출력 (빈도순)
    for cat, count in sorted(total_stats.items(), key=lambda x: x[1], reverse=True):
        pct = (count / valid_count * 100) if valid_count > 0 else 0
        print(f"  {cat:15s}: {count:7,}개  ({pct:6.2f}%)")
    
    # 누락된 샘플
    missing = total_samples - valid_count
    if missing > 0:
        print(f"\n⚠️  Category 누락: {missing:,}개")
    
    print(f"{'='*60}\n")


def get_dataset_summary(file_path: str) -> Dict[str, Any]:
    """
    데이터셋 요약 정보를 dict로 반환
    
    Returns:
        {
            'total_samples': int,
            'valid_samples': int,
            'missing_samples': int,
            'categories': {'category': count, ...}
        }
    """
    data = load_jsonl(file_path)
    stats = get_category_distribution(data)
    
    total = len(data)
    valid = sum(stats.values())
    
    return {
        'total_samples': total,
        'valid_samples': valid,
        'missing_samples': total - valid,
        'categories': stats
    }


if __name__ == "__main__":
    print_dataset_info("data/instruction/train/train_abb_vqa.jsonl")
    # remove_human_video_prompts("data/instruction/train/train_abb_vqa_train.jsonl")
