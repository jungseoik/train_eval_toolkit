import argparse
import json
import sys
from collections import Counter
from typing import Any, Dict, Iterable, Optional, Union

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

def safe_loads_json_maybe(val: Union[str, Dict[str, Any], None]) -> Optional[Dict[str, Any]]:
    """val이 JSON 문자열이면 json.loads, 이미 dict면 그대로 반환, 아니면 None."""
    if val is None:
        return None
    if isinstance(val, dict):
        return val
    if isinstance(val, str):
        s = val.strip()
        if not s:
            return None
        try:
            obj = json.loads(s)
            return obj if isinstance(obj, dict) else None
        except json.JSONDecodeError:
            return None
    return None

def extract_gpt_category(item: Dict[str, Any]) -> Optional[str]:
    """
    한 레코드에서 마지막 "from":"gpt"의 value를 JSON으로 파싱해 category를 반환.
    실패 시 None.
    """
    conv = item.get("conversations")
    if not isinstance(conv, list):
        return None

    gpt_entries = [c for c in conv if isinstance(c, dict) and c.get("from") == "gpt"]
    if not gpt_entries:
        return None

    for c in reversed(gpt_entries):  # 마지막 gpt부터 시도
        value = c.get("value")
        obj = safe_loads_json_maybe(value)
        if obj and "category" in obj:
            return str(obj["category"]).strip().lower()
    return None

def iter_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    """JSONL 파일을 줄 단위로 dict로 yield. 잘못된 줄은 경고 후 스킵."""
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    yield lineno, obj
                else:
                    print(f"[경고] {lineno}행: dict가 아닌 JSON - 스킵", file=sys.stderr)
            except json.JSONDecodeError as e:
                print(f"[경고] {lineno}행 JSON 파싱 실패: {e} - 스킵", file=sys.stderr)

def main():
    ap = argparse.ArgumentParser(description="JSONL의 GPT category(violence/normal) 비율 집계")
    ap.add_argument("jsonl_path", help="입력 JSONL 파일 경로")
    ap.add_argument("--verbose", action="store_true", help="스킵/이상치 상세 로그 출력")
    args = ap.parse_args()

    counts = Counter()
    total_valid = 0
    skipped = 0

    for lineno, item in iter_jsonl(args.jsonl_path):
        cat = extract_gpt_category(item)
        if cat in ("violence", "normal"):
            counts[cat] += 1
            total_valid += 1
        else:
            skipped += 1
            if args.verbose:
                print(f"[스킵] {lineno}행: category 없음/파싱 실패/정의 외 값", file=sys.stderr)

    # 콘솔 요약 출력
    print("\n=== Category Composition ===")
    print(f"총 유효 샘플: {total_valid}")
    for cat in ("violence", "normal"):
        n = counts.get(cat, 0)
        pct = (n / total_valid * 100.0) if total_valid > 0 else 0.0
        print(f"- {cat:8s}: {n:6d}  ({pct:6.2f}%)")
    print(f"기타/누락(파싱 실패·정의 외 값): {skipped}\n")

if __name__ == "__main__":
    
    remove_human_video_prompts("data/instruction/evaluation/test_rwf2000.jsonl")
    # main()
    # python src/utils/jsonl_inform_check.py configs/instruction/exper000.jsonl
