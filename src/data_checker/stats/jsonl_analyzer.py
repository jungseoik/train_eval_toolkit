from __future__ import annotations

import json
import os
from collections import Counter
from dataclasses import dataclass
from typing import Iterator, Optional


@dataclass
class JSONLFileStats:
    """한 개 JSONL 파일에 대한 요약 통계."""

    file_path: str
    total_rows: int
    category_counts: Counter
    error_count: int


def _extract_category(row: dict) -> Optional[str]:
    """conversations의 마지막 응답에서 category를 뽑아낸다."""
    try:
        conversations = row["conversations"]
        response_value = conversations[-1]["value"]
        response_json = json.loads(response_value)
    except (KeyError, IndexError, json.JSONDecodeError):
        return None

    return response_json.get("category")


def analyze_jsonl_file(file_path: str) -> JSONLFileStats:
    """
    JSONL 파일 하나를 읽어 카테고리 빈도/총 개수를 계산한다.
    parsing 오류 개수까지 함께 반환한다.
    """

    categories = []
    error_count = 0
    with open(file_path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue

            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                error_count += 1
                continue

            category = _extract_category(row)
            if category is None:
                error_count += 1
                continue

            categories.append(category)

    return JSONLFileStats(
        file_path=file_path,
        total_rows=len(categories),
        category_counts=Counter(categories),
        error_count=error_count,
    )


def analyze_jsonl_folder(folder_path: str) -> Iterator[JSONLFileStats]:
    """
    폴더 내부의 *.jsonl 파일을 순회하며 통계를 yield 한다.
    """

    for entry in sorted(os.listdir(folder_path)):
        if not entry.endswith(".jsonl"):
            continue

        file_path = os.path.join(folder_path, entry)
        if not os.path.isfile(file_path):
            continue

        yield analyze_jsonl_file(file_path)


def print_category_report(stats: JSONLFileStats) -> None:
    """카테고리 통계를 보기 좋게 출력."""

    total = stats.total_rows
    print(f"\n📂 File: {os.path.basename(stats.file_path)}")
    print(f"총 유효 row 개수: {total}")

    if stats.error_count:
        print(f"⚠️ 파싱 실패 row: {stats.error_count}")

    if total == 0:
        print("- 카테고리 데이터가 없습니다.")
        return

    for category, count in stats.category_counts.items():
        ratio = (count / total) * 100
        print(f"- {category}: {count}개 ({ratio:.2f}%)")
