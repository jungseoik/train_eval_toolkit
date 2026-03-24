"""
JSON 라벨 파일 카테고리 분포 점검 모듈.

하위 폴더를 재귀 탐색하며 카테고리 분포 통계 및 낮은 비율 카테고리를 탐지한다.

사용법:
    from src.data_checker.stats.json_checker import check_json_directory
    check_json_directory("data/processed/gangnam")
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from typing import Dict, List, Tuple


def _get_category_stats_in_dir(directory: str) -> Dict[str, int]:
    """디렉토리 내 JSON 파일의 category 빈도수를 계산한다."""
    category_counts: Dict[str, int] = defaultdict(int)

    for entry in os.listdir(directory):
        if not entry.endswith(".json"):
            continue
        full_path = os.path.join(directory, entry)
        if not os.path.isfile(full_path):
            continue

        try:
            with open(full_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if "category" in data and isinstance(data["category"], str):
                category = data["category"].strip().lower()
                category_counts[category] += 1
        except (json.JSONDecodeError, Exception):
            continue

    return dict(category_counts)


def check_json_directory(target_path: str, low_threshold: float = 0.49) -> None:
    """
    대상 경로를 재귀 탐색하며 JSON 라벨 파일의 카테고리 분포를 점검한다.

    Args:
        target_path: 점검할 최상위 디렉토리 경로.
        low_threshold: 낮은 비율로 간주할 기준 (0.0~1.0). 기본 0.49 (49%).
    """
    abs_path = os.path.abspath(target_path)

    if not os.path.isdir(abs_path):
        print(f"오류: 유효한 디렉토리 경로가 아닙니다: {abs_path}")
        return

    print("=" * 60)
    print(f"  JSON 라벨 점검 시작: {abs_path}")
    print("=" * 60)

    total_json = 0
    low_proportion_files: Dict[str, List[Tuple[str, str]]] = defaultdict(list)

    for root, _dirs, _files in os.walk(abs_path):
        category_counts = _get_category_stats_in_dir(root)
        if not category_counts:
            continue

        total_in_dir = sum(category_counts.values())
        total_json += total_in_dir
        proportions = {
            cat: cnt / total_in_dir for cat, cnt in category_counts.items()
        }

        low_categories = [
            cat for cat, prop in proportions.items() if prop < low_threshold
        ]

        # 낮은 비율 파일 수집
        if low_categories:
            for entry in os.listdir(root):
                if not entry.endswith(".json"):
                    continue
                full_path = os.path.join(root, entry)
                if not os.path.isfile(full_path):
                    continue
                try:
                    with open(full_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    if "category" in data and isinstance(data["category"], str):
                        cat = data["category"].strip().lower()
                        if cat in low_categories:
                            rel = os.path.relpath(root, abs_path)
                            low_proportion_files[cat].append(
                                (os.path.abspath(full_path), rel)
                            )
                except (json.JSONDecodeError, Exception):
                    continue

        # 폴더별 통계 출력
        rel_path = os.path.relpath(root, abs_path)

        print(f"\n  [{rel_path}] JSON {total_in_dir}개")
        for cat, prop in sorted(proportions.items(), key=lambda x: x[1], reverse=True):
            cnt = category_counts[cat]
            pct = prop * 100
            marker = " (* 낮은 비율)" if cat in low_categories else ""
            print(f"    {cat:15s}: {cnt}개 ({pct:.1f}%){marker}")

    # --- 최종 요약 ---
    print("\n" + "=" * 60)
    print("  점검 결과 요약")
    print("=" * 60)

    print(f"\n  총 JSON 파일: {total_json}개")

    if low_proportion_files:
        print(f"\n  [낮은 비율 카테고리] (기준: < {low_threshold * 100:.0f}%)")
        for cat, file_list in low_proportion_files.items():
            files_by_folder: Dict[str, List[str]] = defaultdict(list)
            for fpath, rel_folder in file_list:
                files_by_folder[rel_folder].append(fpath)

            print(f"\n    카테고리: {cat.upper()} ({len(file_list)}개)")
            for folder, paths in files_by_folder.items():
                print(f"      폴더: {folder} - {len(paths)}개")
                for p in paths[:5]:
                    print(f"        {os.path.basename(p)}")
                if len(paths) > 5:
                    print(f"        ... 외 {len(paths) - 5}개")
    else:
        print(f"\n  [낮은 비율 카테고리] 없음")

    print(f"\n{'=' * 60}")
    print("  점검 완료")
    print("=" * 60)
