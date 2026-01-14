import os
import json
import argparse
from collections import defaultdict
from typing import Dict, List, Tuple, Any

class FolderCategoryStats:
    """
    지정된 경로 내의 하위 폴더를 탐색하며, JSON 파일의 'category' 키 값에
    대한 통계를 계산하고, 비율이 낮은 카테고리에 속하는 파일의 절대 경로와
    해당 파일이 속한 폴더 정보를 수집 및 출력하는 클래스.
    """

    # 비율이 낮은 것으로 간주할 기준값 (예: 5.0% 미만)
    LOW_PROPORTION_THRESHOLD = 0.49

    def __init__(self, target_path: str):
        """
        초기화 메서드. 통계를 계산할 대상 경로를 설정합니다.

        Args:
            target_path (str): 통계를 계산할 최상위 경로.
        """
        self.target_path = target_path
        # 비율이 낮은 파일 정보를 저장:
        # 키: 카테고리 이름
        # 값: List[Tuple(JSON 절대 경로, PNG 절대 경로, 상대 폴더 경로)]
        self.low_proportion_files: Dict[str, List[Tuple[str, str, str]]] = defaultdict(list)


    def _get_category_stats_in_dir(self, directory: str) -> Dict[str, int]:
        """
        주어진 디렉토리 내의 모든 JSON 파일에서 'category' 빈도수를 계산합니다.
        """
        category_counts = defaultdict(int)
        
        for entry in os.listdir(directory):
            full_path = os.path.join(directory, entry)
            
            if entry.endswith('.json') and os.path.isfile(full_path):
                try:
                    with open(full_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        if 'category' in data and isinstance(data['category'], str):
                            category = data['category'].strip().lower()
                            category_counts[category] += 1
                except (json.JSONDecodeError, Exception):
                    continue
                    
        return dict(category_counts)

    def _calculate_category_proportions(self, category_counts: Dict[str, int]) -> Dict[str, float]:
        """
        카테고리 빈도수 딕셔너리를 기반으로 비율을 계산합니다.
        """
        total_files = sum(category_counts.values())
        if total_files == 0:
            return {}
        
        proportions = {
            category: count / total_files
            for category, count in category_counts.items()
        }
        return proportions
    
    def _get_low_proportion_files(self, directory: str, low_categories: List[str]):
        """
        주어진 디렉토리에서 비율이 낮은 카테고리에 해당하는 JSON 및 PNG 파일 경로를 수집합니다.
        파일 경로와 함께 파일이 속한 폴더의 상대 경로를 함께 저장합니다.

        Args:
            directory (str): 현재 디렉토리 경로.
            low_categories (List[str]): 비율이 낮은 카테고리 이름 리스트.
        """
        
        file_entries = os.listdir(directory)
        json_files = {os.path.splitext(f)[0]: f for f in file_entries if f.endswith('.json')}
        png_files = {os.path.splitext(f)[0]: f for f in file_entries if f.endswith('.png')}
        
        # 파일이 속한 폴더의 경로를 최상위 경로 대비 상대 경로로 계산
        relative_folder_path = os.path.relpath(directory, self.target_path)

        for base_name, json_name in json_files.items():
            full_json_path = os.path.join(directory, json_name)
            
            try:
                with open(full_json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                    if 'category' in data and isinstance(data['category'], str):
                        category = data['category'].strip().lower()
                        
                        if category in low_categories:
                            
                            json_path = os.path.abspath(full_json_path)
                            
                            png_name = png_files.get(base_name)
                            if png_name:
                                full_png_path = os.path.join(directory, png_name)
                                png_path = os.path.abspath(full_png_path)
                            else:
                                png_path = "PNG 파일 없음"

                            # 저장: (JSON 절대 경로, PNG 절대 경로, 상대 폴더 경로)
                            self.low_proportion_files[category].append((json_path, png_path, relative_folder_path))

            except (json.JSONDecodeError, Exception):
                continue

    def _print_low_proportion_files(self):
        """
        수집된 낮은 비율의 파일 목록을 카테고리별/폴더별로 그룹화하여 출력합니다.
        """
        
        if not self.low_proportion_files:
            print("\n🔍 낮은 비율의 카테고리에 속하는 파일을 찾지 못했습니다.")
            return

        print("\n\n==================================================")
        print("🚨 낮은 비율 카테고리 파일 절대 경로 (기준: < %.1f%%)" % (self.LOW_PROPORTION_THRESHOLD * 100))
        print("==================================================")
        
        for category, file_list in self.low_proportion_files.items():
            
            # 카테고리 내에서 폴더별로 그룹화
            files_by_folder = defaultdict(list)
            for json_path, png_path, relative_folder_path in file_list:
                files_by_folder[relative_folder_path].append((json_path, png_path))
            
            print(f"\n### 🗂️ 카테고리: **{category.upper()}**")
            print("--------------------------------------------------")
            
            # 폴더 경로를 기준으로 출력
            for folder_path, files in files_by_folder.items():
                
                # 폴더 이름과 상대 경로를 표시
                folder_name_display = os.path.basename(folder_path) if folder_path != '.' else os.path.basename(self.target_path)
                print(f"  📁 폴더: **{folder_name_display}** ({folder_path}) - {len(files)} 개")
                
                # 파일별 상세 경로 출력
                for json_path, png_path in files:
                    print(f"    - **JSON**: {os.path.basename(json_path)}")
                    print(f"      - 경로: {json_path}")
                    print(f"      - 이미지 경로: {png_path}")
                print("-" * 35)


    def analyze(self) -> None:
        """
        대상 경로를 탐색하고 폴더별 카테고리 통계를 계산, 수집 및 출력합니다.
        """
        print(f"==================================================")
        print(f"📊 대상 경로 분석 시작: {self.target_path}")
        print(f"==================================================")

        if not os.path.isdir(self.target_path):
            print(f"오류: 유효한 디렉토리 경로가 아닙니다: {self.target_path}")
            return
        
        for root, dirs, files in os.walk(self.target_path):
            category_counts = self._get_category_stats_in_dir(root)
            
            if not category_counts:
                continue

            proportions = self._calculate_category_proportions(category_counts)
            
            low_categories = [
                category for category, proportion in proportions.items()
                if proportion < self.LOW_PROPORTION_THRESHOLD
            ]
            
            if low_categories:
                self._get_low_proportion_files(root, low_categories)

            # 폴더별 통계 출력
            folder_name = os.path.basename(root) if root != self.target_path else self.target_path
            relative_path = os.path.relpath(root, self.target_path)
            
            print(f"\n📁 폴더: {folder_name} ({relative_path})")
            print(f"--------------------------------------------------")
            total_processed = sum(category_counts.values())
            print(f"   > 총 처리된 JSON 파일 수: {total_processed} 개")
            print(f"   > 카테고리 비율:")
            
            for category, proportion in sorted(proportions.items(), key=lambda item: item[1], reverse=True):
                count = category_counts[category]
                percentage = proportion * 100
                
                marker = " (* 낮은 비율)" if category in low_categories else ""
                print(f"     - **{category.ljust(15)}**: {count} 개 ({percentage:.2f}%){marker}")

        self._print_low_proportion_files()
        
        print(f"\n==================================================")
        print(f"✅ 분석 완료")
        print(f"==================================================")


# --- 독립적인 실행을 위한 Main 블록 및 Argparser ---

def main():
    """
    명령줄 인수를 처리하고 FolderCategoryStats 클래스를 실행하는 메인 함수.
    """
    parser = argparse.ArgumentParser(
        description="특정 경로 내의 하위 폴더별 JSON 파일의 'category' 비율 통계를 계산하고, 낮은 비율 파일의 절대 경로를 폴더별로 그룹화하여 출력합니다."
    )
    
    parser.add_argument(
        "target_directory",
        type=str,
        help="분석을 시작할 최상위 디렉토리 경로 (예: ./data)"
    )

    args = parser.parse_args()
    
    abs_path = os.path.abspath(args.target_directory)
    
    stats_analyzer = FolderCategoryStats(target_path=abs_path)
    stats_analyzer.analyze()

if __name__ == "__main__":
    main()
    # python src/stats/json_category_stats.py data/processed/gangnam/gaepo4/clean
    # python src/stats/json_category_stats.py data/processed/gangnam/samsung/clean
    # python src/stats/json_category_stats.py data/processed/gangnam/gaepo1
    # python src/stats/json_category_stats.py data/processed/gangnam/yeoksam2
    # python src/stats/json_category_stats.py data/processed/gangnam/gaepo1_v2
    # python src/stats/json_category_stats.py data/processed/gangnam/yeoksam2_v2
    # python src/stats/json_category_stats.py data/processed/gangnam/yeoksam2_v2