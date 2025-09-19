import os
from collections import defaultdict


KEYWORDS = ['falldown', 'normal', 'violence']
VIDEO_EXTENSIONS = ('.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv')

def analyze_video_directory(root_path: str):
    """
    지정된 경로의 하위 폴더들을 분석하여 비디오 파일 개수와 키워드별 통계를 출력합니다.

    Args:
        root_path (str): 분석을 시작할 최상위 폴더 경로
    """
    # 1. 경로 유효성 검사
    if not os.path.isdir(root_path):
        print(f"오류: '{root_path}'는 유효한 폴더가 아닙니다.")
        return

    print(f"'{os.path.basename(root_path)}' 폴더 분석을 시작합니다...\n")

    # 2. 통계 저장을 위한 변수 초기화
    total_video_count = 0
    # defaultdict을 사용하면 키가 없을 때 자동으로 0으로 초기화해줍니다.
    keyword_counts = defaultdict(int)

    print("## 폴더별 비디오 개수")
    
    # 3. 하위 폴더 순회 및 분석
    # root_path 바로 아래에 있는 항목들만 가져옴
    for subfolder_name in sorted(os.listdir(root_path)):
        subfolder_path = os.path.join(root_path, subfolder_name)

        if os.path.isdir(subfolder_path):
            current_folder_video_count = 0
            
            # --- 수정된 부분 ---
            # os.listdir() 대신 os.walk()를 사용하여 해당 폴더 내의 모든 하위 폴더를 재귀적으로 탐색
            for dirpath, dirnames, filenames in os.walk(subfolder_path):
                # 현재 탐색 중인 폴더 내의 모든 파일 이름을 확인
                for filename in filenames:
                    # 파일 확장자가 비디오 확장자 목록에 포함되는지 확인
                    if filename.lower().endswith(VIDEO_EXTENSIONS):
                        current_folder_video_count += 1
            # --- 수정 끝 ---
            
            # 폴더별 결과 출력
            print(f"- '{subfolder_name}': {current_folder_video_count}개")
            
            # 전체 개수 업데이트
            total_video_count += current_folder_video_count
            
            # 키워드별 개수 업데이트 (폴더 이름을 소문자로 바꿔서 확인)
            for keyword in KEYWORDS:
                if keyword in subfolder_name.lower():
                    keyword_counts[keyword] += current_folder_video_count
    
    # 4. 최종 결과 요약 출력
    print("\n--------------------")
    print("분석 결과 요약")
    print("--------------------")

    print(f"\n## 전체 비디오 개수: {total_video_count}개")

    print("\n## 키워드별 비디오 개수")
    for keyword in KEYWORDS:
        count = keyword_counts[keyword]
        print(f"- '{keyword}': {count}개")
    print("--------------------")


# --- 예제 실행 코드 ---
if __name__ == "__main__":
    target_directory = "data/raw"  
    # target_directory = "data/processed" 
    analyze_video_directory(target_directory)