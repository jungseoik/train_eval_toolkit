import os
import re
import shutil

def analyze_and_move_videos(root_folder_path, dest_root):
    """
    비디오를 분석하고, 30프레임 미만과 600프레임 이상 비디오를 지정된 폴더로 이동.
    동일한 JSON 파일도 같이 이동.
    
    Args:
        root_folder_path (str): 분석할 최상위 폴더
        dest_root (str): 이동할 루트 폴더
    """
    print(f"📁 분석 대상 폴더: {root_folder_path}")
    if not os.path.isdir(root_folder_path):
        print(f"❌ 오류: '{root_folder_path}' 폴더가 존재하지 않습니다.")
        return

    video_data = []
    pattern = re.compile(r'_(\d+)_(\d+)\.mp4$')

    for root, dirs, files in os.walk(root_folder_path):
        file_set = set(files)
        for filename in files:
            match = pattern.search(filename)
            if not match:
                continue

            base_name, _ = os.path.splitext(filename)
            json_filename = base_name + ".json"
            if json_filename not in file_set:
                continue

            try:
                start_frame = int(match.group(1))
                end_frame = int(match.group(2))
                frame_length = end_frame - start_frame
                if frame_length < 0:
                    continue

                full_path = os.path.join(root, filename)
                video_data.append({
                    'path': full_path,
                    'length': frame_length
                })
            except Exception:
                continue

    if not video_data:
        print("📊 분석 결과: 조건에 맞는 비디오가 없습니다.")
        return

    # 조건별 필터링
    short_videos = [v for v in video_data if v['length'] < 30]
    long_videos = [v for v in video_data if v['length'] >= 600]

    print("📌 프레임 길이 조건별 통계")
    print(f"   - 30프레임 미만 비디오: {len(short_videos)}개")
    print(f"   - 600프레임 이상 비디오: {len(long_videos)}개")
    print("-" * 60)

    # 이동 함수
    def move_files(videos, label):
        for v in videos:
            video_path = v['path']
            json_path = os.path.splitext(video_path)[0] + ".json"

            # 원본 경로에서 root_folder_path 이후 부분만 떼오기
            rel_path = os.path.relpath(video_path, root_folder_path)
            dest_video_path = os.path.join(dest_root, rel_path)

            # 디렉토리 생성
            os.makedirs(os.path.dirname(dest_video_path), exist_ok=True)

            # 파일 이동
            shutil.move(video_path, dest_video_path)
            if os.path.exists(json_path):
                dest_json_path = os.path.splitext(dest_video_path)[0] + ".json"
                shutil.move(json_path, dest_json_path)

            print(f"📦 {label}: {video_path} -> {dest_video_path}")

    # 실제 이동
    if short_videos:
        move_files(short_videos, "SHORT(<30)")
    if long_videos:
        move_files(long_videos, "LONG(>=600)")


# --- 사용 예시 ---
if __name__ == "__main__":
    SRC_ROOT = "data/raw/ai_hub_cctv"
    DEST_ROOT = "data/raw/ai_hub_cctv_except"  
    analyze_and_move_videos(SRC_ROOT, DEST_ROOT)
