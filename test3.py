import os
import re

def analyze_video_frames_recursively(root_folder_path):
    """
    지정된 폴더와 모든 하위 폴더를 재귀적으로 탐색하여 비디오 파일을 분석합니다.
    분석 조건:
    1. 파일 이름이 '..._{시작프레임}_{종료프레임}.mp4' 형식과 일치
    2. 동일한 이름의 .json 파일이 같은 폴더 내에 존재
    """
    
    print(f"📁 분석 대상 폴더 (하위 폴더 포함): {root_folder_path}")
    
    # 폴더가 존재하는지 확인
    if not os.path.isdir(root_folder_path):
        print(f"❌ 오류: '{root_folder_path}' 폴더를 찾을 수 없습니다. 경로를 확인해주세요.")
        return

    video_data = []
    skipped_files = {
        'pattern_mismatch': [],
        'json_missing': []
    }

    pattern = re.compile(r'_(\d+)_(\d+)\.mp4$')

    for root, dirs, files in os.walk(root_folder_path):
        file_set = set(files)
        
        for filename in files:
            match = pattern.search(filename)
            
            if not match:
                if filename.endswith('.mp4'):
                    skipped_files['pattern_mismatch'].append(os.path.join(root, filename))
                continue

            base_name, _ = os.path.splitext(filename)
            json_filename = base_name + ".json"
            
            if json_filename in file_set:
                try:
                    start_frame = int(match.group(1))
                    end_frame = int(match.group(2))
                    frame_length = end_frame - start_frame
                    
                    if frame_length >= 0:
                        full_path = os.path.join(root, filename)
                        video_data.append({
                            'path': full_path,
                            'length': frame_length
                        })
                    else:
                        skipped_files['pattern_mismatch'].append(os.path.join(root, filename))

                except (ValueError, IndexError):
                    skipped_files['pattern_mismatch'].append(os.path.join(root, filename))
            
            else:
                skipped_files['json_missing'].append(os.path.join(root, filename))
    
    print("=" * 60)

    if not video_data:
        print("📊 분석 결과: 조건을 만족하는 비디오 파일을 찾을 수 없습니다.")
        return

    sorted_videos = sorted(video_data, key=lambda x: x['length'], reverse=True)
    
    total_videos = len(sorted_videos)
    total_frames = sum(v['length'] for v in sorted_videos)
    avg_frames = total_frames / total_videos if total_videos > 0 else 0

    print("📊 전체 통계 (JSON 파일이 있는 비디오 기준)")
    print(f"   - 분석된 비디오 수: {total_videos}개")
    print(f"   - 전체 프레임 합계: {total_frames:,} frames")
    print(f"   - 평균 프레임 길이: {avg_frames:.1f} frames")
    print("-" * 60)

    # ✅ 추가된 부분: 프레임 길이 조건별 개수
    short_videos = [v for v in sorted_videos if v['length'] < 30]
    long_videos = [v for v in sorted_videos if v['length'] >= 600]

    print("📌 프레임 길이 조건별 통계")
    print(f"   - 30프레임 미만 비디오: {len(short_videos)}개")
    print(f"   - 600프레임 이상 비디오: {len(long_videos)}개")
    print("-" * 60)

    longest_video = sorted_videos[0]
    shortest_video = sorted_videos[-1]
    
    print(f"🏆 가장 긴 비디오 (TOP 1)")
    print(f"   - 경로: {longest_video['path']}")
    print(f"   - 프레임 길이: {longest_video['length']:,} frames")
    print()
    
    print(f"📉 가장 짧은 비디오")
    print(f"   - 경로: {shortest_video['path']}")
    print(f"   - 프레임 길이: {shortest_video['length']:,} frames")
    print("-" * 60)
        
    total_skipped = len(skipped_files['pattern_mismatch']) + len(skipped_files['json_missing'])
    if total_skipped > 0:
        print(f"\n⚠️  분석에서 제외된 파일 총 {total_skipped}개")
        
        if skipped_files['json_missing']:
            print(f"\n   - 사유: JSON 파일 없음 ({len(skipped_files['json_missing'])}개)")
            for file_path in skipped_files['json_missing'][:5]:
                print(f"     - {file_path}")
            if len(skipped_files['json_missing']) > 5:
                print(f"     ... 외 {len(skipped_files['json_missing']) - 5}개")
        
        if skipped_files['pattern_mismatch']:
            print(f"\n   - 사유: 파일 이름 패턴 불일치 ({len(skipped_files['pattern_mismatch'])}개)")
            for file_path in skipped_files['pattern_mismatch'][:5]:
                print(f"     - {file_path}")
            if len(skipped_files['pattern_mismatch']) > 5:
                print(f"     ... 외 {len(skipped_files['pattern_mismatch']) - 5}개")

# --- 사용 예시 ---
if __name__ == "__main__":
    VIDEO_ROOT_FOLDER = "data/raw/ai_hub_cctv" 
    analyze_video_frames_recursively(VIDEO_ROOT_FOLDER)

