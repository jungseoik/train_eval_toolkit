import os
import decord
from tqdm import tqdm
import multiprocessing
import argparse # 커맨드 라인 인자를 받기 위해 추가

def check_single_video(video_path: str):
    """
    단일 비디오 파일을 검사하는 함수. 멀티프로세싱의 각 worker가 이 함수를 실행합니다.
    파일이 손상되었으면 파일 경로를, 정상이면 None을 반환합니다.
    """
    try:
        # ctx=decord.cpu(0) 옵션은 GPU 대신 CPU를 사용하게 하여 GPU 관련 오류를 방지합니다.
        vr = decord.VideoReader(video_path, ctx=decord.cpu(0))
        # 첫 프레임을 실제로 읽어봐야 정확한 오류를 잡을 수 있습니다.
        _ = vr[0]
        return None # 성공 시 None 반환
    except (decord.DECORDError, Exception):
        # decord 오류 또는 기타 예외 발생 시 파일 경로 반환
        return video_path

def find_corrupted_videos_multi(directory: str, num_processes: int):
    """
    지정된 디렉토리에서 멀티프로세싱을 사용하여 손상된 비디오 파일을 찾습니다.

    Args:
        directory (str): 검사를 시작할 최상위 폴더 경로
        num_processes (int): 사용할 프로세스의 개수
    """
    # 1. 지원할 비디오 확장자 정의
    video_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.webm')
    
    # 2. 전체 비디오 파일 목록을 재귀적으로 수집
    print(f"🔍 지정된 폴더에서 비디오 파일을 찾는 중...: {directory}")
    all_video_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(video_extensions):
                full_path = os.path.join(root, file)
                all_video_files.append(full_path)

    if not all_video_files:
        print("❌ 해당 경로에서 비디오 파일을 찾을 수 없습니다.")
        return

    total_files = len(all_video_files)
    print(f"✅ 총 {total_files}개의 비디오 파일을 찾았습니다. {num_processes}개의 프로세스로 검사를 시작합니다.")

    # 3. 멀티프로세싱 Pool을 사용하여 병렬 처리
    corrupted_files_list = []
    # with 구문을 사용하면 Pool을 안전하게 생성하고 종료할 수 있습니다.
    with multiprocessing.Pool(processes=num_processes) as pool:
        # imap_unordered는 작업을 분배하고 완료되는 순서대로 결과를 반환하여 효율적입니다.
        # tqdm을 여기에 적용하여 전체 진행 상황을 실시간으로 보여줍니다.
        results = tqdm(
            pool.imap_unordered(check_single_video, all_video_files),
            total=total_files,
            desc="🎥 비디오 파일 검사 중",
            unit="file"
        )
        
        # 각 프로세스로부터 반환된 결과를 취합
        for result in results:
            if result is not None: # 결과가 None이 아니면 (즉, 파일 경로가 반환되면) 손상된 파일임
                corrupted_files_list.append(result)

    # 4. 최종 결과 리포트 출력
    print("\n" + "="*50)
    print("✨ 검사가 완료되었습니다! ✨")
    print("="*50)
    print(f"📊 전체 비디오 파일 수: {total_files}개")
    print(f"💔 손상 의심 파일 수: {len(corrupted_files_list)}개")
    print("="*50)

    if corrupted_files_list:
        print("\n📋 아래는 손상되었거나 읽을 수 없는 파일 목록입니다:")
        # 보기 좋게 정렬해서 출력
        corrupted_files_list.sort()
        for path in corrupted_files_list:
            print(path)
    else:
        print("\n🎉 모든 비디오 파일을 성공적으로 읽었습니다!")

if __name__ == "__main__":


    find_corrupted_videos_multi("data/raw/ai_hub_cctv", 128)

