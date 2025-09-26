import os
import decord
import multiprocessing
import argparse
from tqdm import tqdm
from pebble import ProcessPool
from concurrent.futures import TimeoutError
from typing import Optional

# decord 초기화 시 로그 레벨을 조용하게 설정
decord.logging.set_level(decord.logging.ERROR)

def test_decord_seek(video_path: str) -> Optional[str]:
    """
    Decord로 비디오를 열고 여러 지점을 탐색(seek)하여 읽어보는 스트레스 테스트.
    성공하면 None, 실패하거나 멈추면 파일 경로를 반환합니다.
    """
    try:
        vr = decord.VideoReader(video_path, ctx=decord.cpu(0))
        total_frames = len(vr)
        
        # 파일이 너무 짧으면 테스트 스킵
        if total_frames < 10:
            return None
            
        # 파일의 여러 지점(시작, 1/4, 중간, 3/4, 끝)을 강제로 읽어보게 함
        indices_to_check = [
            0, 
            total_frames // 4, 
            total_frames // 2, 
            (total_frames * 3) // 4, 
            total_frames - 1
        ]
        
        for idx in indices_to_check:
            _ = vr[idx] # 프레임 읽기 시도
            
        return None # 모든 탐색 및 읽기 성공
    except Exception:
        # Decord가 파일을 읽는 도중 에러를 발생시키는 경우
        return video_path

def main(directory: str, num_processes: int, timeout: int, output_file: str):
    video_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.webm')
    
    print(f"🔍 지정된 폴더에서 비디오 파일을 찾는 중...: {directory}")
    all_video_files = [
        os.path.join(root, file)
        for root, _, files in os.walk(directory)
        for file in files if file.lower().endswith(video_extensions)
    ]

    if not all_video_files:
        print("❌ 해당 경로에서 비디오 파일을 찾을 수 없습니다.")
        return

    total_files = len(all_video_files)
    print(f"✅ 총 {total_files}개의 비디오 파일을 찾았습니다. {num_processes}개의 프로세스로 Decord 호환성 검사를 시작합니다.")
    print(f"⏱️ 각 파일당 타임아웃은 {timeout}초로 설정됩니다.")

    hanging_files = []
    # Pebble ProcessPool을 사용하여 타임아웃 기능 구현
    with ProcessPool(max_workers=num_processes) as pool:
        # 각 파일에 대해 test_decord_seek 함수를 실행하고 timeout을 설정
        future = pool.map(test_decord_seek, all_video_files, timeout=timeout)
        iterator = future.result()

        pbar = tqdm(total=total_files, desc=" compat Decord 호환성 검사 중", unit="file")
        
        while True:
            try:
                result = next(iterator)
                if result is not None:
                    hanging_files.append(result)
            except StopIteration:
                break # 모든 작업 완료
            except TimeoutError as error:
                # 타임아웃이 발생한 파일의 인덱스를 찾아 파일 경로를 특정
                file_index = error.args[1]
                timed_out_file = all_video_files[file_index]
                hanging_files.append(timed_out_file)
                print(f"\n⚠️ 타임아웃 발생! 파일: {timed_out_file}")
            except Exception as error:
                # 그 외 다른 오류
                print(f"\n🔥 처리 중 오류 발생: {error}")
            
            pbar.update(1)
        pbar.close()

    print("\n" + "="*60)
    print("✨ 검사가 완료되었습니다! ✨")
    print("="*60)
    print(f"📊 전체 비디오 파일 수: {total_files}개")
    print(f"💔 Decord 행(Hang) 유발 의심 파일 수: {len(hanging_files)}개")
    print("="*60)

    if hanging_files:
        hanging_files.sort()
        print(f"\n📋 아래는 Decord를 멈추게 하거나 오류를 유발하는 파일 목록입니다.")
        print(f"💾 해당 목록을 '{output_file}' 파일에 저장합니다.")
        with open(output_file, 'w', encoding='utf-8') as f:
            for path in hanging_files:
                f.write(f"{path}\n")
        print(f"✅ 저장이 완료되었습니다.")
    else:
        print("\n🎉 모든 비디오 파일이 Decord와 호환되는 것으로 보입니다!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Decord 라이브러리와 호환되지 않아 행(hang)을 유발하는 비디오 파일을 찾습니다.")
    parser.add_argument("directory", type=str, help="검사를 시작할 최상위 폴더 경로")
    parser.add_argument("-p", "--processes", type=int, default=multiprocessing.cpu_count(), help="사용할 병렬 프로세스의 개수")
    parser.add_argument("-t", "--timeout", type=int, default=60, help="각 파일당 최대 처리 시간(초)")
    parser.add_argument("-o", "--output", type=str, default="decord_hanging_files.txt", help="문제 파일 목록을 저장할 텍스트 파일")
    args = parser.parse_args()
    
    main(args.directory, args.processes, args.timeout, args.output)