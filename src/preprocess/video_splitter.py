import cv2
import os
import glob
import decord
from concurrent.futures import ProcessPoolExecutor
from typing import List, Optional
from functools import partial

def _split_single_video(video_path: str, output_base_dir: str, seconds_per_clip: int = 1) -> str:
    """(내부 함수) 단일 비디오 파일을 지정된 시간 단위의 클립으로 분할합니다.

    `decord`를 사용하여 비디오 프레임을 효율적으로 읽고, `cv2`를 사용하여
    결과 클립을 MP4 파일로 저장합니다. 출력 파일은 원본 비디오 이름으로 생성된
    하위 폴더에 저장됩니다.

    예: `output_base_dir/video1/video1_0_29.mp4`

    Args:
        video_path (str): 처리할 원본 비디오 파일의 전체 경로.
        output_base_dir (str): 분할된 클립들이 저장될 최상위 폴더 경로.
        seconds_per_clip (int): 각 클립의 길이 (초 단위).

    Returns:
        str: 클립들이 성공적으로 저장된 폴더의 경로. 오류 발생 시 빈 문자열을 반환합니다.
    """
    try:
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        output_video_dir = os.path.join(output_base_dir, video_name)
        os.makedirs(output_video_dir, exist_ok=True)

        vr = decord.VideoReader(video_path)
        fps = int(round(vr.get_avg_fps()))
        total_frames = len(vr)
        
        height, width, _ = vr[0].asnumpy().shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        
        frames_per_clip = fps * seconds_per_clip

        for start_frame in range(0, total_frames, frames_per_clip):
            end_frame = min(start_frame + frames_per_clip - 1, total_frames - 1)
            
            if start_frame > end_frame:
                continue

            frame_indices = list(range(start_frame, end_frame + 1))
            frames = vr.get_batch(frame_indices).asnumpy()
            
            output_filename = f"{video_name}_{start_frame}_{end_frame}.mp4"
            output_filepath = os.path.join(output_video_dir, output_filename)
            
            out = cv2.VideoWriter(output_filepath, fourcc, fps, (width, height))
            for frame in frames:
                out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            out.release()

        print(f"✅ Successfully processed {video_path} -> {output_video_dir}")
        return output_video_dir

    except Exception as e:
        print(f"Error processing {video_path}: {e}")
        return ""

def _process_videos_in_parallel(
    video_paths: List[str], 
    output_dir: str, 
    seconds_per_clip: int = 1, 
    num_workers: Optional[int] = None
):
    """(내부 함수) 비디오 처리 작업을 병렬로 실행합니다.

    `concurrent.futures.ProcessPoolExecutor`를 사용하여 `_split_single_video` 함수를
    여러 프로세스에 분배합니다. 이를 통해 다수의 비디오를 동시에 처리하여
    전체 작업 시간을 단축합니다.

    Args:
        video_paths (List[str]): 처리할 모든 비디오 파일의 경로 리스트.
        output_dir (str): 분할된 클립들이 저장될 최상위 폴더 경로.
        seconds_per_clip (int): 각 클립의 길이 (초 단위).
        num_workers (Optional[int]): 사용할 CPU 프로세스의 수.
            None으로 설정하면 사용 가능한 모든 CPU 코어를 사용합니다.
    """
    worker_func = partial(_split_single_video, output_base_dir=output_dir, seconds_per_clip=seconds_per_clip)

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = executor.map(worker_func, video_paths)
    
    for result in results:
        if not result:
            print("A video processing task failed.")

def preprocess_video_chunk_split_folder(
    input_folder: str, 
    output_folder: str, 
    seconds_per_clip: int = 1, 
    num_workers: Optional[int] = None
):
    """지정된 폴더 내의 모든 비디오를 찾아 병렬로 클립 분할을 수행합니다.

    이 함수는 모듈의 메인 인터페이스 역할을 합니다. 입력 폴더에서 지원하는 확장자
    (`.mp4`, `.avi`, `.mov`, `.mkv`)를 가진 모든 비디오 파일을 찾은 뒤,
    이 파일들을 여러 CPU 코어를 사용하여 동시에 처리합니다.

    Args:
        input_folder (str): 원본 비디오 파일들이 들어있는 폴더의 경로.
        output_folder (str): 분할된 클립들을 저장할 목적지 폴더의 경로.
        seconds_per_clip (int, optional): 각 클립의 길이 (초 단위). 기본값은 1입니다.
        num_workers (Optional[int], optional): 사용할 CPU 코어의 수.
            기본값은 None이며, 이 경우 시스템의 모든 가용 코어를 사용합니다.
    """
    supported_extensions = ["*.mp4", "*.avi", "*.mov", "*.mkv"]
    
    videos_to_process = []
    for ext in supported_extensions:
        search_path = os.path.join(input_folder, ext)
        videos_to_process.extend(glob.glob(search_path))

    if not videos_to_process:
        print(f"No video files found in {input_folder}")
        return


    print(f"Found {len(videos_to_process)} videos to process in '{input_folder}'.")
    print(f"Using {num_workers if num_workers else 'all available'} CPU cores.")

    _process_videos_in_parallel(
        videos_to_process, 
        output_folder, 
        seconds_per_clip=seconds_per_clip,
        num_workers=num_workers
    )
    
    print(f"All videos from '{input_folder}' have been processed and saved to '{output_folder}'.")