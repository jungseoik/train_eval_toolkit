import os
import argparse
import multiprocessing
from typing import Optional, Tuple

# --- 필요한 라이브러리 임포트 ---
import torch
import numpy as np
from PIL import Image
from decord import VideoReader, cpu
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from pebble import ProcessPool
from concurrent.futures import TimeoutError
from tqdm import tqdm

# decord 라이브러리의 로그 출력을 최소화하여 화면을 깨끗하게 유지
import decord
decord.logging.set_level(decord.logging.ERROR)

# --- 제공된 비디오 로딩 및 전처리 함수 (이전과 동일) ---
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size: int = 448):
    return T.Compose([
        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        tgt_ar = ratio[0] / ratio[1]
        diff = abs(aspect_ratio - tgt_ar)
        if diff < best_ratio_diff or (diff == best_ratio_diff and area > 0.5 * image_size * image_size * ratio[0] * ratio[1]):
            best_ratio_diff = diff
            best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    ow, oh = image.size
    aspect_ratio = ow / oh
    target_ratios = sorted(
        {(i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if min_num <= i * j <= max_num},
        key=lambda x: x[0] * x[1],
    )
    ratio = find_closest_aspect_ratio(aspect_ratio, target_ratios, ow, oh, image_size)
    tw, th = image_size * ratio[0], image_size * ratio[1]
    blocks = ratio[0] * ratio[1]
    resized = image.resize((tw, th))
    tiles = [
        resized.crop(
            (
                (idx % (tw // image_size)) * image_size,
                (idx // (tw // image_size)) * image_size,
                ((idx % (tw // image_size)) + 1) * image_size,
                ((idx // (tw // image_size)) + 1) * image_size,
            )
        )
        for idx in range(blocks)
    ]
    if use_thumbnail and blocks != 1:
        tiles.append(image.resize((image_size, image_size)))
    return tiles

def get_index(bound, fps, max_frame, first_idx=0, num_segments=32):
    if bound:
        start, end = bound[0], bound[1]
    else:
        start, end = -100000, 100000
    start_idx = max(first_idx, round(start * fps))
    end_idx = min(round(end * fps), max_frame)
    seg_size = float(end_idx - start_idx) / num_segments
    frame_indices = np.array([
        int(start_idx + (seg_size / 2) + np.round(seg_size * idx))
        for idx in range(num_segments)
    ])
    return frame_indices

def load_video(video_path, bound=None, input_size=448, max_num=1, num_segments=12):
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    max_frame = len(vr) - 1
    fps = float(vr.get_avg_fps())

    pixel_values_list, num_patches_list = [], []
    transform = build_transform(input_size=input_size)
    frame_indices = get_index(bound, fps, max_frame, first_idx=0, num_segments=num_segments)
    for frame_index in frame_indices:
        safe_frame_index = min(frame_index, max_frame)
        img = Image.fromarray(vr[safe_frame_index].asnumpy()).convert('RGB')
        img = dynamic_preprocess(img, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pixel_values = [transform(tile) for tile in img]
        pixel_values = torch.stack(pixel_values)
        num_patches_list.append(pixel_values.shape[0])
        pixel_values_list.append(pixel_values)
    pixel_values = torch.cat(pixel_values_list)
    return pixel_values, num_patches_list

def test_video_with_full_logic(video_path: str) -> Optional[Tuple[str, str]]:
    try:
        _, _ = load_video(video_path, num_segments=12)
        return None
    except Exception as e:
        return (video_path, f"Exception: {str(e)}")

# --- 메인 실행 로직 (실시간 로그 출력 기능 추가) ---
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
    print(f"✅ 총 {total_files}개의 비디오 파일을 찾았습니다. {num_processes}개의 프로세스로 최종 검사를 시작합니다.")
    print(f"⏱️ 각 파일당 타임아웃은 {timeout}초로 설정됩니다.")

    problematic_files = []
    with ProcessPool(max_workers=num_processes) as pool:
        future = pool.map(test_video_with_full_logic, all_video_files, timeout=timeout)
        iterator = future.result()

        pbar = tqdm(total=total_files, desc="⚙️ 실제 로직으로 최종 검사 중", unit="file", mininterval=1.0)
        
        while True:
            try:
                result = next(iterator)
                if result is not None:
                    path, reason = result
                    # ▼▼▼▼▼ 실시간 로그 출력 ▼▼▼▼▼
                    # tqdm.write는 진행률 표시줄을 방해하지 않고 메시지를 출력합니다.
                    pbar.write(f"\n🚨 문제 발견 (에러) 🚨\n    - 파일: {path}\n    - 원인: {reason}")
                    problematic_files.append(result)
            except StopIteration:
                break # 모든 작업 완료
            except TimeoutError as error:
                file_index = error.args[1]
                timed_out_file = all_video_files[file_index]
                # ▼▼▼▼▼ 실시간 로그 출력 ▼▼▼▼▼
                pbar.write(f"\n🚨 문제 발견 (타임아웃) 🚨\n    - 파일: {timed_out_file}\n    - 원인: 처리 시간 초과 (무한 루프 의심)")
                problematic_files.append((timed_out_file, "Timeout: 무한 루프 또는 처리 시간 초과"))
            except Exception as error:
                pbar.write(f"\n🔥 처리 중 심각한 오류 발생: {error}")
            
            pbar.update(1) # 진행률 1 증가
        pbar.close()

    # --- 최종 결과 리포트 및 파일 저장 (이전과 동일) ---
    print("\n" + "="*60)
    print("✨ 최종 검사가 완료되었습니다! ✨")
    print("="*60)
    print(f"📊 전체 비디오 파일 수: {total_files}개")
    print(f"💔 문제 유발 의심 파일 수: {len(problematic_files)}개")
    print("="*60)

    if problematic_files:
        problematic_files.sort()
        print(f"\n📋 문제 유발 의심 파일 전체 목록을 '{output_file}' 파일에 저장합니다.")
        with open(output_file, 'w', encoding='utf-8') as f:
            for path, reason in problematic_files:
                f.write(f"{path} | 원인: {reason}\n")
        print(f"✅ 저장이 완료되었습니다.")
    else:
        print("\n🎉 모든 비디오 파일이 실제 로딩 로직을 통과했습니다!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="실제 운영 로직과 타임아웃을 사용하여 Decord 문제를 유발하는 비디오를 최종적으로 찾아냅니다.")
    parser.add_argument("directory", type=str, help="검사를 시작할 최상위 폴더 경로")
    parser.add_argument("-p", "--processes", type=int, default=multiprocessing.cpu_count(), help="사용할 병렬 프로세스의 개수")
    parser.add_argument("-t", "--timeout", type=int, default=30, help="각 파일당 최대 처리 시간(초)")
    parser.add_argument("-o", "--output", type=str, default="problematic_videos.txt", help="문제 파일 목록을 저장할 텍스트 파일")
    args = parser.parse_args()
    
    main(args.directory, args.processes, args.timeout, args.output)