import os
import json
import cv2
from multiprocessing import Pool, cpu_count
from decord import VideoReader, cpu
from tqdm import tqdm


def _save_clip(video_path, output_path, start_frame, end_frame, fps):
    """주어진 프레임 구간을 잘라서 mp4 클립으로 저장 (resume 기본)"""
    if os.path.exists(output_path):  # resume 기능
        return

    try:
        vr = VideoReader(video_path, ctx=cpu(0))
        total_frames = len(vr)

        # ✅ 안전하게 보정
        start_frame = max(0, start_frame)
        end_frame = min(end_frame, total_frames - 1)
        if start_frame >= total_frames or end_frame < start_frame:
            print(f"⚠️ {os.path.basename(video_path)} 구간[{start_frame}-{end_frame}] 잘못된 범위 → 스킵")
            return

        frames = vr.get_batch(range(start_frame, end_frame + 1)).asnumpy()
    except Exception as e:
        print(f"⚠️ {os.path.basename(video_path)} 구간[{start_frame}-{end_frame}] 읽기 오류: {e}")
        return

    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()


def process_video_json_pair(task_args):
    """하나의 (비디오, JSON) 쌍을 처리"""
    json_path, output_root = task_args
    base_name = os.path.splitext(os.path.basename(json_path))[0]
    source_dir = os.path.dirname(json_path)

    video_file = next(
        (f"{base_name}{ext}" for ext in [".mp4", ".avi", ".mov", ".mkv"]
         if os.path.exists(os.path.join(source_dir, f"{base_name}{ext}"))),
        None
    )
    if not video_file:
        print(f"⚠️ '{base_name}' 비디오 파일 없음 → 스킵")
        return

    video_path = os.path.join(source_dir, video_file)

    violence_output_path = os.path.join(output_root, "violence")
    normal_output_path = os.path.join(output_root, "normal")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # ✅ 메타데이터 확보
    video_info = data.get("video_info", {})
    fps = video_info.get("fps", None)
    total_frames = video_info.get("total_frame", None)

    # 만약 JSON에 없으면, 직접 비디오에서 가져오기
    if fps is None or total_frames is None:
        try:
            vr = VideoReader(video_path, ctx=cpu(0))
            fps = fps or vr.get_avg_fps()
            total_frames = total_frames or len(vr)
        except Exception as e:
            print(f"⚠️ {os.path.basename(video_path)} 메타데이터 추출 실패: {e}")
            return

    # violence 구간 처리
    violence_intervals = []
    violence_full_video_used = False  # ✅ 전체 영상 저장 여부 플래그

    for clip_name, clip_info in data["clips"].items():
        if clip_info["category"] == "violence":
            start_frame, end_frame = clip_info["timestamp"]

            # ✅ JSON timestamp(30fps 기준) → 실제 fps 기준 변환
            start_frame = int(start_frame * fps / 30)
            end_frame = int(end_frame * fps / 30)

            # ✅ 구간 길이가 2초 미만이면 전체 영상 저장
            if end_frame - start_frame + 1 < 2 * fps:
                start_frame, end_frame = 0, total_frames - 1
                output_filename = f"{base_name}_frame-0_frame-{end_frame}.mp4"
                violence_full_video_used = True
            else:
                output_filename = f"{base_name}_frame-{start_frame}_frame-{end_frame}.mp4"

            output_filepath = os.path.join(violence_output_path, output_filename)

            # 🎥 비디오 클립 저장
            _save_clip(video_path, output_filepath, start_frame, end_frame, fps)

            # 📝 JSON 저장 (en_caption -> description)
            if "en_caption" in clip_info and clip_info["en_caption"]:
                json_output = {
                    "category": "violence",
                    "description": clip_info["en_caption"][0]  # 첫 번째 캡션 사용
                }
                json_filename = output_filename.replace(".mp4", ".json")
                json_filepath = os.path.join(violence_output_path, json_filename)
                with open(json_filepath, "w", encoding="utf-8") as jf:
                    json.dump(json_output, jf, ensure_ascii=False, indent=4)

            violence_intervals.append((start_frame, end_frame))

    num_violence = len(violence_intervals)

    # ✅ normal 구간은 violence 전체 영상 모드일 때는 생성하지 않음
    if not violence_full_video_used:
        # normal 구간 계산
        violence_intervals.sort()
        normal_intervals = []
        last_end_frame = 0
        for start, end in violence_intervals:
            if start > last_end_frame:
                normal_intervals.append((last_end_frame, start - 1))
            last_end_frame = end + 1
        if last_end_frame < total_frames:
            normal_intervals.append((last_end_frame, total_frames - 1))

        # normal 구간 클립 생성 (앞에서부터, violence 개수 이하만)
        min_clip_frames = int(1 * fps)
        max_clip_frames = int(2 * fps)

        normal_count = 0
        for start, end in normal_intervals:
            if normal_count >= num_violence:
                break  # violence 개수 이상이면 중단

            current_pos = start
            while current_pos < end and normal_count < num_violence:
                clip_end = min(current_pos + max_clip_frames - 1, end)
                if clip_end - current_pos + 1 >= min_clip_frames:
                    output_filename = f"{base_name}_frame-{current_pos}_frame-{clip_end}.mp4"
                    output_filepath = os.path.join(normal_output_path, output_filename)
                    _save_clip(video_path, output_filepath, current_pos, clip_end, fps)
                    normal_count += 1
                current_pos = clip_end + 1


def run_tasks(tasks, num_processes):
    """비디오 작업 병렬 처리 + tqdm 진행률 표시"""
    if not tasks:
        print("ℹ️ 처리할 비디오 작업 없음")
        return

    if num_processes == 0:
        num_processes = cpu_count()
    elif num_processes < 0:
        num_processes = 1

    if num_processes <= 1:
        for task in tqdm(tasks, desc="Processing videos", unit="video"):
            process_video_json_pair(task)
    else:
        effective_processes = min(num_processes, len(tasks))
        with Pool(processes=effective_processes) as pool:
            list(tqdm(pool.imap_unordered(process_video_json_pair, tasks),
                      total=len(tasks),
                      desc="Processing videos", unit="video"))


def process_videos_clips_aihub_store(source_root, output_root, num_processes=1):
    """모든 (비디오, JSON) 쌍 처리"""
    if not os.path.isdir(source_root):
        print(f"❌ 소스 디렉토리 없음: {source_root}")
        return

    os.makedirs(os.path.join(output_root, "violence"), exist_ok=True)
    os.makedirs(os.path.join(output_root, "normal"), exist_ok=True)

    tasks = []
    all_files = os.listdir(source_root)
    json_files = [f for f in all_files if f.endswith(".json")]

    for json_file in json_files:
        base_name = os.path.splitext(json_file)[0]
        video_exists = any(f"{base_name}{ext}" in all_files for ext in [".mp4", ".avi", ".mov", ".mkv"])
        if video_exists:
            json_path = os.path.join(source_root, json_file)
            tasks.append((json_path, output_root))
        else:
            print(f"⚠️ '{json_file}' 대응 비디오 없음 → 건너뜀")

    run_tasks(tasks, num_processes)
    print("\n🎉 모든 작업 완료!")


if __name__ == "__main__":
    SOURCE_DIRECTORY = "data/raw/gj_violence"
    OUTPUT_DIRECTORY = "data/processed/gj_violence"
    process_videos_clips_aihub_store(SOURCE_DIRECTORY, OUTPUT_DIRECTORY, num_processes=4)