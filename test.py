import os
import sys
import re
import csv
import time
import json
import math
import queue
import glob
import multiprocessing as mp
from dataclasses import dataclass
from typing import Dict, List, Tuple

# ---------------- 공유 유틸/모델 (1단계 코드에서 가져온 것 + 일부 확장) ----------------
import torch
from transformers import AutoModel, AutoTokenizer
from PIL import Image
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from decord import VideoReader, cpu
import numpy as np

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size: int = 448):
    return T.Compose(
        [
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )

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

def build_video_reader(video_path: str):
    """decord VideoReader 생성(메타만 보고 곧바로 닫지 않고 필요 시 프레임 단위로 접근)."""
    return VideoReader(video_path, ctx=cpu(0), num_threads=1)

def get_indices_by_frame_range(start_idx: int, end_idx: int, num_segments: int) -> np.ndarray:
    """
    [start_idx, end_idx] (inclusive) 범위에서 균등 간격으로 num_segments개 인덱스 선택.
    범위 길이가 짧으면 중복이 생기지 않도록 클램프 처리.
    """
    start = int(start_idx)
    end = int(end_idx)
    if end < start:
        end = start
    length = end - start + 1
    num = max(1, min(num_segments, length))
    # 균등 샘플링 (중앙 보정)
    step = length / float(num)
    idxs = [start + int(step * i + step / 2) for i in range(num)]
    # 범위 클램프
    idxs = [min(max(start, x), end) for x in idxs]
    return np.array(idxs, dtype=int)

def load_video_window(video_path: str,
                      start_frame: int,
                      end_frame: int,
                      input_size: int = 448,
                      max_num: int = 1,
                      num_segments: int = 12):
    """
    주어진 프레임 범위[start_frame, end_frame]에서 num_segments만큼만 샘플링해
    InternVL 입력 텐서를 만듭니다. (윈도우만 디코딩)
    """
    vr = build_video_reader(video_path)
    max_frame = len(vr) - 1
    s = max(0, min(start_frame, max_frame))
    e = max(0, min(end_frame, max_frame))
    indices = get_indices_by_frame_range(s, e, num_segments=num_segments)

    pixel_values_list, num_patches_list = [], []
    transform = build_transform(input_size=input_size)
    for frame_index in indices:
        img = Image.fromarray(vr[frame_index].asnumpy()).convert('RGB')
        tiles = dynamic_preprocess(img, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pixel_values = [transform(tile) for tile in tiles]
        pixel_values = torch.stack(pixel_values)
        num_patches_list.append(pixel_values.shape[0])
        pixel_values_list.append(pixel_values)
    pixel_values = torch.cat(pixel_values_list)
    return pixel_values, num_patches_list

class InternVL3Inferencer:
    def __init__(self, model_path="OpenGVLab/InternVL3-2B", device="cuda:0"):
        print(f"[INFO] ({os.getpid()}) InternVL 모델 로딩 중... device={device}, model={model_path}")
        self.model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            use_flash_attn=False,
            trust_remote_code=True
        ).eval().to(device)

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
        self.device = device
        self.generation_config = dict(max_new_tokens=50, do_sample=False)
        print(f"[INFO] ({os.getpid()}) InternVL 모델 로딩 완료.")

    def infer_window(self,
                     video_path: str,
                     prompt: str,
                     start_frame: int,
                     end_frame: int,
                     num_segments: int = 12,
                     input_size: int = 448) -> str:
        pixel_values, num_patches_list = load_video_window(
            video_path, start_frame, end_frame, input_size=input_size, max_num=1, num_segments=num_segments
        )
        pixel_values = pixel_values.to(torch.bfloat16).cuda()
        video_prefix = ''.join([f'Frame{i+1}: <image>\n' for i in range(len(num_patches_list))])
        question = video_prefix + prompt
        response = self.model.chat(self.tokenizer, pixel_values, question, self.generation_config)
        return response

# ---------------- 파서 (사용자 제공 로직 그대로) ----------------
# def parse_prediction(pred_str: str) -> str:
#     try:
#         clean_str = pred_str
#         if '```json' in clean_str:
#             clean_str = clean_str.split('```json')[1].split('```')[0]
#         elif '```' in clean_str:
#             clean_str = clean_str.split('```')[1].split('```')[0]
#         clean_str = clean_str.strip()
#         start_brace = clean_str.find('{')
#         end_brace = clean_str.rfind('}')
#         if start_brace != -1 and end_brace != -1 and start_brace < end_brace:
#             json_part = clean_str[start_brace:end_brace + 1]
#             data = json.loads(json_part)
#             category = data.get('category')
#             if category in ['violence', 'normal']:
#                 return category
#         cat_match = re.search(r'["\']category["\']\s*:\s*["\'](violence|normal)["\']', clean_str)
#         if cat_match:
#             return cat_match.group(1)
#         return 'violence'
#     except Exception:
#         return 'violence'
    
def parse_prediction(pred_str: str) -> str:
    if not isinstance(pred_str, str): return 'parsing_failed'
    try:
        clean_str = pred_str
        if '```json' in clean_str: clean_str = clean_str.split('```json')[1].split('```')[0]
        elif '```' in clean_str: clean_str = clean_str.split('```')[1].split('```')[0]
        clean_str = clean_str.strip()
        start_brace, end_brace = clean_str.find('{'), clean_str.rfind('}')
        if start_brace != -1 and end_brace != -1 and start_brace < end_brace:
            json_part = clean_str[start_brace : end_brace + 1]
            try:
                data = json.loads(json_part)
                category = data.get('category')
                if isinstance(category, str) and category.lower() in ['violence', 'normal']:
                    return category.lower()
            except json.JSONDecodeError: pass
        cat_match = re.search(r'["\']category["\']\s*:\s*["\'](violence|normal)["\']', clean_str, re.IGNORECASE)
        if cat_match: return cat_match.group(1).lower()
        return 'no_json_found'
    except Exception: return 'parsing_failed'

# ---------------- CUDA 체크 ----------------
def check_and_set_cuda_device(gpu_id: int) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")
    visible_count = torch.cuda.device_count()
    if visible_count == 0:
        raise RuntimeError("No visible CUDA devices.")
    if gpu_id < 0 or gpu_id >= visible_count:
        raise RuntimeError(f"Invalid gpu_id={gpu_id}. Visible={visible_count}")
    torch.cuda.set_device(gpu_id)
    return {
        "gpu_id": gpu_id,
        "device": f"cuda:{gpu_id}",
        "name": torch.cuda.get_device_name(gpu_id),
        "capability": torch.cuda.get_device_capability(gpu_id),
        "visible_count": visible_count,
    }

# ---------------- 워커 IPC 프로토콜 ----------------
CMD_PING = "ping"
CMD_STOP = "stop"
CMD_READY = "ready"
CMD_INFER = "infer"

@dataclass
class WorkerHandle:
    gpu_id: int
    index_on_gpu: int
    proc: mp.Process
    ctrl_q: mp.Queue
    resp_q: mp.Queue

def _torch_mem(device: int) -> Tuple[int, int]:
    try:
        return torch.cuda.memory_allocated(device), torch.cuda.memory_reserved(device)
    except Exception:
        return (0, 0)

def model_worker_main(gpu_id: int, ctrl_q: mp.Queue, resp_q: mp.Queue, model_path: str):
    info = check_and_set_cuda_device(gpu_id)
    device_str = info["device"]

    try:
        inferencer = InternVL3Inferencer(model_path=model_path, device=device_str)
        resp_q.put({"type": CMD_READY, "pid": os.getpid(), "gpu_id": gpu_id, "device": device_str, "ok": True})
    except Exception as e:
        resp_q.put({"type": CMD_READY, "pid": os.getpid(), "gpu_id": gpu_id, "device": device_str, "ok": False, "error": repr(e)})
        return

    while True:
        try:
            cmd = ctrl_q.get()
        except (EOFError, KeyboardInterrupt):
            break
        if not isinstance(cmd, dict):
            continue

        op = cmd.get("op")

        if op == CMD_PING:
            alloc, reserved = _torch_mem(gpu_id)
            resp_q.put({
                "type": "pong",
                "pid": os.getpid(),
                "gpu_id": gpu_id,
                "device": device_str,
                "allocated_bytes": int(alloc),
                "reserved_bytes": int(reserved),
                "status": "alive",
            })

        elif op == CMD_STOP:
            resp_q.put({"type": "stopped", "pid": os.getpid(), "gpu_id": gpu_id, "device": device_str})
            break

        elif op == CMD_INFER:
            # 필수 페이로드
            video_path = cmd["video_path"]
            start_frame = int(cmd["start_frame"])
            end_frame   = int(cmd["end_frame"])
            prompt      = cmd["prompt"]
            num_segments = int(cmd.get("num_segments", 12))
            input_size   = int(cmd.get("input_size", 448))

            t0 = time.time()
            ok = True
            err = None
            raw = ""
            label = "violence"
            try:
                raw = inferencer.infer_window(
                    video_path=video_path,
                    prompt=prompt,
                    start_frame=start_frame,
                    end_frame=end_frame,
                    num_segments=num_segments,
                    input_size=input_size,
                )
                label = parse_prediction(raw)
            # except Exception as e:
            #     ok = False
            #     err = repr(e)
            except Exception as e:
                # ---↓ 에러를 터미널에 직접 출력하는 코드 추가 ↓---
                print(f"\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                print(f"!!! INFERENCE ERROR on '{video_path}' (frames {start_frame}-{end_frame})")
                print(f"!!! ERROR: {e}")
                print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")
                # ---↑-----------------------------------------↑---
                ok = False
                err = repr(e)
                
                
            resp_q.put({
                "type": "infer_result",
                "pid": os.getpid(),
                "gpu_id": gpu_id,
                "video_path": video_path,
                "start_frame": start_frame,
                "end_frame": end_frame,
                "ok": ok,
                "elapsed": time.time() - t0,
                "label": label if ok else "violence",
                "raw": raw if ok else "",
                "error": err,
            })

        else:
            resp_q.put({"type": "error", "pid": os.getpid(), "gpu_id": gpu_id, "device": device_str, "error": f"unknown op: {op}"})


# ---------------- 풀/스케줄러 ----------------
def _make_worker(gpu_id: int, index_on_gpu: int, model_path: str) -> WorkerHandle:
    ctrl_q: mp.Queue = mp.Queue()
    resp_q: mp.Queue = mp.Queue()
    proc = mp.Process(target=model_worker_main, args=(gpu_id, ctrl_q, resp_q, model_path), daemon=True)
    proc.start()
    return WorkerHandle(gpu_id=gpu_id, index_on_gpu=index_on_gpu, proc=proc, ctrl_q=ctrl_q, resp_q=resp_q)

def start_model_pool(gpu_to_nproc: Dict[int, int], model_path: str) -> List[WorkerHandle]:
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    handles: List[WorkerHandle] = []
    for gpu_id, nproc in sorted(gpu_to_nproc.items()):
        for k in range(max(0, int(nproc))):
            handles.append(_make_worker(gpu_id, k, model_path))
    return handles

def wait_until_ready(handles: List[WorkerHandle], timeout_per_worker: float = 600.0) -> None:
    total = len(handles)
    received = 0
    deadline = time.time() + timeout_per_worker * max(1, total)

    while received < total and time.time() < deadline:
        for h in handles:
            try:
                msg = h.resp_q.get_nowait()
            except queue.Empty:
                continue
            if isinstance(msg, dict) and msg.get("type") == CMD_READY:
                received += 1
                print(f"[READY] ({received}/{total}) pid={msg.get('pid')} gpu={msg.get('gpu_id')} ok={msg.get('ok')}")
                if not msg.get("ok", False):
                    print("       -> ERROR:", msg.get("error"))
        time.sleep(0.05)
    if received < total:
        print(f"[WARN] READY 미수신 워커 있음: {received}/{total}")

def stop_all(handles: List[WorkerHandle], timeout: float = 20.0):
    for h in handles:
        if h.proc.is_alive():
            h.ctrl_q.put({"op": CMD_STOP})
    t0 = time.time()
    for h in handles:
        while h.proc.is_alive() and (time.time() - t0) < timeout:
            try:
                _ = h.resp_q.get_nowait()
            except queue.Empty:
                pass
            time.sleep(0.02)
        if h.proc.is_alive():
            h.proc.kill()
        h.proc.join(timeout=1.0)

# ---------------- 작업 생성/집계/CSV ----------------
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}

def list_videos(video_dir: str) -> List[str]:
    paths = []
    for ext in VIDEO_EXTS:
        paths.extend(glob.glob(os.path.join(video_dir, f"*{ext}")))
    return sorted(paths)

def video_meta(video_path: str) -> Tuple[int, float]:
    vr = build_video_reader(video_path)
    return len(vr), float(vr.get_avg_fps())

def make_windows(n_frames: int, window_size: int) -> List[Tuple[int, int]]:
    """
    0..n_frames-1 범위에서 [s, e] 윈도우를 window_size 간격으로 생성.
    마지막 구간이 부족하면 남은 만큼 사용.
    """
    wins = []
    s = 0
    while s < n_frames:
        e = min(n_frames - 1, s + window_size - 1)
        wins.append((s, e))
        s += window_size
    return wins

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

# def write_csv(video_path: str, csv_save_dir: str, labels_per_frame: List[int]):
#     """
#     CSV 형식:
#     frame,falldown,fire,violence
#     violence/normal만 사용하는 경우: violence만 0/1로 기록
#     """
#     ensure_dir(csv_save_dir)
#     base = os.path.splitext(os.path.basename(video_path))[0]
#     out_path = os.path.join(csv_save_dir, f"{base}.csv")
#     with open(out_path, "w", newline="") as f:
#         w = csv.writer(f)
#         w.writerow(["frame", "violence"])
#         for idx, v in enumerate(labels_per_frame):
#             # 현재 설계: violence만 사용 (falldown/fire는 0)
#             w.writerow([idx, int(v)])
#     return out_path

# vvvvvvvvvvvv ---- 아래 코드로 교체하세요 ---- vvvvvvvvvvvvv
def write_csv(video_path: str, csv_save_dir: str, labels_per_frame: List[int], raws_per_frame: List[str]):
    """
    CSV 형식:
    frame,violence,raw
    """
    ensure_dir(csv_save_dir)
    base = os.path.splitext(os.path.basename(video_path))[0]
    out_path = os.path.join(csv_save_dir, f"{base}.csv")
    with open(out_path, "w", newline="", encoding='utf-8') as f: # utf-8 인코딩 추가
        w = csv.writer(f)
        # 헤더에 'raw' 컬럼 추가
        w.writerow(["frame", "violence", "raw"])
        # zip을 사용해 레이블과 raw 텍스트를 함께 반복
        for idx, (label, raw_text) in enumerate(zip(labels_per_frame, raws_per_frame)):
            w.writerow([idx, int(label), raw_text])
    return out_path


# ---------------- 메인 파이프라인 ----------------
def run_inference(
    video_dir: str,
    csv_save_dir: str,
    window_size: int,
    gpu_to_nproc: Dict[int, int],
    prompt: str,
    model_path: str = "OpenGVLab/InternVL3-2B",
    num_segments: int = 12,
    input_size: int = 448,
):
    """
    - video_dir 안의 비디오를 모두 처리
    - 각 비디오는 window_size로 나눠 infer → 윈도우 레이블을 프레임 전체에 전파 → CSV 저장
    - 진행률 출력(전체/개별)
    """
    print("[STEP] 모델 풀 시작")
    handles = start_model_pool(gpu_to_nproc, model_path=model_path)

    print("[STEP] 로드 대기")
    wait_until_ready(handles)
    if not handles:
        print("[ERROR] 워커가 없습니다.")
        return

    # 작업 생성
    videos = list_videos(video_dir)
    if not videos:
        print(f"[WARN] 비디오가 없습니다: {video_dir}")
        stop_all(handles)
        return

    # 비디오별 메타/윈도우
    video_windows: Dict[str, List[Tuple[int, int]]] = {}
    video_frames: Dict[str, int] = {}
    for vp in videos:
        n_frames, fps = video_meta(vp)
        wins = make_windows(n_frames, window_size)
        video_frames[vp] = n_frames
        video_windows[vp] = wins

    total_windows = sum(len(ws) for ws in video_windows.values())
    print(f"[INFO] 총 비디오: {len(videos)}, 총 윈도우: {total_windows}, window_size={window_size}")

    # 진행률 상태
    overall_done = 0
    per_video_done = {vp: 0 for vp in videos}
    per_video_total = {vp: len(video_windows[vp]) for vp in videos}

    # 프레임 단위 violence 라벨(0/1) 초기화
    per_video_frame_labels = {vp: [0] * video_frames[vp] for vp in videos}

    # <<< -------------------- 여기를 추가하세요 -------------------- >>>
    per_video_frame_raws = {vp: [""] * video_frames[vp] for vp in videos}
    # <<< ----------------------------------------------------------- >>>

    # 간단 라운드로빈 디스패치
    # 현재는 거친 스케줄링: 비디오별 윈도우를 평평하게 풀에 흘려보냄
    job_queue = []
    for vp in videos:
        for s, e in video_windows[vp]:
            job_queue.append((vp, s, e))
    job_idx = 0

    # 먼저 워커 수만큼 채워넣기
    in_flight = 0
    for h in handles:
        if job_idx < len(job_queue):
            vp, s, e = job_queue[job_idx]
            h.ctrl_q.put({
                "op": CMD_INFER,
                "video_path": vp,
                "start_frame": s,
                "end_frame": e,
                "prompt": prompt,
                "num_segments": num_segments,
                "input_size": input_size,
            })
            job_idx += 1
            in_flight += 1

    # 수신 루프: 결과 받을 때마다 다음 작업 투입
    t_last_print = 0
    while overall_done < total_windows:
        for h in handles:
            try:
                msg = h.resp_q.get_nowait()
            except queue.Empty:
                continue

            if not isinstance(msg, dict):
                continue

            typ = msg.get("type")

            if typ == "infer_result":
                vp = msg["video_path"]
                s = msg["start_frame"]
                e = msg["end_frame"]
                ok = msg["ok"]
                label = msg.get("label", "violence")
                # <<< -------------------- 여기를 추가하세요 -------------------- >>>
                raw_text = msg.get("raw", "")
                # <<< ----------------------------------------------------------- >>>

                # 라벨 적용: violence → 1, normal → 0
                vflag = 1 if label == "violence" else 0
                for idx in range(s, e + 1):
                    if 0 <= idx < len(per_video_frame_labels[vp]):
                        # 논리 OR (다른 윈도우가 violence면 유지)
                        per_video_frame_labels[vp][idx] = per_video_frame_labels[vp][idx] or vflag
                        # <<< -------------------- 여기를 추가하세요 -------------------- >>>
                        # 해당 프레임에 raw text 저장 (단순 덮어쓰기)
                        per_video_frame_raws[vp][idx] = raw_text
                        # <<< ----------------------------------------------------------- >>>

                per_video_done[vp] += 1
                overall_done += 1
                in_flight -= 1

                # 다음 작업 투입
                if job_idx < len(job_queue):
                    nvp, ns, ne = job_queue[job_idx]
                    h.ctrl_q.put({
                        "op": CMD_INFER,
                        "video_path": nvp,
                        "start_frame": ns,
                        "end_frame": ne,
                        "prompt": prompt,
                        "num_segments": num_segments,
                        "input_size": input_size,
                    })
                    job_idx += 1
                    in_flight += 1

            # (선택) 기타 메시지 소비
            elif typ in {"pong", "error", "stopped"}:
                pass

        # 진행률 출력 (0.5초마다)
        now = time.time()
        if now - t_last_print > 0.5:
            overall_pct = 100.0 * overall_done / max(1, total_windows)
            detail = " | ".join(
                [f"{os.path.basename(vp)}: {per_video_done[vp]}/{per_video_total[vp]}"
                 for vp in videos[:6]]  # 너무 길어지지 않게 앞 6개만 보여줌
            )
            print(f"[PROGRESS] overall {overall_done}/{total_windows} ({overall_pct:5.1f}%) :: {detail}")
            t_last_print = now

        time.sleep(0.01)

    # 모든 윈도우 처리 완료 → CSV 저장
    print("[STEP] CSV 저장")
    for vp in videos:
        # out = write_csv(vp, csv_save_dir, per_video_frame_labels[vp])
        # vvvvvvvvvvvv ---- 이 부분을 수정하세요 ---- vvvvvvvvvvvvv
        out = write_csv(vp, csv_save_dir, per_video_frame_labels[vp], per_video_frame_raws[vp])
        # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^        

        print(f"  - saved: {out}")

    print("[STEP] 종료")
    stop_all(handles)
    print("[DONE]")

# ---------------- 간단 실행 예시 ----------------
if __name__ == "__main__":
    """
    예시 실행:
      python multi_gpu_infer.py
    아래 값만 수정해서 테스트 하세요.
    """
    VIDEO_DIR = "/home/piawsa6000/nas192/datasets/projects/huggingface_benchmarks_dataset/Leaderboard_bench/PIA_Violence/dataset/violence"         # a6000
    VIDEO_DIR = "/mnt/nas_192/datasets/projects/huggingface_benchmarks_dataset/Leaderboard_bench/PIA_Violence/dataset/violence"         # h100
    # VIDEO_DIR = "data/test/test2"         # h100

    CSV_DIR   = "results/eval_hf_result/InternVL3-2B_gangnam_vietnam_aihubstore_gj_space_no_split"       # CSV 저장 폴더
    CSV_DIR   = "results/eval_hf_result/InternVL3-2B_gangnam_rwf2000_gj_cctv_scvdALL_NOweapon_no_split"
    CSV_DIR   = "results/eval_test/test6"

    WINDOW    = 12                        # 윈도우 크기(프레임 개수)
    PROMPT    = """
    Watch this short video clip and respond with exactly one JSON object.\n\n[Rules]\n- The category must be either 'violence' or 'normal'.  \n- Classify as violence if any of the following actions are present:  \n  * Punching  \n  * Kicking  \n  * Weapon Threat\n  * Weapon Attack\n  * Falling/Takedown  \n  * Pushing/Shoving  \n  * Brawling/Group Fight  \n- If none of the above are observed, classify as normal.  \n- The following cases must always be classified as normal:  \n  * Affection (hugging, holding hands, light touches)  \n  * Helping (supporting, assisting)  \n  * Accidental (unintentional bumping)  \n  * Playful (non-aggressive playful contact)  \n\n[Output Format]\n- Output exactly one JSON object.  \n- The object must contain only two keys: \"category\" and \"description\".  \n- The description should briefly and objectively describe the scene.  \n\nExample (violence):  \n{\"category\":\"violence\",\"description\":\"A man in a black jacket punches another man, who stumbles backward.\"}\n\nExample (normal):  \n{\"category\":\"normal\",\"description\":\"Two people are hugging inside an elevator
    """
    GPU_PROC  = {0:8, 1:8, 2:8, 3:8}                    # GPU0에 프로세스 2개
    MODEL_ID  = "ckpts/InternVL3-2B_gangnam_vietnam_rwf2000_aihubstore_gj_space_no_split"
    MODEL_ID  = "ckpts/InternVL3-2B"
    MODEL_ID  = "ckpts/InternVL3-2B_gangnam_rwf2000_gj_cctv_scvdALL_NOweapon_no_split"

    
    run_inference(
        video_dir=VIDEO_DIR,
        csv_save_dir=CSV_DIR,
        window_size=WINDOW,
        gpu_to_nproc=GPU_PROC,
        prompt=PROMPT,
        model_path=MODEL_ID,
        num_segments=12,
        input_size=448,
    )


# export OMP_NUM_THREADS=1
# export MKL_NUM_THREADS=1
# CUDA_VISIBLE_DEVICES=0,1 \
# PYTHONPATH="$(pwd)" torchrun --nproc_per_node=2 src/evaluation/evaluate_pia_hf_bench_eff.py \
#     --checkpoint ckpts/InternVL3-2B_gangnam_vietnam_aihubstore_gj_space_no_split \
#     --video-root /home/piawsa6000/nas192/datasets/projects/huggingface_benchmarks_dataset/Leaderboard_bench/PIA_Violence/dataset/violence \
#     --out-dir results/eval_hf_result/InternVL3-2B_gangnam_vietnam_aihubstore_gj_space_no_split \
#     --window-size 15 \
#     --workers-per-gpu 0 \
#     --procs-per-gpu 4