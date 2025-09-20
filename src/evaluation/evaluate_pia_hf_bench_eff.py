import argparse
import os
import random
import torch
from torch.utils.data import Dataset, DataLoader
from decord import VideoReader
from PIL import Image
from tqdm import tqdm
import numpy as np
import json
import re
import pandas as pd
import multiprocessing as mp
from math import ceil

from transformers import AutoModelForCausalLM, AutoTokenizer
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
# ----------------------------
# Preprocessing Utils (원본과 동일)
# ----------------------------
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def ensure_rgb(img):
    return img.convert("RGB") if getattr(img, "mode", None) != "RGB" else img

def build_transform(input_size: int = 448):
    return T.Compose(
        [
            T.Lambda(ensure_rgb),  # ← lambda 제거
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
        if diff < best_ratio_diff or (
            diff == best_ratio_diff and area > 0.5 * image_size * image_size * ratio[0] * ratio[1]
        ):
            best_ratio_diff = diff
            best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    ow, oh = image.size
    aspect_ratio = ow / oh
    # NOTE: For typical 16:9 or 4:3 video frames, this complex tiling is often not triggered
    # and it will fall back to a single 1x1 tile. We keep the logic for consistency.
    target_ratios = sorted(
        {
            (i, j)
            for n in range(min_num, max_num + 1)
            for i in range(1, n + 1)
            for j in range(1, n + 1)
            if min_num <= i * j <= max_num
        },
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


# ----------------------------
# Dataset and DataLoader Setup
# ----------------------------

class VideoClipDataset(Dataset):
    def __init__(self, video_root, window_size, transform_builder, image_size):
        self.video_root = video_root
        self.window_size = window_size
        self.transform_builder = transform_builder
        self.image_size = image_size
        self.transform = self.transform_builder(self.image_size) # Create one transform instance

        self.video_files = sorted([os.path.join(video_root, f) for f in os.listdir(video_root)
                                   if f.lower().endswith(('.mp4', '.avi', '.mov'))])
        
        print("Scanning videos to create clips...")
        self.clips = []
        self.video_lengths = {}
        for video_path in tqdm(self.video_files, desc="Scanning Videos"):
            try:
                vr = VideoReader(video_path, num_threads=1)
                vlen = len(vr)
                self.video_lengths[video_path] = vlen
                for start in range(0, vlen, self.window_size):
                    self.clips.append({'video_path': video_path, 'start_frame': start})
            except Exception as e:
                print(f"Warning: Could not read video info from {video_path}: {e}")

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        clip_info = self.clips[idx]
        video_path = clip_info['video_path']
        start_frame = clip_info['start_frame']
        
        vr = VideoReader(video_path, num_threads=1)
        end_frame = min(start_frame + self.window_size, self.video_lengths[video_path])
        
        pixel_values_list = []
        num_patches_list = []
        
        frame_indices = list(range(start_frame, end_frame))
        for frame_idx in frame_indices:
            img = Image.fromarray(vr[frame_idx].asnumpy()).convert("RGB")
            
            # ⭐️⭐️⭐️ [핵심 수정 사항] ⭐️⭐️⭐️
            # 원본 코드의 dynamic_preprocess 로직을 여기에 다시 적용합니다.
            # max_num=1로 설정하여 각 프레임을 하나의 타일로 처리하도록 강제합니다.
            tiles = dynamic_preprocess(
                img, 
                image_size=self.image_size, 
                use_thumbnail=True, 
                max_num=1 # 각 프레임은 단일 이미지로 취급
            )
            
            pv = [self.transform(tile) for tile in tiles]
            pv = torch.stack(pv)
            num_patches_list.append(pv.shape[0])
            pixel_values_list.append(pv)

        pixel_values = torch.cat(pixel_values_list)
        
        return {
            'pixel_values': pixel_values,
            'num_patches_list': num_patches_list,
            'video_path': video_path,
            'start_frame': start_frame,
            'end_frame': end_frame
        }

def custom_collate_fn(batch):
    return batch


# ----------------------------
# Main Logic
# ----------------------------

def load_model_and_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, trust_remote_code=True, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint,
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16,
        load_in_8bit=args.load_in_8bit,
        load_in_4bit=args.load_in_4bit,
        trust_remote_code=True
    ).eval()
    return model, tokenizer

def parse_prediction(pred_str: str) -> str:
    try:
        clean_str = pred_str
        if '```json' in clean_str:
            clean_str = clean_str.split('```json')[1].split('```')[0]
        elif '```' in clean_str:
            clean_str = clean_str.split('```')[1].split('```')[0]
        clean_str = clean_str.strip()
        start_brace = clean_str.find('{')
        end_brace = clean_str.rfind('}')
        if start_brace != -1 and end_brace != -1 and start_brace < end_brace:
            json_part = clean_str[start_brace:end_brace + 1]
            data = json.loads(json_part)
            category = data.get('category')
            if category in ['violence', 'normal']:
                return category
        cat_match = re.search(r'["\']category["\']\s*:\s*["\'](violence|normal)["\']', clean_str)
        if cat_match:
            return cat_match.group(1)
        return 'violence'
    except Exception:
        return 'violence'
def worker_process(local_rank, device_id, args, image_size, clip_chunk, video_lengths):
    import os
    import torch
    from decord import VideoReader
    from PIL import Image

    torch.cuda.set_device(device_id)
    device = torch.device(f"cuda:{device_id}") if torch.cuda.is_available() else torch.device("cpu")

    # 각 워커가 자기 모델/토크나이저 로드
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, trust_remote_code=True, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint,
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16,
        load_in_8bit=args.load_in_8bit,
        load_in_4bit=args.load_in_4bit,
        trust_remote_code=True
    ).eval()
    if not args.load_in_8bit and not args.load_in_4bit:
        model = model.to(device)

    transform = build_transform(image_size)

    # 프롬프트 닫힘 수정 (마지막에 ." } 로 끝나야 함)
    prompt = (
    """
        "Watch this short video clip and respond with exactly one JSON object.\n\n[Rules]\n- The category must be either 'violence' or 'normal'.  \n- Classify as violence if any of the following actions are present:  \n  * Punching  \n  * Kicking  \n  * Weapon Threat\n  * Weapon Attack\n  * Falling/Takedown  \n  * Pushing/Shoving  \n  * Brawling/Group Fight  \n- If none of the above are observed, classify as normal.  \n- The following cases must always be classified as normal:  \n  * Affection (hugging, holding hands, light touches)  \n  * Helping (supporting, assisting)  \n  * Accidental (unintentional bumping)  \n  * Playful (non-aggressive playful contact)  \n\n[Output Format]\n- Output exactly one JSON object.  \n- The object must contain only two keys: \"category\" and \"description\".  \n- The description should briefly and objectively describe the scene.  \n\nExample (violence):  \n{\"category\":\"violence\",\"description\":\"A man in a black jacket punches another man, who stumbles backward.\"}\n\nExample (normal):  \n{\"category\":\"normal\",\"description\":\"Two people are hugging inside an elevator"
    """
    )

    local_results = {}
    processed_clips = 0

    for clip in clip_chunk:
        video_path = clip['video_path']
        start_frame = clip['start_frame']

        vlen = video_lengths[video_path]
        end_frame = min(start_frame + args.window_size, vlen)

        try:
            vr = VideoReader(video_path, num_threads=1)
        except Exception as e:
            # 깨진 스트림 등은 스킵
            processed_clips += 1
            continue

        frame_indices = list(range(start_frame, end_frame))

        pixel_values_list = []
        num_patches_list = []
        for frame_idx in frame_indices:
            try:
                img = Image.fromarray(vr[frame_idx].asnumpy()).convert("RGB")
            except Exception:
                # 프레임 읽기 실패 시 해당 클립만 스킵
                pixel_values_list = []
                break

            tiles = dynamic_preprocess(
                img,
                image_size=image_size,
                use_thumbnail=True,
                max_num=1
            )
            pv = torch.stack([transform(t) for t in tiles])
            num_patches_list.append(pv.shape[0])
            pixel_values_list.append(pv)

        if not pixel_values_list:
            processed_clips += 1
            continue

        pixel_values = torch.cat(pixel_values_list).to(device, dtype=torch.bfloat16, non_blocking=True)

        video_prefix = ''.join([f'Frame{i+1}: <image>\n' for i in range(len(num_patches_list))])
        final_question = video_prefix + prompt

        with torch.no_grad():
            pred = model.chat(
                tokenizer=tokenizer,
                pixel_values=pixel_values,
                question=final_question,
                generation_config=dict(num_beams=1, max_new_tokens=50, min_new_tokens=5),
                num_patches_list=num_patches_list,
                verbose=False,
            )

        category = parse_prediction(pred)
        prediction_value = 1 if category != 'normal' else 0

        if video_path not in local_results:
            length = video_lengths[video_path]
            local_results[video_path] = torch.zeros(length, dtype=torch.int32, device=device)

        local_results[video_path][start_frame:end_frame] = prediction_value
        processed_clips += 1

        # 메모리 정리
        del vr

    # CPU로 변환해 상위 프로세스로 반환 + ✅ 이번 워커가 처리한 클립 수 반환
    return {k: v.to('cpu') for k, v in local_results.items()}, processed_clips


def evaluate_videos(model, tokenizer, args, device, local_rank):
    # 이 rank가 사용할 GPU id
    device_id = device.index if device.type == 'cuda' else None

    # 이미지 크기
    image_size = getattr(model.config, "force_image_size", None)
    if image_size is None:
        vc = getattr(model.config, "vision_config", None)
        image_size = getattr(vc, "image_size", 448)

    # Dataset (로더는 쓰지 않음)
    dataset = VideoClipDataset(args.video_root, args.window_size, build_transform, image_size)

    world_size = dist.get_world_size() if dist.is_initialized() else 1
    rank = dist.get_rank() if dist.is_initialized() else 0

    # 1차 분할 (랭크별)
    N = len(dataset)
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    rank_indices = list(range(rank, N, world_size))

    # 넘겨줄 “간단한” 클립 리스트를 만들기 (pickle-friendly)
    clips = [dataset.clips[i] for i in rank_indices]   # 각 원소: {'video_path': ..., 'start_frame': ...}
    video_lengths = dataset.video_lengths               # dict[str,int] (pickle 가능)

    # 2차 분할 (GPU당 K개 워커)
    procs = max(1, getattr(args, "procs_per_gpu", 1))
    if procs > 1:
        chunk_size = ceil(len(clips) / procs)
        chunks = [clips[i*chunk_size : (i+1)*chunk_size] for i in range(procs) if i*chunk_size < len(clips)]

        # 랭크/디바이스 상태 출력
        print(f"[rank {rank}] cuda:{device.index} -> spawning {len(chunks)} workers "
              f"(procs-per-gpu={procs}), total clips={len(clips)}, chunk_size={chunk_size}")

        mp.set_start_method("spawn", force=True)
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=len(chunks)) as pool:
            # ✅ 진행바: 이 랭크에 할당된 '클립 수' 기준
            pbar = tqdm(total=len(clips), desc=f"Rank {rank} on cuda:{device.index}", position=rank, dynamic_ncols=True)
            jobs = [
                pool.apply_async(
                    worker_process,
                    (local_rank, device.index, args, image_size, chunk, video_lengths)
                )
                for chunk in chunks
            ]

            worker_outputs = []
            for j, chunk in zip(jobs, chunks):
                # 워커 하나가 끝날 때마다 결과 수신 + 진행도 업데이트
                out_dict, processed_cnt = j.get()
                worker_outputs.append(out_dict)
                # processed_cnt가 chunk 길이와 다를 수도 있으니, 워커 보고대로 올림
                pbar.update(processed_cnt)

            pbar.close()
    else:
        print(f"[rank {rank}] cuda:{device.index} -> single worker, total clips={len(clips)}")
        out_dict, processed_cnt = worker_process(local_rank, device.index, args, image_size, clips, video_lengths)
        worker_outputs = [out_dict]

    # rank 내부 병합(MAX)
    # 먼저 모든 비디오에 대한 텐서를 준비 (GPU 텐서로)
    rank_results = {
        path: torch.zeros(length, dtype=torch.int32, device=device)
        for path, length in dataset.video_lengths.items()
    }

    # 워커가 보낸 CPU 텐서를 합치기
    for wout in worker_outputs:
        for vpath, cpu_tensor in wout.items():
            t = cpu_tensor.to(device)
            # MAX 병합: 어느 워커든 1이면 1
            rank_results[vpath] = torch.maximum(rank_results[vpath], t)

    # 전 랭크 병합(all_reduce MAX)
    if dist.is_initialized():
        for k in rank_results:
            dist.all_reduce(rank_results[k], op=dist.ReduceOp.MAX)

    # 저장은 rank 0만
    is_main = (not dist.is_initialized()) or (dist.get_rank() == 0)
    if is_main:
        print("\nSaving results to CSV files...")
        os.makedirs(args.out_dir, exist_ok=True)
        for video_path, results in tqdm(rank_results.items(), desc="Saving CSVs"):
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            save_path = os.path.join(args.out_dir, f"{video_name}.csv")
            arr = results.detach().to('cpu').numpy()
            df = pd.DataFrame({"frame": np.arange(len(arr)), "violence": arr})
            df.to_csv(save_path, index=False)
        print("Evaluation finished successfully.")

    if dist.is_initialized():
        dist.barrier()

def init_distributed():
    # torchrun이 LOCAL_RANK 등을 넣어줍니다.
    if 'LOCAL_RANK' not in os.environ:
        return None  # 단일 GPU/CPU 실행
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # ... (argparse 설정은 이전과 동일)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--video-root', type=str, required=True)
    parser.add_argument('--out-dir', type=str, default='results/csv_outputs')
    parser.add_argument('--window-size', type=int, default=15, help='Number of frames to process in one clip.')
    parser.add_argument('--batch-size', type=int, default=4, help='Number of clips to process in a single model forward pass.')
    parser.add_argument('--workers-per-gpu', type=int, default=4, help='Number of worker processes for data loading.')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--load-in-8bit', action='store_true')
    parser.add_argument('--load-in-4bit', action='store_true')
    parser.add_argument('--procs-per-gpu', type=int, default=4, help='Number of model worker processes per GPU rank.')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    
    local_rank = init_distributed()
    if torch.cuda.is_available():
        if local_rank is None:
            device = torch.device("cuda:0")
        else:
            device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")
    
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    model, tokenizer = load_model_and_tokenizer(args)
    if not args.load_in_8bit and not args.load_in_4bit:
        model = model.to(device)

    evaluate_videos(model, tokenizer, args, device, local_rank)