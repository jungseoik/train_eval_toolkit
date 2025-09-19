import argparse
import os
import random
import time
import torch
import torch.multiprocessing as mp
from decord import VideoReader
from PIL import Image
from tqdm import tqdm
import numpy as np
import json
import re
import pandas as pd

from src.training.internvl.model.internvl_chat import InternVLChatModel
from transformers import AutoTokenizer
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CLIP_MEAN = (0.4814546, 0.4578275, 0.40821073)
CLIP_STD = (0.2686295, 0.2613025, 0.2757711)

worker_globals = {}


def build_transform(is_train, input_size, normalize_type='imagenet'):
    if normalize_type == 'imagenet':
        MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    elif normalize_type == 'clip':
        MEAN, STD = CLIP_MEAN, CLIP_STD
    else:
        raise NotImplementedError
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform


def load_model_and_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, trust_remote_code=True, use_fast=False)
    model = InternVLChatModel.from_pretrained(
        args.checkpoint, low_cpu_mem_usage=True, torch_dtype=torch.bfloat16,
        load_in_8bit=args.load_in_8bit, load_in_4bit=args.load_in_4bit).eval()
    return model, tokenizer


def init_worker(model, tokenizer, args, local_rank):
    torch.cuda.set_device(local_rank)
    worker_globals['model'] = model
    worker_globals['tokenizer'] = tokenizer
    worker_globals['args'] = args
    worker_globals['local_rank'] = local_rank
    worker_globals['prompt'] = (
        "Watch this short video clip and respond with exactly one JSON object.\n\n"
        "- category must be 'violence' or 'normal'.\n"
        "- If parsing fails, assume violence.\n"
        "Output format: {\"category\":\"violence|normal\",\"description\":\"...\"}"
    )
    worker_globals['image_size'] = model.config.force_image_size or model.config.vision_config.image_size
    worker_globals['transform'] = build_transform(is_train=False, input_size=worker_globals['image_size'])


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
        return 'violence'  # fallback: violence
    except Exception:
        return 'violence'


def process_video(video_path):
    model = worker_globals['model']
    tokenizer = worker_globals['tokenizer']
    args = worker_globals['args']
    prompt = worker_globals['prompt']
    transform = worker_globals['transform']
    local_rank = worker_globals['local_rank']

    try:
        vr = VideoReader(video_path, num_threads=1)
    except Exception as e:
        print(f"Error loading video {video_path}: {e}")
        return None

    vlen = len(vr)
    results = np.zeros(vlen, dtype=int)

    ws = args.window_size

    # stride = window_size (겹치지 않게)
    for start in range(0, vlen, ws):
        end = min(start + ws, vlen)

        # 프레임 인덱스
        frame_indices = list(range(start, end))

        # 남는 프레임 처리: 마지막 프레임 반복해서 window_size 맞추기
        if len(frame_indices) < ws:
            frame_indices += [frame_indices[-1]] * (ws - len(frame_indices))

        frames = vr.get_batch(frame_indices).asnumpy()
        frames = [Image.fromarray(frames[i]) for i in range(frames.shape[0])]
        pixel_values = torch.stack([transform(f) for f in frames])
        pixel_values = pixel_values.to(f'cuda:{local_rank}', dtype=torch.bfloat16)

        pred = model.chat(
            tokenizer=tokenizer,
            pixel_values=pixel_values,
            question=prompt,
            generation_config=dict(num_beams=1, max_new_tokens=50, min_new_tokens=5),
            verbose=False
        )

        category = parse_prediction(pred)

        # 윈도우 단위 결과 → start~end 범위 프레임 전체 채우기
        if category != 'normal':  # violence or parsing 실패 → violence
            results[start:end] = 1
        else:
            results[start:end] = 0

    # CSV 저장
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    save_path = os.path.join(args.out_dir, f"{video_name}.csv")
    df = pd.DataFrame({"frame": np.arange(vlen), "violence": results})
    df.to_csv(save_path, index=False)
    print(f"Saved {save_path}")
    return save_path


def evaluate_videos(local_rank, model, tokenizer, args):
    video_files = [os.path.join(args.video_root, f) for f in os.listdir(args.video_root)
                   if f.lower().endswith(('.mp4', '.avi', '.mov'))]

    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=args.workers_per_gpu,
                  initializer=init_worker,
                  initargs=(model, tokenizer, args, local_rank)) as pool:
        list(tqdm(pool.imap_unordered(process_video, video_files),
                  total=len(video_files), desc="Processing Videos"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--video-root', type=str, required=True)
    parser.add_argument('--out-dir', type=str, default='results/csv_outputs')
    parser.add_argument('--window-size', type=int, default=15)
    parser.add_argument('--workers-per-gpu', type=int, default=2)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--load-in-8bit', action='store_true')
    parser.add_argument('--load-in-4bit', action='store_true')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    local_rank = int(os.getenv('LOCAL_RANK', 0))
    torch.cuda.set_device(local_rank)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    model, tokenizer = load_model_and_tokenizer(args)
    if not args.load_in_8bit and not args.load_in_4bit:
        model = model.to(local_rank)

    evaluate_videos(local_rank, model, tokenizer, args)
