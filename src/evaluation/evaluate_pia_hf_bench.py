import argparse
import os
import random
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


# ----------------------------
# Preprocessing Utils
# ----------------------------
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size: int = 448):
    """Return torchvision transform matching InternVL pre-training."""
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
        if diff < best_ratio_diff or (
            diff == best_ratio_diff and area > 0.5 * image_size * image_size * ratio[0] * ratio[1]
        ):
            best_ratio_diff = diff
            best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    """Split arbitrarily-sized image into ≤12 tiles sized 448×448 (InternVL spec)."""
    ow, oh = image.size
    aspect_ratio = ow / oh
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
# Model / Worker setup
# ----------------------------
worker_globals = {}


def load_model_and_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, trust_remote_code=True, use_fast=False)
    model = InternVLChatModel.from_pretrained(
        args.checkpoint,
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16,
        load_in_8bit=args.load_in_8bit,
        load_in_4bit=args.load_in_4bit,
    ).eval()
    return model, tokenizer

def init_worker(model, tokenizer, args, local_rank):
    torch.cuda.set_device(local_rank)
    worker_globals['model'] = model
    worker_globals['tokenizer'] = tokenizer
    worker_globals['args'] = args
    worker_globals['local_rank'] = local_rank
    worker_globals['prompt'] = (
        "Watch this short video clip and respond with exactly one JSON object.\n\n[Rules]\n- The category must be either 'violence' or 'normal'.  \n- Classify as violence if any of the following actions are present:  \n  * Punching  \n  * Kicking  \n  * Weapon Threat\n  * Weapon Attack\n  * Falling/Takedown  \n  * Pushing/Shoving  \n  * Brawling/Group Fight  \n- If none of the above are observed, classify as normal.  \n- The following cases must always be classified as normal:  \n  * Affection (hugging, holding hands, light touches)  \n  * Helping (supporting, assisting)  \n  * Accidental (unintentional bumping)  \n  * Playful (non-aggressive playful contact)  \n\n[Output Format]\n- Output exactly one JSON object.  \n- The object must contain only two keys: \"category\" and \"description\".  \n- The description should briefly and objectively describe the scene.  \n\nExample (violence):  \n{\"category\":\"violence\",\"description\":\"A man in a black jacket punches another man, who stumbles backward.\"}\n\nExample (normal):  \n{\"category\":\"normal\",\"description\":\"Two people are hugging inside an elevator"
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
        frame_indices = list(range(start, end))

        # ---- InternVL 전처리 (dynamic_preprocess) ----
        pixel_values_list, num_patches_list = [], []
        for idx in frame_indices:
            img = Image.fromarray(vr[idx].asnumpy()).convert("RGB")
            tiles = dynamic_preprocess(
                img,
                image_size=worker_globals["image_size"],
                use_thumbnail=True,
                max_num=1,
            )
            pv = [transform(tile) for tile in tiles]
            pv = torch.stack(pv)
            num_patches_list.append(pv.shape[0])
            pixel_values_list.append(pv)

        pixel_values = torch.cat(pixel_values_list).to(f"cuda:{local_rank}", dtype=torch.bfloat16)
        # --- 수정된 부분 시작 ---
        # 각 프레임에 대한 <image> 플레이스홀더를 동적으로 생성합니다.
        # num_patches_list의 길이가 현재 처리 중인 프레임 수를 의미합니다.
        video_prefix = ''.join([f'Frame{i+1}: <image>\n' for i in range(len(num_patches_list))])
        # 생성된 prefix와 기존 프롬프트를 결합하여 최종 질문을 만듭니다.
        final_question = video_prefix + prompt
        # --- 수정된 부분 끝 ---

        # 모델 추론
        pred = model.chat(
            tokenizer=tokenizer,
            pixel_values=pixel_values,
            question=final_question,
            generation_config=dict(num_beams=1, max_new_tokens=50, min_new_tokens=5),
            num_patches_list=num_patches_list,
            verbose=False,
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
