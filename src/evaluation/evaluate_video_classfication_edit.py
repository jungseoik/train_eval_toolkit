import argparse
import itertools
import json
import os
import random
import time
from functools import partial

import torch
from decord import VideoReader, cpu
from PIL import Image
from tqdm import tqdm
import numpy as np
import re

from src.training.internvl.model.internvl_chat import InternVLChatConfig, InternVLChatModel
from transformers import AutoTokenizer
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

### MODIFICATION START ###
# 새로운 비디오 전처리 및 로딩 함수들을 여기에 추가합니다.
# 기존의 get_frame_indices, read_frames_decord, build_transform 함수는
# 아래의 새로운 함수들로 대체되므로 제거하거나 주석 처리합니다.

def build_transform(input_size=448):
    """Return torchvision transform matching InternVL pre‑training."""
    return T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
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
    """Split arbitrarily‑sized image into ≤12 tiles sized 448×448 (InternVL spec)."""
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
    # Ensure indices are within bounds
    frame_indices = np.clip(frame_indices, 0, max_frame).astype(int)
    return frame_indices

def load_video(video_path, bound=None, input_size=448, max_num=1, num_segments=32):
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    max_frame = len(vr) - 1
    fps = float(vr.get_avg_fps())

    pixel_values_list, num_patches_list = [], []
    transform = build_transform(input_size=input_size)
    frame_indices = get_index(bound, fps, max_frame, first_idx=0, num_segments=num_segments)
    
    for frame_index in frame_indices:
        img = Image.fromarray(vr[frame_index].asnumpy()).convert('RGB')
        # 각 프레임을 dynamic_preprocess로 처리 (max_num=1이면 타일링 안 함)
        img_tiles = dynamic_preprocess(img, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pixel_values = [transform(tile) for tile in img_tiles]
        pixel_values = torch.stack(pixel_values)
        num_patches_list.append(pixel_values.shape[0])
        pixel_values_list.append(pixel_values)
        
    pixel_values = torch.cat(pixel_values_list)
    return pixel_values, num_patches_list

### MODIFICATION END ###


def load_model_and_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, trust_remote_code=True, use_fast=False)
    model = InternVLChatModel.from_pretrained(
        args.checkpoint, low_cpu_mem_usage=True, torch_dtype=torch.bfloat16,
        load_in_8bit=args.load_in_8bit, load_in_4bit=args.load_in_4bit).eval()
    if not args.load_in_8bit and not args.load_in_4bit:
        model = model.cuda()
    return model, tokenizer


### MODIFICATION START ###
# VideoClassificationDataset과 collate_fn을 새로운 비디오 처리 방식에 맞게 수정합니다.

class VideoClassificationDataset(torch.utils.data.Dataset):
    def __init__(self, annotation_path, num_frames=16, input_size=224, video_root='', max_tiles=1):
        self.data = []
        with open(annotation_path, 'r') as f:
            for line in f:
                self.data.append(json.loads(line))
        
        self.num_frames = num_frames
        self.input_size = input_size
        self.video_root = video_root
        self.max_tiles = max_tiles # 각 프레임을 몇 개의 타일로 나눌지 결정
        print(f"Found {len(self.data)} videos in {annotation_path}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        video_path = os.path.join(self.video_root, item['video'])
        video_id = item.get('id', video_path)

        try:
            gt_value_str = item['conversations'][0]['value']
            ground_truth_category = json.loads(gt_value_str)['category']
        except (json.JSONDecodeError, KeyError, IndexError) as e:
            print(f"Error parsing ground truth for video_id {video_id}: {e}")
            ground_truth_category = "parsing_error"

        try:
            # 새로운 load_video 함수 사용
            pixel_values, num_patches_list = load_video(
                video_path, 
                num_segments=self.num_frames, 
                input_size=self.input_size,
                max_num=self.max_tiles
            )
        except Exception as e:
            print(f"Error processing video {video_path}: {e}")
            # 에러 발생 시 더미 데이터 반환
            pixel_values = torch.zeros((self.num_frames, 3, self.input_size, self.input_size))
            num_patches_list = [1] * self.num_frames

        return {
            'video_id': video_id,
            'pixel_values': pixel_values,
            'num_patches_list': num_patches_list,
            'ground_truth': ground_truth_category,
        }

def collate_fn(inputs):
    # 배치 사이즈가 1로 고정되어 있으므로, 리스트의 첫 번째 원소만 사용
    batch = inputs[0]
    pixel_values = batch['pixel_values']
    num_patches_list = batch['num_patches_list']
    video_id = batch['video_id']
    ground_truth = batch['ground_truth']
    
    # DataLoader가 배치 차원을 추가하지 않도록 단일 아이템을 직접 반환
    # 단, 나중에 루프에서 일관성을 위해 리스트로 감싸줍니다.
    return pixel_values, [num_patches_list], [video_id], [ground_truth]

### MODIFICATION END ###


class InferenceSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, size):
        self._size = int(size)
        assert size > 0
        self._rank = torch.distributed.get_rank()
        self._world_size = torch.distributed.get_world_size()
        self._local_indices = self._get_local_indices(size, self._world_size, self._rank)

    @staticmethod
    def _get_local_indices(total_size, world_size, rank):
        shard_size = total_size // world_size
        left = total_size % world_size
        shard_sizes = [shard_size + int(r < left) for r in range(world_size)]
        begin = sum(shard_sizes[:rank])
        end = min(sum(shard_sizes[:rank + 1]), total_size)
        return range(begin, end)

    def __iter__(self):
        yield from self._local_indices

    def __len__(self):
        return len(self._local_indices)

def parse_prediction(pred_str: str) -> str:
    """모델의 출력 문자열에서 JSON을 파싱하여 category 값을 추출합니다."""
    try:
        if '```json' in pred_str:
            pred_str = pred_str.split('```json')[1].split('```')[0]
        elif '```' in pred_str:
            pred_str = pred_str.split('```')[1].split('```')[0]
            
        match = re.search(r'\{.*\}', pred_str, re.DOTALL)
        if match:
            json_str = match.group(0)
            data = json.loads(json_str)
            return data.get('category', 'parsing_failed')
        else:
            return 'no_json_found'
    except Exception:
        return 'parsing_failed'

def evaluate_video_classification():
    # 모델에 비디오에 대한 분류를 요청하는 프롬프트 (이제 <image> 플레이스홀더는 동적으로 생성됨)
    prompt = (
        "Watch this short video clip and respond with exactly one JSON object.\n\n[Rules]\n- The category must be either 'violence' or 'normal'.  \n- Classify as violence if any of the following actions are present:  \n  * Punching  \n  * Kicking  \n  * Weapon Threat\n  * Weapon Attack\n  * Falling/Takedown  \n  * Pushing/Shoving  \n  * Brawling/Group Fight  \n- If none of the above are observed, classify as normal.  \n- The following cases must always be classified as normal:  \n  * Affection (hugging, holding hands, light touches)  \n  * Helping (supporting, assisting)  \n  * Accidental (unintentional bumping)  \n  * Playful (non-aggressive playful contact)  \n\n[Output Format]\n- Output exactly one JSON object.  \n- The object must contain only two keys: \"category\" and \"description\".  \n- The description should briefly and objectively describe the scene.  \n\nExample (violence):  \n{\"category\":\"violence\",\"description\":\"A man in a black jacket punches another man, who stumbles backward.\"}\n\nExample (normal):  \n{\"category\":\"normal\",\"description\":\"Two people are hugging inside an elevator"
      
    )
    print('Prompt Body:', prompt)

    ### MODIFICATION START ###
    # 데이터셋 생성 시 새로운 인자(max_tiles)를 전달합니다.
    dataset = VideoClassificationDataset(
        annotation_path=args.annotation,
        num_frames=args.num_frames,
        input_size=image_size,
        video_root=args.video_root,
        max_tiles=args.max_tiles
    )
    ### MODIFICATION END ###
    
    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        sampler=InferenceSampler(len(dataset)),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_fn,
    )

    results_list = []
    ### MODIFICATION START ###
    # Dataloader의 반환값이 변경되었으므로 루프 변수를 수정합니다.
    for pixel_values, num_patches_lists, video_ids, ground_truths in tqdm(dataloader):
        pixel_values = pixel_values.to(torch.bfloat16).cuda()
        
        # 배치 사이즈가 1이므로 첫 번째 아이템의 num_patches_list를 사용
        num_patches_list = num_patches_lists[0]
        
        # 비디오 프레임에 대한 프롬프트 접두사 생성
        video_prefix = ''.join([f'Frame{i+1}: <image>\n' for i in range(len(num_patches_list))])
        # 최종 질문 구성
        question = video_prefix + prompt
        
        pred = model.chat(
            tokenizer=tokenizer,
            pixel_values=pixel_values, # 이제 pixel_values는 (T*tiles, C, H, W) 형태
            question=question,
            generation_config=dict(
                num_beams=args.num_beams,
                max_new_tokens=50,
                min_new_tokens=5,
            ),
            verbose=False
        )
    ### MODIFICATION END ###
        
        predicted_category = parse_prediction(pred)
        
        results_list.append({
            'video_id': video_ids[0],
            'prediction': predicted_category,
            'ground_truth': ground_truths[0],
            'is_correct': predicted_category == ground_truths[0]
        })

    torch.distributed.barrier()

    world_size = torch.distributed.get_world_size()
    merged_results = [None for _ in range(world_size)]
    torch.distributed.all_gather_object(merged_results, results_list)

    if torch.distributed.get_rank() == 0:
        merged_results = list(itertools.chain.from_iterable(merged_results))
        
        total_samples = len(merged_results)
        correct_predictions = sum(1 for r in merged_results if r['is_correct'])
        accuracy = (correct_predictions / total_samples) * 100 if total_samples > 0 else 0
        
        print("\n--- Evaluation Summary ---")
        print(f"Total Videos Evaluated: {total_samples}")
        print(f"Correct Predictions: {correct_predictions}")
        print(f"Accuracy: {accuracy:.2f}%")
        print("--------------------------\n")

        time_prefix = time.strftime('%y%m%d%H%M%S', time.localtime())
        results_file_path = os.path.join(args.out_dir, f'classification_results_{time_prefix}.json')
        with open(results_file_path, 'w') as f:
            json.dump(merged_results, f, indent=4)
        print(f"Detailed results saved to: {results_file_path}")

        summary_path = os.path.join(args.out_dir, 'evaluation_summary.txt')
        with open(summary_path, 'a') as f:
            f.write(f"Checkpoint: {args.checkpoint}\n")
            f.write(f"Annotation: {args.annotation}\n")
            f.write(f"Total: {total_samples}, Correct: {correct_predictions}, Accuracy: {accuracy:.2f}%\n")
            f.write("-" * 20 + "\n")
        print(f"Summary saved to: {summary_path}")

    torch.distributed.barrier()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True, help="Path to the model checkpoint")
    parser.add_argument('--annotation', type=str, required=True, help="Path to the annotation .jsonl file")
    parser.add_argument('--video-root', type=str, default='', help="Root directory for video files")
    parser.add_argument('--out-dir', type=str, default='results', help="Directory to save evaluation results")
    parser.add_argument('--num-frames', type=int, default=16, help="Number of frames to sample from each video")
    
    ### MODIFICATION START ###
    # 각 프레임을 타일로 나눌 최대 개수 인자 추가
    parser.add_argument('--max-tiles', type=int, default=1, help="Maximum number of tiles to split each frame into. 1 means no tiling.")
    ### MODIFICATION END ###

    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--num-beams', type=int, default=1)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--load-in-8bit', action='store_true')
    parser.add_argument('--load-in-4bit', action='store_true')
    args = parser.parse_args()

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir, exist_ok=True)
    
    assert args.batch_size == 1, 'Only batch size 1 is supported for this evaluation script.'

    torch.distributed.init_process_group(
        backend='nccl',
        world_size=int(os.getenv('WORLD_SIZE', '1')),
        rank=int(os.getenv('RANK', '0')),
    )
    torch.cuda.set_device(int(os.getenv('LOCAL_RANK', 0)))
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    model, tokenizer = load_model_and_tokenizer(args)
    image_size = model.config.force_image_size or model.config.vision_config.image_size
    
    print(f'[INFO] Model loaded from: {args.checkpoint}')
    print(f'[INFO] Image size: {image_size}')
    
    evaluate_video_classification()