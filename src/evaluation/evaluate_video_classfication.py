import argparse
import itertools
import json
import os
import random
import time
from functools import partial

import torch
from decord import VideoReader
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
CLIP_MEAN = (0.4814546, 0.4578275, 0.40821073)
CLIP_STD = (0.2686295, 0.2613025, 0.2757711)
SIGLIP_MEAN = (0.5, 0.5, 0.5)
SIGLIP_STD = (0.5, 0.5, 0.5)

def get_frame_indices(num_frames, vlen, sample='rand', fix_start=None, input_fps=1, max_num_frames=-1):
    if sample in ['rand', 'uniform']:
        acc_samples = min(num_frames, vlen)
        intervals = np.linspace(start=0, stop=vlen, num=acc_samples + 1).astype(int)
        ranges = []
        for idx, interv in enumerate(intervals[:-1]):
            ranges.append((interv, intervals[idx + 1] - 1))
        if sample == 'rand':
            try:
                frame_indices = [random.choice(range(x[0], x[1])) for x in ranges]
            except:
                frame_indices = np.random.permutation(vlen)[:acc_samples]
                frame_indices.sort()
                frame_indices = list(frame_indices)
        elif sample == 'uniform': # 중앙 프레임 추출로 변경
            frame_indices = [(x[0] + x[1]) // 2 for x in ranges]
        else:
            raise NotImplementedError

        if len(frame_indices) < num_frames:
            padded_frame_indices = [frame_indices[-1]] * num_frames
            padded_frame_indices[:len(frame_indices)] = frame_indices
            frame_indices = padded_frame_indices
    else:
        raise ValueError
    return frame_indices

def read_frames_decord(video_path, num_frames, sample='rand'):
    video_reader = VideoReader(video_path, num_threads=1)
    vlen = len(video_reader)
    frame_indices = get_frame_indices(num_frames, vlen, sample=sample)
    frames = video_reader.get_batch(frame_indices).asnumpy()
    frames = [Image.fromarray(frames[i]) for i in range(frames.shape[0])]
    return frames

def build_transform(is_train, input_size, normalize_type='imagenet'):
    if normalize_type == 'imagenet':
        MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    elif normalize_type == 'clip':
        MEAN, STD = CLIP_MEAN, CLIP_STD
    else:
        raise NotImplementedError
    
    # 평가 시에는 데이터 증강(augmentation)을 사용하지 않음
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

# --- 모델 로딩 (제공된 코드) ---
# split_model 함수는 제공되지 않아, `auto` 옵션을 사용하지 않는다고 가정하고
# 해당 부분을 제외했습니다. 필요 시 `split_model` 함수를 추가해야 합니다.
def load_model_and_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, trust_remote_code=True, use_fast=False)
    model = InternVLChatModel.from_pretrained(
        args.checkpoint, low_cpu_mem_usage=True, torch_dtype=torch.bfloat16,
        load_in_8bit=args.load_in_8bit, load_in_4bit=args.load_in_4bit).eval()
    if not args.load_in_8bit and not args.load_in_4bit:
        model = model.cuda()
    return model, tokenizer

# --- 데이터셋 및 평가 로직 (수정된 부분) ---

class VideoClassificationDataset(torch.utils.data.Dataset):
    def __init__(self, annotation_path, num_frames=16, input_size=224, video_root=''):
        self.data = []
        with open(annotation_path, 'r') as f:
            for line in f:
                self.data.append(json.loads(line))
        
        self.num_frames = num_frames
        self.video_root = video_root
        self.transform = build_transform(is_train=False, input_size=input_size)
        print(f"Found {len(self.data)} videos in {annotation_path}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        video_path = os.path.join(self.video_root, item['video'])
        video_id = item.get('id', video_path)

        # 정답 카테고리 파싱
        try:
            gt_value_str = item['conversations'][0]['value']
            ground_truth_category = json.loads(gt_value_str)['category']
        except (json.JSONDecodeError, KeyError, IndexError) as e:
            print(f"Error parsing ground truth for video_id {video_id}: {e}")
            ground_truth_category = "parsing_error" # 에러 처리

        # 비디오 프레임 샘플링 및 전처리
        try:
            frames = read_frames_decord(video_path, num_frames=self.num_frames, sample='uniform')
            pixel_values = torch.stack([self.transform(frame) for frame in frames])
        except Exception as e:
            print(f"Error processing video {video_path}: {e}")
            # 에러 발생 시 더미 데이터 반환
            pixel_values = torch.zeros((self.num_frames, 3, self.transform.transforms[1].size[0], self.transform.transforms[1].size[1]))

        return {
            'video_id': video_id,
            'pixel_values': pixel_values,
            'ground_truth': ground_truth_category,
        }

def collate_fn(inputs):
    pixel_values = torch.stack([_['pixel_values'] for _ in inputs])
    video_ids = [_['video_id'] for _ in inputs]
    ground_truths = [_['ground_truth'] for _ in inputs]
    return pixel_values, video_ids, ground_truths


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
        # 출력에 포함될 수 있는 ```json ... ``` 같은 마크다운 제거
        if '```json' in pred_str:
            pred_str = pred_str.split('```json')[1].split('```')[0]
        elif '```' in pred_str:
            pred_str = pred_str.split('```')[1].split('```')[0]
            
        # 가장 깊은 레벨의 중괄호 {} 사이의 내용을 찾습니다.
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
    # 모델에 비디오에 대한 분류를 요청하는 프롬프트
    prompt = (
        "Watch this short video clip (1–2 seconds) and respond with exactly one JSON object.\n\n[Rules]\n- The category must be either 'violence' or 'normal'.  \n- Classify as violence if any of the following actions are present:  \n  * Punching  \n  * Kicking  \n  * Weapon Threat  \n  * Falling/Takedown  \n  * Pushing/Shoving  \n  * Brawling/Group Fight  \n- If none of the above are observed, classify as normal.  \n- The following cases must always be classified as normal:  \n  * Affection (hugging, holding hands, light touches)  \n  * Helping (supporting, assisting)  \n  * Accidental (unintentional bumping)  \n  * Playful (non-aggressive playful contact)  \n  * Sports (contact within sports rules)  \n\n[Output Format]\n- Output exactly one JSON object.  \n- The object must contain only two keys: \"category\" and \"description\".  \n- The description should briefly and objectively describe the scene.  \n\nExample (violence):  \n{\"category\":\"violence\",\"description\":\"A man in a black jacket punches another man, who stumbles backward.\"}\n\nExample (normal):  \n{\"category\":\"normal\",\"description\":\"Two people are hugging inside an elevator."
    )
    print('Prompt:', prompt)

    dataset = VideoClassificationDataset(
        annotation_path=args.annotation,
        num_frames=args.num_frames,
        input_size=image_size,
        video_root=args.video_root,
    )
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
    for pixel_values, video_ids, ground_truths in tqdm(dataloader):
        pixel_values = pixel_values.to(torch.bfloat16).cuda()
        
        # 모델 추론
        # 배치 사이즈가 1이므로, 첫 번째 아이템만 사용
        pred = model.chat(
            tokenizer=tokenizer,
            pixel_values=pixel_values[0], # (B, T, C, H, W) -> (T, C, H, W)
            question=prompt,
            generation_config=dict(
                num_beams=args.num_beams,
                max_new_tokens=50, # JSON 응답은 짧으므로 길이를 줄임
                min_new_tokens=5,
            ),
            verbose=False # 로그 출력을 줄임
        )
        
        predicted_category = parse_prediction(pred)
        
        results_list.append({
            'video_id': video_ids[0],
            'prediction': predicted_category,
            'ground_truth': ground_truths[0],
            'is_correct': predicted_category == ground_truths[0]
        })

    torch.distributed.barrier()

    # 모든 GPU의 결과 취합
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

        # 결과 파일 저장
        time_prefix = time.strftime('%y%m%d%H%M%S', time.localtime())
        results_file_path = os.path.join(args.out_dir, f'classification_results_{time_prefix}.json')
        with open(results_file_path, 'w') as f:
            json.dump(merged_results, f, indent=4)
        print(f"Detailed results saved to: {results_file_path}")

        # 요약 파일 저장
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
    
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--num-beams', type=int, default=1)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--load-in-8bit', action='store_true')
    parser.add_argument('--load-in-4bit', action='store_true')
    args = parser.parse_args()

    # 결과 디렉토리 생성
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir, exist_ok=True)
    
    # 배치 사이즈는 1로 고정 (model.chat이 배치 처리를 지원하지 않을 수 있음)
    assert args.batch_size == 1, 'Only batch size 1 is supported for this evaluation script.'

    # 분산 환경 설정
    torch.distributed.init_process_group(
        backend='nccl',
        world_size=int(os.getenv('WORLD_SIZE', '1')),
        rank=int(os.getenv('RANK', '0')),
    )
    torch.cuda.set_device(int(os.getenv('LOCAL_RANK', 0)))
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # 모델 및 토크나이저 로드
    model, tokenizer = load_model_and_tokenizer(args)
    image_size = model.config.force_image_size or model.config.vision_config.image_size
    
    print(f'[INFO] Model loaded from: {args.checkpoint}')
    print(f'[INFO] Image size: {image_size}')
    
    # 평가 실행
    evaluate_video_classification()