import argparse
import itertools
import json
import os
import random
import time

import torch
import torch.multiprocessing as mp
from decord import VideoReader
from PIL import Image
from tqdm import tqdm
import numpy as np
import re
from sklearn.metrics import f1_score, precision_score, recall_score
from src.training.internvl.model.internvl_chat import InternVLChatConfig, InternVLChatModel
from transformers import AutoTokenizer
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CLIP_MEAN = (0.4814546, 0.4578275, 0.40821073)
CLIP_STD = (0.2686295, 0.2613025, 0.2757711)

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
        elif sample == 'uniform':
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
    # MODIFIED: Move to device is handled by the main process
    # if not args.load_in_8bit and not args.load_in_4bit:
    #     model = model.cuda()
    return model, tokenizer

# 전역 변수 및 워커 초기화 함수
worker_globals = {}

# MODIFIED: local_rank를 인자로 받도록 수정
def init_worker(model, tokenizer, args, local_rank):
    """각 워커 프로세스를 초기화하는 함수"""
    # ADDED: 워커의 GPU 장치를 명시적으로 설정!
    torch.cuda.set_device(local_rank)
    
    worker_globals['model'] = model
    worker_globals['tokenizer'] = tokenizer
    worker_globals['args'] = args
    worker_globals['local_rank'] = local_rank # 나중에 사용하기 위해 저장
    worker_globals['prompt'] = (
        "Watch this short video clip and respond with exactly one JSON object.\n\n[Rules]\n- The category must be either 'violence' or 'normal'.  \n- Classify as violence if any of the following actions are present:  \n  * Punching  \n  * Kicking  \n  * Weapon Threat\n  * Weapon Attack\n  * Falling/Takedown  \n  * Pushing/Shoving  \n  * Brawling/Group Fight  \n- If none of the above are observed, classify as normal.  \n- The following cases must always be classified as normal:  \n  * Affection (hugging, holding hands, light touches)  \n  * Helping (supporting, assisting)  \n  * Accidental (unintentional bumping)  \n  * Playful (non-aggressive playful contact)  \n\n[Output Format]\n- Output exactly one JSON object.  \n- The object must contain only two keys: \"category\" and \"description\".  \n- The description should briefly and objectively describe the scene.  \n\nExample (violence):  \n{\"category\":\"violence\",\"description\":\"A man in a black jacket punches another man, who stumbles backward.\"}\n\nExample (normal):  \n{\"category\":\"normal\",\"description\":\"Two people are hugging inside an elevator"
    )
    worker_globals['image_size'] = model.config.force_image_size or model.config.vision_config.image_size
    worker_globals['transform'] = build_transform(is_train=False, input_size=worker_globals['image_size'])

# 단일 비디오 처리 워커 함수
def process_video(item):
    """워커 프로세스가 실행할 단일 비디오 처리 로직"""
    model = worker_globals['model']
    tokenizer = worker_globals['tokenizer']
    args = worker_globals['args']
    prompt = worker_globals['prompt']
    transform = worker_globals['transform']
    local_rank = worker_globals['local_rank'] # 저장된 local_rank 사용

    video_path = os.path.join(args.video_root, item['video'])
    video_id = item.get('id', video_path)
    video_path_log = item['video']

    try:
        gt_value_str = item['conversations'][0]['value']
        ground_truth_category = json.loads(gt_value_str)['category']
    except Exception:
        ground_truth_category = "parsing_error"

    try:    
        frames = read_frames_decord(video_path, num_frames=args.num_frames, sample='uniform')
        pixel_values = torch.stack([transform(frame) for frame in frames])
        # MODIFIED: 명시적인 장치와 데이터 타입으로 텐서를 이동
        pixel_values = pixel_values.to(f'cuda:{local_rank}', dtype=torch.bfloat16)
    except Exception as e:
        print(f"Error processing video {video_path}: {e}")
        return None

    pred = model.chat(
        tokenizer=tokenizer,
        pixel_values=pixel_values,
        question=prompt,
        generation_config=dict(
            num_beams=args.num_beams,
            max_new_tokens=50,
            min_new_tokens=5,
        ),
        verbose=False
    )

    predicted_category = parse_prediction(pred)

    return {
        'video_path_log' : video_path_log,
        'video_id': video_id,
        'prediction': predicted_category,
        'ground_truth': ground_truth_category,
        'is_correct': predicted_category == ground_truth_category,
        'pred_result' : pred
    }

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
    """
    모델의 출력 문자열에서 다양한 케이스를 고려하여 'category' 값을 파싱합니다.
    """
    try:
        # 1단계: 입력 문자열 정규화 (마크다운 및 공백 제거)
        clean_str = pred_str
        if '```json' in clean_str:
            clean_str = clean_str.split('```json')[1].split('```')[0]
        elif '```' in clean_str:
            clean_str = clean_str.split('```')[1].split('```')[0]
        clean_str = clean_str.strip()

        # 2단계: 완전한 JSON 객체 파싱 시도
        # 문자열에서 첫 '{'와 마지막 '}'를 찾아 JSON 부분만 추출
        start_brace = clean_str.find('{')
        end_brace = clean_str.rfind('}')
        if start_brace != -1 and end_brace != -1 and start_brace < end_brace:
            json_part = clean_str[start_brace : end_brace + 1]
            try:
                data = json.loads(json_part)
                category = data.get('category')
                if category in ['violence', 'normal']:
                    return category
            except json.JSONDecodeError:
                # JSON 파싱에 실패하면 다음 단계로 넘어감
                pass

        # 3단계: 정규표현식을 이용한 키-값 직접 추출 (Fallback)
        # "category" : "value" 패턴을 직접 찾음 (따옴표 종류, 공백 유연하게 처리)
        # 예: "category":"normal", 'category' : 'violence' 등
        cat_match = re.search(r'["\']category["\']\s*:\s*["\'](violence|normal)["\']', clean_str)
        if cat_match:
            return cat_match.group(1)  # "violence" 또는 "normal" 반환

        # 4단계: 모든 시도가 실패한 경우
        return 'no_json_found'

    except Exception:
        # 함수 실행 중 예상치 못한 에러 발생 시
        return 'parsing_failed'

# MODIFIED: local_rank, model, tokenizer, args를 인자로 받도록 수정
def evaluate_video_classification_parallel(local_rank, model, tokenizer, args):
    all_data = []
    with open(args.annotation, 'r') as f:
        for line in f:
            all_data.append(json.loads(line))

    sampler = InferenceSampler(len(all_data))
    my_data_indices = list(sampler)
    my_data = [all_data[i] for i in my_data_indices]
    
    rank = torch.distributed.get_rank()
    print(f"Rank {rank} (GPU {local_rank}) processing {len(my_data)} videos.")

    results_list = []
    ctx = mp.get_context('spawn')
    with ctx.Pool(
        processes=args.workers_per_gpu, 
        initializer=init_worker, 
        # MODIFIED: init_worker에 local_rank 전달
        initargs=(model, tokenizer, args, local_rank)
    ) as pool:
        for result in tqdm(pool.imap_unordered(process_video, my_data), total=len(my_data), desc=f"Rank {rank} Progress"):
            if result is not None:
                results_list.append(result)

    torch.distributed.barrier()

    world_size = torch.distributed.get_world_size()
    merged_results = [None for _ in range(world_size)]
    torch.distributed.all_gather_object(merged_results, results_list)
    
    if rank == 0:
        merged_results = list(itertools.chain.from_iterable(merged_results))
        total_samples = len(merged_results)

        y_true = []
        y_pred = []
        valid_gt_labels = {'violence', 'normal'}
        
        for r in merged_results:
            # 정답 레이블이 유효한 경우에만 평가 데이터에 포함시킵니다.
            if r['ground_truth'] in valid_gt_labels:
                y_true.append(r['ground_truth'])
                
                # 새로운 규칙 적용: 예측이 'normal'이 아니면 모두 'violence'로 간주
                if r['prediction'] == 'normal':
                    y_pred.append('normal')
                else:
                    y_pred.append('violence')
        
        # 2. 유효한 데이터 기준으로 각종 지표를 다시 계산합니다.
        valid_samples = len(y_true)
        invalid_gt_samples = total_samples - valid_samples

        correct_predictions = sum(1 for gt, pred in zip(y_true, y_pred) if gt == pred)
        accuracy = (correct_predictions / valid_samples) * 100 if valid_samples > 0 else 0.0
        
        precision = precision_score(y_true, y_pred, pos_label='violence', zero_division=0)
        recall = recall_score(y_true, y_pred, pos_label='violence', zero_division=0)
        f1 = f1_score(y_true, y_pred, pos_label='violence', zero_division=0)
        
        print("\n--- Evaluation Summary ---")
        print(f"GPUs used: {world_size}")
        print(f"Workers per GPU: {args.workers_per_gpu}")
        print(f"Total Videos Submitted: {total_samples}")
        print(f"Invalid Ground-Truth Samples (excluded): {invalid_gt_samples}")
        print(f"Valid Videos Evaluated: {valid_samples}")
        print("------------------------------------")
        print(f"Accuracy: {accuracy:.2f}% ({correct_predictions}/{valid_samples})")
        print(f"Precision (violence): {precision:.4f}")
        print(f"Recall (violence):    {recall:.4f}")
        print(f"F1-Score (violence):  {f1:.4f}")
        print("------------------------------------\n")
        
        # 4. 파일 저장 로직은 그대로 유지합니다.
        time_prefix = time.strftime('%Y_%m_%d', time.localtime())
        model_name = os.path.basename(args.checkpoint)
        dataset_name = os.path.splitext(os.path.basename(args.annotation))[0]
        new_filename = f"{time_prefix}_{model_name}_{dataset_name}.json"
        results_file_path = os.path.join(args.out_dir, new_filename)

        with open(results_file_path, 'w') as f:
            json.dump(merged_results, f, indent=4) # 전체 원본 결과는 그대로 저장
        print(f"Detailed results saved to: {results_file_path}")

        # 5. 요약 파일에도 새로운 정보를 추가하여 저장합니다.
        summary_path = os.path.join(args.out_dir, 'evaluation_summary.txt')
        with open(summary_path, 'a') as f:
            f.write(f"Checkpoint: {args.checkpoint}\n")
            f.write(f"Annotation: {args.annotation}\n")
            f.write(f"Total Samples: {total_samples} (Valid: {valid_samples}, Invalid GT: {invalid_gt_samples})\n")
            f.write(f"Accuracy: {accuracy:.2f}%\n")
            f.write(f"Precision: {precision:.4f}\n")
            f.write(f"Recall: {recall:.4f}\n")
            f.write(f"F1-Score: {f1:.4f}\n")
            f.write("-" * 20 + "\n")
        print(f"Summary saved to: {summary_path}")
    torch.distributed.barrier()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--annotation', type=str, required=True)
    parser.add_argument('--video-root', type=str, default='')
    parser.add_argument('--out-dir', type=str, default='results/eval_result')
    parser.add_argument('--num-frames', type=int, default=16)
    parser.add_argument('--workers-per-gpu', type=int, default=4, help="Number of parallel processes per GPU.")
    parser.add_argument('--num-beams', type=int, default=1)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--load-in-8bit', action='store_true')
    parser.add_argument('--load-in-4bit', action='store_true')
    args = parser.parse_args()
    
    timestamp = time.strftime('%Y_%m_%d_%H-%M-%S')
    new_out_dir = os.path.join(args.out_dir, timestamp)
    args.out_dir = new_out_dir
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir, exist_ok=True)
    
    torch.distributed.init_process_group(
        backend='nccl',
        world_size=int(os.getenv('WORLD_SIZE', '1')),
        rank=int(os.getenv('RANK', '0')),
    )
    
    # MODIFIED: local_rank를 변수로 저장
    local_rank = int(os.getenv('LOCAL_RANK', 0))
    torch.cuda.set_device(local_rank)
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    model, tokenizer = load_model_and_tokenizer(args)
    # MODIFIED: Move model to the correct device after loading
    if not args.load_in_8bit and not args.load_in_4bit:
        model = model.to(local_rank)

    print(f'[INFO] Rank {torch.distributed.get_rank()} (GPU {local_rank}) loaded model from: {args.checkpoint}')
    
    # MODIFIED: local_rank와 로드된 객체들을 평가 함수에 인자로 전달
    evaluate_video_classification_parallel(local_rank, model, tokenizer, args)