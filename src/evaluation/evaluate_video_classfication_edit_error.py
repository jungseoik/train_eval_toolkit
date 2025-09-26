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
import torch
from transformers import AutoModel, AutoTokenizer
from PIL import Image
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from decord import VideoReader, cpu
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
import numpy as np

def build_transform(input_size: int = 448):
    """Return torchvision transform matching InternVL pre‑training."""
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
        img = dynamic_preprocess(img, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pixel_values = [transform(tile) for tile in img]
        pixel_values = torch.stack(pixel_values)
        num_patches_list.append(pixel_values.shape[0])
        pixel_values_list.append(pixel_values)
    pixel_values = torch.cat(pixel_values_list)
    return pixel_values, num_patches_list


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
    worker_globals['transform'] = build_transform(input_size=worker_globals['image_size'])

def process_video(item):
    """워커 프로세스가 실행할 단일 비디오 처리 로직"""
    video_path_log = item['video']
    worker_pid = os.getpid() # 현재 프로세스 ID
    
    # [로그 STEP 1]: 처리 시작
    print(f"\n[PID: {worker_pid}] [VIDEO: {video_path_log}] [STEP 1] ==> Process START", flush=True)

    model = worker_globals['model']
    tokenizer = worker_globals['tokenizer']
    args = worker_globals['args']
    transform = worker_globals['transform']
    local_rank = worker_globals['local_rank']

    video_path = os.path.join(args.video_root, item['video'])
    video_id = item.get('id', video_path)

    try:
        gt_value_str = item['conversations'][0]['value']
        ground_truth_category = json.loads(gt_value_str)['category']
    except Exception:
        ground_truth_category = "parsing_error"
    
    try:
        # [로그 STEP 2]: 비디오 로딩 시작
        print(f"[PID: {worker_pid}] [VIDEO: {video_path_log}] [STEP 2] -- Calling load_video...", flush=True)
        pixel_values, num_patches_list = load_video(
            video_path, 
            num_segments=args.num_frames, 
            input_size=worker_globals['image_size'],
            max_num=1
        )
        # [로그 STEP 3]: 비디오 로딩 성공
        print(f"[PID: {worker_pid}] [VIDEO: {video_path_log}] [STEP 3] -- load_video SUCCEEDED. Tensor shape: {pixel_values.shape}", flush=True)
        
        # [로그 STEP 4]: GPU로 데이터 이동 시작
        print(f"[PID: {worker_pid}] [VIDEO: {video_path_log}] [STEP 4] -- Moving tensor to CUDA device...", flush=True)
        pixel_values = pixel_values.to(f'cuda:{local_rank}', dtype=torch.bfloat16)
        # [로그 STEP 5]: GPU로 데이터 이동 성공
        print(f"[PID: {worker_pid}] [VIDEO: {video_path_log}] [STEP 5] -- Move to CUDA SUCCEEDED.", flush=True)
        
    except Exception as e:
        print(f"[PID: {worker_pid}] [VIDEO: {video_path_log}] [CRITICAL ERROR] during video loading: {e}", flush=True)
        return None

    # <<< CHANGED: 프롬프트를 동적으로 생성
    # 1. 처리된 프레임 수에 맞춰 <image> 토큰을 포함한 비디오 프리픽스를 만듭니다.
    video_prefix = ''.join([f'Frame{i+1}: <image>\n' for i in range(len(num_patches_list))])

    # 2. 기존의 상세한 지시사항(템플릿)을 정의합니다.
    prompt_template = (
        "Watch this short video clip and respond with exactly one JSON object.\n\n[Rules]\n- The category must be either 'violence' or 'normal'.  \n- Classify as violence if any of the following actions are present:  \n  * Punching  \n  * Kicking  \n  * Weapon Threat\n  * Weapon Attack\n  * Falling/Takedown  \n  * Pushing/Shoving  \n  * Brawling/Group Fight  \n- If none of the above are observed, classify as normal.  \n- The following cases must always be classified as normal:  \n  * Affection (hugging, holding hands, light touches)  \n  * Helping (supporting, assisting)  \n  * Accidental (unintentional bumping)  \n  * Playful (non-aggressive playful contact)  \n\n[Output Format]\n- Output exactly one JSON object.  \n- The object must contain only two keys: \"category\" and \"description\".  \n- The description should briefly and objectively describe the scene.  \n\nExample (violence):  \n{\"category\":\"violence\",\"description\":\"A man in a black jacket punches another man, who stumbles backward.\"}\n\nExample (normal):  \n{\"category\":\"normal\",\"description\":\"Two people are hugging inside an elevator"
    )

    # 3. 비디오 프리픽스와 템플릿을 합쳐 최종 질문을 완성합니다.
    question = video_prefix + prompt_template
    # <<< END OF CHANGES

    # [로그 STEP 6]: 모델 추론 시작
    print(f"[PID: {worker_pid}] [VIDEO: {video_path_log}] [STEP 6] -- Calling model.chat...", flush=True)
    pred = model.chat(
        tokenizer=tokenizer,
        pixel_values=pixel_values,
        question=question,
        generation_config=dict(
            num_beams=args.num_beams,
            max_new_tokens=50,
            min_new_tokens=5,
        ),
        verbose=False
    )
    # [로그 STEP 7]: 모델 추론 성공
    print(f"[PID: {worker_pid}] [VIDEO: {video_path_log}] [STEP 7] -- model.chat SUCCEEDED. Raw prediction: {pred[:50]}...", flush=True)

    predicted_category = parse_prediction(pred)

    # [로그 STEP 8]: 처리 완료
    print(f"[PID: {worker_pid}] [VIDEO: {video_path_log}] [STEP 8] ==> Process END", flush=True)

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

    # --- ▼▼▼▼▼ 여기에 코드 추가 ▼▼▼▼▼ ---
    # 디버깅을 위해 추가된 코드입니다.
    # 현재 실행 옵션(--nproc_per_node=1)에서는 rank 0이 모든 데이터를 처리하므로, 
    # 아래 코드는 정확하게 3330번 인덱스부터 시작하게 만듭니다.
    start_index = 3330 
    if rank == 0 and len(my_data) > start_index:
        original_count = len(my_data)
        my_data = my_data[start_index:]
        print(f"--- DEBUG MODE: Skipped to index {start_index}. "
              f"Processing {len(my_data)} items out of {original_count}. ---", flush=True)
    # --- ▲▲▲▲▲ 여기까지 추가 ▲▲▲▲▲ ---

    print(f"Rank {rank} (GPU {local_rank}) processing {len(my_data)} videos.")



    results_list = []
    ctx = mp.get_context('spawn')
    
    ## 수정파트
    # 1. Pool을 with 구문 밖에서 직접 생성합니다.
    pool = ctx.Pool(
        processes=args.workers_per_gpu, 
        initializer=init_worker, 
        initargs=(model, tokenizer, args, local_rank)
    )
    
    # 2. try...finally 구문으로 핵심 로직과 자원 해제를 감쌉니다.
    try:
        # 비디오 처리 로직은 try 블록 안에 위치시킵니다.
        results_iterator = pool.imap_unordered(process_video, my_data)
        for result in tqdm(results_iterator, total=len(my_data), desc=f"Rank {rank} Progress"):
            if result is not None:
                results_list.append(result)
    finally:
        # 3. 이 부분은 try 블록에서 에러가 발생하더라도 '항상' 실행됩니다.
        #    따라서 좀비 프로세스가 남는 것을 방지할 수 있습니다.
        print(f"Rank {rank} is finalizing the worker pool...")
        pool.close()
        pool.join()

    # 4. 모든 워커 프로세스가 100% 안전하게 종료된 후에 분산 동기화를 시작합니다.
    print(f"Rank {rank} has finished its jobs and is waiting at the barrier.")
    torch.distributed.barrier()

    world_size = torch.distributed.get_world_size()
    merged_results = [None for _ in range(world_size)]
    torch.distributed.all_gather_object(merged_results, results_list)
    #### 수정파트
    # with ctx.Pool(
    #     processes=args.workers_per_gpu, 
    #     initializer=init_worker, 
    #     # MODIFIED: init_worker에 local_rank 전달
    #     initargs=(model, tokenizer, args, local_rank)
    # ) as pool:
    #     for result in tqdm(pool.imap_unordered(process_video, my_data), total=len(my_data), desc=f"Rank {rank} Progress"):
    #         if result is not None:
    #             results_list.append(result)

    # torch.distributed.barrier()

    # world_size = torch.distributed.get_world_size()
    # merged_results = [None for _ in range(world_size)]
    # torch.distributed.all_gather_object(merged_results, results_list)
    
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

import datetime # 파일 상단에 추가
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
    
    # 기본 타임아웃은 30분입니다. 이것을 2시간 등으로 늘려줍니다.
    timeout_delta = datetime.timedelta(hours=2)


    torch.distributed.init_process_group(
        backend='nccl',
        world_size=int(os.getenv('WORLD_SIZE', '1')),
        rank=int(os.getenv('RANK', '0')),
        timeout=timeout_delta
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