import os, json, math, time, re
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, AutoModel, AutoConfig

# -----------------------
# 재사용: 이미지 전처리/프롬프트/파서 (변경 없음)
# -----------------------
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

PROMPT_TEMPLATE = """<video>
Watch this short video clip (1–2 seconds) and respond with exactly one JSON object.

[Rules]
- The category must be either 'violence' or 'normal'.
- Classify as violence if any of the following actions are present:
  * Punching
  * Kicking
  * Weapon Threat
  * Weapon Attack
  * Falling/Takedown
  * Pushing/Shoving
  * Brawling/Group Fight
- If none of the above are observed, classify as normal.
- The following cases must always be classified as normal:
  * Affection (hugging, holding hands, light touches)
  * Helping (supporting, assisting)
  * Accidental (unintentional bumping)
  * Playful (non-aggressive playful contact)

[Output Format]
- Output exactly one JSON object.
- The object must contain only two keys: "category" and "description".
- The description should briefly and objectively describe the scene.

Example (violence):
{"category":"violence","description":"A man in a black jacket punches another man, who stumbles backward."}

Example (normal):
{"category":"normal","description":"Two people are hugging inside an elevator"}
"""

import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

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

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=True):
    ow, oh = image.size
    aspect_ratio = ow / oh
    target_ratios = sorted(
        {(i, j) for n in range(min_num, max_num + 1)
         for i in range(1, n + 1) for j in range(1, n + 1)
         if min_num <= i * j <= max_num},
        key=lambda x: x[0] * x[1],
    )
    ratio = find_closest_aspect_ratio(aspect_ratio, target_ratios, ow, oh, image_size)
    tw, th = image_size * ratio[0], image_size * ratio[1]
    blocks = ratio[0] * ratio[1]
    resized = image.resize((tw, th))
    tiles = []
    for idx in range(blocks):
        x = (idx % (tw // image_size)) * image_size
        y = (idx // (tw // image_size)) * image_size
        tiles.append(resized.crop((x, y, x + image_size, y + image_size)))
    if use_thumbnail and blocks != 1:
        tiles.append(image.resize((image_size, image_size)))
    return tiles

def parse_prediction(pred_str: str) -> str:
    try:
        clean = pred_str
        if '```json' in clean:
            clean = clean.split('```json')[1].split('```')[0]
        elif '```' in clean:
            clean = clean.split('```')[1].split('```')[0]
        clean = clean.strip()
        s, e = clean.find('{'), clean.rfind('}')
        if s != -1 and e != -1 and s < e:
            json_part = clean[s:e+1]
            try:
                data = json.loads(json_part)
                cat = data.get('category')
                if cat in ['violence', 'normal']:
                    return cat
            except json.JSONDecodeError:
                pass
        m = re.search(r'["\']category["\']\s*:\s*["\'](violence|normal)["\']', clean)
        if m:
            return m.group(1)
        return 'no_json_found'
    except Exception:
        return 'parsing_failed'

# -----------------------
# 모델 로딩 (변경 없음)
# -----------------------
def split_model(model_path):
    world_size = torch.cuda.device_count()
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    num_layers = config.llm_config.num_hidden_layers
    num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
    num_layers_per_gpu = [num_layers_per_gpu] * world_size
    num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.5)
    device_map, layer_cnt = {}, 0
    for i, num_layer in enumerate(num_layers_per_gpu):
        for _ in range(num_layer):
            device_map[f'language_model.model.layers.{layer_cnt}'] = i
            layer_cnt += 1
    device_map.update({
        'vision_model': 0, 'mlp1': 0,
        'language_model.model.tok_embeddings': 0,
        'language_model.model.embed_tokens': 0,
        'language_model.output': 0,
        'language_model.model.norm': 0,
        'language_model.model.rotary_emb': 0,
        'language_model.lm_head': 0,
        f'language_model.model.layers.{num_layers - 1}': 0
    })
    return device_map

def load_model_and_tokenizer(checkpoint: str, multi_gpu: bool):
    print(f"Loading model from: {checkpoint}")
    if multi_gpu:
        device_map = split_model(checkpoint)
        model = AutoModel.from_pretrained(
            checkpoint, torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True, trust_remote_code=True,
            device_map=device_map
        ).eval()
    else:
        model = AutoModel.from_pretrained(
            checkpoint, torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True, trust_remote_code=True
        ).eval().cuda()
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True, use_fast=False)
    image_size = model.config.force_image_size or model.config.vision_config.image_size
    return model, tokenizer, image_size

# -----------------------
# 비디오 처리 유틸 (변경 없음)
# -----------------------
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".m4v"}

def list_videos_recursive(root_dir: str) -> List[Path]:
    root = Path(root_dir)
    files = [p for p in root.rglob("*") if p.suffix.lower() in VIDEO_EXTS]
    return sorted(files)

def ensure_parent_dirs(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)

def pil_from_frame(frame_rgb: np.ndarray) -> Image.Image:
    return Image.fromarray(frame_rgb)

def frames_to_pixel_values(frames_rgb: List[np.ndarray], input_size: int) -> Tuple[torch.Tensor, List[int]]:
    """한 '윈도우'의 프레임들을 전처리하여 타일 텐서와 각 프레임별 타일 개수 리스트를 반환"""
    transform = build_transform(input_size)
    all_tensors, num_patches_list = [], []
    for fr in frames_rgb:
        img = pil_from_frame(fr)
        tiles = dynamic_preprocess(img, image_size=input_size, use_thumbnail=True, max_num=12)
        px = [transform(t) for t in tiles]
        px = torch.stack(px)
        all_tensors.append(px)
        num_patches_list.append(px.size(0)) # 각 '프레임'의 타일 수
    cat = torch.cat(all_tensors, dim=0) # 한 윈도우의 모든 타일을 이어붙임
    return cat, num_patches_list

def overlay_text_top_center(frame_bgr: np.ndarray, text: str) -> np.ndarray:
    h, w, _ = frame_bgr.shape
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.6, w / 1280.0 * 0.8)
    thickness = 2
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    x = (w - tw) // 2
    y = max(10 + th, 10 + th)
    pad = 10
    x1, y1 = max(0, x - pad), max(0, y - th - pad)
    x2, y2 = min(w, x + tw + pad), min(h, y + pad)
    overlay = frame_bgr.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 0), -1)
    alpha = 0.45
    cv2.addWeighted(overlay, alpha, frame_bgr, 1 - alpha, 0, frame_bgr)
    color = (255, 255, 255) if text == "normal" else (0, 180, 255)
    cv2.putText(frame_bgr, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)
    return frame_bgr

# -----------------------
# <<< NEW: 핵심 로직 수정 >>>
# -----------------------
def process_video(
    video_path: Path,
    input_root: Path,
    output_root: Path,
    window_size: int,
    batch_size: int,
    model,
    tokenizer,
    image_size: int,
):
    """
    한 개 비디오를 배치 단위로 추론하고 결과를 오버레이하여 저장.
    1. 비디오를 `window_size` 크기의 여러 윈도우로 분할.
    2. 윈도우들을 `batch_size` 만큼 묶어 `model.batch_chat`으로 일괄 추론.
    3. 추론된 레이블을 각 윈도우의 모든 프레임에 오버레이하여 새 비디오 파일로 저장.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"⚠️ Cannot open video: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    rel = video_path.relative_to(input_root)
    out_path = (output_root / rel).with_suffix(".mp4")
    ensure_parent_dirs(out_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))

    def read_window_frames_rgb(start_idx: int, end_idx: int) -> List[np.ndarray]:
        frames = []
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_idx)
        for _ in range(start_idx, end_idx):
            ok, frame_bgr = cap.read()
            if not ok or frame_bgr is None: break
            frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        return frames

    # 1. 전체 비디오에 대한 윈도우 경계 계산
    window_ranges = []
    for s in range(0, total_frames, window_size):
        e = min(s + window_size, total_frames)
        if s < e: window_ranges.append((s, e))

    # 2. 모든 윈도우에 대한 카테고리를 배치 추론으로 채우기
    categories_per_window = ["normal"] * len(window_ranges)
    device = next(model.parameters()).device
    
    # tqdm을 배치 루프에 적용
    pbar_desc = f"Batch infer for {rel.as_posix()}"
    batch_iter = range(0, len(window_ranges), batch_size)
    for i in tqdm(batch_iter, desc=pbar_desc):
        batch_window_ranges = window_ranges[i:i + batch_size]
        
        # 현재 배치에서 추론할 데이터 준비
        batch_pixel_values_list = []
        batch_num_patches_list = [] # `batch_chat`에 전달될 최종 리스트
        batch_questions = []
        valid_indices_in_batch = [] # 실제 추론이 수행된 윈도우의 원본 인덱스

        for j, (ws, we) in enumerate(batch_window_ranges):
            frames_rgb = read_window_frames_rgb(ws, we)
            if not frames_rgb:
                continue

            # 한 윈도우의 모든 프레임을 하나의 샘플로 간주하고 전처리
            pixel_values_one_window, num_patches_per_frame = frames_to_pixel_values(frames_rgb, image_size)
            
            batch_pixel_values_list.append(pixel_values_one_window)
            batch_num_patches_list.append(pixel_values_one_window.size(0)) # 윈도우 전체의 타일 개수
            
            # 프롬프트 생성
            video_prefix = ''.join([f'Frame{k+1}: <image>\n' for k in range(len(frames_rgb))])
            question = video_prefix + PROMPT_TEMPLATE
            batch_questions.append(question)
            
            valid_indices_in_batch.append(i + j)

        if not valid_indices_in_batch:
            continue

        # 배치 추론 수행
        final_pixel_values = torch.cat(batch_pixel_values_list, dim=0).to(device).to(torch.bfloat16)

        responses = model.batch_chat(
            tokenizer,
            pixel_values=final_pixel_values,
            num_patches_list=batch_num_patches_list,
            questions=batch_questions,
            generation_config=dict(num_beams=1, max_new_tokens=15, min_new_tokens=5),
        )

        # 결과 파싱 및 할당
        for original_idx, res in zip(valid_indices_in_batch, responses):
            cat = parse_prediction(res)
            categories_per_window[original_idx] = cat if cat in ("violence", "normal") else "normal"

    # 3. 추론 결과를 사용하여 비디오 프레임에 오버레이 및 저장
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    with tqdm(total=total_frames, desc=f"Writing {rel.as_posix()}") as pbar:
        for fidx in range(total_frames):
            ok, frame_bgr = cap.read()
            if not ok or frame_bgr is None: break
            
            win_id = fidx // window_size
            if 0 <= win_id < len(categories_per_window):
                label = categories_per_window[win_id]
            else:
                label = "normal" # 혹시 모를 인덱스 오류 방지

            frame_bgr = overlay_text_top_center(frame_bgr, label)
            writer.write(frame_bgr)
            pbar.update(1)

    writer.release()
    cap.release()

# -----------------------
# 파이프라인 실행 함수 (변경 없음)
# -----------------------
def run_pipeline(
    input_root: str,
    output_root: str,
    checkpoint: str,
    window_size: int = 15,
    batch_size: int = 20,
    multi_gpu: bool = False,
):
    input_root = Path(input_root).resolve()
    output_root = Path(output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    model, tokenizer, image_size = load_model_and_tokenizer(checkpoint, multi_gpu)

    videos = list_videos_recursive(str(input_root))
    if not videos:
        print(f"❌ No videos found under {input_root}")
        return
    print(f"Found {len(videos)} videos under {input_root}")

    for v in tqdm(videos, desc="Processing videos"):
        process_video(
            video_path=v,
            input_root=input_root,
            output_root=output_root,
            window_size=window_size,
            batch_size=batch_size,
            model=model,
            tokenizer=tokenizer,
            image_size=image_size,
        )

# -----------------------
# 메인 실행 블록 (변경 없음)
# -----------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=str, required=True, help="재귀적으로 탐색할 입력 비디오 루트 경로")
    parser.add_argument("--output-root", type=str, required=True, help="동일 폴더 구조로 저장할 출력 루트 경로")
    parser.add_argument("--checkpoint", type=str, required=True, help="InternVL 체크포인트 경로")
    parser.add_argument("--window-size", type=int, default=15, help="한 번에 추론할 비디오 프레임 묶음(윈도우) 크기")
    parser.add_argument("--batch-size", type=int, default=20, help="한 번에 추론할 윈도우의 수 (배치 크기)")
    parser.add_argument("--multi-gpu", action="store_true", help="멀티 GPU 분산 로딩 사용")
    args = parser.parse_args()

    run_pipeline(
        input_root=args.input_root,
        output_root=args.output_root,
        checkpoint=args.checkpoint,
        window_size=args.window_size,
        batch_size=args.batch_size,
        multi_gpu=args.multi_gpu,
    )