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
# 재사용: 이미지 전처리/프롬프트/파서
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
# 모델 로딩 (기존 함수 형태 유지)
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
# 비디오 처리 유틸
# -----------------------

# --- NEW: multi-image(window) inference --------------------------------------
def infer_category_for_window(model, tokenizer, image_size, frames_rgb: List[np.ndarray]) -> str:
    """
    한 윈도우의 연속 프레임들을 <image> 시퀀스로 묶어 단일 질의로 추론.
    - frames_rgb: RGB ndarray 리스트 (윈도우 내 모든 프레임)
    - 반환: "violence" 또는 "normal" (파싱 실패 시 "normal")
    """
    # 기존 유틸 재사용: 여러 프레임 -> (concat pixel_values, num_patches_list)
    pixel_values_tensor, num_patches_list = frames_to_pixel_values(frames_rgb, image_size)

    device = next(model.parameters()).device
    pixel_values_tensor = pixel_values_tensor.to(device).to(torch.bfloat16)

    # Frame1..N 프리픽스 + 기존 PROMPT_TEMPLATE 사용
    video_prefix = ''.join([f'Frame{i+1}: <image>\n' for i in range(len(num_patches_list))])
    question = video_prefix + PROMPT_TEMPLATE

    # 단일 대화 호출 (히스토리 없음)
    response = model.chat(
        tokenizer,
        pixel_values=pixel_values_tensor,
        question=question,
        generation_config=dict(num_beams=1, max_new_tokens=15, min_new_tokens=5),
        num_patches_list=num_patches_list,
        history=None,
        return_history=False
    )

    cat = parse_prediction(response)
    return cat if cat in ("violence", "normal") else "normal"

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".m4v"}

def list_videos_recursive(root_dir: str) -> List[Path]:
    root = Path(root_dir)
    files = [p for p in root.rglob("*") if p.suffix.lower() in VIDEO_EXTS]
    return sorted(files)

def ensure_parent_dirs(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)

def pil_from_frame(frame_rgb: np.ndarray) -> Image.Image:
    # decord/cv2 없이도 호환되도록: frame은 RGB ndarray라고 가정
    return Image.fromarray(frame_rgb)

def frames_to_pixel_values(frames_rgb: List[np.ndarray], input_size: int) -> Tuple[torch.Tensor, List[int]]:
    """여러 프레임(RGB ndarray) -> InternVL 배치 입력 텐서와 각 샘플의 타일 개수 리스트"""
    transform = build_transform(input_size)
    all_tensors, num_patches_list = [], []
    for fr in frames_rgb:
        img = pil_from_frame(fr)
        tiles = dynamic_preprocess(img, image_size=input_size, use_thumbnail=True, max_num=12)
        px = [transform(t) for t in tiles]
        px = torch.stack(px)  # [num_tiles, 3, H, W]
        all_tensors.append(px)
        num_patches_list.append(px.size(0))
    cat = torch.cat(all_tensors, dim=0)  # 모든 타일을 이어붙임
    return cat, num_patches_list

def infer_categories_for_frames(model, tokenizer, image_size, frames_rgb: List[np.ndarray]) -> List[str]:
    # 프레임들을 InternVL 입력으로 변환
    pixel_values_tensor, num_patches_list = frames_to_pixel_values(frames_rgb, image_size)
    device = next(model.parameters()).device
    pixel_values_tensor = pixel_values_tensor.to(device).to(torch.bfloat16)

    questions = [PROMPT_TEMPLATE] * len(frames_rgb)
    responses = model.batch_chat(
        tokenizer,
        pixel_values=pixel_values_tensor,
        num_patches_list=num_patches_list,
        questions=questions,
        generation_config=dict(num_beams=1, max_new_tokens=15, min_new_tokens=5),
    )
    cats = [parse_prediction(r) for r in responses]
    # 안전장치: 파싱 실패는 normal로 폴백
    cats = [c if c in ("violence", "normal") else "normal" for c in cats]
    return cats

def overlay_text_top_center(frame_bgr: np.ndarray, text: str) -> np.ndarray:
    """상단 중앙에 반투명 박스+텍스트"""
    h, w, _ = frame_bgr.shape
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.6, w / 1280.0 * 0.8)
    thickness = 2
    # 텍스트 크기 계산
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    x = (w - tw) // 2
    y = max(10 + th, 10 + th)  # 상단 여백
    # 배경 박스
    pad = 10
    x1, y1 = max(0, x - pad), max(0, y - th - pad)
    x2, y2 = min(w, x + tw + pad), min(h, y + pad)
    overlay = frame_bgr.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 0), -1)
    alpha = 0.45
    cv2.addWeighted(overlay, alpha, frame_bgr, 1 - alpha, 0, frame_bgr)
    # 텍스트
    color = (255, 255, 255) if text == "normal" else (0, 180, 255)  # normal: white, violence: orange-ish
    cv2.putText(frame_bgr, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)
    return frame_bgr

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
    """한 개 비디오 처리:
       - window_size마다 마지막 프레임 인덱스를 대표로 추론
       - 배치 단위로 묶어 추론
       - 윈도우 범위의 모든 프레임에 동일 라벨 오버레이
       - 출력 루트에 원 경로 구조 그대로 mp4 저장
    """
    # 입력 비디오 오픈 (cv2 VideoCapture 사용: 안정적 코덱/프레임 접근)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"⚠️ Cannot open video: {video_path}")
        return

    # 비디오 속성
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 출력 경로 (동일 구조 미러링)
    rel = video_path.relative_to(input_root)
    out_path = (output_root / rel).with_suffix(".mp4")
    ensure_parent_dirs(out_path)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))

    # --- CHANGED: 윈도우 단위로 프레임 전체를 받아 추론 -----------------------
    def read_window_frames_rgb(start_idx: int, end_idx: int) -> List[np.ndarray]:
        """[start_idx, end_idx) 구간을 RGB 프레임 리스트로 읽기"""
        frames = []
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_idx)
        for _f in range(start_idx, end_idx):
            ok, frame_bgr = cap.read()
            if not ok or frame_bgr is None:
                break
            frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        return frames

    # 윈도우 경계 나누기
    window_ranges: List[Tuple[int, int]] = []
    for s in range(0, total_frames, window_size):
        e = min(s + window_size, total_frames)
        window_ranges.append((s, e))

    # 각 윈도우별 레이블 추론
    categories_per_window: List[str] = []
    for (ws, we) in tqdm(window_ranges, desc=f"Infer windows for {rel.as_posix()}"):
        frames_rgb = read_window_frames_rgb(ws, we)
        if not frames_rgb:
            categories_per_window.append("normal")
            continue
        cat = infer_category_for_window(model, tokenizer, image_size, frames_rgb)
        categories_per_window.append(cat)

    # --- 쓰기(오버레이) 단계: 기존 로직 유지 -----------------------------------
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    with tqdm(total=total_frames, desc=f"Writing {rel.as_posix()}") as pbar:
        for fidx in range(total_frames):
            ok, frame_bgr = cap.read()
            if not ok or frame_bgr is None:
                break
            win_id = min(fidx // window_size, len(categories_per_window)-1)
            label = categories_per_window[win_id] if win_id >= 0 else "normal"
            frame_bgr = overlay_text_top_center(frame_bgr, label)
            writer.write(frame_bgr)
            pbar.update(1)

    writer.release()
    cap.release()

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

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=str, required=True, help="재귀적으로 탐색할 입력 비디오 루트 경로")
    parser.add_argument("--output-root", type=str, required=True, help="동일 폴더 구조로 저장할 출력 루트 경로")
    parser.add_argument("--checkpoint", type=str, required=True, help="InternVL 체크포인트 경로")
    parser.add_argument("--window-size", type=int, default=15, help="윈도우 크기 (윈도우 마지막 프레임이 대표)")
    parser.add_argument("--batch-size", type=int, default=20, help="한 번에 추론할 대표 프레임 수")
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
