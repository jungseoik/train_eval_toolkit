import gradio as gr
import torch
from transformers import AutoModel, AutoTokenizer
from PIL import Image
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from decord import VideoReader, cpu
import numpy as np
import cv2
import tempfile
import os
from configs.constants import APP_PROMPT
# 기존 코드들
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

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
    # start_idx = max(first_idx, round(start * fps))
    # end_idx = min(round(end * fps), max_frame)
    start_idx = max(first_idx, start)
    end_idx = min(end, max_frame)
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

class InternVL3Inferencer:
    # def __init__(self, model_path="ckpts/InternVL3-2B_total_vio", device="cuda:0"):
    def __init__(self, model_path="ckpts/InternVL3-2B_gangnam_rwf2000_gj_cctv_scvdALL_NOweapon_no_split"
                 , device="cuda:0"):
        
    # def __init__(self, model_path="ckpts/merge_result", device="cuda:0"):
    # def __init__(self, model_path="ckpts/InternVL3-2B", device="cuda:0"):
    
        print("[INFO] InternVL 모델 로딩 중...")
        self.model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            use_flash_attn=False,
            trust_remote_code=True
        ).eval().to(device)
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
        self.device = device
        self.generation_config = dict(max_new_tokens=1024, do_sample=False)
        print("[INFO] InternVL 모델 로딩 완료.")

    def infer(self, video_path: str, template: str, num_segments: int = 12, bound=None) -> str:
        pixel_values, num_patches_list = load_video(video_path, bound=bound, num_segments=num_segments, max_num=1)
        pixel_values = pixel_values.to(torch.bfloat16).cuda()
        video_prefix = ''.join([f'Frame{i+1}: <image>\n' for i in range(len(num_patches_list))])
        question = video_prefix + template
        response = self.model.chat(self.tokenizer, pixel_values, question, self.generation_config)
        return response

# 전역 변수들
current_video_reader = None
current_video_path = None
# inferencer = None
inferencer = InternVL3Inferencer()

def load_video_for_display(video_path):
    """비디오를 로드하고 전역 변수에 저장"""
    global current_video_reader, current_video_path
    if video_path != current_video_path:
        current_video_reader = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        current_video_path = video_path
    return current_video_reader

def get_video_info(video_path):
    """비디오 정보 반환 (총 프레임 수, FPS 등)"""
    if video_path is None:
        return 0, 0, 0
    
    vr = load_video_for_display(video_path)
    total_frames = len(vr)
    fps = float(vr.get_avg_fps())
    duration = total_frames / fps
    
    return total_frames, fps, duration

def get_frame_at_index(video_path, frame_idx):
    """특정 프레임 인덱스의 이미지 반환"""
    if video_path is None:
        return None
    
    vr = load_video_for_display(video_path)
    frame_idx = max(0, min(frame_idx, len(vr) - 1))
    frame = vr[frame_idx].asnumpy()
    return Image.fromarray(frame)

def update_frame_slider_range(video_path):
    """비디오가 업로드되면 프레임 슬라이더 범위 업데이트"""
    if video_path is None:
        return gr.update(maximum=0, value=0), None, [], ""
    
    total_frames, fps, duration = get_video_info(video_path)
    
    # 첫 번째 프레임 표시
    first_frame = get_frame_at_index(video_path, 0)
    
    info_text = f"총 프레임: {total_frames}, FPS: {fps:.2f}, 길이: {duration:.2f}초"
    
    return (
        gr.update(maximum=total_frames-1, value=0),
        first_frame,
        [],
        info_text
    )

def update_frame_display(video_path, frame_idx, window_size):
    """프레임 슬라이더가 변경되면 이미지 및 갤러리 업데이트"""
    if video_path is None:
        return None, []
    
    # 현재 프레임 표시
    current_frame = get_frame_at_index(video_path, int(frame_idx))
    
    # window_size만큼의 프레임들을 갤러리로 표시
    vr = load_video_for_display(video_path)
    total_frames = len(vr)
    
    # 현재 프레임을 기준으로 뒤로 window_size만큼 프레임 추출
    start_idx = max(0, int(frame_idx) - window_size + 1)
    end_idx = int(frame_idx) + 1
    
    gallery_images = []
    for i in range(start_idx, end_idx):
        if i < total_frames:
            frame_img = get_frame_at_index(video_path, i)
            gallery_images.append((frame_img, f"Frame {i}"))
    
    return current_frame, gallery_images

def initialize_model():
    """모델 초기화"""
    global inferencer
    if inferencer is None:
        inferencer = InternVL3Inferencer()
    return "모델이 준비되었습니다!"

def run_inference(video_path, template, num_segments, frame_idx, window_size):
    """추론 실행"""
    global inferencer
    
    if video_path is None:
        return "비디오를 먼저 업로드하세요."
    
    if inferencer is None:
        return "모델을 먼저 초기화하세요."
    
    if not template.strip():
        return "질문을 입력하세요."
    
    try:
        # 현재 프레임을 기준으로 window_size만큼의 구간을 bound로 설정
        vr = load_video_for_display(video_path)
        fps = float(vr.get_avg_fps())
        
        start_frame = max(0, int(frame_idx) - window_size + 1)
        end_frame = int(frame_idx) + 1
        
        # 프레임을 시간으로 변환
        # start_time = start_frame / fps
        # end_time = end_frame / fps
        # bound = [start_time, end_time]
        bound = [start_frame, end_frame]
        
        result = inferencer.infer(
            video_path=video_path,
            template=template,
            num_segments=num_segments,
            bound=bound
        )
        return result
    except Exception as e:
        return f"추론 중 오류가 발생했습니다: {str(e)}"

# Gradio 인터페이스 구성
with gr.Blocks(title="Video Analysis") as demo:
    gr.Markdown("# Video Analysis Tool")
    with gr.Row():
        with gr.Column(scale=1):
            video_input = gr.File(
                label="비디오 업로드", 
                file_types=["video"],
                type="filepath"
            )
            
            init_btn = gr.Button("모델 초기화", variant="secondary")
            inference_btn = gr.Button("추론 실행", variant="primary")
            init_status = gr.Textbox(label="모델 상태", value="모델이 초기화되지 않았습니다.", interactive=False)
            video_info = gr.Textbox(label="비디오 정보", interactive=False)
            
            # 프레임 탐색 슬라이더
            frame_slider = gr.Slider(
                minimum=0, 
                maximum=100, 
                value=0, 
                step=1,
                label="프레임 탐색",
                interactive=True
            )
            example_image = gr.Image(label="예시" ,value="assets/edit_video.png")
            # 구간 선택을 위한 window_size 입력
            window_size_input = gr.Number(
                label="Window Size (프레임 개수)",
                value=12,
                minimum=1,
                maximum=50,
                step=1,
                info="현재 프레임을 기준으로 뒤로 몇 개의 프레임을 포함할지 설정"
            )
            
            # num_segments 설정
            num_segments_input = gr.Slider(
                minimum=1,
                maximum=32,
                value=12,
                step=1,
                label="window 안에서 추출할 프레임 개수 (num_segments)"
            )
            
            # 질문 입력
            template_input = gr.Textbox(
                label="질문/템플릿",
                value=APP_PROMPT,
                placeholder="비디오에 대해 묻고 싶은 내용을 입력하세요...",
                lines=3
            )
            

        
        # 오른쪽 열: 이미지 표시 및 결과
        with gr.Column(scale=1):
            # 현재 프레임 표시
            frame_display = gr.Image(
                label="현재 프레임",
                type="pil",
                height=300
            )
            
            # 프레임 갤러리 (window_size만큼의 프레임들)
            frame_gallery = gr.Gallery(
                label="프레임 시퀀스",
                columns=5,
                object_fit="contain"
            )
            
            # 추론 결과
            result_output = gr.Textbox(
                label="추론 결과",
                lines=8,
                max_lines=15,
                interactive=False
            )
    
    # 이벤트 핸들러들
    init_btn.click(
        initialize_model,
        outputs=init_status
    )
    
    video_input.change(
        update_frame_slider_range,
        inputs=video_input,
        outputs=[frame_slider, frame_display, frame_gallery, video_info]
    )
    
    frame_slider.change(
        update_frame_display,
        inputs=[video_input, frame_slider, window_size_input],
        outputs=[frame_display, frame_gallery]
    )
    
    window_size_input.change(
        update_frame_display,
        inputs=[video_input, frame_slider, window_size_input],
        outputs=[frame_display, frame_gallery]
    )
    
    inference_btn.click(
        run_inference,
        inputs=[video_input, template_input, num_segments_input, frame_slider, window_size_input],
        outputs=result_output
    )

if __name__ == "__main__":
    demo.launch( debug=True)