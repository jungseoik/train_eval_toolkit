import os
from src.preprocess.video_splitter import preprocess_video_chunk_split_folder
from configs.config_preprocess import INPUT_VIDEO_DIRECTORY, OUTPUT_CLIPS_DIRECTORY, CLIP_DURATION, NUM_CORES
from src.autolabel.auto_labeler import autolabel_videos_recursively
if __name__ == '__main__':
    # # 전처리 프로세스
    # print("--- Running video_splitter module as a standalone script for demonstration ---")
    # os.makedirs(INPUT_VIDEO_DIRECTORY, exist_ok=True)
    # preprocess_video_chunk_split_folder(
    #     input_folder=INPUT_VIDEO_DIRECTORY,
    #     output_folder=OUTPUT_CLIPS_DIRECTORY,
    #     seconds_per_clip=CLIP_DURATION,
    #     num_workers=NUM_CORES
    # )
    
    ### 오토라벨 프로세스
    INPUT_ROOT_DIR = "data/processed/violence_inside_elevator_clips_2sec"
    FAILURE_LOG_DIR = "assets/logs"
    NUM_CORES = 8
    
    print("--- Running Gemini Autolabeler (Recursive) ---")
    autolabel_videos_recursively(
        input_folder=INPUT_ROOT_DIR,
        failure_log_dir=FAILURE_LOG_DIR,
        num_workers=NUM_CORES
    )
    
    