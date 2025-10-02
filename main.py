import os
from src.preprocess.video_splitter import preprocess_video_chunk_split_folder
from configs.config_preprocess import INPUT_VIDEO_DIRECTORY, OUTPUT_CLIPS_DIRECTORY, CLIP_DURATION, NUM_CORES
from src.autolabel.auto_labeler import autolabel_videos_recursively
import argparse


from src.preprocess.label_id_sorting import load_jsonl, reindex_by_id, save_jsonl
def run_jsonl_reindex_sorting(args):
    rows = load_jsonl(args.input_file)
    rows_reindexed = reindex_by_id(rows)
    save_jsonl(rows_reindexed, args.output_file)

from src.utils.merge_jsonl import JSONLMerger    
def run_merge_jsonl(args):
    merger = JSONLMerger()
    file1 = args.file1
    file2 = args.file2
    output = args.output
    merger.merge_jsonl_files(file1, file2, output)
from src.utils.jsonl_inform_check import print_dataset_info
def run_jsonl_inform_check(args):
    print_dataset_info(args.files)

################################################################################################
from src.preprocess.gj.gj_split import process_videos_clips
def run_gj_video_split(args):
    source_dir = args.input_dir
    output_dir = args.output_dir
    num_processes = args.num_processes
    process_videos_clips(source_dir , output_dir, num_processes)

from src.preprocess.train_test_split import split_dataset_final
def run_split_train_test_dataset(args):
    split_dataset_final(args.input_file , args.ratio, args.output_dir)
        
from src.preprocess.aihub.store.aihub_store_split import process_videos_clips_aihub_store
def run_aihub_store_video_split(args):
    process_videos_clips_aihub_store(args.input_dir, args.output_dir, args.num_processes)
from src.preprocess.label2jsonl import label_to_jsonl_result_save
def run_label_to_jsonl(args):
    label_to_jsonl_result_save(args.input_dir, args.output_file, args.option , args.data_type)

def run_preprocess(args): # 이제 각 함수는 args를 받을 수 있습니다.
    """전처리 프로세스를 실행합니다."""
    print("--- Running video_splitter module ---")
    # 향후 커맨드라인에서 값을 받고 싶다면 args.input_folder 처럼 사용 가능합니다.
    os.makedirs(INPUT_VIDEO_DIRECTORY, exist_ok=True) 
    preprocess_video_chunk_split_folder(
        input_folder=INPUT_VIDEO_DIRECTORY,
        output_folder=OUTPUT_CLIPS_DIRECTORY,
        seconds_per_clip=CLIP_DURATION,
        num_workers=NUM_CORES
    )
    print("--- Preprocessing finished ---")
    

def run_autolabel(args):
    """오토라벨 프로세스를 실행합니다."""
    # INPUT_ROOT_DIR = "data/processed/violence_inside_elevator_clips_2sec"
    # data/raw/rwf2000/RWF-2000/train/NonFight
    # data/raw/rwf2000/RWF-2000/train/Fight
    FAILURE_LOG_DIR = "assets/logs"
    NUM_CORES_LABEL = 8
    
    print("--- Running Gemini Autolabeler ---")
    autolabel_videos_recursively(
        input_folder=args.input_dir,
        failure_log_dir=FAILURE_LOG_DIR,
        num_workers=args.num_process,
        options=args.options
    )
    print("--- Autolabeling finished ---")
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="main script with commands for video processing, train, eval")
    
    subparsers = parser.add_subparsers(dest='command' , required=True, help="Available commands")
    
    # --- 'preprocess' 서브 파서 ---
    parser_preprocess = subparsers.add_parser('preprocess', help='Run the video preprocessing process.')
    # 나중에 preprocess에만 필요한 옵션이 생기면 여기에 추가합니다.
    # 예: parser_preprocess.add_argument('--input', type=str, help='Input video directory')
    parser_preprocess.set_defaults(func=run_preprocess) # 실행할 함수를 지정

    # --- 'autolabel' 서브 파서 ---
    parser_autolabel = subparsers.add_parser('autolabel', help='Run the autolabeling process.')
    parser_autolabel.add_argument('-i', '--input-dir', type=str, required=True,
                                 help='Input directory containing video clips')
    parser_autolabel.add_argument('-opt','--options', choices=['vio', 'normal', 'basic','vio_timestamp', 
                                                               "aihub_space" , "gj_normal" , "gj_violence",
                                                               "cctv_normal" , "cctv_violence", "scvd_normal" , "scvd_violence"],  
                                 required=True, help='Labeling mode')
    parser_autolabel.add_argument('-n','--num_process', type=int, default=8, 
                                 required=False, help='Num processes')
    parser_autolabel.set_defaults(func=run_autolabel)
    
    # --- 'jsonl_reindex_sorting' 서브 파서 ---
    parser_jsonl_reindex_sort = subparsers.add_parser('jsonl_reindex' , help="Run the jsonl reindex and id sorting")
    parser_jsonl_reindex_sort.add_argument('-i' , '--input_file' , type=str, required=True, help='Input JSONL file path')
    parser_jsonl_reindex_sort.add_argument('-o' , '--output_file' , type=str, required=True, help='Save JSONL file path')
    
    parser_jsonl_reindex_sort.set_defaults(func=run_jsonl_reindex_sorting)
    
    # --- 'jsonl_merge' 서브 파서 ---
    parser_merge = subparsers.add_parser('merge_jsonl', help="Merge two JSONL files")
    parser_merge.add_argument('-1', '--file1', type=str, required=True,
                            help='First JSONL file to merge')
    parser_merge.add_argument('-2', '--file2', type=str, required=True,
                            help='Second JSONL file to merge') 
    parser_merge.add_argument('-o', '--output', type=str, required=True,
                            help='Output merged file')
    parser_merge.set_defaults(func=run_merge_jsonl)
    
    # --- 'label_to_jsonl' 서브 파서 ---
    parser_label_to_jsonl = subparsers.add_parser('label2jsonl', help="Convert labels to jsonl ")
    parser_label_to_jsonl.add_argument('-i', '--input_dir', type=str, required=True,
                            help='A directory containing labels')
    parser_label_to_jsonl.add_argument('-o', '--output_file', type=str, required=True,
                            help='Location of jsonl file to save') 
    parser_label_to_jsonl.add_argument('-opt', '--option', default="train", type=str, required=False,
                            help='train or test extract mode select') 
    parser_label_to_jsonl.add_argument('-dt', '--data_type', default="video", type=str, required=False,
                            help='image or video type select') 

    parser_label_to_jsonl.set_defaults(func=run_label_to_jsonl)    
    
    # --- 'jsonl_information_check' 서브 파서 ---
    # remove_human_video_prompts("data/instruction/evaluation/test_rwf2000.jsonl")
    parser_json_inform_check = subparsers.add_parser('jsonl_inform_check', help="jsonl information check")
    parser_json_inform_check.add_argument('-i', "--files",nargs='+', help='분석할 JSONL 파일 경로 (여러 개 가능)', required=True)

    parser_json_inform_check.set_defaults(func=run_jsonl_inform_check)    

    ####################################################################################################################################


    # --- 'gj_split_video' 서브 파서 ---
    parser_gj_split = subparsers.add_parser('gj_split', help="split video ,Only gang-jin labeling format")
    parser_gj_split.add_argument('-i', '--input_dir', type=str, required=True,
                            help='Input directory containing video & label')
    parser_gj_split.add_argument('-o', '--output_dir', type=str, required=True,
                            help='Output save clip directory path')
    parser_gj_split.add_argument('-p', '--num_processes', type=int, required=True,
                            help='Number of processes to use for parallel processing')
    parser_gj_split.set_defaults(func=run_gj_video_split)

    # --- 'train_test_split' 서브 파서 ---
    parser_train_test_split = subparsers.add_parser('train_test_split', help="split video ,Only gang-jin labeling format")
    parser_train_test_split.add_argument('-i', '--input_file', type=str, required=True,
                            help='Input Video Annotation full file jsonl')
    parser_train_test_split.add_argument('-r', '--ratio', type=float, required=True,
                            help='Test dataset ratio')
    parser_train_test_split.add_argument('-o', '--output_dir', type=str, required=True,
                            help='train, test jsonl file save path ')
    parser_train_test_split.set_defaults(func=run_split_train_test_dataset)

    # --- 'aihub_store_split_video' 서브 파서 ---
    aihub_store_split_video = subparsers.add_parser('aihub_store_split', help="split video ,Only aihub-store labeling format")
    aihub_store_split_video.add_argument('-i', '--input_dir', type=str, required=True,
                            help='Input directory containing video & label')
    aihub_store_split_video.add_argument('-o', '--output_dir', type=str, required=True,
                            help='Output save clip directory path')
    aihub_store_split_video.add_argument('-p', '--num_processes', type=int, required=True,
                            help='Number of processes to use for parallel processing')
    aihub_store_split_video.set_defaults(func=run_aihub_store_video_split)



    args = parser.parse_args()
    args.func(args)

