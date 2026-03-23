import os
from src.preprocess.video_splitter import preprocess_video_chunk_split_folder
from configs.config_preprocess import INPUT_VIDEO_DIRECTORY, OUTPUT_CLIPS_DIRECTORY, CLIP_DURATION, NUM_CORES
from src._autolabeling import autolabel_files_recursively, translate_descriptions_recursively
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

from src.data_checker.stats.json_checker import check_json_directory
def run_data_check(args):
    """데이터 점검을 실행합니다. JSON 디렉토리 또는 JSONL 파일을 점검합니다."""
    if args.type == "json":
        check_json_directory(args.input, low_threshold=args.threshold)
    elif args.type == "jsonl":
        print_dataset_info(args.input if isinstance(args.input, list) else [args.input])

from src.preprocess.train_test_split import split_dataset_final
def run_split_train_test_dataset(args):
    split_dataset_final(args.input_file , args.ratio, args.output_dir)
from src.preprocess.label2jsonl import label_to_jsonl_result_save
def run_label_to_jsonl(args):
     label_to_jsonl_result_save(
         input_dir=args.input_dir,
         output_file_path=args.output_file,
         mode=args.mode,
         data_type=args.data_type,
         base_dir=args.base_dir,
         item_type=args.item_type,
         item_task=args.item_task,
         task_name=args.task_name,
     )

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
    FAILURE_LOG_DIR = "assets/logs"

    print("--- Running Gemini Autolabeler ---")
    autolabel_files_recursively(
        input_folder=args.input_dir,
        failure_log_dir=FAILURE_LOG_DIR,
        num_workers=args.num_process,
        options=args.options,
        mode=args.mode,
        model_name=args.model,
    )
    print("--- Autolabeling finished ---")


def run_translate(args):
    """JSON description 영→한 번역 프로세스를 실행합니다."""
    FAILURE_LOG_DIR = "assets/logs"

    print("--- Running JSON Description Translator ---")
    translate_descriptions_recursively(
        input_folder=args.input_dir,
        failure_log_dir=FAILURE_LOG_DIR,
        num_workers=args.num_process,
        model_name=args.model,
    )
    print("--- Translation finished ---")
    
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
                                                               "cctv_normal" , "cctv_violence", "scvd_normal" , "scvd_violence", 
                                                               "gangnam","hyundai_normal" , "hyundai_falldown" ],  
                                 required=True, help='Labeling mode')
    parser_autolabel.add_argument('-n','--num_process', type=int, default=8, 
                                 required=False, help='Num processes')
    parser_autolabel.add_argument('-m','--mode', choices=['video', 'image'] , default='video',
                                 required=False, help='Labeling mode type')
    parser_autolabel.add_argument('--model', type=str, default=None,
                                 help='Gemini 모델명 (기본: gemini-3-pro-preview)')
    parser_autolabel.set_defaults(func=run_autolabel)

    # --- 'translate' 서브 파서 ---
    parser_translate = subparsers.add_parser('translate', help='Translate JSON description fields (English → Korean).')
    parser_translate.add_argument('-i', '--input-dir', type=str, required=True,
                                 help='Input directory containing JSON label files')
    parser_translate.add_argument('-n', '--num_process', type=int, default=8,
                                 required=False, help='Num processes')
    parser_translate.add_argument('--model', type=str, default=None,
                                 help='Gemini 모델명 (기본: gemini-2.0-flash)')
    parser_translate.set_defaults(func=run_translate)

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
    parser_label_to_jsonl.add_argument('-opt', '--mode', default="train", type=str,
                            help='변환 모드 (train: 학습용, test: 평가용)')
    parser_label_to_jsonl.add_argument('-dt', '--data_type', default="video", type=str,
                            help='미디어 타입 (video / image)')
    parser_label_to_jsonl.add_argument('-ity', '--item_type', default="clip", type=str,
                            help='JSONL type 필드값 (예: clip, capture_frame)')
    parser_label_to_jsonl.add_argument('-itk', '--item_task', default="caption", type=str,
                            help='JSONL task 필드값 (예: caption)')
    parser_label_to_jsonl.add_argument('-tn', '--task_name', default="violence", type=str,
                            help='분류 작업명 — 프롬프트 선택 키 (예: violence, falldown)')
    parser_label_to_jsonl.add_argument('--base-dir', default="data/", type=str,
                            help='미디어 상대 경로 기준 디렉토리 (기본: data/)')

    parser_label_to_jsonl.set_defaults(func=run_label_to_jsonl)    
    
    # --- 'jsonl_information_check' 서브 파서 (하위 호환) ---
    parser_json_inform_check = subparsers.add_parser('jsonl_inform_check', help="jsonl information check")
    parser_json_inform_check.add_argument('-i', "--files",nargs='+', help='분석할 JSONL 파일 경로 (여러 개 가능)', required=True)
    parser_json_inform_check.set_defaults(func=run_jsonl_inform_check)

    # --- 'data_check' 서브 파서 (통합 데이터 점검) ---
    parser_data_check = subparsers.add_parser('data_check', help="JSON 라벨/JSONL 데이터 통합 점검")
    parser_data_check.add_argument('-i', '--input', type=str, required=True,
                                   help='점검할 디렉토리(json) 또는 파일 경로(jsonl)')
    parser_data_check.add_argument('-t', '--type', choices=['json', 'jsonl'], required=True,
                                   help='점검 유형: json(라벨 디렉토리), jsonl(어노테이션 파일)')
    parser_data_check.add_argument('--threshold', type=float, default=0.49,
                                   help='낮은 비율 카테고리 기준값 (기본: 0.49)')
    parser_data_check.set_defaults(func=run_data_check)

    # --- 'train_test_split' 서브 파서 ---
    parser_train_test_split = subparsers.add_parser('train_test_split', help="split video ,Only gang-jin labeling format")
    parser_train_test_split.add_argument('-i', '--input_file', type=str, required=True,
                            help='Input Video Annotation full file jsonl')
    parser_train_test_split.add_argument('-r', '--ratio', type=float, required=True,
                            help='Test dataset ratio')
    parser_train_test_split.add_argument('-o', '--output_dir', type=str, required=True,
                            help='train, test jsonl file save path ')
    parser_train_test_split.set_defaults(func=run_split_train_test_dataset)

    args = parser.parse_args()
    args.func(args)

