#!/usr/bin/env python3
"""
멀티프로세스 비디오 재인코딩 스크립트
사용법: python video_encoder.py <입력폴더> <출력폴더> [동시실행수]
예: python video_encoder.py /data/raw/ai_hub_cctv /data/processed 8
"""

import os
import sys
import shutil
import subprocess
import tempfile
from pathlib import Path
from multiprocessing import Pool, cpu_count
import argparse
from tqdm import tqdm
import time

def process_single_video(args_tuple):
    """단일 비디오 파일 처리"""
    video_path, output_base_dir, input_base_dir = args_tuple  # input_base_dir 추가
    
    try:
        video_path = Path(video_path)
        output_base_dir = Path(output_base_dir)
        input_base_dir = Path(input_base_dir)  # 추가
        
        print(f"🔄 처리 시작: {video_path.name}")
        
        # 새로운 코드: input_dir 기준 상대 경로 계산
        relative_path = video_path.parent.relative_to(input_base_dir)
        
        # 출력 디렉토리 생성
        target_dir = output_base_dir / relative_path
        target_dir.mkdir(parents=True, exist_ok=True)
        
        # 파일 경로들
        output_video = target_dir / video_path.name
        json_filename = video_path.stem + '.json'
        input_json = video_path.parent / json_filename
        output_json = target_dir / json_filename
        
        # 임시 파일 생성
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_file:
            temp_video_path = temp_file.name
        
        try:
            # FFmpeg 명령어 구성
            ffmpeg_cmd = [
                'ffmpeg', '-y', '-i', str(video_path),
                '-c:v', 'libx264',
                '-profile:v', 'baseline',
                '-level', '4.2',
                '-pix_fmt', 'yuv420p',
                '-g', '30',
                '-keyint_min', '30',
                '-sc_threshold', '0',
                '-x264-params', 'bframes=0:repeat-headers=1',
                '-preset', 'veryfast',
                '-movflags', '+faststart',
                '-c:a', 'aac',
                '-b:a', '128k',
                temp_video_path
            ]
            
            # FFmpeg 실행 (출력 숨기기)
            result = subprocess.run(
                ffmpeg_cmd, 
                capture_output=True, 
                text=True,
                check=True
            )
            
            # 성공시 최종 위치로 이동
            shutil.move(temp_video_path, output_video)
            print(f"✅ 비디오 완료: {output_video}")
            
            # JSON 파일 복사 (있다면)
            if input_json.exists():
                shutil.copy2(input_json, output_json)
                print(f"📄 JSON 복사됨: {output_json}")
            
            print(f"🎉 전체 완료: {video_path.name}")
            return True, str(video_path)
            
        except subprocess.CalledProcessError as e:
            # FFmpeg 실패시
            if os.path.exists(temp_video_path):
                os.unlink(temp_video_path)
            print(f"❌ FFmpeg 실패: {video_path}")
            print(f"   에러: {e.stderr}")
            return False, str(video_path)
            
        except Exception as e:
            # 기타 에러
            if os.path.exists(temp_video_path):
                os.unlink(temp_video_path)
            print(f"❌ 처리 실패: {video_path}")
            print(f"   에러: {str(e)}")
            return False, str(video_path)
            
    except Exception as e:
        print(f"❌ 심각한 오류: {video_path}")
        print(f"   에러: {str(e)}")
        return False, str(video_path)

def find_video_files(input_dir):
    """재귀적으로 모든 .mp4 파일 찾기"""
    input_path = Path(input_dir)
    video_files = list(input_path.rglob("*.mp4"))
    return [str(f) for f in video_files]

def main():
    parser = argparse.ArgumentParser(description='멀티프로세스 비디오 재인코딩')
    parser.add_argument('input_dir', help='입력 폴더 경로')
    parser.add_argument('output_dir', help='출력 폴더 경로') 
    parser.add_argument('-j', '--jobs', type=int, default=cpu_count(), 
                       help=f'동시 실행 프로세스 수 (기본값: {cpu_count()})')
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    max_jobs = args.jobs
    
    # 입력 폴더 검사
    if not input_dir.exists():
        print(f"❌ 입력 폴더가 존재하지 않습니다: {input_dir}")
        sys.exit(1)
    
    # 출력 폴더 생성
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("🚀 멀티프로세스 비디오 처리 시작")
    print(f"📁 입력 폴더: {input_dir}")
    print(f"📁 출력 폴더: {output_dir}")
    print(f"⚡ 동시 실행: {max_jobs} 개")
    print("==================================")
    
    # 모든 비디오 파일 찾기
    print("🔍 비디오 파일 검색 중...")
    video_files = find_video_files(input_dir)
    
    if not video_files:
        print("❌ 처리할 .mp4 파일이 없습니다.")
        sys.exit(1)
    
    print(f"📊 발견된 비디오 파일: {len(video_files)}개")
    
    # 처리할 인자 준비
    # process_args = [(video_path, str(output_dir)) for video_path in video_files]
    process_args = [(video_path, str(output_dir), str(input_dir)) for video_path in video_files]
    # 시작 시간 기록
    start_time = time.time()
    
    # 멀티프로세싱으로 처리
    with Pool(processes=max_jobs) as pool:
        # 진행상황 표시와 함께 처리
        results = []
        with tqdm(total=len(video_files), desc="처리 진행", unit="files") as pbar:
            for result in pool.imap(process_single_video, process_args):
                results.append(result)
                pbar.update(1)
    
    # 결과 분석
    successful = [r for r in results if r[0]]
    failed = [r for r in results if not r[0]]
    
    # 처리 시간 계산
    end_time = time.time()
    processing_time = end_time - start_time
    
    print("==================================")
    print("🎊 모든 작업 완료!")
    print(f"📊 처리 결과: {len(successful)}/{len(video_files)} 개 파일 성공")
    print(f"⏱️  총 처리 시간: {processing_time:.1f}초")
    
    if failed:
        print(f"❌ 실패한 파일들:")
        for _, failed_file in failed[:5]:  # 최대 5개만 표시
            print(f"   - {Path(failed_file).name}")
        if len(failed) > 5:
            print(f"   ... 및 {len(failed) - 5}개 더")

if __name__ == "__main__":
    main()
    # python encoding.py results/eval_quality_gangnam/video_quality/yeoksam2st/falldown results/eval_quality_encoding/GangNam/yeoksam2st/falldown -j 64