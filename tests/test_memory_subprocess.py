"""
메모리 누적 검증 테스트.

기존 방식(in-process)과 subprocess 방식의 메모리 사용량을 비교한다.
lmdeploy 서버 없이 프레임 추출 + base64 인코딩만으로 메모리 누적 패턴을 검증.

실행:
    python tests/test_memory_subprocess.py
"""

import asyncio
import base64
import gc
import json
import multiprocessing
import os
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace


def get_rss_mb() -> float:
    """현재 프로세스의 RSS(Resident Set Size)를 MB 단위로 반환."""
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS"):
                return int(line.split()[1]) / 1024
    return 0


def simulate_video_evaluation(video_idx: int, num_frames: int = 100, concurrency: int = 50):
    """비디오 1개의 평가를 시뮬레이션. (lmdeploy 서버 없이 메모리 패턴만 재현)

    실제 evaluate_benchmark과 동일한 구조:
    - asyncio.run() 으로 이벤트 루프 생성
    - Semaphore로 동시성 제어
    - cv2 프레임 추출 → base64 인코딩 → httpx payload 크기 데이터 생성
    """

    async def _evaluate_async():
        sem = asyncio.Semaphore(concurrency)
        results = {}

        async def process_frame(fidx: int):
            async with sem:
                # 실제 코드와 동일한 메모리 할당 패턴
                # 1920x1080x3 프레임 시뮬레이션
                frame_data = bytearray(1920 * 1080 * 3)
                # JPEG 인코딩 시뮬레이션 (~500KB)
                jpeg_data = bytearray(500 * 1024)
                # base64 인코딩 (~670KB)
                b64 = base64.b64encode(jpeg_data).decode("ascii")
                # httpx payload 시뮬레이션
                payload = json.dumps({"image": b64, "text": "test prompt" * 10})
                # 응답 시뮬레이션
                await asyncio.sleep(0.001)
                del frame_data, jpeg_data, b64, payload
                return fidx, 1

        tasks = [asyncio.create_task(process_frame(f)) for f in range(num_frames)]
        for coro in asyncio.as_completed(tasks):
            fidx, pred = await coro
            results[fidx] = pred

        return results

    return asyncio.run(_evaluate_async())


# ============================================================
# 테스트 1: 기존 방식 (in-process) - 메모리 누적 확인
# ============================================================

def test_inprocess(num_videos: int = 30, num_frames: int = 100):
    """기존 방식: 같은 프로세스에서 모든 비디오를 처리."""
    print("=" * 60)
    print("테스트 1: 기존 방식 (in-process)")
    print(f"  비디오 수: {num_videos}, 프레임/비디오: {num_frames}")
    print("=" * 60)

    rss_start = get_rss_mb()
    print(f"  시작 RSS: {rss_start:.0f} MB")

    for i in range(num_videos):
        simulate_video_evaluation(i, num_frames=num_frames)
        gc.collect()
        if (i + 1) % 10 == 0:
            rss = get_rss_mb()
            print(f"  [{i+1}/{num_videos}] RSS: {rss:.0f} MB (+{rss - rss_start:.0f} MB)")

    rss_end = get_rss_mb()
    print(f"  최종 RSS: {rss_end:.0f} MB (증가: +{rss_end - rss_start:.0f} MB)")
    return rss_end - rss_start


# ============================================================
# 테스트 2: subprocess 방식 - 메모리 리셋 확인
# ============================================================

def _subprocess_benchmark_worker(video_start: int, video_count: int, num_frames: int, result_file: str):
    """subprocess에서 실행되는 워커."""
    for i in range(video_count):
        simulate_video_evaluation(video_start + i, num_frames=num_frames)
        gc.collect()

    with open(result_file, "w") as f:
        json.dump({"done": True, "rss_mb": get_rss_mb()}, f)


def test_subprocess(num_videos: int = 30, num_frames: int = 100, videos_per_subprocess: int = 10):
    """subprocess 방식: 일정 수의 비디오마다 새 프로세스에서 실행."""
    print("=" * 60)
    print("테스트 2: subprocess 방식")
    print(f"  비디오 수: {num_videos}, 프레임/비디오: {num_frames}")
    print(f"  subprocess당 비디오: {videos_per_subprocess}")
    print("=" * 60)

    ctx = multiprocessing.get_context("spawn")
    rss_start = get_rss_mb()
    print(f"  부모 시작 RSS: {rss_start:.0f} MB")

    for batch_start in range(0, num_videos, videos_per_subprocess):
        batch_count = min(videos_per_subprocess, num_videos - batch_start)

        fd, result_file = tempfile.mkstemp(suffix=".json")
        os.close(fd)

        proc = ctx.Process(
            target=_subprocess_benchmark_worker,
            args=(batch_start, batch_count, num_frames, result_file),
        )
        proc.start()
        proc.join()

        # 자식 프로세스의 메모리 확인
        try:
            with open(result_file) as f:
                child_result = json.load(f)
            child_rss = child_result.get("rss_mb", 0)
        except Exception:
            child_rss = 0
        finally:
            os.unlink(result_file)

        parent_rss = get_rss_mb()
        batch_end = batch_start + batch_count
        print(f"  [{batch_end}/{num_videos}] 부모 RSS: {parent_rss:.0f} MB (+{parent_rss - rss_start:.0f} MB) | 자식 RSS: {child_rss:.0f} MB")

    rss_end = get_rss_mb()
    print(f"  부모 최종 RSS: {rss_end:.0f} MB (증가: +{rss_end - rss_start:.0f} MB)")
    return rss_end - rss_start


# ============================================================
# 메인
# ============================================================

if __name__ == "__main__":
    print("\n메모리 누적 비교 테스트")
    print("(lmdeploy 서버 없이 메모리 할당 패턴만 검증)\n")

    NUM_VIDEOS = 30
    NUM_FRAMES = 100

    leak_inprocess = test_inprocess(NUM_VIDEOS, NUM_FRAMES)
    print()
    leak_subprocess = test_subprocess(NUM_VIDEOS, NUM_FRAMES, videos_per_subprocess=10)

    print("\n" + "=" * 60)
    print("결과 비교")
    print("=" * 60)
    print(f"  in-process 메모리 증가: +{leak_inprocess:.0f} MB")
    print(f"  subprocess 메모리 증가: +{leak_subprocess:.0f} MB")
    print(f"  절감율: {(1 - leak_subprocess / max(leak_inprocess, 1)) * 100:.0f}%")
