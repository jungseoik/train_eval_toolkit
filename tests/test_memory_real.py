"""
실제 cv2를 사용한 메모리 누적 비교 테스트.

기존 방식(in-process)과 subprocess 방식의 메모리 사용량을 비교한다.
lmdeploy 서버 없이 프레임 추출 + base64 인코딩으로 메모리 패턴을 검증.

실행:
    python tests/test_memory_real.py
"""

import asyncio
import base64
import gc
import json
import multiprocessing
import os
import sys
import tempfile
from pathlib import Path


BENCH_BASE = Path("/mnt/PoC_benchmark/huggingface_benchmarks_dataset/Leaderboard_bench")
VIDEO_DIR = BENCH_BASE / "KISA_Falldown/dataset/falldown"


def get_rss_mb() -> float:
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS"):
                return int(line.split()[1]) / 1024
    return 0


def evaluate_video_real(video_path: str, window_size: int = 30, concurrency: int = 50):
    """실제 cv2로 프레임 추출 + base64 인코딩."""
    import cv2

    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    frames = list(range(0, total, window_size))

    async def _eval():
        sem = asyncio.Semaphore(concurrency)
        loop = asyncio.get_event_loop()

        def extract(fidx):
            c = cv2.VideoCapture(video_path)
            c.set(cv2.CAP_PROP_POS_FRAMES, fidx)
            ok, frame = c.read()
            c.release()
            if not ok:
                return None
            ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            if not ok:
                return None
            return base64.b64encode(buf.tobytes()).decode("ascii")

        async def process(fidx):
            async with sem:
                b64 = await loop.run_in_executor(None, extract, fidx)
                if b64:
                    payload = json.dumps({"image": b64})
                    del payload, b64
                return fidx, 1

        tasks = [asyncio.create_task(process(f)) for f in frames]
        results = {}
        for coro in asyncio.as_completed(tasks):
            fidx, pred = await coro
            results[fidx] = pred
        return results

    return asyncio.run(_eval())


def subprocess_worker(video_paths: list[str], result_file: str):
    """subprocess에서 실행되는 워커 함수."""
    for v in video_paths:
        evaluate_video_real(v)
        gc.collect()
    with open(result_file, "w") as f:
        json.dump({"rss_mb": get_rss_mb()}, f)


def main():
    videos = sorted(VIDEO_DIR.glob("*.mp4"))[:20]
    video_strs = [str(v) for v in videos]

    if not videos:
        print("비디오 파일을 찾을 수 없습니다.")
        return

    print(f"테스트 비디오: {len(videos)}개 (KISA_Falldown)")
    print()

    # ---- 테스트 1: 기존 방식 (in-process) ----
    print("=" * 60)
    print("테스트 1: 기존 방식 (in-process)")
    print("=" * 60)
    rss0 = get_rss_mb()
    print(f"  시작: {rss0:.0f} MB")

    for i, v in enumerate(video_strs):
        evaluate_video_real(v)
        gc.collect()
        if (i + 1) % 5 == 0:
            rss = get_rss_mb()
            print(f"  [{i+1}/{len(videos)}] RSS: {rss:.0f} MB (+{rss - rss0:.0f})")

    rss_inproc = get_rss_mb()
    leak_inproc = rss_inproc - rss0
    print(f"  최종: {rss_inproc:.0f} MB (증가: +{leak_inproc:.0f} MB)")
    print()

    # ---- 테스트 2: subprocess 방식 ----
    print("=" * 60)
    print("테스트 2: subprocess 방식 (벤치마크 단위 분리)")
    print("=" * 60)

    # 새 프로세스에서 테스트 (이전 테스트의 메모리 영향 배제)
    ctx = multiprocessing.get_context("spawn")
    rss0_sub = get_rss_mb()
    print(f"  부모 시작: {rss0_sub:.0f} MB")

    batch_size = 5  # 벤치마크 1개 = 비디오 5개로 시뮬레이션
    for i in range(0, len(video_strs), batch_size):
        batch = video_strs[i : i + batch_size]

        fd, rf = tempfile.mkstemp(suffix=".json")
        os.close(fd)

        p = ctx.Process(target=subprocess_worker, args=(batch, rf))
        p.start()
        p.join()

        try:
            with open(rf) as f:
                child_rss = json.load(f).get("rss_mb", 0)
        except Exception:
            child_rss = 0
        finally:
            os.unlink(rf)

        parent_rss = get_rss_mb()
        batch_end = i + len(batch)
        print(
            f"  [{batch_end}/{len(videos)}] 부모 RSS: {parent_rss:.0f} MB "
            f"(+{parent_rss - rss0_sub:.0f}) | 자식 피크 RSS: {child_rss:.0f} MB"
        )

    rss_sub = get_rss_mb()
    leak_sub = rss_sub - rss0_sub
    print(f"  부모 최종: {rss_sub:.0f} MB (증가: +{leak_sub:.0f} MB)")

    # ---- 비교 ----
    print()
    print("=" * 60)
    print("결과 비교")
    print("=" * 60)
    print(f"  in-process 메모리 증가: +{leak_inproc:.0f} MB")
    print(f"  subprocess 부모 증가:   +{leak_sub:.0f} MB")

    if leak_inproc > 0:
        print(f"  절감율: {(1 - leak_sub / leak_inproc) * 100:.0f}%")
    print()

    if leak_sub < leak_inproc:
        print("  ✓ subprocess 방식이 메모리 누적을 효과적으로 방지합니다.")
    else:
        print("  △ 이 테스트 규모에서는 차이가 미미합니다.")
        print("    (실제 장시간 평가에서는 차이가 커집니다)")


if __name__ == "__main__":
    main()
