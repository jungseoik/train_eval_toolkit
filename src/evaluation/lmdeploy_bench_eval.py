"""
lmdeploy_bench_eval.py

LMDeploy 서버를 이용한 프레임 단위 벤치마크 평가 모듈.
파인튜닝 완료된 InternVL3 계열 로컬 모델의 최종 벤치마크 평가 전용.
(테스트셋 평가 Precision/Recall/F1 과는 별개 프로세스)

LMDeploy는 OpenAI 호환 API(/v1/chat/completions)를 제공하므로
vllm_bench_eval.py와 동일한 추론 로직을 사용.

사용법:
    python src/evaluation/lmdeploy_bench_eval.py --config configs/lmdeploy_eval/config.py

    # 특정 벤치마크만 평가
    python src/evaluation/lmdeploy_bench_eval.py --config configs/lmdeploy_eval/config.py \
        --benchmarks Innodep_Falldown KhonKaen_Smoke
"""

import argparse
import asyncio
import base64
import csv
import importlib.util
import json
import re
import sys
import time
import warnings
from pathlib import Path
from types import ModuleType
from typing import Optional

import cv2
import httpx


# ============================================================
# 설정 로딩
# ============================================================

def load_config(config_path: str) -> ModuleType:
    """config.py 파일을 동적으로 로드하여 모듈 객체로 반환."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    spec = importlib.util.spec_from_file_location("lmdeploy_eval_config", path)
    cfg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cfg)
    return cfg


# ============================================================
# 벤치마크 데이터 탐색
# ============================================================

def get_category_from_bench(bench_name: str) -> str:
    """벤치마크 폴더명에서 카테고리 추출. 예: Innodep_Falldown -> falldown"""
    parts = bench_name.split("_")
    return parts[-1].lower() if len(parts) >= 2 else bench_name.lower()


def find_video_gt_pairs(bench_path: Path, category: str) -> list[tuple[Path, Path]]:
    """
    dataset/{category}/ 아래의 (mp4, csv) 쌍을 모두 반환.
    @eaDir 같은 시스템 폴더는 자동으로 건너뜀.
    """
    dataset_dir = bench_path / "dataset" / category
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    VIDEO_EXTENSIONS = (".mp4", ".MP4", ".mov", ".MOV", ".avi", ".AVI", ".mkv", ".MKV")

    pairs: list[tuple[Path, Path]] = []
    for csv_file in sorted(dataset_dir.glob("*.csv")):
        video_file = None
        for ext in VIDEO_EXTENSIONS:
            candidate = csv_file.with_suffix(ext)
            if candidate.exists():
                video_file = candidate
                break
        if video_file is not None:
            pairs.append((video_file, csv_file))

    return pairs


# ============================================================
# 프레임 추출 / 인코딩 (동기, 스레드 풀에서 실행됨)
# ============================================================

def _extract_frame_jpeg_sync(video_path: Path, frame_idx: int, jpeg_quality: int) -> Optional[str]:
    """
    비디오에서 frame_idx 번 프레임을 읽어 JPEG base64 문자열로 반환.
    실패 시 None 반환.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        return None
    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
    if not ok:
        return None
    return base64.b64encode(buf.tobytes()).decode("ascii")


# ============================================================
# LMDeploy 추론 (OpenAI 호환 API)
# ============================================================

async def _infer_frame(
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    api_url: str,
    model: str,
    image_b64: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    seed: int,
) -> str:
    """이미지 1장을 LMDeploy 서버로 보내고 응답 텍스트를 반환."""
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "seed": seed,
    }
    async with semaphore:
        resp = await client.post(api_url, json=payload)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]


# ============================================================
# 파서
# 두 단계로 분리: parse_model_output -> classify
# ============================================================

def parse_model_output(raw_text: str, valid_values: list[str]) -> str:
    """
    모델 raw 출력(JSON / 마크다운 코드블록 / 일반 텍스트)에서
    카테고리 값을 추출하여 반환.

    valid_values: 허용되는 값 목록 (예: ["falldown", "normal"])

    반환값:
      - valid_values 중 하나 (파싱 성공)
      - "no_json_found"   (JSON 구조를 찾았으나 category 키 없음)
      - "parsing_failed"  (예상치 못한 에러)
    """
    try:
        # 1단계: 마크다운 코드블록 제거
        clean = raw_text
        if "```json" in clean:
            clean = clean.split("```json")[1].split("```")[0]
        elif "```" in clean:
            clean = clean.split("```")[1].split("```")[0]
        clean = clean.strip()

        # 2단계: JSON 객체 추출 및 파싱
        start = clean.find("{")
        end = clean.rfind("}")
        if start != -1 and end != -1 and start < end:
            json_part = clean[start: end + 1]
            try:
                data = json.loads(json_part)
                category = data.get("category")
                if category in valid_values:
                    return category
            except json.JSONDecodeError:
                pass

        # 3단계: 정규식 fallback -- "category": "value" 패턴 직접 탐색
        pattern = r'["\']category["\']\s*:\s*["\'](' + "|".join(re.escape(v) for v in valid_values) + r')["\']'
        m = re.search(pattern, clean)
        if m:
            return m.group(1)

        return "no_json_found"

    except Exception:
        return "parsing_failed"


def classify(parsed: str, positive_label: str) -> int:
    """
    parse_model_output() 의 반환값을 받아 0 또는 1로 판별.

    parsed == positive_label 이면 1, 그 외 모두 0.
    (미감지 레이블, no_json_found, parsing_failed 모두 0 처리)
    """
    return 1 if parsed == positive_label else 0


# ============================================================
# 단일 비디오 평가
# ============================================================

async def _evaluate_video_async(
    video_path: Path,
    gt_csv_path: Path,
    category: str,
    valid_values: list[str],
    cfg: ModuleType,
    video_label: str,
) -> dict[int, int]:
    """
    비디오 1개를 프레임 단위로 추론.

    반환: {frame_idx: pred} 전체 프레임 예측값 딕셔너리
    """
    # GT에서 total_frames 파악
    gt: dict[int, int] = {}
    with open(gt_csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if category not in (reader.fieldnames or []):
            raise ValueError(
                f"GT CSV does not have column '{category}': {gt_csv_path}"
            )
        for row in reader:
            gt[int(row["frame"])] = int(row[category])

    total_frames = max(gt.keys()) + 1
    sampled_frames = list(range(0, total_frames, cfg.WINDOW_SIZE))

    # 프롬프트: 카테고리별 템플릿 조회, 없으면 "default" fallback
    template = cfg.PROMPT_TEMPLATES.get(category, cfg.PROMPT_TEMPLATES["default"])
    prompt = template.format(category=category) if "{category}" in template else template

    api_url = cfg.API_BASE.rstrip("/") + "/chat/completions"
    semaphore = asyncio.Semaphore(cfg.CONCURRENCY)
    limits = httpx.Limits(
        max_connections=cfg.CONCURRENCY,
        max_keepalive_connections=cfg.CONCURRENCY,
    )

    sampled_preds: dict[int, int] = {}
    loop = asyncio.get_event_loop()

    async def process_frame(fidx: int) -> tuple[int, int]:
        b64 = await loop.run_in_executor(
            None, _extract_frame_jpeg_sync, video_path, fidx, cfg.JPEG_QUALITY
        )
        if b64 is None:
            warnings.warn(f"Frame extraction failed: {video_path.name} frame={fidx}")
            return fidx, 0
        try:
            raw_text = await _infer_frame(
                client, semaphore, api_url,
                cfg.MODEL, b64, prompt,
                cfg.MAX_TOKENS, cfg.TEMPERATURE, cfg.SEED,
            )
            pred = classify(parse_model_output(raw_text, valid_values), category)
        except Exception as exc:
            warnings.warn(f"Inference failed for {video_path.name} frame={fidx}: {exc}")
            pred = 0
        return fidx, pred

    async with httpx.AsyncClient(limits=limits, timeout=300.0) as client:
        tasks = [asyncio.create_task(process_frame(fidx)) for fidx in sampled_frames]
        done = 0
        for coro in asyncio.as_completed(tasks):
            fidx, pred = await coro
            sampled_preds[fidx] = pred
            done += 1
            print(
                f"    {video_label}  frame {fidx:5d}/{total_frames-1}"
                f"  pred={pred}  [{done}/{len(sampled_frames)}]",
                end="\r",
            )

    print()

    # --------------------------------------------------------
    # 인터폴레이션 -> 전체 프레임 예측값 생성
    # --------------------------------------------------------
    all_preds: dict[int, int] = {}
    sorted_sampled = sorted(sampled_preds.keys())

    if cfg.INTERPOLATION == "forward":
        for i, fidx in enumerate(sorted_sampled):
            end = sorted_sampled[i + 1] if i + 1 < len(sorted_sampled) else total_frames
            for f in range(fidx, end):
                all_preds[f] = sampled_preds[fidx]
    elif cfg.INTERPOLATION == "backward":
        prev_end = 0
        for fidx in sorted_sampled:
            for f in range(prev_end, fidx):
                all_preds[f] = 0
            all_preds[fidx] = sampled_preds[fidx]
            prev_end = fidx + 1
        if sorted_sampled:
            last_pred = sampled_preds[sorted_sampled[-1]]
            for f in range(prev_end, total_frames):
                all_preds[f] = last_pred
    else:
        raise ValueError(f"Unknown INTERPOLATION: {cfg.INTERPOLATION!r}. Use 'forward' or 'backward'.")

    return all_preds


# ============================================================
# 단일 벤치마크 평가
# ============================================================

def evaluate_benchmark(bench_name: str, cfg: ModuleType) -> None:
    """
    벤치마크 1개의 모든 비디오를 순차 평가하고
    {OUTPUT_PATH}/{RUN_NAME}/{BenchmarkName}/{video_stem}.csv 에 결과를 저장.

    CSV 포맷:
        frame,{category}
        0,0
        1,0
        2,1
        ...
    """
    bench_path = Path(cfg.BENCH_BASE_PATH) / bench_name
    if not bench_path.exists():
        print(f"[SKIP] Benchmark path not found: {bench_path}")
        return

    category = get_category_from_bench(bench_name)

    print(f"\n{'='*60}")
    print(f"Benchmark : {bench_name}")
    print(f"Category  : {category}")
    print(f"Window    : every {cfg.WINDOW_SIZE} frames  ({cfg.INTERPOLATION} fill)")
    print(f"{'='*60}")

    try:
        pairs = find_video_gt_pairs(bench_path, category)
    except FileNotFoundError as exc:
        print(f"[ERROR] {exc}")
        return

    if not pairs:
        print("[SKIP] No (mp4, csv) pairs found.")
        return

    print(f"Videos    : {len(pairs)}")

    # 파서에 전달할 허용값: [감지 카테고리, 미감지 레이블]
    valid_values = [category, cfg.NEGATIVE_LABEL]

    output_dir = Path(cfg.OUTPUT_PATH) / cfg.RUN_NAME / bench_name
    output_dir.mkdir(parents=True, exist_ok=True)

    for i, (video_path, gt_csv_path) in enumerate(pairs):
        label = f"[{i+1}/{len(pairs)}] {video_path.name[:50]:<50}"
        print(f"\n  {label}")
        t_start = time.perf_counter()

        try:
            all_preds = asyncio.run(
                _evaluate_video_async(video_path, gt_csv_path, category, valid_values, cfg, label)
            )
        except Exception as exc:
            print(f"  [ERROR] {exc}")
            continue

        elapsed = time.perf_counter() - t_start
        total_frames = max(all_preds.keys()) + 1
        print(f"  elapsed: {elapsed:.1f}s")

        # CSV 저장: frame,{category}
        output_csv = output_dir / f"{video_path.stem}.csv"
        with open(output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["frame", category])
            for frame_idx in range(total_frames):
                writer.writerow([frame_idx, all_preds.get(frame_idx, 0)])

        print(f"  Saved -> {output_csv}")

    print(f"\n  Output dir -> {output_dir}")


# ============================================================
# 진입점
# ============================================================

def _resolve_config_path(raw: str) -> Path:
    """절대경로면 그대로, 상대경로면 레포 루트 기준으로 탐색."""
    p = Path(raw)
    if p.is_absolute():
        return p
    repo_root = Path(__file__).parent.parent.parent
    candidate = repo_root / p
    if candidate.exists():
        return candidate
    cwd_candidate = Path.cwd() / p
    if cwd_candidate.exists():
        return cwd_candidate
    return p


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LMDeploy 서버를 이용한 프레임 단위 벤치마크 평가"
    )
    parser.add_argument(
        "--config",
        default="configs/lmdeploy_eval/config.py",
        help="config.py 경로 (절대 또는 레포 루트 기준 상대경로)",
    )
    parser.add_argument(
        "--benchmarks",
        nargs="*",
        metavar="BENCH",
        help="평가할 벤치마크 이름 (미지정 시 config.BENCHMARKS 사용)",
    )
    args = parser.parse_args()

    config_path = _resolve_config_path(args.config)
    try:
        cfg = load_config(str(config_path))
    except FileNotFoundError as exc:
        print(f"[ERROR] {exc}")
        sys.exit(1)

    benchmarks: list[str] = args.benchmarks or cfg.BENCHMARKS

    print("=" * 60)
    print("LMDeploy Benchmark Evaluation")
    print("=" * 60)
    print(f"Config      : {config_path}")
    print(f"API         : {cfg.API_BASE}  model={cfg.MODEL}")
    print(f"Benchmarks  : {benchmarks}")
    print(f"Output      : {cfg.OUTPUT_PATH}/{cfg.RUN_NAME}")
    print(f"Window      : {cfg.WINDOW_SIZE} frames  interpolation={cfg.INTERPOLATION}")
    print(f"Concurrency : {cfg.CONCURRENCY}")

    wall_start = time.perf_counter()
    for bench_name in benchmarks:
        evaluate_benchmark(bench_name, cfg)

    total_elapsed = time.perf_counter() - wall_start
    print(f"\nAll done. Total elapsed: {total_elapsed:.1f}s")


if __name__ == "__main__":
    main()
