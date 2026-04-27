"""
model_and_combiner.py

두 모델의 프레임별 예측 CSV를 AND 연산으로 결합하는 모듈.

벤치마크 폴더 구조:
    {BENCH_BASE_PATH}/{bench_name}/models/{model_name}/CFG/*/alarm/{category}/{video}.csv

사용법:
    python src/evaluation/model_and_combiner.py \\
        --bench_name PIA_Falldown \\
        --m1 FalldownCls_v3.0.0 \\
        --m2 Blue-VLM-TF_PE-Core-L14-336_Zero-shot_APOv1 \\
        --output_path /path/to/results

    # 여러 벤치마크 한번에 처리
    python src/evaluation/model_and_combiner.py \\
        --bench_name PIA_Falldown Kumho_Falldown \\
        --m1 FalldownCls_v3.0.0 \\
        --m2 Blue-VLM-TF_PE-Core-L14-336_Zero-shot_APOv1 \\
        --output_path /path/to/results
"""

import argparse
from pathlib import Path

import pandas as pd


BENCH_BASE_PATH = "/mnt/PoC_benchmark/huggingface_benchmarks_dataset/Leaderboard_bench"

EXCLUDE_DIRS = {"@eaDir", "huggingface_benchmarks_original_dataset"}


# ============================================================
# CSV 탐색
# ============================================================

def find_alarm_csvs(bench_base: Path, bench_name: str, model_name: str) -> dict[str, Path]:
    """
    모델의 alarm CSV를 탐색하여 {파일명(확장자 제외): Path} 딕셔너리로 반환.

    경로 패턴: {bench_base}/{bench_name}/models/{model_name}/CFG/*/alarm/{category}/*.csv
    cfg_name이 모델마다 다르므로 glob(*) 으로 탐색.
    """
    model_dir = bench_base / bench_name / "models" / model_name
    if not model_dir.exists():
        raise FileNotFoundError(f"모델 폴더를 찾을 수 없습니다: {model_dir}")

    csv_map: dict[str, Path] = {}
    alarm_csvs = list(model_dir.glob("CFG/*/alarm/*/*.csv"))

    if not alarm_csvs:
        raise FileNotFoundError(
            f"alarm CSV를 찾을 수 없습니다: {model_dir}/CFG/*/alarm/*/*.csv"
        )

    for csv_path in alarm_csvs:
        csv_map[csv_path.stem] = csv_path

    return csv_map


# ============================================================
# AND 연산
# ============================================================

def and_combine(m1_path: Path, m2_path: Path) -> pd.DataFrame:
    """
    두 alarm CSV를 AND 연산으로 결합.

    - frame 컬럼 기준으로 merge (outer join, 없는 프레임은 0 처리)
    - 카테고리 컬럼에 대해 element-wise AND (1 & 1 = 1, 나머지 = 0)
    - 결과는 frame 오름차순 정렬
    """
    df1 = pd.read_csv(m1_path)
    df2 = pd.read_csv(m2_path)

    category_cols = [c for c in df1.columns if c != "frame"]

    merged = pd.merge(df1, df2, on="frame", how="outer", suffixes=("_m1", "_m2"))
    merged = merged.sort_values("frame").reset_index(drop=True)

    result = pd.DataFrame()
    result["frame"] = merged["frame"].astype(int)

    for col in category_cols:
        col_m1 = f"{col}_m1" if f"{col}_m1" in merged.columns else col
        col_m2 = f"{col}_m2" if f"{col}_m2" in merged.columns else col

        s1 = merged[col_m1].fillna(0).astype(int)
        s2 = merged[col_m2].fillna(0).astype(int)
        result[col] = (s1 & s2).astype(int)

    return result


# ============================================================
# 메인 처리
# ============================================================

def run_and_combine(
    bench_name: str,
    m1: str,
    m2: str,
    output_path: str,
    bench_base: str = BENCH_BASE_PATH,
) -> None:
    """
    단일 벤치마크에 대해 M1, M2 alarm CSV를 AND 결합하고 저장.

    출력 경로: {output_path}/{m1}_{m2}_AND/{bench_name}/
    """
    base = Path(bench_base)
    out_dir = Path(output_path) / f"{m1}_{m2}_AND" / bench_name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[{bench_name}] AND 결합 시작")
    print(f"  M1: {m1}")
    print(f"  M2: {m2}")
    print(f"  출력: {out_dir}")

    m1_csvs = find_alarm_csvs(base, bench_name, m1)
    m2_csvs = find_alarm_csvs(base, bench_name, m2)

    common = sorted(set(m1_csvs.keys()) & set(m2_csvs.keys()))
    only_m1 = set(m1_csvs.keys()) - set(m2_csvs.keys())
    only_m2 = set(m2_csvs.keys()) - set(m1_csvs.keys())

    if only_m1:
        print(f"  [경고] M1에만 있는 파일 {len(only_m1)}개 → 스킵")
    if only_m2:
        print(f"  [경고] M2에만 있는 파일 {len(only_m2)}개 → 스킵")
    print(f"  매칭된 CSV: {len(common)}개")

    for stem in common:
        result_df = and_combine(m1_csvs[stem], m2_csvs[stem])
        out_file = out_dir / f"{stem}.csv"
        result_df.to_csv(out_file, index=False)

    print(f"  완료: {len(common)}개 저장 → {out_dir}")


# ============================================================
# CLI
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="두 모델의 alarm CSV를 AND 연산으로 결합합니다."
    )
    parser.add_argument(
        "--bench_name",
        nargs="+",
        required=True,
        help="처리할 벤치마크 이름 (여러 개 가능, 예: PIA_Falldown Kumho_Falldown)",
    )
    parser.add_argument("--m1", required=True, help="첫 번째 모델 이름")
    parser.add_argument("--m2", required=True, help="두 번째 모델 이름")
    parser.add_argument("--output_path", required=True, help="결과 저장 최상위 경로")
    parser.add_argument(
        "--bench_base",
        default=BENCH_BASE_PATH,
        help=f"벤치마크 루트 경로 (기본값: {BENCH_BASE_PATH})",
    )
    args = parser.parse_args()

    skipped: list[str] = []
    for bench in args.bench_name:
        try:
            run_and_combine(
                bench_name=bench,
                m1=args.m1,
                m2=args.m2,
                output_path=args.output_path,
                bench_base=args.bench_base,
            )
        except FileNotFoundError as e:
            skipped.append(f"{bench}: {e}")

    if skipped:
        print(f"\n[스킵 목록] 총 {len(skipped)}개")
        for s in skipped:
            print(f"  - {s}")


if __name__ == "__main__":
    main()
