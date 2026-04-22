"""
configs/vllm_pipeline/**/*.yaml와 configs/lmdeploy_pipeline/**/*.yaml 파일에
`pipeline.mode` 필드를 일괄 주입한다.

- 이미 `mode`가 있으면 skip (멱등)
- 텍스트 기반 라인 삽입 → 주석/들여쓰기 보존
- `--dry-run` 옵션으로 변경 대상만 미리 확인

사용법:
    python scripts/utils/add_pipeline_mode.py                # vllm + lmdeploy 자동 처리
    python scripts/utils/add_pipeline_mode.py --dry-run
    python scripts/utils/add_pipeline_mode.py --root configs/vllm_pipeline --mode vllm
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


MODE_BY_ROOT = {
    "configs/vllm_pipeline": "vllm",
    "configs/lmdeploy_pipeline": "lmdeploy",
}

PIPELINE_START_RE = re.compile(r"^pipeline\s*:\s*$")
# 최상위 키 (pipeline 블록의 끝을 판단). 2칸 들여쓰기의 하위 키와 구분 필요.
TOP_LEVEL_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*\s*:")
NAME_KEY_RE = re.compile(r"^(\s+)name\s*:")
MODE_KEY_RE = re.compile(r"^\s+mode\s*:")


def _iter_yaml_files(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*.yaml") if p.is_file())


def inject_mode(path: Path, mode: str, dry_run: bool) -> str:
    """YAML 파일에 `pipeline.mode: <mode>` 라인을 삽입. 상태 문자열 반환."""
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # pipeline: 블록 범위 찾기
    pipeline_start = None
    for i, line in enumerate(lines):
        if PIPELINE_START_RE.match(line):
            pipeline_start = i
            break
    if pipeline_start is None:
        return "skip (no pipeline block)"

    # 블록 끝 찾기 (다음 top-level key 또는 파일 끝)
    pipeline_end = len(lines)
    for j in range(pipeline_start + 1, len(lines)):
        s = lines[j]
        if s.strip() == "" or s.lstrip().startswith("#"):
            continue
        if TOP_LEVEL_KEY_RE.match(s):
            pipeline_end = j
            break

    block = lines[pipeline_start + 1 : pipeline_end]

    # 이미 mode가 있는지 검사
    for line in block:
        if MODE_KEY_RE.match(line):
            existing = line.split(":", 1)[1].strip().strip('"').strip("'")
            if existing == mode:
                return f"skip (already mode={existing})"
            return f"WARN existing mode={existing}, expected {mode}"

    # 삽입 위치: name: 라인 직후 (있으면), 없으면 pipeline: 바로 다음
    indent = "  "  # 기본 2칸
    insert_at = pipeline_start + 1

    for idx, line in enumerate(block):
        m = NAME_KEY_RE.match(line)
        if m:
            indent = m.group(1)
            insert_at = pipeline_start + 1 + idx + 1
            break
    else:
        # name 없음 → 블록 첫 키의 들여쓰기를 상속
        for line in block:
            if line.strip() and not line.lstrip().startswith("#"):
                m = re.match(r"^(\s+)", line)
                if m:
                    indent = m.group(1)
                break

    new_line = f'{indent}mode: "{mode}"\n'
    lines.insert(insert_at, new_line)

    if dry_run:
        return f"would insert mode={mode} at line {insert_at + 1}"

    with open(path, "w", encoding="utf-8") as f:
        f.writelines(lines)
    return f"inserted mode={mode} at line {insert_at + 1}"


def main() -> int:
    parser = argparse.ArgumentParser(description="YAML pipeline.mode 일괄 주입")
    parser.add_argument("--root", help="특정 폴더만 대상 (예: configs/vllm_pipeline)")
    parser.add_argument("--mode", choices=["vllm", "lmdeploy"], help="--root 사용 시 강제 지정")
    parser.add_argument("--dry-run", action="store_true", help="변경 없이 미리보기")
    args = parser.parse_args()

    if args.root:
        root = Path(args.root)
        resolved_mode = args.mode or MODE_BY_ROOT.get(str(root))
        if not resolved_mode:
            print(
                f"ERROR: --root={root}에 대한 기본 mode 추론 불가. --mode 명시 필요.",
                file=sys.stderr,
            )
            return 2
        targets = [(root, resolved_mode)]
    else:
        targets = [(Path(p), m) for p, m in MODE_BY_ROOT.items() if Path(p).exists()]

    if not targets:
        print("ERROR: 대상 폴더 없음", file=sys.stderr)
        return 2

    total = 0
    changed = 0
    warned = 0
    for root, mode in targets:
        print(f"\n[{root}] mode={mode}")
        for yaml_path in _iter_yaml_files(root):
            rel = yaml_path.relative_to(root.parent)
            total += 1
            result = inject_mode(yaml_path, mode, args.dry_run)
            if result.startswith("inserted") or result.startswith("would"):
                changed += 1
            elif result.startswith("WARN"):
                warned += 1
            print(f"  {rel}: {result}")

    print(f"\nSummary: total={total}, changed={changed}, warned={warned}, dry_run={args.dry_run}")
    return 0 if warned == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
