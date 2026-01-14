import argparse
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path


MEDIA_EXTS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".gif",
    ".mp4",
    ".avi",
    ".mov",
    ".mkv",
    ".webm",
}
MISSING_JSON_LOG_LIMIT = 20


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _ensure_empty_or_missing(dir_path: Path) -> None:
    if dir_path.exists():
        has_any = any(dir_path.rglob("*"))
        if has_any:
            raise RuntimeError(f"Error: '{dir_path}' already exists and is not empty.")


def _collect_media_pairs(
    root: Path,
    train_dir: Path,
    test_dir: Path,
    max_missing_json: int,
    log_limit: int,
):
    groups = defaultdict(list)
    missing_json = []

    for media_path in root.rglob("*"):
        if not media_path.is_file():
            continue
        if media_path.suffix.lower() not in MEDIA_EXTS:
            continue
        if _is_within(media_path, train_dir) or _is_within(media_path, test_dir):
            continue

        json_path = media_path.with_suffix(".json")
        if not json_path.exists():
            missing_json.append(media_path)
            continue

        rel_dir = media_path.parent.relative_to(root)
        groups[rel_dir].append((media_path, json_path))

    if missing_json:
        missing_count = len(missing_json)
        sample = missing_json[:log_limit]
        sample_lines = [f"  - {p}" for p in sample]
        more = missing_count - len(sample)

        if missing_count > max_missing_json:
            lines = [
                f"Error: missing json for {missing_count} media files "
                f"(max allowed: {max_missing_json}).",
                "Sample:",
            ]
            lines.extend(sample_lines)
            if more > 0:
                lines.append(f"  ... and {more} more")
            raise RuntimeError("\n".join(lines))

        print(
            f"Warning: missing json for {missing_count} media files "
            f"(<= max allowed: {max_missing_json}). Skipping them."
        )
        if sample_lines:
            print("Sample:")
            print("\n".join(sample_lines))
            if more > 0:
                print(f"  ... and {more} more")

    return groups


def _plan_split(groups, test_ratio: float, seed: int):
    rng = random.Random(seed)
    splits = {}

    for rel_dir, pairs in groups.items():
        pairs_sorted = sorted(pairs, key=lambda x: x[0].name)
        rng.shuffle(pairs_sorted)

        num_test = int(round(len(pairs_sorted) * test_ratio))
        num_test = min(num_test, len(pairs_sorted))

        test_pairs = pairs_sorted[:num_test]
        train_pairs = pairs_sorted[num_test:]
        splits[rel_dir] = (train_pairs, test_pairs)

    return splits


def _move_file(src: Path, dst: Path) -> None:
    try:
        src.rename(dst)
    except OSError:
        shutil.move(str(src), str(dst))


def _execute_moves(moves):
    for src, dst in moves:
        dst.parent.mkdir(parents=True, exist_ok=True)
        _move_file(src, dst)


def _remove_empty_dirs(root: Path, keep_dirs):
    dirs = [p for p in root.rglob("*") if p.is_dir()]
    dirs_sorted = sorted(dirs, key=lambda p: len(p.parts), reverse=True)
    for dir_path in dirs_sorted:
        if any(_is_within(dir_path, k) for k in keep_dirs):
            continue
        try:
            dir_path.rmdir()
        except OSError:
            pass


def _remove_non_train_test(root: Path, keep_dirs):
    for entry in root.iterdir():
        if any(entry == k for k in keep_dirs):
            continue
        try:
            if entry.is_dir():
                shutil.rmtree(entry)
            else:
                entry.unlink()
        except OSError as exc:
            raise RuntimeError(f"Error: failed to remove '{entry}': {exc}") from exc


def _assert_only_train_test(root: Path, keep_dirs):
    leftovers = [p for p in root.iterdir() if not any(p == k for k in keep_dirs)]
    if leftovers:
        lines = ["Error: non train/test entries remain in input_dir:"]
        lines.extend([f"  - {p}" for p in leftovers])
        raise RuntimeError("\n".join(lines))


def _build_tree_counts(root: Path):
    counts = defaultdict(int)
    for media_path in root.rglob("*"):
        if media_path.is_file() and media_path.suffix.lower() in MEDIA_EXTS:
            for parent in media_path.parents:
                if _is_within(parent, root) or parent == root:
                    counts[parent] += 1
    return counts


def _print_tree(root: Path, counts):
    lines = [f"{root.name}/"]

    split_dirs = [root / "train", root / "test"]
    split_dirs = [d for d in split_dirs if d.exists()]

    def walk(dir_path: Path, prefix: str, children_override=None):
        if children_override is None:
            children = [d for d in dir_path.iterdir() if d.is_dir()]
        else:
            children = children_override
        children = sorted(children, key=lambda p: p.name)

        for idx, child in enumerate(children):
            is_last = idx == len(children) - 1
            connector = "└── " if is_last else "├── "
            count = counts.get(child, 0)
            suffix = f" ({count} pairs)" if count > 0 else ""
            lines.append(f"{prefix}{connector}{child.name}/{suffix}")
            extension = "    " if is_last else "│   "
            walk(child, prefix + extension)

    walk(root, "", children_override=split_dirs)
    return "\n".join(lines)


def split_folder(input_dir: str, test_ratio: float, seed: int, max_missing_json: int):
    root = Path(input_dir).resolve()
    if not root.exists() or not root.is_dir():
        raise RuntimeError(f"Error: input_dir not found or not a directory: {root}")

    if not (0.0 <= test_ratio <= 1.0):
        raise RuntimeError("Error: test_ratio must be between 0.0 and 1.0.")
    if max_missing_json < 0:
        raise RuntimeError("Error: max_missing_json must be >= 0.")

    train_dir = root / "train"
    test_dir = root / "test"

    _ensure_empty_or_missing(train_dir)
    _ensure_empty_or_missing(test_dir)

    groups = _collect_media_pairs(
        root, train_dir, test_dir, max_missing_json, MISSING_JSON_LOG_LIMIT
    )
    if not groups:
        raise RuntimeError("Error: no media files found to split.")

    splits = _plan_split(groups, test_ratio, seed)

    moves = []
    for rel_dir, (train_pairs, test_pairs) in splits.items():
        for media_path, json_path in train_pairs:
            dest_media = train_dir / rel_dir / media_path.name
            dest_json = train_dir / rel_dir / json_path.name
            moves.append((media_path, dest_media))
            moves.append((json_path, dest_json))
        for media_path, json_path in test_pairs:
            dest_media = test_dir / rel_dir / media_path.name
            dest_json = test_dir / rel_dir / json_path.name
            moves.append((media_path, dest_media))
            moves.append((json_path, dest_json))

    for _, dst in moves:
        if dst.exists():
            raise RuntimeError(f"Error: destination already exists: {dst}")

    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    _execute_moves(moves)
    _remove_non_train_test(root, [train_dir, test_dir])
    _remove_empty_dirs(root, [train_dir, test_dir])
    _assert_only_train_test(root, [train_dir, test_dir])

    total_pairs = sum(len(v[0]) + len(v[1]) for v in splits.values())
    total_test = sum(len(v[1]) for v in splits.values())
    total_train = sum(len(v[0]) for v in splits.values())

    print("\n--- Split Summary ---")
    print(f"Input: {root}")
    print(f"Total pairs: {total_pairs}")
    print(f"Train pairs: {total_train}")
    print(f"Test pairs: {total_test}")
    if total_pairs > 0:
        ratio = total_test / total_pairs
        print(f"Actual test ratio: {ratio:.4f}")

    print("\n--- Per-Folder Summary ---")
    for rel_dir in sorted(splits.keys(), key=lambda p: str(p)):
        train_pairs, test_pairs = splits[rel_dir]
        total = len(train_pairs) + len(test_pairs)
        label = str(rel_dir) if str(rel_dir) != "." else "."
        print(f"- {label}: total {total}, train {len(train_pairs)}, test {len(test_pairs)}")

    counts = _build_tree_counts(root)
    print("\n--- Result Tree ---")
    print(_print_tree(root, counts))


def main():
    parser = argparse.ArgumentParser(
        description="Split a folder into train/test by moving paired media+json files."
    )
    parser.add_argument(
        "-i",
        "--input-dir",
        required=True,
        help="Root folder to split (train/test will be created here).",
    )
    parser.add_argument(
        "-r",
        "--ratio",
        type=float,
        default=0.1,
        help="Test split ratio (default: 0.1).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible splits (default: 42).",
    )
    parser.add_argument(
        "--max-missing-json",
        type=int,
        default=0,
        help="Allow up to N media files without json before abort (default: 0).",
    )

    args = parser.parse_args()
    try:
        split_folder(args.input_dir, args.ratio, args.seed, args.max_missing_json)
    except RuntimeError as exc:
        print(str(exc))
        sys.exit(1)


if __name__ == "__main__":
    main()
