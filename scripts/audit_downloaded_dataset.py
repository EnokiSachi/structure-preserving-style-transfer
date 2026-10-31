import argparse
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

from PIL import Image, ImageOps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    splits = ("trainA", "trainB", "testA", "testB")
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    for split in splits:
        if not (root / split).is_dir():
            raise FileNotFoundError(root / split)

    records, errors = [], []
    groups = defaultdict(list)
    dimensions = Counter()
    counts = Counter()

    for split in splits:
        paths = sorted(
            p for p in (root / split).rglob("*")
            if p.is_file() and p.suffix.lower() in extensions
        )
        for path in paths:
            relative = path.relative_to(root).as_posix()
            try:
                with Image.open(path) as source:
                    source.load()
                    rgb = ImageOps.exif_transpose(source).convert("RGB")
                    width, height = rgb.size
                    digest = hashlib.sha256()
                    digest.update(f"{width}x{height}:".encode("ascii"))
                    digest.update(rgb.tobytes())

                pixel_hash = digest.hexdigest()
                records.append({
                    "path": relative,
                    "split": split,
                    "width": width,
                    "height": height,
                    "pixel_sha256": pixel_hash,
                })
                groups[pixel_hash].append(relative)
                dimensions[f"{width}x{height}"] += 1
                counts[split] += 1
            except Exception as exc:
                errors.append({"path": relative, "error": str(exc)})

        print(f"{split}: {counts[split]} readable / {len(paths)} files",
              flush=True)

    duplicates = [paths for paths in groups.values() if len(paths) > 1]
    overlaps = [
        paths for paths in duplicates
        if any(p.startswith("train") for p in paths)
        and any(p.startswith("test") for p in paths)
    ]

    repository = Path(__file__).resolve().parents[1]
    output = repository / "outputs" / (
        "dataset_audit_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    )
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "dataset_root": str(root),
        "readable_counts": dict(counts),
        "dimensions": dict(dimensions),
        "records": records,
        "errors": errors,
        "exact_duplicate_groups": duplicates,
        "train_test_overlap_groups": overlaps,
        "near_duplicates_checked": False,
    }
    with (output / "audit.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)

    print("Unreadable images:", len(errors))
    print("Exact duplicate groups:", len(duplicates))
    print("Train/test exact overlap groups:", len(overlaps))
    print("Most common dimensions:", dimensions.most_common(5))
    for group in overlaps[:10]:
        print("Overlap:", " | ".join(group))
    for error in errors[:10]:
        print("Read error:", error)
    print("Report:", output / "audit.json")
    print("No source images were modified.")
    print("Resized, cropped and recompressed duplicates need further review.")


if __name__ == "__main__":
    main()
