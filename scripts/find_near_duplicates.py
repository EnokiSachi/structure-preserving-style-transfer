import argparse
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from scipy.fft import dctn


def perceptual_hash(path):
    with Image.open(path) as source:
        gray = ImageOps.exif_transpose(source).convert("L")
        gray = gray.resize((32, 32), Image.Resampling.LANCZOS)
        pixels = np.asarray(gray, dtype=np.float64)
    coefficients = dctn(pixels, type=2, norm="ortho")[:8, :8].ravel()
    # Ignore the DC term, which mainly describes average brightness.
    bits = coefficients[1:] > np.median(coefficients[1:])
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", type=Path, required=True)
    args = parser.parse_args()
    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    if audit["errors"]:
        raise RuntimeError("Resolve unreadable images before continuing.")

    root = Path(audit["dataset_root"])
    groups = defaultdict(list)
    for record in audit["records"]:
        groups[record["pixel_sha256"]].append(record["path"])

    images = []
    for index, (digest, aliases) in enumerate(sorted(groups.items()), 1):
        aliases = sorted(aliases)
        representative = aliases[0]
        images.append({
            "pixel_sha256": digest,
            "path": representative,
            "aliases": aliases,
            "phash": perceptual_hash(root / representative),
        })
        if index % 1000 == 0:
            print(f"Hashed {index}/{len(groups)} unique images", flush=True)

    candidates = []
    for index, left in enumerate(images):
        for right in images[index + 1:]:
            distance = (left["phash"] ^ right["phash"]).bit_count()
            if distance <= 6:
                candidates.append({
                    "distance": distance,
                    "left": left["path"],
                    "right": right["path"],
                    "left_aliases": left["aliases"],
                    "right_aliases": right["aliases"],
                })

    candidates.sort(
        key=lambda item: (item["distance"], item["left"], item["right"])
    )
    repository = Path(__file__).resolve().parents[1]
    output = repository / "outputs" / (
        "near_duplicate_review_"
        + datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    )
    output.mkdir(parents=True, exist_ok=False)

    with (output / "candidates.json").open("x", encoding="utf-8") as stream:
        json.dump({
            "dataset_root": str(root),
            "source_audit": str(args.audit.resolve()),
            "method": "63-bit grayscale DCT perceptual hash",
            "maximum_hamming_distance": 6,
            "unique_images_checked": len(images),
            "manual_review_required": True,
            "candidates": candidates,
        }, stream, ensure_ascii=False, indent=2)

    print("Unique images checked:", len(images))
    print("Candidate pairs:", len(candidates))
    for candidate in candidates[:15]:
        print(
            f"distance={candidate['distance']}: "
            f"{candidate['left']} | {candidate['right']}"
        )
    print("Report:", output / "candidates.json")
    print("No source images were modified.")
    print("Candidates are not confirmed duplicates.")
    print("This screening may miss cropped or heavily edited copies.")


if __name__ == "__main__":
    main()
