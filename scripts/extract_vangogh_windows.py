"""Reproduce Windows-safe numbering without changing archived image bytes."""
import argparse
import hashlib
import json
import shutil
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath


EXPECTED_SHA256 = "362c3473e70be0944bda08d49d660eafdc2854df55d697835b8186d1d79caaff"


def archive_digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def extract(archive, target):
    if target.exists():
        raise FileExistsError(f"Target already exists; nothing overwritten: {target}")
    digest = archive_digest(archive)
    if digest != EXPECTED_SHA256:
        raise ValueError("Archive differs from the locally audited dataset version")
    counts, records = Counter(), []
    with zipfile.ZipFile(archive) as zf:
        entries = []
        for info in zf.infolist():
            if info.is_dir():
                continue
            parts = PurePosixPath(info.filename).parts
            if (len(parts) != 3 or parts[0] != "vangogh2photo" or
                    parts[1] not in {"trainA", "trainB", "testA", "testB"} or
                    PurePosixPath(parts[2]).suffix.lower() not in
                    {".jpg", ".jpeg", ".png", ".bmp", ".webp"}):
                raise ValueError(f"Unexpected archive entry: {info.filename}")
            entries.append(info)
        if not entries:
            raise ValueError("Archive has no image entries")
        target.mkdir(parents=True, exist_ok=False)
        for info in sorted(entries, key=lambda item: item.filename):
            source = PurePosixPath(info.filename)
            split = source.parts[1]
            counts[split] += 1
            relative = Path(split) / f"{counts[split]:06d}{source.suffix.lower()}"
            destination = target / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as reader, destination.open("xb") as writer:
                shutil.copyfileobj(reader, writer)
            if destination.stat().st_size != info.file_size:
                raise ValueError(f"Extracted size differs: {destination}")
            records.append({"original_archive_path": info.filename,
                            "extracted_path": relative.as_posix(),
                            "bytes": info.file_size,
                            "zip_crc32": f"{info.CRC:08x}"})
    with (target / "extraction_manifest.json").open("x", encoding="utf-8") as stream:
        json.dump({"archive_name": archive.name, "archive_sha256": digest,
                   "files": records}, stream, ensure_ascii=False, indent=2)
    for split, count in sorted(counts.items()):
        print(f"{split}: {count} images")
    print("Saved:", target.resolve())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    args = parser.parse_args()
    extract(args.archive, args.target)


if __name__ == "__main__":
    main()
