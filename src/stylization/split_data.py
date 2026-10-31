"""Manifest-based datasets for separated training and evaluation."""

import hashlib
import json
import random
from pathlib import Path

from PIL import Image, ImageOps
from torch.utils.data import Dataset
from torchvision import transforms


def checked_paths(records, root):
    if not records:
        raise ValueError("Image records must not be empty")

    paths = []
    for record in records:
        path = (root / record["path"]).resolve()

        if not path.is_relative_to(root):
            raise ValueError(f"Image path is outside the repository: {path}")

        if not path.is_file():
            raise FileNotFoundError(path)

        with Image.open(path) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
            digest = hashlib.sha256()
            digest.update(str(image.size).encode("ascii"))
            digest.update(image.tobytes())

        if digest.hexdigest() != record["pixel_sha256"]:
            raise ValueError(f"Image has changed since splitting: {path}")

        paths.append(path)

    return paths


class ManifestDataset(Dataset):
    def __init__(
        self,
        manifest_path,
        repository_root,
        split,
        image_size=256,
        seed=42,
    ):
        if split not in {"train", "validation", "test"}:
            raise ValueError(f"Unknown split: {split}")
        if image_size <= 0 or image_size % 4 != 0:
            raise ValueError("image_size must be positive and divisible by 4")

        root = Path(repository_root).resolve()
        with Path(manifest_path).open(encoding="utf-8") as stream:
            manifest = json.load(stream)

        if manifest.get("schema_version") != 1:
            raise ValueError("Unsupported manifest schema")
        if manifest.get("path_base") != "repository_root":
            raise ValueError("Unsupported path base")

        # Reject exact image leakage recorded in the manifest.
        groups = {}
        for name in ("train", "validation", "test"):
            section = manifest["splits"][name]
            groups[name] = {
                item["pixel_sha256"]
                for role in ("content", "style")
                for item in section[role]
            }

        for first, second in (
            ("train", "validation"),
            ("train", "test"),
            ("validation", "test"),
        ):
            if groups[first] & groups[second]:
                raise ValueError(f"Split overlap: {first}, {second}")

        section = manifest["splits"][split]
        self.content_files = checked_paths(section["content"], root)
        self.style_files = checked_paths(section["style"], root)
        self.split = split
        self.seed = int(seed)
        self.epoch = 0

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,) * 3, (0.5,) * 3),
        ])

    def set_epoch(self, epoch):
        if self.split != "train":
            raise ValueError("Evaluation pairing must remain fixed")
        if epoch < 0:
            raise ValueError("epoch must be non-negative")
        self.epoch = int(epoch)

    def __len__(self):
        if self.split == "train":
            return len(self.content_files)
        return len(self.content_files) * len(self.style_files)

    def pair_paths(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)

        if self.split == "train":
            content_index = index
            rng = random.Random(f"{self.seed}:{self.epoch}:{index}")
            style_index = rng.randrange(len(self.style_files))
        else:
            content_index, style_index = divmod(
                index, len(self.style_files)
            )

        return (
            self.content_files[content_index],
            self.style_files[style_index],
        )

    def read_image(self, path):
        with Image.open(path) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
            return self.transform(image)

    def __getitem__(self, index):
        content_path, style_path = self.pair_paths(index)
        return {
            "content": self.read_image(content_path),
            "style": self.read_image(style_path),
            "content_path": str(content_path),
            "style_path": str(style_path),
        }
