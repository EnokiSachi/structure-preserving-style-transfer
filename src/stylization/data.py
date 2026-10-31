"""Reproducible unpaired sampling for training only."""

import random
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def list_images(directory):
    directory = Path(directory).resolve()
    if not directory.is_dir():
        raise FileNotFoundError(f"Image directory not found: {directory}")

    files = sorted(
        (
            path for path in directory.iterdir()
            if path.is_file()
            and path.suffix.lower() in IMAGE_EXTENSIONS
        ),
        key=lambda path: path.name,
    )

    if not files:
        raise ValueError(f"No supported images found in: {directory}")

    return files


class UnpairedTrainingDataset(Dataset):
    """Visit every content image and sample one style for each item."""

    def __init__(
        self,
        content_dir,
        style_dir,
        image_size=256,
        seed=42,
    ):
        if image_size <= 0 or image_size % 4 != 0:
            raise ValueError("image_size must be positive and divisible by 4")

        self.content_files = list_images(content_dir)
        self.style_files = list_images(style_dir)
        self.seed = int(seed)
        self.epoch = 0

        # Keep the original resize policy for the corrected baseline.
        # Random augmentation will be configured separately.
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.5, 0.5, 0.5),
                (0.5, 0.5, 0.5),
            ),
        ])

    def __len__(self):
        return len(self.content_files)

    def set_epoch(self, epoch):
        if epoch < 0:
            raise ValueError("epoch must be non-negative")
        self.epoch = int(epoch)

    def style_index(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)

        rng = random.Random(f"{self.seed}:{self.epoch}:{index}")
        return rng.randrange(len(self.style_files))

    def read_image(self, path):
        with Image.open(path) as image:
            return self.transform(image.convert("RGB"))

    def __getitem__(self, index):
        content_path = self.content_files[index]
        style_path = self.style_files[self.style_index(index)]

        return {
            "content": self.read_image(content_path),
            "style": self.read_image(style_path),
            "content_path": str(content_path),
            "style_path": str(style_path),
        }
