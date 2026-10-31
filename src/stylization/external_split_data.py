"""Datasets for schema-2 manifests with an explicitly supplied image root."""
import hashlib
import json
from pathlib import Path, PurePosixPath

from PIL import Image, ImageOps
from torchvision import transforms

from stylization.split_data import ManifestDataset


HASH_METHOD = "SHA256(width + 'x' + height + ':' + EXIF-transposed RGB bytes)"


def validate_manifest(manifest):
    if manifest.get("schema_version") != 2:
        raise ValueError("Expected manifest schema 2")
    if manifest.get("path_base") != "external_dataset_root":
        raise ValueError("Expected external_dataset_root path base")
    if manifest.get("pixel_hash_method") != HASH_METHOD:
        raise ValueError("Unsupported pixel fingerprint method")
    if set(manifest["splits"]) != {"train", "validation", "test"}:
        raise ValueError("Expected train, validation and test splits")
    hashes, paths, groups = set(), set(), {}
    for split, section in manifest["splits"].items():
        if set(section) != {"content", "style"}:
            raise ValueError("Expected content and style sections")
        for role, records in section.items():
            if not records:
                raise ValueError(f"Empty image section: {split}/{role}")
            for record in records:
                path = record["path"]
                parts = PurePosixPath(path).parts
                allowed = {"trainB", "testB"} if role == "content" else {"trainA", "testA"}
                if (len(parts) != 2 or parts[0] not in allowed or
                        "\\" in path or parts[1] in {".", ".."} or
                        ":" in path):
                    raise ValueError(f"Invalid image path: {path}")
                digest, group = record["pixel_sha256"], record["group_id"]
                for value in (digest, group):
                    if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                        raise ValueError("Malformed image or group fingerprint")
                if path in paths or digest in hashes:
                    raise ValueError("Duplicate image in manifest")
                paths.add(path)
                hashes.add(digest)
                if groups.setdefault(group, (split, role)) != (split, role):
                    raise ValueError("Related-image group crosses splits or roles")
                if record["width"] <= 0 or record["height"] <= 0:
                    raise ValueError("Invalid image dimensions")
    if paths & {r["path"] for r in manifest.get("excluded", [])}:
        raise ValueError("An excluded image was selected")


def checked_external_paths(records, root):
    paths = []
    for record in records:
        path = (root / record["path"]).resolve()
        if not path.is_relative_to(root):
            raise ValueError(f"Image escapes dataset root: {path}")
        with Image.open(path) as source:
            source.load()
            image = ImageOps.exif_transpose(source).convert("RGB")
            width, height = image.size
            if (width, height) != (record["width"], record["height"]):
                raise ValueError(f"Image dimensions changed: {path}")
            digest = hashlib.sha256(f"{width}x{height}:".encode("ascii"))
            digest.update(image.tobytes())
        if digest.hexdigest() != record["pixel_sha256"]:
            raise ValueError(f"Image fingerprint changed: {path}")
        paths.append(path)
    return paths


class ExternalManifestDataset(ManifestDataset):
    """Reuse debug dataset pairing/transforms, with schema-2 validation.

    Training visits each content image once per epoch, sampling one style.
    Evaluation enumerates all content/style pairs in a fixed order.
    Only the requested split's images are opened during construction.
    """

    def __init__(self, manifest_path, dataset_root, split, image_size=256, seed=42):
        if split not in {"train", "validation", "test"}:
            raise ValueError(f"Unknown split: {split}")
        if not isinstance(image_size, int) or image_size <= 0 or image_size % 4:
            raise ValueError("image_size must be positive and divisible by 4")
        root = Path(dataset_root).resolve()
        if not root.is_dir():
            raise NotADirectoryError(root)
        manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        validate_manifest(manifest)
        section = manifest["splits"][split]
        self.content_files = checked_external_paths(section["content"], root)
        self.style_files = checked_external_paths(section["style"], root)
        self.split = split
        self.seed = int(seed)
        self.epoch = 0
        self.manifest_path = Path(manifest_path).resolve()
        self.dataset_root = root
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,) * 3, (0.5,) * 3),
        ])
