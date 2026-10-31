"""Bounded train/validation selections for the reference-model smoke run."""
import hashlib
import json
from pathlib import Path

from torch.utils.data import Dataset

from stylization.external_split_data import ExternalManifestDataset, validate_manifest


class SelectedDataset(Dataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = tuple(indices)
        self.split = dataset.split

    def set_epoch(self, epoch):
        self.dataset.set_epoch(epoch)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        return self.dataset[self.indices[index]]


def validate_selection(config, manifest):
    validate_manifest(manifest)
    selection = config["selection"]
    if selection["algorithm"] != "sha256_rank_unique_groups_v1":
        raise ValueError("Unknown selection algorithm")
    train = manifest["splits"]["train"]["content"]
    val = manifest["splits"]["validation"]
    ti, vi = selection["train_content_indices"], selection["validation_pair_indices"]
    for indices, length in ((ti, len(train)), (vi, len(val["content"]) * len(val["style"]))):
        if len(indices) != 8 or any(type(i) is not int or not 0 <= i < length for i in indices):
            raise ValueError("Smoke selection needs eight valid indices")
        if len(set(indices)) != len(indices):
            raise ValueError("Repeated selection index")
    if [train[i]["path"] for i in ti] != selection["train_content_paths"]:
        raise ValueError("Training selection paths do not match manifest")
    if len({train[i]["group_id"] for i in ti}) != 8:
        raise ValueError("Training selection repeats related-image groups")
    pairs = [divmod(i, len(val["style"])) for i in vi]
    expected = [{"content": val["content"][c]["path"], "style": val["style"][s]["path"]}
                for c, s in pairs]
    if expected != selection["validation_pairs"]:
        raise ValueError("Validation paths do not match selected indices")
    content_ids, style_ids = {c for c, _ in pairs}, {s for _, s in pairs}
    if len(content_ids) != 4 or len(style_ids) != 2 or set(pairs) != {
            (c, s) for c in content_ids for s in style_ids}:
        raise ValueError("Validation selection must be four content images by two styles")
    if len({val["content"][c]["group_id"] for c in content_ids}) != 4 or len({
            val["style"][s]["group_id"] for s in style_ids}) != 2:
        raise ValueError("Validation selection repeats related-image groups")


def build_reference_datasets(config, manifest_path, dataset_root):
    raw = Path(manifest_path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != config["manifest_sha256"]:
        raise ValueError("Manifest fingerprint differs from training configuration")
    manifest = json.loads(raw.decode("utf-8"))
    validate_selection(config, manifest)
    datasets = {}
    for split, key in (("train", "train_content_indices"),
                       ("validation", "validation_pair_indices")):
        base = ExternalManifestDataset(manifest_path, dataset_root, split,
                                       image_size=config["image_size"], seed=config["seed"])
        datasets[split] = SelectedDataset(base, config["selection"][key])
    return datasets
