"""Versioned model checkpoints for inference and evaluation."""

import hashlib
import json
import os
import tempfile
from pathlib import Path

import torch


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_model_checkpoint(
    path,
    model,
    *,
    model_name,
    completed_epochs,
    global_step,
    config,
    manifest_path,
):
    if completed_epochs < 0 or global_step < 0:
        raise ValueError("Training counters must be non-negative")

    # Require portable configuration values and reject NaN/Infinity.
    portable_config = json.loads(json.dumps(config, allow_nan=False))

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "format_version": 1,
        "checkpoint_kind": "model_only",
        "model_name": model_name,
        "completed_epochs": int(completed_epochs),
        "global_step": int(global_step),
        "config": portable_config,
        "manifest_sha256": file_sha256(manifest_path),
        "torch_version": str(torch.__version__),
        "model_state": {
            name: tensor.detach().cpu().clone()
            for name, tensor in model.state_dict().items()
        },
    }

    # Write to a temporary file before replacing the destination.
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent,
            prefix=path.name + ".",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)

        torch.save(payload, temporary_path)
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def load_model_checkpoint(
    path,
    model,
    *,
    expected_model_name,
    expected_manifest_path=None,
):
    payload = torch.load(
        Path(path),
        map_location="cpu",
        weights_only=True,
    )

    if not isinstance(payload, dict):
        raise ValueError("Invalid checkpoint")

    if payload.get("format_version") != 1:
        raise ValueError("Unsupported checkpoint format")

    if payload.get("checkpoint_kind") != "model_only":
        raise ValueError("Expected a model-only checkpoint")

    if payload.get("model_name") != expected_model_name:
        raise ValueError("Checkpoint model name does not match")

    if expected_manifest_path is not None:
        expected_hash = file_sha256(expected_manifest_path)
        if payload.get("manifest_sha256") != expected_hash:
            raise ValueError("Data manifest does not match the checkpoint")

    model.load_state_dict(payload["model_state"], strict=True)

    return {
        key: value
        for key, value in payload.items()
        if key != "model_state"
    }
