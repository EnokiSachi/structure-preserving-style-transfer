"""Training state saved only at completed epoch boundaries."""

import json
import os
import random
import tempfile
from pathlib import Path

import numpy as np
import torch

from stylization.checkpoints import file_sha256


def save_training_state(
    path, generator, discriminator, optimizer_g, optimizer_d,
    order_generator, *, completed_epochs, global_step,
    config, manifest_path,
):
    if completed_epochs < 0 or global_step < 0:
        raise ValueError("Training counters must be non-negative")

    numpy_state = np.random.get_state()

    payload = {
        "format_version": 1,
        "kind": "epoch_boundary_training_state",
        "completed_epochs": int(completed_epochs),
        "global_step": int(global_step),
        "config": json.loads(json.dumps(config, allow_nan=False)),
        "manifest_sha256": file_sha256(manifest_path),
        "torch_version": str(torch.__version__),
        "generator": generator.state_dict(),
        "discriminator": discriminator.state_dict(),
        "optimizer_g": optimizer_g.state_dict(),
        "optimizer_d": optimizer_d.state_dict(),
        "python_rng": random.getstate(),
        "numpy_rng": {
            "name": numpy_state[0],
            "keys": numpy_state[1].tolist(),
            "position": int(numpy_state[2]),
            "has_gauss": int(numpy_state[3]),
            "cached_gaussian": float(numpy_state[4]),
        },
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": (
            torch.cuda.get_rng_state_all()
            if torch.cuda.is_available() else []
        ),
        "order_rng": order_generator.get_state(),
    }

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None

    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, suffix=".tmp", delete=False
        ) as temporary:
            temporary_path = Path(temporary.name)

        torch.save(payload, temporary_path)
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def restore_training_state(
    path, generator, discriminator, optimizer_g, optimizer_d,
    order_generator, *, config, manifest_path,
):
    state = torch.load(path, map_location="cpu", weights_only=True)

    if state.get("format_version") != 1:
        raise ValueError("Unsupported training-state format")
    if state.get("kind") != "epoch_boundary_training_state":
        raise ValueError("Not an epoch-boundary training state")
    if state["config"] != config:
        raise ValueError("Configuration differs from saved training state")
    if state["manifest_sha256"] != file_sha256(manifest_path):
        raise ValueError("Data manifest differs from saved training state")
    if state["torch_version"] != str(torch.__version__):
        raise ValueError("PyTorch version differs from saved training state")

    cuda_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if len(state["cuda_rng"]) != cuda_count:
        raise ValueError("CUDA device count differs from saved training state")

    generator.load_state_dict(state["generator"], strict=True)
    discriminator.load_state_dict(state["discriminator"], strict=True)
    optimizer_g.load_state_dict(state["optimizer_g"])
    optimizer_d.load_state_dict(state["optimizer_d"])

    random.setstate(state["python_rng"])

    numpy_state = state["numpy_rng"]
    np.random.set_state((
        numpy_state["name"],
        np.array(numpy_state["keys"], dtype=np.uint32),
        numpy_state["position"],
        numpy_state["has_gauss"],
        numpy_state["cached_gaussian"],
    ))

    torch.set_rng_state(state["torch_rng"])
    if state["cuda_rng"]:
        torch.cuda.set_rng_state_all(state["cuda_rng"])
    order_generator.set_state(state["order_rng"])

    return {
        "completed_epochs": state["completed_epochs"],
        "global_step": state["global_step"],
    }
