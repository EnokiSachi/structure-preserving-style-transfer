"""Mean of differentiable edge losses at fixed average-pooled scales."""
import math
from numbers import Real

import torch
from torch.nn import functional as F

from stylization.structure_losses import single_scale_structure_loss


def multiscale_structure_loss(generated, content, scales=(1, 2, 4), weights=None):
    """Compare signed Sobel responses at each spatial scale in float32.

    Scale 1 uses the original image. Scale k uses nonoverlapping k-by-k average
    pooling of the original image, not repeated resizing. Return the normalized
    weighted mean of per-scale losses. Content is detached at every scale.
    """
    if not isinstance(scales, (tuple, list)) or not scales:
        raise ValueError("Scales must be a nonempty list or tuple")
    if any(type(scale) is not int or scale < 1 for scale in scales):
        raise ValueError("Scales must be positive integers")
    if list(scales) != sorted(set(scales)):
        raise ValueError("Scales must be unique and ascending")
    if weights is None:
        weights = [1.0] * len(scales)
    if not isinstance(weights, (tuple, list)) or len(weights) != len(scales):
        raise ValueError("Weights must match scales")
    if any(not isinstance(w, Real) or isinstance(w, bool) or
           not math.isfinite(w) or w < 0 for w in weights):
        raise ValueError("Weights must be finite nonnegative numbers")
    total_weight = sum(weights)
    if not math.isfinite(total_weight) or total_weight <= 0:
        raise ValueError("Weights must have a finite positive sum")
    if not isinstance(generated, torch.Tensor) or not isinstance(content, torch.Tensor):
        raise ValueError("Generated and content must be tensors")
    if generated.shape != content.shape or generated.device != content.device:
        raise ValueError("Generated and content shapes/devices must match")
    for image in (generated, content):
        if (image.ndim != 4 or image.shape[0] < 1 or image.shape[1] != 3 or
                not image.is_floating_point()):
            raise ValueError("Expected nonempty floating RGB NCHW images")
        if not torch.isfinite(image).all().item():
            raise ValueError("Images must be finite")
    height, width = generated.shape[-2:]
    for scale in scales:
        if height % scale or width % scale or min(height // scale, width // scale) < 3:
            raise ValueError("Each scale must divide image dimensions and leave at least 3x3")

    with torch.autocast(device_type=generated.device.type, enabled=False):
        generated = generated.float()
        target = content.detach().float()
        losses = []
        for scale, weight in zip(scales, weights):
            if weight == 0:
                continue
            if scale == 1:
                prediction, reference = generated, target
            else:
                prediction = F.avg_pool2d(generated, kernel_size=scale, stride=scale)
                reference = F.avg_pool2d(target, kernel_size=scale, stride=scale)
            losses.append(single_scale_structure_loss(prediction, reference)
                          * (weight / total_weight))
        return torch.stack(losses).sum()
