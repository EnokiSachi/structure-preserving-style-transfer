"""Single-scale differentiable luminance-edge preservation baseline."""
import torch
from torch.nn import functional as F


def luminance_sobel(images):
    """Return signed x/y Sobel responses, shape [N, 2, H-2, W-2].

    Inputs are floating RGB images in [-1, 1]. Compute in float32 without
    autocast. Valid convolution excludes the one-pixel border instead of
    introducing artificial edges through padding. No thresholding is used.
    """
    if (not isinstance(images, torch.Tensor) or images.ndim != 4 or
            images.shape[0] < 1 or images.shape[1] != 3 or
            min(images.shape[-2:]) < 3 or not images.is_floating_point()):
        raise ValueError("Expected nonempty floating RGB NCHW images, at least 3x3")
    if not torch.isfinite(images).all().item():
        raise ValueError("Images must be finite")
    with torch.autocast(device_type=images.device.type, enabled=False):
        pixels = (images.float() + 1.0) / 2.0
        weights = pixels.new_tensor([0.299, 0.587, 0.114]).reshape(1, 3, 1, 1)
        gray = (pixels * weights).sum(dim=1, keepdim=True)
        kernels = pixels.new_tensor([
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
        ]).unsqueeze(1) / 8.0
        return F.conv2d(gray, kernels)


def single_scale_structure_loss(generated, content):
    """Mean absolute difference between signed luminance edge responses.

    The target is detached. This is a fixed-filter baseline, not a learned
    perceptual metric or a guarantee of semantic structure preservation.
    """
    if not isinstance(generated, torch.Tensor) or not isinstance(content, torch.Tensor):
        raise ValueError("Generated and content must be tensors")
    if generated.shape != content.shape or generated.device != content.device:
        raise ValueError("Generated and content shapes/devices must match")
    return F.l1_loss(luminance_sobel(generated), luminance_sobel(content.detach()))
