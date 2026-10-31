"""Explicit output-blend strength baseline, not learned feature control."""
from numbers import Real

import torch


def stylize_with_strength(generator, content, style, strength=1.0):
    """Blend content with reference-conditioned output in normalized RGB space.

    strength: a number, floating scalar tensor, or floating tensor shaped [N].
    It must be finite and in [0, 1]. Tensor strengths retain their gradients.
    The generator is evaluated at every strength, including zero. This helper
    does not change model mode, device, precision or gradient context.
    """
    if (not isinstance(content, torch.Tensor) or content.ndim != 4 or
            not content.is_floating_point() or content.shape[0] < 1):
        raise ValueError("Content must be a nonempty floating NCHW tensor")
    if (not torch.isfinite(content).all().item() or
            content.min().item() < -1 or content.max().item() > 1):
        raise ValueError("Content must be finite and in [-1, 1]")
    if isinstance(strength, torch.Tensor):
        if not strength.is_floating_point():
            raise ValueError("Tensor strength must use floating point")
        if strength.ndim == 0:
            alpha = strength.reshape(1, 1, 1, 1)
        elif strength.ndim == 1 and strength.shape[0] == content.shape[0]:
            alpha = strength.reshape(-1, 1, 1, 1)
        else:
            raise ValueError("Strength tensor must be scalar or shaped [batch_size]")
        alpha = alpha.to(device=content.device, dtype=content.dtype)
    elif isinstance(strength, Real) and not isinstance(strength, bool):
        if not 0 <= strength <= 1:
            raise ValueError("Strength must be finite and in [0, 1]")
        alpha = content.new_tensor(float(strength)).reshape(1, 1, 1, 1)
    else:
        raise TypeError("Strength must be a number or floating tensor")
    if (not torch.isfinite(alpha).all().item() or
            alpha.min().item() < 0 or alpha.max().item() > 1):
        raise ValueError("Strength must be finite and in [0, 1]")
    generated = generator(content, style)
    if (generated.shape != content.shape or generated.device != content.device or
            generated.dtype != content.dtype):
        raise ValueError("Generated output must match content shape, device and dtype")
    if (not torch.isfinite(generated).all().item() or
            generated.min().item() < -1 or generated.max().item() > 1):
        raise ValueError("Generated output must be finite and in [-1, 1]")
    return torch.lerp(content, generated, alpha)
