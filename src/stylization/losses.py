"""Basic losses for the corrected fixed-style baseline."""

import torch
import torch.nn.functional as F


def content_l1_loss(generated, content):
    if generated.shape != content.shape:
        raise ValueError("Generated and content shapes must match")
    return F.l1_loss(generated, content.detach())


def gram_matrix(features):
    if features.ndim != 4:
        raise ValueError("Expected features with shape [B, C, H, W]")

    batch, channels, height, width = features.shape
    flat = features.float().reshape(batch, channels, height * width)
    return torch.bmm(flat, flat.transpose(1, 2)) / (
        channels * height * width
    )


def relative_gram_style_loss(
    generated_features,
    style_features,
    eps=1e-8,
):
    if eps <= 0:
        raise ValueError("eps must be positive")

    if len(generated_features) == 0:
        raise ValueError("Feature lists must not be empty")

    if len(generated_features) != len(style_features):
        raise ValueError("Feature lists must have equal lengths")

    losses = []

    for generated, style in zip(generated_features, style_features):
        generated_gram = gram_matrix(generated)
        style_gram = gram_matrix(style.detach())

        if generated_gram.shape != style_gram.shape:
            raise ValueError("Gram matrix shapes must match")

        # Normalize separately for each sample.
        numerator = (generated_gram - style_gram).square().mean(
            dim=(1, 2)
        )
        denominator = style_gram.square().mean(
            dim=(1, 2)
        ).clamp_min(eps)

        losses.append((numerator / denominator).mean())

    return torch.stack(losses).mean()


def total_variation_loss(images):
    if images.ndim != 4:
        raise ValueError("Expected images with shape [B, C, H, W]")

    if images.shape[-2] < 2 or images.shape[-1] < 2:
        raise ValueError("Image height and width must be at least 2")

    horizontal = (
        images[:, :, :, 1:] - images[:, :, :, :-1]
    ).abs().mean()

    vertical = (
        images[:, :, 1:, :] - images[:, :, :-1, :]
    ).abs().mean()

    return horizontal + vertical
