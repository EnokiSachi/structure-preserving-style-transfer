"""Training and validation loops for the reference-conditioned model."""

import math

import torch
import torch.nn.functional as F

from stylization.structure_losses import single_scale_structure_loss
from stylization.multiscale_structure import multiscale_structure_loss

from stylization.losses import (
    content_l1_loss,
    relative_gram_style_loss,
    total_variation_loss,
)


def checked_step(loss, model, optimizer):
    if not torch.isfinite(loss).item():
        raise FloatingPointError("Non-finite training loss")

    loss.backward()

    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    if not gradients or any(gradient is None for gradient in gradients):
        raise RuntimeError("Missing gradient in trainable model")

    if not all(torch.isfinite(g).all().item() for g in gradients):
        raise FloatingPointError("Non-finite model gradient")

    optimizer.step()


def train_one_epoch(
    generator,
    discriminator,
    extractor,
    loader,
    optimizer_g,
    optimizer_d,
    config,
    device,
    epoch_index,
    global_step,
):
    if loader.dataset.split != "train":
        raise ValueError("Training requires the train split")
    if len(loader) == 0:
        raise ValueError("Training loader is empty")

    loader.dataset.set_epoch(epoch_index)
    generator.train()
    discriminator.train()
    extractor.eval()
    discriminator.requires_grad_(True)

    weights = config["loss_weights"]
    structure_mode = config.get("structure_loss", "single_scale_sobel_v1")
    if structure_mode not in {"single_scale_sobel_v1", "multiscale_sobel_124_v1"}:
        raise ValueError("Unsupported structure loss mode")
    structure_function = (single_scale_structure_loss if structure_mode == "single_scale_sobel_v1"
                          else multiscale_structure_loss)
    structure_weight = weights.get("structure", 0.0)
    if not math.isfinite(structure_weight) or structure_weight < 0:
        raise ValueError("Structure weight must be finite and nonnegative")
    update_count = config["generator_updates_per_batch"]
    if update_count <= 0:
        raise ValueError("Generator update count must be positive")

    totals = {
        "d_loss": 0.0,
        "g_loss": 0.0,
        "content": 0.0,
        "style": 0.0,
        "tv": 0.0,
        "structure": 0.0,
        "structure_weighted": 0.0,
    }
    samples = 0

    for batch_index, batch in enumerate(loader):
        content = batch["content"].to(device)
        style = batch["style"].to(device)
        batch_size = content.shape[0]

        with torch.no_grad():
            reference_features = extractor(style)
            fake_for_d = generator(content, style)

        # One discriminator update.
        optimizer_d.zero_grad(set_to_none=True)
        real_scores = discriminator(style)
        fake_scores = discriminator(fake_for_d)

        d_loss = 0.5 * (
            F.binary_cross_entropy_with_logits(
                real_scores,
                torch.full_like(real_scores, config["real_label"]),
            )
            + F.binary_cross_entropy_with_logits(
                fake_scores,
                torch.full_like(fake_scores, config["fake_label"]),
            )
        )
        checked_step(d_loss, discriminator, optimizer_d)
        optimizer_d.zero_grad(set_to_none=True)

        # Freeze D parameters, not its computation graph.
        discriminator.requires_grad_(False)
        batch_totals = {
            "g_loss": 0.0, "content": 0.0, "style": 0.0, "tv": 0.0,
            "structure": 0.0, "structure_weighted": 0.0
        }

        try:
            for _ in range(update_count):
                optimizer_g.zero_grad(set_to_none=True)

                generated = generator(content, style)
                scores = discriminator(generated)

                adversarial = F.binary_cross_entropy_with_logits(
                    scores,
                    torch.full_like(scores, config["real_label"]),
                )
                content_loss = content_l1_loss(generated, content)
                style_loss = relative_gram_style_loss(
                    extractor(generated), reference_features
                )
                tv_loss = total_variation_loss(generated)
                structure_loss = structure_function(
                    generated if structure_weight > 0 else generated.detach(), content
                )
                weighted_structure = structure_weight * structure_loss

                g_loss = (
                    weights["adversarial"] * adversarial
                    + weights["content"] * content_loss
                    + weights["style"] * style_loss
                    + weights["tv"] * tv_loss
                )
                # Preserve the original objective graph exactly at zero weight.
                if structure_weight > 0:
                    g_loss = g_loss + weighted_structure
                checked_step(g_loss, generator, optimizer_g)

                for name, value in (
                    ("g_loss", g_loss),
                    ("content", content_loss),
                    ("style", style_loss),
                    ("tv", tv_loss),
                    ("structure", structure_loss),
                    ("structure_weighted", weighted_structure),
                ):
                    batch_totals[name] += value.detach().item()
        finally:
            discriminator.requires_grad_(True)

        totals["d_loss"] += d_loss.detach().item() * batch_size
        for name, value in batch_totals.items():
            totals[name] += value / update_count * batch_size

        samples += batch_size
        global_step += 1

        print(
            f"epoch={epoch_index + 1} "
            f"batch={batch_index + 1}/{len(loader)} "
            f"step={global_step} "
            f"D={d_loss.item():.4f} "
            f"G={batch_totals['g_loss'] / update_count:.4f} "
            f"structure={batch_totals['structure'] / update_count:.6f}",
            flush=True,
        )

    return {
        name: value / samples for name, value in totals.items()
    }, global_step


@torch.no_grad()
def validate(generator, extractor, loader, device):
    if loader.dataset.split != "validation":
        raise ValueError("Validation requires the validation split")
    if len(loader) == 0:
        raise ValueError("Validation loader is empty")

    generator.eval()
    extractor.eval()

    totals = {"content": 0.0, "style": 0.0, "tv": 0.0,
              "structure": 0.0, "structure_multiscale": 0.0}
    samples = 0

    for batch in loader:
        content = batch["content"].to(device)
        style = batch["style"].to(device)
        generated = generator(content, style)

        values = {
            "content": content_l1_loss(generated, content),
            "style": relative_gram_style_loss(
                extractor(generated), extractor(style)
            ),
            "tv": total_variation_loss(generated),
            "structure": single_scale_structure_loss(generated, content),
            "structure_multiscale": multiscale_structure_loss(generated, content),
        }

        for name, value in values.items():
            if not torch.isfinite(value).item():
                raise FloatingPointError(
                    f"Non-finite validation value: {name}"
                )
            totals[name] += value.item() * content.shape[0]

        samples += content.shape[0]

    return {
        name: value / samples for name, value in totals.items()
    }
