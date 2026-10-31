"""Frozen VGG19 style features with gradients preserved for inputs."""

import torch
from torch import nn
from torchvision.models import vgg19, VGG19_Weights


class VGG19StyleFeatures(nn.Module):
    def __init__(self):
        super().__init__()

        network = vgg19(weights=VGG19_Weights.IMAGENET1K_V1)
        self.features = network.features[:29]

        # Preserve convolution outputs without later in-place mutation.
        for layer in self.features:
            if isinstance(layer, nn.ReLU):
                layer.inplace = False

        self.layer_indices = {0, 5, 10, 19, 28}

        self.register_buffer(
            "mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
        )

        self.requires_grad_(False)
        self.eval()

    def train(self, mode=True):
        # Keep the pretrained feature extractor in evaluation mode.
        return super().train(False)

    def forward(self, images):
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("Expected RGB images with shape [B, 3, H, W]")

        if min(images.shape[-2:]) < 32:
            raise ValueError("Image height and width must be at least 32")

        x = (images.float() + 1.0) / 2.0
        x = (x - self.mean) / self.std

        outputs = []
        for index, layer in enumerate(self.features):
            x = layer(x)
            if index in self.layer_indices:
                outputs.append(x)

        return outputs
