"""Learned reference conditioning; not a reproduction of pretrained AdaIN.

All weights are randomly initialized. Pixel inputs use RGB in [-1, 1].
"""
import torch
from torch import nn
from torch.nn import functional as F


MODEL_NAME = "reference_generator_v1"


class StyleEncoder(nn.Module):
    def __init__(self, style_dim=256):
        super().__init__()
        # No instance normalization: preserve reference feature statistics.
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.SiLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.SiLU(),
        )
        self.project = nn.Sequential(nn.Linear(512, style_dim), nn.SiLU())

    def forward(self, image):
        features = self.features(image)
        mean = features.mean(dim=(2, 3))
        std = (features.var(dim=(2, 3), unbiased=False) + 1e-5).sqrt()
        return self.project(torch.cat((mean, std), dim=1))


class ReferenceNorm(nn.Module):
    def __init__(self, channels, style_dim=256):
        super().__init__()
        self.norm = nn.InstanceNorm2d(channels, affine=False)
        self.affine = nn.Linear(style_dim, channels * 2)
        # Nonzero weights allow gradients into the reference encoder immediately.
        nn.init.normal_(self.affine.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.affine.bias)

    def forward(self, features, style_code):
        gamma, beta = self.affine(style_code).chunk(2, dim=1)
        return self.norm(features) * (1 + gamma[:, :, None, None]) + beta[:, :, None, None]


class ReferenceResidualBlock(nn.Module):
    def __init__(self, channels, style_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm1 = ReferenceNorm(channels, style_dim)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm2 = ReferenceNorm(channels, style_dim)

    def forward(self, features, style_code):
        residual = F.relu(self.norm1(self.conv1(features), style_code))
        residual = self.norm2(self.conv2(residual), style_code)
        return features + residual


class ReferenceUpsample(nn.Module):
    def __init__(self, in_channels, out_channels, style_dim):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm = ReferenceNorm(out_channels, style_dim)

    def forward(self, features, style_code):
        features = F.interpolate(features, scale_factor=2, mode="nearest")
        return F.relu(self.norm(self.conv(features), style_code))


class ReferenceGenerator(nn.Module):
    model_name = MODEL_NAME

    def __init__(self):
        super().__init__()
        style_dim = 256
        self.style_encoder = StyleEncoder(style_dim)
        self.content_encoder = nn.Sequential(
            nn.Conv2d(3, 64, 7, padding=3), nn.InstanceNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.InstanceNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.InstanceNorm2d(256), nn.ReLU(),
        )
        self.blocks = nn.ModuleList([
            ReferenceResidualBlock(256, style_dim) for _ in range(3)
        ])
        self.up1 = ReferenceUpsample(256, 128, style_dim)
        self.up2 = ReferenceUpsample(128, 64, style_dim)
        self.output = nn.Conv2d(64, 3, 7, padding=3)

    @staticmethod
    def validate_inputs(content, style):
        for name, image in (("content", content), ("style", style)):
            if not isinstance(image, torch.Tensor) or image.ndim != 4:
                raise ValueError(f"{name} must be an NCHW tensor")
            if image.shape[0] < 1 or image.shape[1] != 3 or min(image.shape[2:]) < 16:
                raise ValueError(f"{name} needs nonempty RGB images of at least 16x16")
            if not image.is_floating_point():
                raise ValueError(f"{name} must use floating-point pixels")
        if any(size % 4 for size in content.shape[2:]):
            raise ValueError("Content height and width must be divisible by 4")
        if style.shape[0] not in (1, content.shape[0]):
            raise ValueError("Style batch must be 1 or match content batch")
        if content.device != style.device or content.dtype != style.dtype:
            raise ValueError("Content and style must share device and dtype")

    def forward(self, content, style):
        self.validate_inputs(content, style)
        style_code = self.style_encoder(style)
        if style_code.shape[0] == 1 and content.shape[0] != 1:
            style_code = style_code.expand(content.shape[0], -1)
        features = self.content_encoder(content)
        for block in self.blocks:
            features = block(features, style_code)
        features = self.up2(self.up1(features, style_code), style_code)
        return torch.tanh(self.output(features))
