"""Models preserved from the original fixed-style prototype."""

from .generator import Generator as LegacyGenerator
from .discriminator import Discriminator as LegacyDiscriminator

__all__ = ["LegacyGenerator", "LegacyDiscriminator"]
