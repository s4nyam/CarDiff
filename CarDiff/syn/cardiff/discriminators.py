"""
Discriminators  D_i  and  D_p
==============================

D_i  – Image discriminator: receives a single radiograph (real or
       synthesized) and outputs a scalar realism probability.

D_p  – Pair discriminator: receives a mask-image pair and outputs a
       joint realism score assessing their spatial correspondence,
       enforcing geometric alignment between lesion segments and
       corresponding regions in the generated radiograph.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class _DiscriminatorBlock(nn.Module):
    """Conv → InstanceNorm → LeakyReLU down-sampling block."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 4, stride=stride, padding=1),
            nn.InstanceNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.block(x)


# ======================================================================

class ImageDiscriminator(nn.Module):
    """
    PatchGAN-style discriminator for radiographic realism.

    Receives synthesised or real radiograph  X  and outputs a
    feature map of scalar logits (one per receptive-field patch).
    """

    def __init__(self, in_channels: int = 1, base_channels: int = 64, n_layers: int = 3):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, base_channels, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        ch = base_channels
        for i in range(1, n_layers):
            next_ch = min(ch * 2, 512)
            layers.append(_DiscriminatorBlock(ch, next_ch))
            ch = next_ch
        # Final 1-channel output
        layers.append(nn.Conv2d(ch, 1, 4, stride=1, padding=1))
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns logits map (B, 1, H', W')."""
        return self.model(x)


# ======================================================================

class PairDiscriminator(nn.Module):
    """
    Conditional PatchGAN discriminator.

    Receives **concatenated** mask + image  (M || X)  and outputs a
    joint realism score assessing spatial correspondence.
    """

    def __init__(
        self,
        img_channels: int = 1,
        mask_channels: int = 1,
        base_channels: int = 64,
        n_layers: int = 3,
    ):
        super().__init__()
        in_ch = img_channels + mask_channels
        layers = [
            nn.Conv2d(in_ch, base_channels, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        ch = base_channels
        for i in range(1, n_layers):
            next_ch = min(ch * 2, 512)
            layers.append(_DiscriminatorBlock(ch, next_ch))
            ch = next_ch
        layers.append(nn.Conv2d(ch, 1, 4, stride=1, padding=1))
        self.model = nn.Sequential(*layers)

    def forward(self, mask: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
        """Returns logits map (B, 1, H', W')."""
        return self.model(torch.cat([mask, image], dim=1))
