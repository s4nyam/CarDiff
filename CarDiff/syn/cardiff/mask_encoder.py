"""
Mask Encoder  C_m(M)
====================

Encodes a segmentation mask  M  (with classes SC, MC, DC + background)
into a dense feature map  F = C_m(M)  that captures spatial and semantic
cues about lesion location and extent.

The feature map F is used both as the anatomical context for FiLM
conditioning and as input for the causal patch-graph construction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskEncoder(nn.Module):
    """
    Light-weight CNN that encodes a single-channel segmentation mask into
    a multi-channel feature map at a reduced spatial resolution matching the
    latent space of the VAE.

    Parameters
    ----------
    in_channels : int
        Number of input mask channels (default 1 for single-channel multi-class).
    base_channels : int
        Width of the first conv layer; doubles at every down-sample.
    out_channels : int
        Number of output feature-map channels (should match the causal module
        and FiLM input dimensionality).
    num_downs : int
        Number of stride-2 down-sampling stages.  The spatial resolution is
        reduced by ``2**num_downs``.
    """

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 64,
        out_channels: int = 256,
        num_downs: int = 4,
    ):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, base_channels, 3, padding=1),
            nn.GroupNorm(8, base_channels),
            nn.SiLU(inplace=True),
        ]
        ch = base_channels
        for _ in range(num_downs):
            next_ch = min(ch * 2, out_channels)
            layers += [
                nn.Conv2d(ch, next_ch, 4, stride=2, padding=1),
                nn.GroupNorm(min(32, next_ch), next_ch),
                nn.SiLU(inplace=True),
            ]
            ch = next_ch

        # Final 1×1 to project to desired out_channels
        layers.append(nn.Conv2d(ch, out_channels, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, mask: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        mask : (B, 1, H, W)  float tensor with class indices normalised to [0,1].

        Returns
        -------
        F : (B, out_channels, H', W')  feature map.
        """
        return self.net(mask)
