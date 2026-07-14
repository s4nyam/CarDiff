"""
Segmentation Consistency Head  S(·)
====================================

Re-segments a synthesized radiograph and compares the prediction with
the conditioning mask via Dice loss.

- Path A:  L_seg   = Dice( S(X_hat), M )
- Path B:  L_self  = Dice( S(X_tilde), M_tilde )
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SegmentationHead(nn.Module):
    """
    Lightweight segmentation decoder that predicts a per-pixel class
    mask from a synthesised radiograph.

    Architecture: small encoder-decoder with skip connections (U-Net
    style) operating directly in pixel space.

    Parameters
    ----------
    in_channels : int
        Image channels (1 for grayscale radiographs).
    num_classes : int
        Number of output classes **including** background.
    base_channels : int
        Channel width of the first encoder level.
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 4,
        base_channels: int = 32,
    ):
        super().__init__()
        c = base_channels

        # Encoder
        self.enc1 = self._block(in_channels, c)
        self.enc2 = self._block(c, c * 2)
        self.enc3 = self._block(c * 2, c * 4)
        self.enc4 = self._block(c * 4, c * 8)

        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = self._block(c * 8, c * 16)

        # Decoder
        self.up4 = nn.ConvTranspose2d(c * 16, c * 8, 2, stride=2)
        self.dec4 = self._block(c * 16, c * 8)
        self.up3 = nn.ConvTranspose2d(c * 8, c * 4, 2, stride=2)
        self.dec3 = self._block(c * 8, c * 4)
        self.up2 = nn.ConvTranspose2d(c * 4, c * 2, 2, stride=2)
        self.dec2 = self._block(c * 4, c * 2)
        self.up1 = nn.ConvTranspose2d(c * 2, c, 2, stride=2)
        self.dec1 = self._block(c * 2, c)

        self.final = nn.Conv2d(c, num_classes, 1)

    @staticmethod
    def _block(in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, 1, H, W) – synthesised radiograph.

        Returns
        -------
        logits : (B, num_classes, H, W)
        """
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        b = self.bottleneck(self.pool(e4))

        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return self.final(d1)
