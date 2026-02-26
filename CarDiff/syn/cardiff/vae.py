"""
Frozen VAE encoder / decoder for latent-space diffusion.

The VAE maps radiographs  X -> z  (encoder) and  z -> X  (decoder).
It is pre-trained on the dental radiograph corpus with a reconstruction +
KL objective and is **frozen** during CarDiff training to provide a stable
latent space.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class _ResBlock(nn.Module):
    """Simple residual block used in both encoder and decoder."""

    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(32, channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(32, channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class _Downsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class _Upsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


# ---------------------------------------------------------------------------
# Encoder / Decoder
# ---------------------------------------------------------------------------

class Encoder(nn.Module):
    """Maps  X (B,C_in,H,W) -> (mu, logvar)  each of shape (B, latent_dim, H/f, W/f)."""

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 64,
        channel_mults: tuple = (1, 2, 4, 4),
        num_res_blocks: int = 2,
        latent_channels: int = 4,
    ):
        super().__init__()
        self.conv_in = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        channels = base_channels
        blocks = []
        for mult in channel_mults:
            out_ch = base_channels * mult
            for _ in range(num_res_blocks):
                blocks.append(_ResBlock(channels) if channels == out_ch else nn.Sequential(
                    nn.Conv2d(channels, out_ch, 1), _ResBlock(out_ch)
                ))
                channels = out_ch
            blocks.append(_Downsample(channels))
        self.blocks = nn.Sequential(*blocks)

        self.norm_out = nn.GroupNorm(32, channels)
        self.conv_out = nn.Conv2d(channels, 2 * latent_channels, 1)  # mu + logvar

    def forward(self, x: torch.Tensor):
        h = self.conv_in(x)
        h = self.blocks(h)
        h = self.norm_out(h)
        h = F.silu(h)
        h = self.conv_out(h)
        mu, logvar = h.chunk(2, dim=1)
        return mu, logvar


class Decoder(nn.Module):
    """Maps  z (B, latent_channels, H/f, W/f) -> X_hat (B, out_channels, H, W)."""

    def __init__(
        self,
        out_channels: int = 1,
        base_channels: int = 64,
        channel_mults: tuple = (1, 2, 4, 4),
        num_res_blocks: int = 2,
        latent_channels: int = 4,
    ):
        super().__init__()
        channels = base_channels * channel_mults[-1]
        self.conv_in = nn.Conv2d(latent_channels, channels, 3, padding=1)

        blocks = []
        for mult in reversed(channel_mults):
            out_ch = base_channels * mult
            for _ in range(num_res_blocks):
                blocks.append(_ResBlock(channels) if channels == out_ch else nn.Sequential(
                    nn.Conv2d(channels, out_ch, 1), _ResBlock(out_ch)
                ))
                channels = out_ch
            blocks.append(_Upsample(channels))
        self.blocks = nn.Sequential(*blocks)

        self.norm_out = nn.GroupNorm(32, channels)
        self.conv_out = nn.Conv2d(channels, out_channels, 3, padding=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.conv_in(z)
        h = self.blocks(h)
        h = self.norm_out(h)
        h = F.silu(h)
        return self.conv_out(h)


# ---------------------------------------------------------------------------
# Full VAE
# ---------------------------------------------------------------------------

class DentalVAE(nn.Module):
    """
    Lightweight VAE for dental radiographs.

    After pre-training, call ``freeze()`` to lock all parameters.
    During CarDiff training only ``encode`` and ``decode`` are used (no
    reparameterization – we feed the mean ``mu`` directly).
    """

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 64,
        channel_mults: tuple = (1, 2, 4),
        num_res_blocks: int = 2,
        latent_channels: int = 4,
    ):
        super().__init__()
        self.encoder = Encoder(in_channels, base_channels, channel_mults, num_res_blocks, latent_channels)
        self.decoder = Decoder(in_channels, base_channels, channel_mults, num_res_blocks, latent_channels)
        self.latent_channels = latent_channels

    # ------------------------------------------------------------------
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return the latent mean (used during frozen inference)."""
        mu, logvar = self.encoder(x)
        return mu

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor):
        """Full forward used only during VAE pre-training."""
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decoder(z)
        return x_hat, mu, logvar

    # ------------------------------------------------------------------
    def freeze(self):
        """Freeze all parameters (call after pre-training)."""
        for p in self.parameters():
            p.requires_grad = False
        self.eval()

    def get_latent_shape(self, image_size: int):
        """Return (C, H, W) of the latent for a given square image size."""
        factor = 2 ** len(self.encoder.blocks)  # approximate
        # More reliable: just run a dummy forward
        with torch.no_grad():
            dummy = torch.zeros(1, 1, image_size, image_size)
            mu, _ = self.encoder(dummy)
        return mu.shape[1:]  # (C, H_lat, W_lat)
