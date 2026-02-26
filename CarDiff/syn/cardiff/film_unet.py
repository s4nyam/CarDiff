"""
FiLM-Conditioned Denoising U-Net  G_{ε_θ}
===========================================

Wraps a standard diffusers ``UNet2DModel`` and injects causal + mask
conditioning via per-layer FiLM modulation.

The U-Net operates in **latent space** (on z_t), not pixel space.
At each denoising step the conditioning vector

    cond = [ C_m(M)_pooled  ||  C_hyb ]

is projected through learned γ_ℓ, β_ℓ MLPs to scale and shift
intermediate U-Net features.
"""

from __future__ import annotations
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import diffusers


class FiLMUNet(nn.Module):
    """
    Denoising U-Net with FiLM conditioning from mask encoder
    and causal module.

    The architecture re-uses the ``diffusers.UNet2DModel`` backbone
    and adds FiLM layers after each resolution block.

    Parameters
    ----------
    image_size : int
        Spatial size of the *latent* (H_lat = W_lat).
    in_channels : int
        Number of latent channels from the VAE.
    out_channels : int
        Same as in_channels (noise prediction target).
    cond_dim : int
        Dimensionality of the combined conditioning vector
        [C_m_pooled || C_hyb].
    block_out_channels : tuple[int]
        Per-level channel counts for the UNet.
    layers_per_block : int
        Number of ResNet layers per UNet block.
    """

    def __init__(
        self,
        image_size: int = 24,
        in_channels: int = 4,
        out_channels: int = 4,
        cond_dim: int = 512,
        block_out_channels: tuple = (128, 256, 256, 512),
        layers_per_block: int = 2,
    ):
        super().__init__()

        self.unet = diffusers.UNet2DModel(
            sample_size=image_size,
            in_channels=in_channels,
            out_channels=out_channels,
            layers_per_block=layers_per_block,
            block_out_channels=block_out_channels,
            down_block_types=tuple(
                "AttnDownBlock2D" if i >= len(block_out_channels) - 2 else "DownBlock2D"
                for i in range(len(block_out_channels))
            ),
            up_block_types=tuple(
                "AttnUpBlock2D" if i < 2 else "UpBlock2D"
                for i in range(len(block_out_channels))
            ),
        )

        # FiLM layers – one per resolution level (down + up)
        from .causal_module import FiLMLayer

        self.film_down = nn.ModuleList([
            FiLMLayer(cond_dim, ch) for ch in block_out_channels
        ])
        self.film_up = nn.ModuleList([
            FiLMLayer(cond_dim, ch) for ch in reversed(block_out_channels)
        ])

        self.cond_dim = cond_dim

    # ------------------------------------------------------------------
    def forward(
        self,
        z_t: torch.Tensor,
        timestep: torch.Tensor,
        cond: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        z_t      : (B, C_lat, H_lat, W_lat) – noisy latent.
        timestep : (B,) or scalar – diffusion timestep.
        cond     : (B, cond_dim) – [C_m_pooled || C_hyb].

        Returns
        -------
        noise_pred : (B, C_lat, H_lat, W_lat)
        """
        # We hook into the UNet's internal block structure.
        # diffusers UNet2DModel stores blocks as:
        #   self.unet.down_blocks, self.unet.mid_block, self.unet.up_blocks

        # --- Timestep embedding ---
        t_emb = self.unet.time_proj(timestep)
        t_emb = self.unet.time_embedding(t_emb)

        # --- Conv in ---
        sample = self.unet.conv_in(z_t)

        # --- Down blocks with FiLM ---
        down_block_res_samples = (sample,)
        for i, down_block in enumerate(self.unet.down_blocks):
            sample, res_samples = down_block(hidden_states=sample, temb=t_emb)
            # Apply FiLM after this resolution level
            if i < len(self.film_down):
                sample = self.film_down[i](sample, cond)
            down_block_res_samples += res_samples

        # --- Mid block ---
        if self.unet.mid_block is not None:
            sample = self.unet.mid_block(sample, t_emb)

        # --- Up blocks with FiLM ---
        for i, up_block in enumerate(self.unet.up_blocks):
            n_resnets = len(up_block.resnets)
            res_samples = down_block_res_samples[-n_resnets:]
            down_block_res_samples = down_block_res_samples[:-n_resnets]

            sample = up_block(
                hidden_states=sample,
                res_hidden_states_tuple=res_samples,
                temb=t_emb,
            )
            if i < len(self.film_up):
                sample = self.film_up[i](sample, cond)

        # --- Conv out ---
        if self.unet.conv_norm_out is not None:
            sample = self.unet.conv_norm_out(sample)
            sample = self.unet.conv_act(sample)
        sample = self.unet.conv_out(sample)

        return sample
