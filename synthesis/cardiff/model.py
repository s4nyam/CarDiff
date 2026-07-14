"""
CarDiffModel — top-level module
================================

Wires together all CarDiff components:

  VAE (frozen)  →  latent-space diffusion
  MaskEncoder   →  C_m(M)
  CausalModule  →  C_hyb
  FiLMUNet      →  G_{ε_θ}  (denoising with FiLM conditioning)
  SegHead       →  S(·)     (segmentation consistency)
  D_i, D_p      →  adversarial feedback

This class exposes a unified interface used by the training loop and
inference pipeline.
"""

from __future__ import annotations
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .vae import DentalVAE
from .mask_encoder import MaskEncoder
from .causal_module import CausalModule
from .film_unet import FiLMUNet
from .discriminators import ImageDiscriminator, PairDiscriminator
from .seg_head import SegmentationHead
from .augmentation import MaskAugmentation


class CarDiffModel(nn.Module):
    """
    Top-level CarDiff model.

    Parameters
    ----------
    image_size : int
        Input radiograph spatial size (e.g., 384).
    img_channels : int
        Number of image channels (1 for grayscale).
    num_classes : int
        Number of segmentation classes **including** background.
    latent_channels : int
        Number of VAE latent channels.
    mask_enc_channels : int
        Mask-encoder output channels (also causal module hidden dim).
    unet_block_channels : tuple[int]
        Per-level channel widths for the FiLM U-Net.
    patch_size : int
        Patch size for the causal graph.
    vae_pretrained_path : str | None
        If given, load pre-trained VAE weights from this path.
    """

    def __init__(
        self,
        image_size: int = 384,
        img_channels: int = 1,
        num_classes: int = 4,
        latent_channels: int = 4,
        mask_enc_channels: int = 256,
        unet_block_channels: tuple = (128, 256, 256, 512),
        patch_size: int = 8,
        vae_pretrained_path: Optional[str] = None,
    ):
        super().__init__()
        self.image_size = image_size
        self.img_channels = img_channels
        self.num_classes = num_classes

        # --- VAE (frozen after init) ---
        self.vae = DentalVAE(
            in_channels=img_channels,
            latent_channels=latent_channels,
        )
        if vae_pretrained_path is not None:
            state = torch.load(vae_pretrained_path, map_location="cpu")
            self.vae.load_state_dict(state, strict=False)
        self.vae.freeze()

        # Compute latent spatial size
        with torch.no_grad():
            dummy = torch.zeros(1, img_channels, image_size, image_size)
            lat_shape = self.vae.encode(dummy).shape  # (1, C_lat, H_lat, W_lat)
        self.latent_shape = lat_shape[1:]  # (C, H, W)
        latent_size = self.latent_shape[1]

        # --- Mask Encoder C_m ---
        self.mask_encoder = MaskEncoder(
            in_channels=1,
            out_channels=mask_enc_channels,
            num_downs=4,
        )

        # --- Causal Module ---
        self.causal = CausalModule(
            feat_dim=mask_enc_channels,
            num_classes=num_classes - 1,  # exclude background
            hidden_dim=mask_enc_channels,
            patch_size=patch_size,
        )

        # --- FiLM U-Net (operates on latents) ---
        cond_dim = mask_enc_channels * 2  # [C_m_pooled || C_hyb]
        self.film_unet = FiLMUNet(
            image_size=latent_size,
            in_channels=latent_channels,
            out_channels=latent_channels,
            cond_dim=cond_dim,
            block_out_channels=unet_block_channels,
        )

        # --- Segmentation Head ---
        self.seg_head = SegmentationHead(
            in_channels=img_channels,
            num_classes=num_classes,
        )

        # --- Discriminators ---
        self.disc_image = ImageDiscriminator(in_channels=img_channels)
        self.disc_pair = PairDiscriminator(img_channels=img_channels, mask_channels=1)

        # --- Mask Augmentation (for Path B) ---
        self.mask_aug = MaskAugmentation()

        # Store dims
        self.mask_enc_channels = mask_enc_channels
        self.cond_dim = cond_dim

    # ------------------------------------------------------------------
    # Forward helpers
    # ------------------------------------------------------------------

    def encode_mask(self, mask: torch.Tensor):
        """
        Encode mask and compute causal context.

        Returns
        -------
        cond : (B, cond_dim)
        F_map : mask feature map
        alphas, h_locals, grid_shapes : for causal regularizers
        """
        F_map = self.mask_encoder(mask)  # (B, C_enc, H', W')
        C_hyb, alphas, h_locals, grid_shapes = self.causal(F_map, mask)
        # Global-average-pool the mask feature map for the FiLM conditioning
        C_m_pooled = F_map.mean(dim=[2, 3])  # (B, mask_enc_channels)
        cond = torch.cat([C_m_pooled, C_hyb], dim=1)  # (B, cond_dim)
        return cond, F_map, alphas, h_locals, grid_shapes

    def predict_noise(
        self, z_t: torch.Tensor, t: torch.Tensor, cond: torch.Tensor
    ) -> torch.Tensor:
        """Run noise prediction through FiLM U-Net."""
        return self.film_unet(z_t, t, cond)

    @torch.no_grad()
    def encode_image(self, x: torch.Tensor) -> torch.Tensor:
        """Encode radiograph to latent (frozen VAE)."""
        return self.vae.encode(x)

    @torch.no_grad()
    def decode_latent(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent to radiograph (frozen VAE)."""
        return self.vae.decode(z)

    def segment(self, x: torch.Tensor) -> torch.Tensor:
        """Re-segment a (possibly synthesized) radiograph."""
        return self.seg_head(x)

    def discriminate_image(self, x: torch.Tensor) -> torch.Tensor:
        return self.disc_image(x)

    def discriminate_pair(self, mask: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # Resize mask to match image if needed
        if mask.shape[-2:] != x.shape[-2:]:
            mask = F.interpolate(mask, size=x.shape[-2:], mode="nearest")
        return self.disc_pair(mask, x)

    def augment_mask(self, mask: torch.Tensor) -> torch.Tensor:
        """Augment masks for self-supervised Path B."""
        return self.mask_aug.augment_batch(mask)

    # ------------------------------------------------------------------
    # Convenience: get parameter groups for separate optimizers
    # ------------------------------------------------------------------

    def generator_parameters(self):
        """Parameters for the generator (everything except discriminators and frozen VAE)."""
        params = []
        for name, module in self.named_children():
            if name not in ("vae", "disc_image", "disc_pair"):
                params.extend(module.parameters())
        return params

    def discriminator_parameters(self):
        """Parameters for discriminators."""
        return list(self.disc_image.parameters()) + list(self.disc_pair.parameters())
