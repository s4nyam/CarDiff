"""
CarDiff Inference / Sampling Pipeline
=======================================

Given a set of segmentation masks, generates radiographs by:
1. Encoding the mask  →  C_m(M) + causal context C_hyb
2. Iteratively denoising a latent  z_T ~ N(0,I)  with the FiLM U-Net
3. Decoding the final latent through the frozen VAE decoder

Compatible with both DDPM and DDIM schedulers.
"""

from __future__ import annotations

import os
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from PIL import Image
from tqdm.auto import tqdm

import diffusers
from diffusers import DiffusionPipeline, ImagePipelineOutput


class CarDiffPipeline:
    """
    Sampling pipeline for CarDiff (not a DiffusionPipeline subclass to
    avoid diffusers registration hassles with custom modules).
    """

    def __init__(
        self,
        cardiff_model,
        scheduler,
        device: str = "cuda",
    ):
        self.model = cardiff_model
        self.scheduler = scheduler
        self.device = device

    @torch.no_grad()
    def __call__(
        self,
        masks: torch.Tensor,
        num_inference_steps: int = 1000,
        generator: Optional[torch.Generator] = None,
        output_type: str = "pil",
    ) -> Union[List[Image.Image], torch.Tensor]:
        """
        Generate radiographs conditioned on segmentation masks.

        Parameters
        ----------
        masks : (B, 1, H, W) float tensor with class values in [0,1].
        num_inference_steps : int
        generator : optional torch.Generator for reproducibility.
        output_type : 'pil' | 'tensor' | 'np'.

        Returns
        -------
        images : list of PIL images  or  tensor  depending on output_type.
        """
        self.model.eval()
        masks = masks.to(self.device)
        B = masks.shape[0]

        # Encode mask → conditioning vector
        cond, _, _, _, _ = self.model.encode_mask(masks)

        # Sample initial noise in latent space
        latent_shape = (B,) + tuple(self.model.latent_shape)
        z = torch.randn(latent_shape, generator=generator, device=self.device)

        # Iterative denoising
        self.scheduler.set_timesteps(num_inference_steps)
        for t in tqdm(self.scheduler.timesteps, desc="Denoising", leave=False):
            t_batch = torch.full((B,), t, device=self.device, dtype=torch.long)
            noise_pred = self.model.predict_noise(z, t_batch, cond)
            z = self.scheduler.step(noise_pred, t, z).prev_sample

        # Decode latent → image
        images = self.model.decode_latent(z)
        images = (images / 2 + 0.5).clamp(0, 1)

        if output_type == "tensor":
            return images

        # Convert to numpy
        images_np = images.cpu().permute(0, 2, 3, 1).numpy()

        if output_type == "np":
            return images_np

        # Convert to PIL
        pil_images = []
        for img in images_np:
            if img.shape[-1] == 1:
                img = img.squeeze(-1)
            img_uint8 = (img * 255).clip(0, 255).astype("uint8")
            if img_uint8.ndim == 2:
                pil_images.append(Image.fromarray(img_uint8, mode="L"))
            else:
                pil_images.append(Image.fromarray(img_uint8))
        return pil_images

    def save_pretrained(self, save_dir: str):
        """Save model weights and scheduler config."""
        os.makedirs(save_dir, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(save_dir, "cardiff_model.pt"))
        self.scheduler.save_pretrained(os.path.join(save_dir, "scheduler"))

    @classmethod
    def from_pretrained(
        cls,
        load_dir: str,
        cardiff_model,
        scheduler_cls=None,
        device: str = "cuda",
    ):
        """Load model weights and scheduler from checkpoint."""
        model_path = os.path.join(load_dir, "cardiff_model.pt")
        state = torch.load(model_path, map_location="cpu")
        cardiff_model.load_state_dict(state, strict=False)
        cardiff_model = cardiff_model.to(device)

        sched_dir = os.path.join(load_dir, "scheduler")
        if scheduler_cls is None:
            scheduler_cls = diffusers.DDPMScheduler
        scheduler = scheduler_cls.from_pretrained(sched_dir)

        return cls(cardiff_model, scheduler, device)
