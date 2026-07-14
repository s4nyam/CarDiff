"""
CarDiff Dual-Path Training Loop
=================================

Implements the training procedure described in the paper:

Path A  –  Paired mask–image supervision.
           Real image → VAE encode → add noise → denoise with FiLM U-Net
           → VAE decode → seg head, discriminators.

Path B  –  Self-supervised generation from augmented masks.
           Augmented mask → condition → denoise pure noise → VAE decode
           → seg head, discriminators.

Both paths are jointly optimised per iteration.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

import diffusers

from .losses import CarDiffLoss


# ======================================================================
# Training Config
# ======================================================================

@dataclass
class CarDiffTrainingConfig:
    """All hyper-parameters for CarDiff training."""

    # --- Architecture knobs ---
    image_size: int = 384
    img_channels: int = 1
    num_classes: int = 4           # including background
    latent_channels: int = 4
    mask_enc_channels: int = 256
    unet_block_channels: tuple = (128, 256, 256, 512)
    patch_size: int = 8

    # --- Training schedule ---
    train_batch_size: int = 4
    eval_batch_size: int = 8
    num_epochs: int = 300
    lr_generator: float = 1e-4
    lr_discriminator: float = 2e-4
    lr_warmup_steps: int = 500
    gradient_accumulation_steps: int = 1

    # --- Loss weights ---
    w_diff: float = 1.0
    w_seg: float = 1.0
    w_adv_i: float = 0.1
    w_adv_p: float = 0.1
    w_self_seg: float = 1.0
    w_sparse: float = 0.01
    w_smooth: float = 0.01

    # --- Diffusion ---
    num_train_timesteps: int = 1000
    model_type: str = "DDPM"  # "DDPM" or "DDIM"

    # --- I/O ---
    output_dir: str = "cardiff-output"
    save_image_epochs: int = 20
    save_model_epochs: int = 20
    resume_epoch: Optional[int] = None
    vae_pretrained_path: Optional[str] = None

    # --- Self-supervised Path B ---
    enable_path_b: bool = True
    path_b_start_epoch: int = 10    # warm up with Path A only first

    # --- Legacy compat ---
    dataset: str = "dc"
    segmentation_guided: bool = True
    segmentation_channel_mode: str = "single"
    num_segmentation_classes: int = 4
    mixed_precision: str = "fp16"
    seed: int = 0
    use_ablated_segmentations: bool = False


# ======================================================================
# Training Loop
# ======================================================================

def cardiff_train_loop(
    config: CarDiffTrainingConfig,
    cardiff_model: nn.Module,
    noise_scheduler,
    train_dataloader,
    eval_dataloader,
    device: str = "cuda",
):
    """
    Main training loop for CarDiff.

    Parameters
    ----------
    config : CarDiffTrainingConfig
    cardiff_model : CarDiffModel instance (already on device).
    noise_scheduler : diffusers noise scheduler.
    train_dataloader, eval_dataloader : PyTorch DataLoaders.
    device : target device string.
    """
    from .model import CarDiffModel

    model: CarDiffModel = cardiff_model

    # Separate optimizers for generator vs. discriminator
    opt_g = torch.optim.AdamW(model.generator_parameters(), lr=config.lr_generator, betas=(0.5, 0.999))
    opt_d = torch.optim.AdamW(model.discriminator_parameters(), lr=config.lr_discriminator, betas=(0.5, 0.999))

    total_steps = len(train_dataloader) * config.num_epochs
    sched_g = diffusers.optimization.get_cosine_schedule_with_warmup(
        opt_g, num_warmup_steps=config.lr_warmup_steps, num_training_steps=total_steps
    )

    loss_fn = CarDiffLoss(
        num_classes=config.num_classes,
        w_diff=config.w_diff,
        w_seg=config.w_seg,
        w_adv_i=config.w_adv_i,
        w_adv_p=config.w_adv_p,
        w_self_seg=config.w_self_seg,
        w_sparse=config.w_sparse,
        w_smooth=config.w_smooth,
    )

    writer = SummaryWriter(comment=f"cardiff-{config.dataset}-{config.image_size}")
    global_step = 0
    start_epoch = config.resume_epoch if config.resume_epoch else 0

    for epoch in range(start_epoch, config.num_epochs):
        progress = tqdm(total=len(train_dataloader), desc=f"Epoch {epoch}")
        model.train()
        # Keep VAE frozen
        model.vae.eval()

        for step, batch in enumerate(train_dataloader):
            images = batch["images"].to(device)      # (B, C, H, W)
            # Build single-channel mask from seg batch
            masks = _extract_mask(batch, device)      # (B, 1, H, W)

            # ==============================================================
            # PATH A  –  Paired learning
            # ==============================================================

            # 1. Encode image to latent
            z = model.encode_image(images)            # (B, C_lat, H_lat, W_lat)

            # 2. Add noise
            noise = torch.randn_like(z)
            bs = z.shape[0]
            t = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bs,), device=device).long()
            z_t = noise_scheduler.add_noise(z, noise, t)

            # 3. Encode mask + causal context
            cond_a, F_map_a, alphas_a, h_locals_a, grid_shapes_a = model.encode_mask(masks)

            # 4. Predict noise
            noise_pred_a = model.predict_noise(z_t, t, cond_a)

            # 5. Decode for seg head & discriminators (use single-step estimate)
            with torch.no_grad():
                # Approximate x0 from noise prediction
                alpha_bar_t = noise_scheduler.alphas_cumprod[t].view(-1, 1, 1, 1).to(device)
                z0_hat = (z_t - torch.sqrt(1 - alpha_bar_t) * noise_pred_a) / torch.sqrt(alpha_bar_t)
            x_hat = model.decode_latent(z0_hat.detach())

            # 6. Segmentation consistency
            seg_logits_a = model.segment(x_hat)

            # Resize mask to match seg head output
            mask_for_seg_a = F.interpolate(masks, size=seg_logits_a.shape[-2:], mode="nearest")

            # 7. Discriminator forward (fake)
            fake_di_a = model.discriminate_image(x_hat)
            fake_dp_a = model.discriminate_pair(masks, x_hat)

            # ==============================================================
            # PATH B  –  Self-supervised (optional, after warm-up)
            # ==============================================================
            do_path_b = config.enable_path_b and epoch >= config.path_b_start_epoch

            noise_b = noise_pred_b = seg_logits_b = mask_b_for_seg = None
            fake_di_b = fake_dp_b = None

            if do_path_b:
                masks_aug = model.augment_mask(masks)  # (B, 1, H, W)

                # Condition from augmented mask
                cond_b, _, alphas_b, h_locals_b, grid_shapes_b = model.encode_mask(masks_aug)

                # Denoise from pure noise
                z_T = torch.randn_like(z)
                noise_b = torch.randn_like(z)
                t_b = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bs,), device=device).long()
                z_t_b = noise_scheduler.add_noise(z_T, noise_b, t_b)

                noise_pred_b = model.predict_noise(z_t_b, t_b, cond_b)

                with torch.no_grad():
                    alpha_bar_tb = noise_scheduler.alphas_cumprod[t_b].view(-1, 1, 1, 1).to(device)
                    z0_hat_b = (z_t_b - torch.sqrt(1 - alpha_bar_tb) * noise_pred_b) / torch.sqrt(alpha_bar_tb)
                x_tilde = model.decode_latent(z0_hat_b.detach())

                seg_logits_b = model.segment(x_tilde)
                mask_b_for_seg = F.interpolate(masks_aug, size=seg_logits_b.shape[-2:], mode="nearest")

                fake_di_b = model.discriminate_image(x_tilde)
                fake_dp_b = model.discriminate_pair(masks_aug, x_tilde)

                # Merge causal regularizers from both paths
                alphas_a = alphas_a + alphas_b
                h_locals_a = h_locals_a + h_locals_b
                grid_shapes_a = grid_shapes_a + grid_shapes_b

            # ==============================================================
            # Generator loss
            # ==============================================================
            g_losses = loss_fn.generator_loss(
                noise=noise,
                noise_pred=noise_pred_a,
                seg_logits_a=seg_logits_a,
                mask_a=mask_for_seg_a,
                fake_logits_di_a=fake_di_a,
                fake_logits_dp_a=fake_dp_a,
                noise_b=noise_b,
                noise_pred_b=noise_pred_b,
                seg_logits_b=seg_logits_b,
                mask_b=mask_b_for_seg,
                fake_logits_di_b=fake_di_b,
                fake_logits_dp_b=fake_dp_b,
                alphas=alphas_a,
                h_locals=h_locals_a,
                grid_shapes=grid_shapes_a,
            )
            g_loss = g_losses["total"]

            opt_g.zero_grad()
            g_loss.backward()
            nn.utils.clip_grad_norm_(model.generator_parameters(), 1.0)
            opt_g.step()
            sched_g.step()

            # ==============================================================
            # Discriminator loss
            # ==============================================================
            with torch.no_grad():
                x_hat_d = x_hat.detach()

            real_di = model.discriminate_image(images)
            fake_di = model.discriminate_image(x_hat_d)
            real_dp = model.discriminate_pair(masks, images)
            fake_dp = model.discriminate_pair(masks, x_hat_d)

            d_losses = loss_fn.discriminator_loss(real_di, fake_di, real_dp, fake_dp)
            d_loss = d_losses["total"]

            opt_d.zero_grad()
            d_loss.backward()
            nn.utils.clip_grad_norm_(model.discriminator_parameters(), 1.0)
            opt_d.step()

            # ==============================================================
            # Logging
            # ==============================================================
            global_step += 1
            writer.add_scalar("loss/g_total", g_loss.item(), global_step)
            writer.add_scalar("loss/d_total", d_loss.item(), global_step)
            for k, v in g_losses.items():
                if k != "total":
                    writer.add_scalar(f"loss/g_{k}", v.item() if torch.is_tensor(v) else v, global_step)

            progress.update(1)
            progress.set_postfix(g=f"{g_loss.item():.4f}", d=f"{d_loss.item():.4f}")

        progress.close()

        # ==============================================================
        # Checkpointing & evaluation
        # ==============================================================
        if (epoch + 1) % config.save_model_epochs == 0 or epoch == config.num_epochs - 1:
            ckpt_dir = os.path.join(config.output_dir, f"checkpoint_epoch_{epoch + 1}")
            os.makedirs(ckpt_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(ckpt_dir, "cardiff_model.pt"))
            # Also save the noise scheduler config
            noise_scheduler.save_pretrained(os.path.join(ckpt_dir, "scheduler"))
            print(f"[Epoch {epoch+1}] Checkpoint saved to {ckpt_dir}")

        if (epoch + 1) % config.save_image_epochs == 0 or epoch == config.num_epochs - 1:
            _save_samples(
                config, model, noise_scheduler, eval_dataloader, epoch, device
            )

    writer.close()
    print("Training complete.")


# ======================================================================
# Helpers
# ======================================================================

def _extract_mask(batch: dict, device: str) -> torch.Tensor:
    """
    Given a batch dict with key(s) like 'seg_all' or 'seg_<type>',
    assemble a single-channel multi-class mask (B, 1, H, W).
    """
    seg_keys = [k for k in batch.keys() if k.startswith("seg_")]
    if not seg_keys:
        raise ValueError("No segmentation keys found in batch")

    # Merge all seg masks into one (non-overlapping assumption)
    masks = torch.zeros_like(batch[seg_keys[0]]).to(device)
    for k in seg_keys:
        seg = batch[k].to(device)
        masks[masks == 0] = seg[masks == 0]
    return masks


@torch.no_grad()
def _save_samples(
    config, model, noise_scheduler, eval_dataloader, epoch, device
):
    """Generate and save sample images for visual inspection."""
    from torchvision.utils import save_image

    model.eval()
    sample_dir = os.path.join(config.output_dir, "samples")
    os.makedirs(sample_dir, exist_ok=True)

    # Get one eval batch
    eval_iter = iter(eval_dataloader)
    batch = next(eval_iter)
    masks = _extract_mask(batch, device)

    # Encode mask
    cond, _, _, _, _ = model.encode_mask(masks)

    # Sample from pure noise
    B = masks.shape[0]
    latent_shape = (B,) + tuple(model.latent_shape)
    z = torch.randn(latent_shape, device=device)

    # Iterative denoising
    noise_scheduler.set_timesteps(config.num_train_timesteps)
    for t in noise_scheduler.timesteps:
        t_batch = torch.full((B,), t, device=device, dtype=torch.long)
        noise_pred = model.predict_noise(z, t_batch, cond)
        z = noise_scheduler.step(noise_pred, t, z).prev_sample

    # Decode
    x_gen = model.decode_latent(z)
    x_gen = (x_gen / 2 + 0.5).clamp(0, 1)

    save_image(x_gen, os.path.join(sample_dir, f"{epoch+1:04d}_gen.png"), nrow=4)
    save_image(masks, os.path.join(sample_dir, f"{epoch+1:04d}_masks.png"), nrow=4, normalize=True)

    if "images" in batch:
        real = batch["images"].to(device)
        real = (real / 2 + 0.5).clamp(0, 1)
        save_image(real, os.path.join(sample_dir, f"{epoch+1:04d}_real.png"), nrow=4)

    model.train()
    model.vae.eval()

    print(f"[Epoch {epoch+1}] Samples saved to {sample_dir}")
