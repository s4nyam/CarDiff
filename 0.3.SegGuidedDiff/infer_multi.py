#!/usr/bin/env python3
"""
Inference script for SSD-Multi (segmentation-guided diffusion) inspired by the provided infer_multi.py flow.

Usage mirrors the SCDM example so you can reuse the same command structure:
  --checkpoint       Path to a trained checkpoint directory (containing 'unet' and 'scheduler')
  --masks-dir        Directory of input segmentation masks
  --output-dir       Directory to write generated images
  --samples-per-mask Number of images to sample per mask
  --device           Target device identifier (e.g. "cuda", "cuda:0", "cpu" or "auto")
  --seed             Random seed
  --use-ema          Ignored (for CLI compatibility only)
  --progression      If enabled, use the same latent initialization for all masks (for disentanglement studies)

Notes:
- Mask pixel values are expected to be grayscale with background 0 and classes encoded as 102, 153, and 255 (as used in this repo).
- Boundaries are overlaid with colors: 102 -> blue, 153 -> green, 255 -> red.
- When --progression is enabled, the same random seed is used to initialize noise for all masks, so you see how 
  different segmentation conditions shape the same underlying synthesis.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch import nn
from PIL import Image

from training import TrainingConfig
from eval import SegGuidedDDPMPipeline, SegGuidedDDIMPipeline

# Reuse loader pattern from fill_mask.py, but keep this file self-sufficient if import paths change.
import os
from diffusers import UNet2DModel, DDPMScheduler, DDIMScheduler

NUM_INFERENCE_STEPS = 200
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run SSD-Multi diffusion inference on segmentation masks (multi-class scenario)."
    )
    parser.add_argument(
        '--checkpoint',
        type=Path,
        default=Path('./ddpm-dc-384-segguided/checkpoint_epoch_320'),
        help='Path to a trained checkpoint directory containing subfolders "unet" and "scheduler".'
    )
    parser.add_argument(
        '--masks-dir',
        type=Path,
        default=Path('/home/sj/working_dir/ssd-multi/progression/masks'),
        help='Directory of input segmentation masks.'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('/home/sj/working_dir/ssd-multi/progression/outputs'),
        help='Directory to write generated images.'
    )
    parser.add_argument(
        '--samples-per-mask',
        type=int,
        default=1,
        help='Number of images to sample per mask.'
    )
    parser.add_argument(
        '--device',
        type=str,
        default="cuda:2",
        help='Target device identifier (e.g. "cuda", "cuda:0", "cpu" or "auto").'
    )
    parser.add_argument('--seed', type=int, default=2234)
    parser.add_argument('--use-ema', action='store_true', help='CLI compatibility only; not used here.')
    parser.add_argument(
        '--progression',
        action='store_true',
        help='If enabled, use the same initial latent/noise for all masks to show how segmentation conditions affect synthesis while keeping base synthesis constant.'
    )
    return parser.parse_args()


def _resolve_device(device_str: str) -> torch.device:
    if device_str == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    try:
        return torch.device(device_str)
    except Exception:
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def _set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def list_masks(directory: Path) -> List[Path]:
    exts = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}
    return sorted([path for path in directory.glob('*') if path.suffix.lower() in exts])


def _compute_boundaries(mask_indices: np.ndarray) -> np.ndarray:
    boundaries = np.zeros_like(mask_indices, dtype=bool)
    if mask_indices.shape[0] > 1:
        boundaries[1:, :] |= mask_indices[1:, :] != mask_indices[:-1, :]
        boundaries[:-1, :] |= mask_indices[:-1, :] != mask_indices[1:, :]
    if mask_indices.shape[1] > 1:
        boundaries[:, 1:] |= mask_indices[:, 1:] != mask_indices[:, :-1]
        boundaries[:, :-1] |= mask_indices[:, :-1] != mask_indices[:, 1:]
    return boundaries


def _overlay_mask(
    image: Image.Image,
    mask_indices: np.ndarray,
    index_to_color: Dict[int, Tuple[int, int, int]],
) -> Image.Image:
    """Overlay colored class boundaries on top of an RGB image.

    Class boundary colors follow SSD-Multi conventions:
      102 -> blue, 153 -> green, 255 -> red.
    """
    base = np.asarray(image).astype(np.uint8)
    overlay = base.copy()
    boundaries = _compute_boundaries(mask_indices)

    contour_colors = {
        102: np.array([0, 0, 255], dtype=np.uint8),   # Blue
        153: np.array([0, 255, 0], dtype=np.uint8),   # Green
        255: np.array([255, 0, 0], dtype=np.uint8),   # Red
    }

    unique_indices = np.unique(mask_indices)
    for idx in unique_indices:
        if int(idx) == 0:
            continue  # skip background
        boundary_region = (mask_indices == idx) & boundaries
        if not np.any(boundary_region):
            continue

        # Try to use a pre-defined contour color, else fall back to palette color
        color_arr = contour_colors.get(int(idx))
        if color_arr is None:
            fallback = index_to_color.get(int(idx), (255, 255, 255))
            color_arr = np.array(fallback, dtype=np.uint8)

        overlay[boundary_region] = color_arr

    return Image.fromarray(overlay)


def _to_grayscale_rgb(image: Image.Image) -> Image.Image:
    grayscale = image.convert('L')
    return grayscale.convert('RGB')


def build_palette_from_values(values: List[int]) -> Dict[int, Tuple[int, int, int]]:
    """Create a simple grayscale palette from class values."""
    palette: Dict[int, Tuple[int, int, int]] = {}
    for v in values:
        palette[int(v)] = (int(v), int(v), int(v))
    return palette


def _colorize_mask(mask_indices: np.ndarray, palette: Dict[int, Tuple[int, int, int]]) -> Image.Image:
    """Convert a 2D indices mask to a color RGB image using a given palette."""
    h, w = mask_indices.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for idx, color in palette.items():
        rgb[mask_indices == int(idx)] = np.array(color, dtype=np.uint8)
    return Image.fromarray(rgb, mode='RGB')


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_model_and_scheduler(model_dir: Path):
    """Load UNet and scheduler from a saved checkpoint directory.

    The directory must contain subfolders:
      - unet
      - scheduler
    """
    model_dir = str(model_dir)
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    unet_path = os.path.join(model_dir, "unet")
    if not os.path.exists(unet_path):
        raise FileNotFoundError(f"UNet model not found in: {unet_path}")

    unet = UNet2DModel.from_pretrained(unet_path, use_safetensors=True)

    scheduler_path = os.path.join(model_dir, "scheduler")
    if not os.path.exists(scheduler_path):
        raise FileNotFoundError(f"Scheduler not found in: {scheduler_path}")

    try:
        scheduler = DDPMScheduler.from_pretrained(scheduler_path)
        model_type = "DDPM"
    except Exception:
        scheduler = DDIMScheduler.from_pretrained(scheduler_path)
        model_type = "DDIM"

    return unet, scheduler, model_type


def main() -> None:
    args = parse_args()

    _set_seed(args.seed)
    device = _resolve_device(args.device)
    _ensure_dir(args.output_dir)

    if args.samples_per_mask < 1:
        raise ValueError("samples-per-mask must be a positive integer.")
    if not args.masks_dir.is_dir():
        raise FileNotFoundError(f"Mask directory not found: {args.masks_dir}")

    # Load model and scheduler
    unet, scheduler, model_type = load_model_and_scheduler(args.checkpoint)

    # Wrap and move to device
    unet = nn.DataParallel(unet)
    unet.to(device)
    unet.eval()

    # Detect seg-guided based on channel configuration
    try:
        in_ch = int(getattr(unet.module.config, "in_channels"))
        out_ch = int(getattr(unet.module.config, "out_channels", 1))
    except Exception:
        in_ch, out_ch = 2, 1
    seg_guided_detected = in_ch > out_ch

    # Determine expected image size from model
    if isinstance(unet.module.config.sample_size, int):
        height = width = int(unet.module.config.sample_size)
    else:
        width = int(unet.module.config.sample_size[0])
        height = int(unet.module.config.sample_size[1])

    # Build minimal config for pipeline
    config = TrainingConfig(
        image_size=height,
        model_type=model_type,
        segmentation_guided=seg_guided_detected,
        segmentation_channel_mode="single" if seg_guided_detected else "none",
        num_segmentation_classes=4,  # 3 caries + 1 background
        eval_batch_size=1,
        dataset="dc",
    )

    # Create pipeline
    if model_type == "DDPM":
        if seg_guided_detected:
            pipeline = SegGuidedDDPMPipeline(
                unet=unet.module,
                scheduler=scheduler,
                eval_dataloader=iter(()),  # not used here
                external_config=config,
            )
        else:
            from diffusers import DDPMPipeline
            pipeline = DDPMPipeline(unet=unet.module, scheduler=scheduler)
            pipeline = pipeline.to(device)
    else:  # DDIM
        if seg_guided_detected:
            pipeline = SegGuidedDDIMPipeline(
                unet=unet.module,
                scheduler=scheduler,
                eval_dataloader=iter(()),  # not used here
                external_config=config,
            )
        else:
            from diffusers import DDIMPipeline
            pipeline = DDIMPipeline(unet=unet.module, scheduler=scheduler)
            pipeline = pipeline.to(device)

    masks = list_masks(args.masks_dir)
    if not masks:
        raise RuntimeError(f"No mask images found in {args.masks_dir}")

    print(f"Model type: {model_type} | Seg-guided: {seg_guided_detected} | Image size: {width}x{height}")
    print(f"Processing {len(masks)} mask(s)...")
    
    # If progression mode, generate a single fixed generator for all masks
    fixed_generator = None
    if args.progression:
        print("Progression mode ENABLED: Using same latent initialization for all masks")
        fixed_generator = torch.Generator(device=device)
        fixed_generator.manual_seed(args.seed)

    for mask_path in masks:
        with Image.open(mask_path) as img:
            img = img.convert('L')
            mask_img = img.resize((width, height), Image.NEAREST)

        # Preserve raw pixel values for overlays/palette
        mask_indices = np.array(mask_img, dtype=np.uint8)
        unique_values = sorted(int(v) for v in np.unique(mask_indices))
        allowed_values = [0, 102, 153, 255]
        if not set(unique_values).issubset(set(allowed_values)):
            print(f"Warning: {mask_path.name} has unexpected pixel values {unique_values}; proceeding anyway.")
        else:
            print(f"Mask {mask_path.name}: {len(unique_values)} unique pixel value(s) -> {unique_values}")

        palette = build_palette_from_values(unique_values)

        # Normalize to [0,1] tensor as expected by training transforms
        mask_tensor = torch.from_numpy(mask_indices.astype(np.float32) / 255.0)  # HxW in [0,1]
        mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0)  # 1x1xHxW

        for sample_idx in range(args.samples_per_mask):
            with torch.no_grad():
                # Reset generator to same seed before each generation if in progression mode
                if args.progression:
                    gen = torch.Generator(device=device)
                    gen.manual_seed(args.seed)
                else:
                    gen = None
                
                if seg_guided_detected:
                    images = pipeline(
                        batch_size=1,
                        generator=gen,
                        seg_batch={
                            'seg_all': mask_tensor,  # 1x1xHxW
                            'image_filenames': [mask_path.name],
                        },
                        num_inference_steps=NUM_INFERENCE_STEPS,
                    ).images
                else:
                    images = pipeline(
                        batch_size=1,
                        generator=gen,
                        num_inference_steps=NUM_INFERENCE_STEPS,
                    ).images

            fake_img: Image.Image = images[0]
            fake_img = _to_grayscale_rgb(fake_img)

            overlay_img = _overlay_mask(fake_img, mask_indices.astype(np.int64), palette)
            mask_color = _colorize_mask(mask_indices, palette)

            stem = mask_path.stem
            suffix = f"_sample{sample_idx:02d}" if args.samples_per_mask > 1 else ""

            fake_path = args.output_dir / f"{stem}{suffix}.png"
            overlay_path = args.output_dir / f"{stem}{suffix}_overlay.png"
            mosaic_path = args.output_dir / f"{stem}{suffix}_mosaic.png"

            fake_img.save(fake_path)
            overlay_img.save(overlay_path)

            # Save simple 1x3 mosaic: [mask_color, fake_img, overlay_img]
            w, h = fake_img.size
            mosaic = Image.new('RGB', size=(w * 3, h))
            mosaic.paste(mask_color.convert('RGB'), box=(0, 0))
            mosaic.paste(fake_img, box=(w, 0))
            mosaic.paste(overlay_img, box=(2 * w, 0))
            mosaic.save(mosaic_path)


if __name__ == '__main__':
    main()
