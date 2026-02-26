from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image

from sisdm import MaskEncoder, load_trainer_from_checkpoint, utils
from sisdm.guided_diffusion import script_util


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SISDM diffusion inference on segmentation masks (multi-class scenario).")
    parser.add_argument('--checkpoint', type=Path, default=Path('./sisdm_runs_multi/checkpoints/checkpoint_step_575000.pt'),
                        help='Path to a trained SISDM checkpoint (.pt).')
    parser.add_argument('--masks-dir', type=Path, default=Path('./test_data/labels00001'),
                        help='Directory of input segmentation masks.')
    parser.add_argument('--output-dir', type=Path, default=Path('./test_data/syn'),
                        help='Directory to write generated images.')
    parser.add_argument('--samples-per-mask', type=int, default=1,
                        help='Number of images to sample per mask.')
    parser.add_argument('--device', type=str, default="cuda",
                        help='Target device identifier (e.g. "cuda", "cuda:0", "cpu" or "auto").')
    parser.add_argument('--seed', type=int, default=2234)
    parser.add_argument('--use-ema', action='store_true', default=False,
                        help='Use EMA weights for inference if available (defaults to off, matching training previews).')
    parser.add_argument('--sample-steps', type=str, default=1000,
                        help='Override sampling steps via timestep respacing (e.g., 250, 100, or ddim50). Keeps base training steps unchanged.')
    
    return parser.parse_args()


def list_masks(directory: Path) -> List[Path]:
    """List all mask image files in the given directory."""
    exts = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}
    return sorted([path for path in directory.glob('*') if path.suffix.lower() in exts])


def _compute_boundaries(mask_indices: np.ndarray) -> np.ndarray:
    """Compute boundary pixels for segmentation mask."""
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
    encoder: MaskEncoder,
    index_to_color: Dict[int, Tuple[int, int, int]],
    edge_alpha: float = 0.85,
) -> Image.Image:
    """Create overlay image with mask boundaries highlighted."""
    base = np.asarray(image).astype(np.float32)
    overlay = base.copy()
    boundaries = _compute_boundaries(mask_indices)

    # Determine which indices correspond to background/ignore so we avoid drawing duplicate borders.
    background_indices: set[int] = set()
    if getattr(encoder, 'class_values', None):
        for idx, value in enumerate(encoder.class_values):
            if value == 0:
                background_indices.add(idx)
            if encoder.ignore_label is not None and value == encoder.ignore_label:
                background_indices.add(idx)
    for idx, color in index_to_color.items():
        if all(int(c) == 0 for c in color):
            background_indices.add(int(idx))

    # Default contour colors for multi-class segmentation (can be customized)
    contour_colors = {
        102: np.array([0, 0, 255], dtype=np.float32),   # Blue
        153: np.array([0, 255, 0], dtype=np.float32),   # Green
        255: np.array([255, 0, 0], dtype=np.float32),   # Red
    }

    unique_indices = np.unique(mask_indices)
    for idx in unique_indices:
        if int(idx) in background_indices:
            continue
        boundary_region = (mask_indices == idx) & boundaries
        if not np.any(boundary_region):
            continue

        # Determine color based on encoder type and class values
        color_arr: np.ndarray
        if not encoder.use_color_map and idx < len(encoder.class_values):
            raw_value = int(encoder.class_values[idx])
            color_arr = contour_colors.get(raw_value, np.array([255, 255, 255], dtype=np.float32))
        elif idx in index_to_color:
            color_arr = np.array(index_to_color[idx], dtype=np.float32)
        else:
            # Fallback color
            color_arr = np.array([255, 255, 255], dtype=np.float32)

        overlay[boundary_region] = (1.0 - edge_alpha) * overlay[boundary_region] + edge_alpha * color_arr

    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    return Image.fromarray(overlay)


def build_palette(metadata: Dict[str, object], encoder: MaskEncoder) -> Dict[int, Tuple[int, int, int]]:
    """Build color palette for visualization."""
    palette_data = metadata.get('palette') if isinstance(metadata, dict) else None
    if isinstance(palette_data, dict) and palette_data:
        return {int(k): tuple(int(c) for c in v) for k, v in palette_data.items()}

    class_values = getattr(encoder, 'class_values', list(range(encoder.label_nc)))
    palette: Dict[int, Tuple[int, int, int]] = {}
    for idx, value in enumerate(class_values):
        gray = int(value)
        palette[idx] = (gray, gray, gray)
    return palette


def _encoder_from_trainer_metadata(trainer) -> Tuple[MaskEncoder, Dict[str, object]]:
    """Build a MaskEncoder and palette metadata from the trainer's stored config."""
    metadata = trainer.cfg.dataset_metadata if isinstance(trainer.cfg.dataset_metadata, dict) else {}
    label_nc = int(metadata.get('label_nc', trainer.cfg.label_nc))
    class_values = list(metadata.get('class_values', list(range(label_nc))))
    ignore_label = metadata.get('ignore_label', 0)
    value_to_index = {int(v): i for i, v in enumerate(class_values)}

    encoder = MaskEncoder(
        label_nc=label_nc,
        ignore_label=ignore_label,
        value_to_index=value_to_index,
        class_values=class_values,
    )
    return encoder, metadata


def load_masks(masks_dir: Path, mask_encoder: MaskEncoder) -> List[torch.Tensor]:
    """Load and encode segmentation masks from directory."""
    mask_files = []
    for ext in ['.png', '.jpg', '.jpeg']:
        mask_files.extend(sorted(masks_dir.glob(f'*{ext}')))
    
    if not mask_files:
        raise ValueError(f"No mask files found in {masks_dir}")
    
    masks = []
    for mask_path in mask_files:
        with Image.open(mask_path) as img:
            mask = img.copy()
        
        # Encode mask
        mask_tensor = mask_encoder.encode(mask)
        masks.append(mask_tensor)
    
    print(f"Loaded {len(masks)} masks from {masks_dir}")
    return masks


def main() -> None:
    args = parse_args()
    
    utils.set_seed(args.seed)
    
    device = utils.resolve_device(args.device)
    
    utils.ensure_dir(args.output_dir)
    
    if args.samples_per_mask < 1:
        raise ValueError("samples-per-mask must be a positive integer.")
    
    if not args.masks_dir.is_dir():
        raise FileNotFoundError(f"Mask directory not found: {args.masks_dir}")
    
    # Load trained model + config
    print(f"Loading trained model from {args.checkpoint}")
    trainer = load_trainer_from_checkpoint(args.checkpoint, device)

    # Build encoder from trainer metadata to match training exactly
    encoder, metadata = _encoder_from_trainer_metadata(trainer)

    # Select model (EMA optional) and move to device
    trainer.model.eval()
    if hasattr(trainer, 'ema_model'):
        trainer.ema_model.eval()

    use_ema = bool(args.use_ema and hasattr(trainer, 'ema_model'))
    if use_ema:
        trainer.ema_model.to(device)
        trainer.model.to('cpu')
        model = trainer.ema_model
        print("Using EMA model for inference")
    else:
        trainer.model.to(device)
        if hasattr(trainer, 'ema_model'):
            trainer.ema_model.to('cpu')
        model = trainer.model
        print("Using non-EMA model (matches training previews)")

    # Optionally override sampling steps by rebuilding diffusion with new respacing
    diffusion = trainer.diffusion
    if args.sample_steps:
        diff_params = dict(trainer.cfg.diffusion_params)
        base_steps = int(diff_params.get('diffusion_steps', 1000))
        timestep_respacing = str(args.sample_steps)
        diffusion = script_util.create_gaussian_diffusion(
            steps=base_steps,
            learn_sigma=bool(diff_params.get('learn_sigma', False)),
            noise_schedule=diff_params.get('noise_schedule', 'linear'),
            use_kl=bool(diff_params.get('use_kl', False)),
            predict_xstart=bool(diff_params.get('predict_xstart', False)),
            rescale_timesteps=bool(diff_params.get('rescale_timesteps', False)),
            rescale_learned_sigmas=bool(diff_params.get('rescale_learned_sigmas', False)),
            timestep_respacing=timestep_respacing,
        )
        print(f"[Info] Sampling with timestep_respacing='{timestep_respacing}' (base diffusion_steps={base_steps}).")

    # Build palette and validation settings
    palette = build_palette(metadata, encoder)
    allowed_indices = set(range(encoder.label_nc))
    
    # List all masks
    masks = list_masks(args.masks_dir)
    if not masks:
        raise RuntimeError(f"No mask images found in {args.masks_dir}")
    
    height, width = trainer.cfg.image_size
    print(f"Model expects {encoder.label_nc} classes. Processing {len(masks)} mask(s)...")
    
    # Process each mask
    for mask_path in masks:
        with Image.open(mask_path) as img:
            if encoder.use_color_map:
                img = img.convert('RGB')
            mask_img = img.resize((width, height), Image.NEAREST)
        
        # Encode mask
        mask_tensor = encoder.encode(mask_img)
        
        # Validate mask values (check against class indices)
        unique_values = set(int(v) for v in np.unique(mask_tensor.cpu().numpy()))
        if not unique_values.issubset(allowed_indices):
            unexpected = sorted(unique_values - allowed_indices)
            print(f"Warning: Mask {mask_path} contains unexpected class indices: {unexpected}")
            print(f"Expected indices: {sorted(allowed_indices)}, Raw class values: {encoder.class_values}")
        
        # Log mask info
        sorted_indices = sorted(unique_values)
        if encoder.use_color_map:
            pixel_values = [tuple(map(int, palette.get(idx, (-1, -1, -1)))) for idx in sorted_indices]
        else:
            pixel_values = [encoder.class_values[idx] if idx < len(encoder.class_values) else idx for idx in sorted_indices]
        print(f"Mask {mask_path.name}: {len(pixel_values)} unique pixel value(s) -> {pixel_values}")
        
        # Prepare conditioning exactly like training
        mask_batch = mask_tensor.unsqueeze(0)  # (1, H, W)
        cond = utils.compute_diffusion_input_conditioning(mask_batch, trainer.cfg).to(device)

        # Generate samples for this mask via diffusion (mirrors training sampling)
        for sample_idx in range(args.samples_per_mask):
            with torch.no_grad():
                sample = diffusion.p_sample_loop(
                    model,
                    (1, 3, height, width),
                    clip_denoised=True,
                    model_kwargs={'y': cond},
                    device=device,
                )

            # Convert to PIL image
            fake_img = utils.tensor_to_pil_image(sample[0])
            
            # Create overlay with mask boundaries
            mask_indices = mask_tensor.cpu().numpy().astype(np.int64)
            overlay_img = _overlay_mask(fake_img, mask_indices, encoder, palette)
            
            # Create colored mask visualization
            mask_color = utils.mask_to_color_image(mask_tensor, palette=palette)
            
            # Save outputs
            stem = mask_path.stem
            suffix = f"_sample{sample_idx:02d}" if args.samples_per_mask > 1 else ""
            
            # Save generated image
            fake_path = args.output_dir / f"{stem}{suffix}.png"
            fake_img.save(fake_path)
            
            # Save overlay
            overlay_path = args.output_dir / f"{stem}{suffix}_overlay.png"
            overlay_img.save(overlay_path)
            
            # Save mosaic (mask, generated, overlay)
            mosaic_path = args.output_dir / f"{stem}{suffix}_mosaic.png"
            utils.save_mosaic([mask_color, fake_img, overlay_img], grid_size=(1, 3), path=mosaic_path)
            
            print(f"Generated sample {sample_idx + 1}/{args.samples_per_mask} for {mask_path.name}")
    
    print(f"All images saved to {args.output_dir}")


if __name__ == "__main__":
    main()


# Keep the model and training schedule intact; only adjust sampling steps:
# python 0.4.SISDM/infer_multi.py \
#   --checkpoint ./sisdm_runs_multi/checkpoints/checkpoint_step_005000.pt \
#   --masks-dir ./test_masks_multi \
#   --output-dir ./infer_outputs_multi \
#   --sample-steps 250

# python 0.4.SISDM/infer_multi.py --sample-steps ddim50


# python 0.4.SISDM/infer_multi.py --use-ema --sample-steps 100




