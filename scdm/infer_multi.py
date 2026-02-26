from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image

from scdm import MaskEncoder, MaskEncoderMetadata, utils
from scdm.guided_diffusion import script_util
from scdm.trainer import load_trainer_from_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SCDM diffusion inference on segmentation masks (multi-class scenario).")
    parser.add_argument('--checkpoint', type=Path, default=Path('./scdm_runs_multi/checkpoints/scdm_step_0392500.pt'), 
                        help='Path to a trained SCDM checkpoint (.pt).')
    parser.add_argument('--masks-dir', type=Path, default=Path('./test_data/temp00005'),
                        help='Directory of input segmentation masks.')
    parser.add_argument('--output-dir', type=Path, default=Path('./test_data/syn'),
                        help='Directory to write generated images.')
    parser.add_argument('--samples-per-mask', type=int, default=1,
                        help='Number of images to sample per mask.')
    parser.add_argument('--device', type=str, default="cuda",
                        help='Target device identifier (e.g. "cuda", "cuda:0", "cpu" or "auto").')
    parser.add_argument('--seed', type=int, default=2234)
    parser.add_argument('--use-ema', action='store_true', help='Use EMA weights for inference if available.')
    parser.add_argument('--sample-steps', type=str, default='1000',
                        help='Number of diffusion sampling steps (e.g. 50, 250 or ddim50). If omitted, uses training timestep_respacing.')
    return parser.parse_args()


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
    encoder: MaskEncoder,
    index_to_color: Dict[int, Tuple[int, int, int]],
) -> Image.Image:
    base = np.asarray(image).astype(np.uint8)
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

    contour_colors = {
        102: np.array([0, 0, 255], dtype=np.uint8),   # Superficial caries -> blue
        153: np.array([0, 255, 0], dtype=np.uint8),   # Medium caries -> green
        255: np.array([255, 0, 0], dtype=np.uint8),   # Deep caries -> red
    }

    unique_indices = np.unique(mask_indices)
    for idx in unique_indices:
        if int(idx) in background_indices:
            continue
        boundary_region = (mask_indices == idx) & boundaries
        if not np.any(boundary_region):
            continue

        raw_value = None
        if not encoder.use_color_map and idx < len(encoder.class_values):
            raw_value = int(encoder.class_values[idx])
        elif encoder.use_color_map and idx in index_to_color:
            raw_value = tuple(int(c) for c in index_to_color[idx])

        if isinstance(raw_value, int):
            color_arr = contour_colors.get(raw_value, np.array([255, 255, 255], dtype=np.uint8))
        elif isinstance(raw_value, tuple) and len(raw_value) == 3:
            if raw_value[0] == raw_value[1] == raw_value[2]:
                color_arr = contour_colors.get(raw_value[0], np.array(raw_value, dtype=np.uint8))
            else:
                color_arr = np.array(raw_value, dtype=np.uint8)
        else:
            fallback = index_to_color.get(idx, (255, 255, 255))
            color_arr = np.array(fallback, dtype=np.uint8)

        overlay[boundary_region] = color_arr

    return Image.fromarray(overlay)


def _to_grayscale_rgb(image: Image.Image) -> Image.Image:
    grayscale = image.convert('L')
    return grayscale.convert('RGB')


def build_palette(metadata: Dict[str, object], encoder: MaskEncoder) -> Dict[int, Tuple[int, int, int]]:
    palette_data = metadata.get('palette') if isinstance(metadata, dict) else None
    if isinstance(palette_data, dict) and palette_data:
        return {int(k): tuple(int(c) for c in v) for k, v in palette_data.items()}

    class_values = getattr(encoder, 'class_values', list(range(encoder.label_nc)))
    palette: Dict[int, Tuple[int, int, int]] = {}
    for idx, value in enumerate(class_values):
        gray = int(value)
        palette[idx] = (gray, gray, gray)
    return palette


def main() -> None:
    args = parse_args()

    utils.set_seed(args.seed)

    device = utils.resolve_device(args.device)

    utils.ensure_dir(args.output_dir)

    if args.samples_per_mask < 1:
        raise ValueError("samples-per-mask must be a positive integer.")

    if not args.masks_dir.is_dir():
        raise FileNotFoundError(f"Mask directory not found: {args.masks_dir}")

    trainer, checkpoint = load_trainer_from_checkpoint(args.checkpoint, device=device)

    trainer.model.eval()
    if hasattr(trainer, 'ema_model'):
        trainer.ema_model.eval()

    use_ema = bool(args.use_ema and hasattr(trainer, 'ema_model'))
    if use_ema:
        trainer.ema_model.to(device)
        trainer.model.to('cpu')
        model = trainer.ema_model
    else:
        trainer.model.to(device)
        if hasattr(trainer, 'ema_model'):
            trainer.ema_model.to('cpu')
        model = trainer.model

    metadata = trainer.cfg.dataset_metadata
    encoder_meta_dict = metadata.get('encoder', {})
    encoder_meta = MaskEncoderMetadata(
        label_nc=int(encoder_meta_dict.get('label_nc', trainer.cfg.label_nc)),
        ignore_label=encoder_meta_dict.get('ignore_label'),
        color_to_index=encoder_meta_dict.get('color_to_index'),
        value_to_index=encoder_meta_dict.get('value_to_index'),
        class_values=list(encoder_meta_dict.get('class_values', list(range(trainer.cfg.label_nc)))),
    )
    encoder = MaskEncoder.from_metadata(encoder_meta)

    palette = build_palette(metadata, encoder)
    allowed_values = set(range(trainer.cfg.label_nc))

    masks = list_masks(args.masks_dir)
    if not masks:
        raise RuntimeError(f"No mask images found in {args.masks_dir}")

    height, width = trainer.cfg.image_size
    print(f"Model expects {trainer.cfg.label_nc} classes. Processing {len(masks)} mask(s)...")
    # Allow overriding the number of sampling steps without changing the base trained schedule.
    diffusion = trainer.diffusion
    if args.sample_steps:
        # Build params from training config; keep base diffusion_steps identical to training to preserve learned noise schedule.
        diff_params = dict(trainer.cfg.diffusion_params)
        base_steps = diff_params.get('diffusion_steps', 1000)
        # Override timestep_respacing with requested sampling steps.
        diff_params['timestep_respacing'] = str(args.sample_steps)
        # create_gaussian_diffusion expects unpacked arguments; mirror script_util.create_gaussian_diffusion usage.
        diffusion = script_util.create_gaussian_diffusion(
            steps=base_steps,
            learn_sigma=diff_params.get('learn_sigma', False),
            noise_schedule=diff_params.get('noise_schedule', 'linear'),
            use_kl=diff_params.get('use_kl', False),
            predict_xstart=diff_params.get('predict_xstart', False),
            rescale_timesteps=diff_params.get('rescale_timesteps', False),
            rescale_learned_sigmas=diff_params.get('rescale_learned_sigmas', False),
            timestep_respacing=diff_params.get('timestep_respacing', ''),
            cond_diffuse=diff_params.get('cond_diffuse', False),
            cond_opt=diff_params.get('cond_opt', ''),
            no_instance=diff_params.get('no_instance', False),
            dataset_mode=diff_params.get('dataset_mode', 'ade20k'),
        )
        print(f"[Info] Sampling with timestep_respacing='{diff_params['timestep_respacing']}' (base steps={base_steps}).")

    for mask_path in masks:
        with Image.open(mask_path) as img:
            if encoder.use_color_map:
                img = img.convert('RGB')
            mask_img = img.resize((width, height), Image.NEAREST)

        mask_tensor = encoder.encode(mask_img)
        unique_values = set(int(v) for v in np.unique(mask_tensor.cpu().numpy()))
        if not unique_values.issubset(allowed_values):
            unexpected = sorted(unique_values - allowed_values)
            raise ValueError(f"Mask {mask_path} contains unexpected pixel values: {unexpected}.")
        sorted_indices = sorted(unique_values)
        if encoder.use_color_map:
            pixel_values = [tuple(map(int, palette.get(idx, (-1, -1, -1)))) for idx in sorted_indices]
        else:
            pixel_values = [encoder.class_values[idx] if idx < len(encoder.class_values) else idx for idx in sorted_indices]
        print(f"Mask {mask_path.name}: {len(pixel_values)} unique pixel value(s) -> {pixel_values}")

        mask_tensor = mask_tensor.unsqueeze(0)
        cond = utils.masks_to_onehot(mask_tensor, trainer.cfg.label_nc)
        if trainer.cfg.condition_with_edges:
            edges = utils.mask_to_edge_map(mask_tensor)
            cond = torch.cat([cond, edges], dim=1)

        cond = cond.to(device)

        for sample_idx in range(args.samples_per_mask):
            with torch.no_grad():
                sample = diffusion.p_sample_loop(
                    model,
                    (1, 3, height, width),
                    clip_denoised=True,
                    model_kwargs={'y': cond},
                    progress=False,
                )
            fake_img = utils.tensor_to_pil(sample[0])
            # Convert the synthetic image to grayscale while keeping three channels for compatibility.
            fake_img = _to_grayscale_rgb(fake_img)

            mask_indices = mask_tensor.squeeze(0).cpu().numpy().astype(np.int64)
            overlay_img = _overlay_mask(fake_img, mask_indices, encoder, palette)
            mask_color = utils.mask_to_color_image(mask_tensor.squeeze(0), palette)

            stem = mask_path.stem
            suffix = f"_sample{sample_idx:02d}" if args.samples_per_mask > 1 else ""
            fake_path = args.output_dir / f"{stem}{suffix}.png"
            fake_img.save(fake_path)

            overlay_path = args.output_dir / f"{stem}{suffix}_overlay.png"
            overlay_img.save(overlay_path)

            mosaic_path = args.output_dir / f"{stem}{suffix}_mosaic.png"
            utils.save_mosaic([mask_color, fake_img, overlay_img], grid_size=(1, 3), path=mosaic_path)


if __name__ == '__main__':
    main()
