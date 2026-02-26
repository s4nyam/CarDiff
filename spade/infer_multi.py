from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import torch
import numpy as np
from PIL import Image

from spade import MaskEncoder, MaskEncoderMetadata, SPADEGenerator, utils


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SPADE generator inference on segmentation masks.")
    parser.add_argument('--checkpoint', type=Path, default=Path('./spade_runs_multi/checkpoints/spade_step_0586000.pt'), help='Path to a trained SPADE checkpoint (.pt).')
    parser.add_argument('--masks-dir', type=Path, default=Path('./test_data/labels'), help='Directory of input segmentation masks.')
    parser.add_argument('--output-dir', type=Path, default=Path('./test_data/syn'), help='Directory to write generated images.')
    parser.add_argument('--samples-per-mask', type=int, default=1, help='Number of images to sample per mask.')
    parser.add_argument('--device', type=str, default="cuda", help='"cpu" or "cuda". Auto if omitted.')
    parser.add_argument('--seed', type=int, default=2234)
    return parser.parse_args()


PIXEL_VALUE_TO_COLOR: Dict[int, tuple[int, int, int]] = {
    0: (0, 0, 0),
    102: (0, 0, 255),  # Superficial caries - blue
    153: (0, 255, 0),  # Medium caries - green
    255: (255, 0, 0),  # Deep caries - red
}


def load_generator(checkpoint_path: Path, device: torch.device) -> tuple[SPADEGenerator, dict, MaskEncoder, dict]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    cfg = checkpoint['config']
    metadata = checkpoint.get('dataset_metadata', {})

    generator = SPADEGenerator(
        label_nc=int(cfg['label_nc']),
        output_nc=3,
        ngf=int(cfg['ngf']),
        z_dim=int(cfg['z_dim']),
        image_size=tuple(cfg['image_size']),
        num_upsampling_layers=cfg.get('num_upsampling_layers', 'normal'),
        param_free_norm=cfg.get('param_free_norm', 'instance'),
    ).to(device)
    generator.load_state_dict(checkpoint['generator'])
    generator.eval()

    encoder_info = metadata.get('encoder') if isinstance(metadata, dict) else None

    if encoder_info is None:
        encoder = MaskEncoder(label_nc=int(cfg['label_nc']))
    else:
        encoder_meta = MaskEncoderMetadata(
            label_nc=int(encoder_info['label_nc']),
            ignore_label=encoder_info.get('ignore_label'),
            color_to_index=encoder_info.get('color_to_index'),
            value_to_index=encoder_info.get('value_to_index'),
            class_values=list(encoder_info.get('class_values', [])) or list(range(int(cfg['label_nc']))),
        )
        encoder = MaskEncoder.from_metadata(encoder_meta)

    return generator, cfg, encoder, metadata if isinstance(metadata, dict) else {}


def _validate_mask_values(mask_image: Image.Image, allowed_values: set[int]) -> None:
    array = np.asarray(mask_image.convert('L'), dtype=np.int32)
    unique_values = set(int(v) for v in np.unique(array))
    if not unique_values.issubset(allowed_values):
        unexpected = sorted(unique_values - allowed_values)
        raise ValueError(
            f"Mask contains unexpected pixel values: {unexpected}. Allowed values: {sorted(allowed_values)}"
        )


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
    index_to_color: Dict[int, tuple[int, int, int]],
    fill_alpha: float = 0.35,
    edge_alpha: float = 0.85,
) -> Image.Image:
    base = np.asarray(image).astype(np.float32)
    overlay = base.copy()
    boundaries = _compute_boundaries(mask_indices)

    for idx, color in index_to_color.items():
        region = mask_indices == idx
        if not np.any(region):
            continue

        color_arr = np.array(color, dtype=np.float32)
        # interior = region & ~boundaries
        # if np.any(interior):
        #     overlay[interior] = (1.0 - fill_alpha) * overlay[interior] + fill_alpha * color_arr

        boundary_region = region & boundaries
        if np.any(boundary_region):
            overlay[boundary_region] = (1.0 - edge_alpha) * overlay[boundary_region] + edge_alpha * color_arr

    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    return Image.fromarray(overlay)


def _build_palette(encoder: MaskEncoder) -> Dict[int, tuple[int, int, int]]:
    class_values = getattr(encoder, 'class_values', list(range(encoder.label_nc)))
    palette: Dict[int, tuple[int, int, int]] = {}
    for idx, value in enumerate(class_values):
        rgb = PIXEL_VALUE_TO_COLOR.get(int(value), (int(value), int(value), int(value)))
        palette[idx] = rgb
    return palette


def list_masks(directory: Path) -> List[Path]:
    exts = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}
    return sorted([path for path in directory.glob('*') if path.suffix.lower() in exts])


def main() -> None:
    args = parse_args()
    utils.set_seed(args.seed)

    device_str = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device(device_str)

    utils.ensure_dir(args.output_dir)

    generator, cfg, encoder, _ = load_generator(args.checkpoint, device)
    height, width = cfg['image_size']
    label_nc = encoder.label_nc
    class_values = [int(v) for v in getattr(encoder, 'class_values', list(range(label_nc)))]

    if label_nc != int(cfg.get('label_nc', label_nc)):
        raise ValueError(
            f"Checkpoint expects {cfg['label_nc']} classes but encoder reports {label_nc}."
        )

    allowed_pixel_values = set(PIXEL_VALUE_TO_COLOR.keys())
    unexpected_from_checkpoint = set(class_values) - allowed_pixel_values
    if unexpected_from_checkpoint:
        raise ValueError(
            "Checkpoint metadata includes unsupported pixel values: "
            f"{sorted(unexpected_from_checkpoint)}. Allowed values: {sorted(allowed_pixel_values)}"
        )

    palette = _build_palette(encoder)
    index_to_color = {
        idx: PIXEL_VALUE_TO_COLOR[int(value)]
        for idx, value in enumerate(class_values)
        if int(value) in PIXEL_VALUE_TO_COLOR and int(value) != 0
    }

    masks = list_masks(args.masks_dir)
    if not masks:
        raise RuntimeError(f"No mask images found in {args.masks_dir}")

    print(f"Discovered {label_nc} classes with mask values: {class_values}")

    for mask_path in masks:
        with Image.open(mask_path) as img:
            if encoder.use_color_map:
                raise ValueError("Multi-class inference expects grayscale masks with predefined pixel values.")
            mask_img = img.resize((width, height), Image.NEAREST)

        _validate_mask_values(mask_img, allowed_pixel_values)
        mask_tensor = encoder.encode(mask_img)
        mask_indices = mask_tensor.cpu().numpy().astype(np.int64)

        onehot = utils.masks_to_onehot(mask_tensor.unsqueeze(0), label_nc).to(device)

        for sample_idx in range(args.samples_per_mask):
            z = torch.randn(1, generator.z_dim, device=device)
            with torch.no_grad():
                fake = generator(onehot, z)

            stem = mask_path.stem
            out_name = f"{stem}_sample{sample_idx:02d}.png" if args.samples_per_mask > 1 else f"{stem}.png"
            out_path = args.output_dir / out_name
            fake_img = utils.tensor_to_pil(fake[0])
            fake_img.save(out_path)

            overlay_img = _overlay_mask(fake_img, mask_indices, index_to_color)
            overlay_name = (
                f"{stem}_overlay_sample{sample_idx:02d}.png" if args.samples_per_mask > 1 else f"{stem}_overlay.png"
            )
            overlay_path = args.output_dir / overlay_name
            overlay_img.save(overlay_path)

            mask_color = utils.mask_to_color_image(mask_tensor, palette)
            mosaic_name = (
                f"{stem}_mosaic_sample{sample_idx:02d}.png" if args.samples_per_mask > 1 else f"{stem}_mosaic.png"
            )
            mosaic_path = args.output_dir / mosaic_name
            utils.save_mosaic([mask_color, fake_img, overlay_img], grid_size=(1, 3), path=mosaic_path)


if __name__ == '__main__':
    main()
