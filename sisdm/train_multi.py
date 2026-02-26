from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import DataLoader

from sisdm import SegmentationGuidedDataset, SISDMTrainer, TrainingConfig, utils, MaskEncoder
from sisdm.guided_diffusion import script_util


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a SISDM diffusion model on paired masks/images for multi-class segmentation-guided synthesis.")
    parser.add_argument('--data-root', type=Path, default=Path('train_data'),
                        help='Root directory containing "images" and "labels" sub-folders.')
    parser.add_argument('--images-dir', type=Path, default=None,
                        help='Override for the images directory.')
    parser.add_argument('--labels-dir', type=Path, default=None,
                        help='Override for the labels directory.')
    parser.add_argument('--output-dir', type=Path, default=Path('./sisdm_runs_multi'),
                        help='Directory to store checkpoints and samples.')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--max-steps', type=int, default=None,
                        help='Optional cap on the number of optimization steps.')
    parser.add_argument('--image-size', type=int, default=384,
                        help='Target square image resolution (supports 64/128/256/384/512).')
    parser.add_argument('--image-height', type=int, default=None,
                        help='Optional specific target height. Overrides --image-size if set.')
    parser.add_argument('--image-width', type=int, default=None,
                        help='Optional specific target width. Overrides --image-size if set.')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight-decay', type=float, default=0.0)
    parser.add_argument('--ema-decay', type=float, default=0.9999)
    parser.add_argument('--save-every', type=int, default=5000,
                        help='Save a checkpoint every N steps.')
    parser.add_argument('--sample-every', type=int, default=5000,
                        help='Write sample images every N steps.')
    parser.add_argument('--log-every', type=int, default=100,
                        help='Update the progress bar with losses every N steps.')
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--label-nc', type=int, default=None,
                        help='Number of semantic classes. If omitted, inferred from masks.')
    parser.add_argument('--ignore-label', type=int, default=0,
                        help='Mask value that should be treated as ignore/background.')
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--device', type=str, default="cuda",
                        help='Target device identifier (e.g. "cuda", "cuda:0", "cpu" or "auto").')
    parser.add_argument('--resume', type=Path, default="./sisdm_runs_multi/checkpoints/checkpoint_step_525000.pt",
                        help='Path to checkpoint to resume training from.')
    parser.add_argument('--gradient-clip', type=float, default=1.0,
                        help='Gradient clipping value. Set to 0 or negative to disable.')
    parser.add_argument('--no-flip', action='store_true', help='Disable random horizontal flips during training.')
    
    # Diffusion model parameters
    parser.add_argument('--diffusion-steps', type=int, default=1000,
                        help='Number of diffusion timesteps.')
    parser.add_argument('--noise-schedule', type=str, default="linear",
                        choices=["linear", "cosine"],
                        help='Noise schedule for diffusion.')
    parser.add_argument('--learn-sigma', action='store_true',
                        help='Learn noise variance schedule.')
    parser.add_argument('--predict-xstart', action='store_true',
                        help='Predict original image instead of noise.')
    parser.add_argument('--use-fp16', action='store_true',
                        help='Use mixed precision training.')
    parser.add_argument('--attention-resolutions', type=str, default="32,16,8",
                        help='Comma-separated list of resolutions for attention.')
    parser.add_argument('--num-channels', type=int, default=128,
                        help='Number of channels in UNet.')
    parser.add_argument('--num-res-blocks', type=int, default=2,
                        help='Number of residual blocks per resolution.')
    parser.add_argument('--channel-mult', type=str, default="1,1,2,2,4,4",
                        help='Channel multiplier for each resolution.')
    parser.add_argument('--dropout', type=float, default=0.0,
                        help='Dropout rate for model.')
    parser.add_argument('--no-edges', dest='include_edges', action='store_false',
                        help='Disable instance-edge conditioning channel.')
    parser.add_argument('--schedule-sampler', choices=['uniform', 'loss-second-moment'], default='uniform',
                        help='Timestep sampling strategy for diffusion training.')
    parser.add_argument('--visualization-grid', type=str, default='2x4',
                        help='Grid layout for sample mosaics, e.g. 2x4.')
    parser.add_argument('--num-visualization-items', type=int, default=8)
    parser.add_argument('--no-preview-current', action='store_true',
                        help='Disable sample mosaics produced by the non-EMA model.')
    parser.set_defaults(include_edges=True)
    
    return parser.parse_args()


def parse_grid(grid_spec: str) -> Tuple[int, int]:
    try:
        rows_str, cols_str = grid_spec.lower().split('x')
        rows = int(rows_str)
        cols = int(cols_str)
        if rows <= 0 or cols <= 0:
            raise ValueError
        return rows, cols
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid grid specification '{grid_spec}'. Use format ROWSxCOLS, e.g. 2x4.") from exc


def resolve_paths(args: argparse.Namespace) -> Tuple[Path, Path]:
    images_dir = Path(args.images_dir) if args.images_dir is not None else Path(args.data_root) / 'images'
    labels_dir = Path(args.labels_dir) if args.labels_dir is not None else Path(args.data_root) / 'labels'

    if not images_dir.is_dir():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    if not labels_dir.is_dir():
        raise FileNotFoundError(f"Labels directory not found: {labels_dir}")

    return images_dir, labels_dir


def create_dataset_and_encoder(args: argparse.Namespace) -> Tuple[SegmentationGuidedDataset, MaskEncoder, int]:
    """Create dataset and mask encoder."""
    # Resolve directory paths
    images_dir, labels_dir = resolve_paths(args)
    
    # Find all mask files for analysis
    mask_files = []
    for ext in ['.png', '.jpg', '.jpeg']:
        mask_files.extend(labels_dir.glob(f'*{ext}'))
    
    if not mask_files:
        raise ValueError(f"No mask files found in {labels_dir}")
    
    print(f"Analyzing {len(mask_files)} mask files...")
    
    # Analyze masks to create encoder
    mask_encoder, palette = MaskEncoder.analyze(mask_files, args.label_nc, args.ignore_label)
    segmentation_classes = mask_encoder.label_nc
    print(f"Detected {segmentation_classes} segmentation classes")

    # Create dataset
    image_size = (
        args.image_height or args.image_size,
        args.image_width or args.image_size
    )
    
    dataset = SegmentationGuidedDataset(
        images_dir=images_dir,
        labels_dir=labels_dir,
        mask_encoder=mask_encoder,
        image_size=image_size,
        random_flip=not args.no_flip,
        random_crop=True,
        is_train=True
    )
    
    return dataset, mask_encoder, segmentation_classes, palette


def build_diffusion_params(args: argparse.Namespace, label_nc: int) -> dict:
    """Build diffusion model parameters."""
    params = script_util.model_and_diffusion_defaults()
    params.update({
        'image_size': args.image_height or args.image_size,
        'class_cond': True,
        'learn_sigma': args.learn_sigma,
        'num_classes': label_nc,
        'no_instance': not args.include_edges,
        'num_channels': args.num_channels,
        'num_res_blocks': args.num_res_blocks,
        'channel_mult': args.channel_mult,
        'attention_resolutions': args.attention_resolutions,
        'dropout': args.dropout,
        'diffusion_steps': args.diffusion_steps,
        'noise_schedule': args.noise_schedule,
        'timestep_respacing': '',
        'use_kl': False,
        'predict_xstart': args.predict_xstart,
        'rescale_timesteps': False,
        'rescale_learned_sigmas': False,
        'use_checkpoint': False,
        'use_scale_shift_norm': True,
        'resblock_updown': False,
        'use_fp16': args.use_fp16,
        'use_new_attention_order': False,
    })
    return params


def main():
    args = parse_args()
    
    # Set seed
    utils.set_seed(args.seed)
    
    # Setup device
    device = utils.resolve_device(args.device)
    print(f"Using device: {device}")
    
    # Create dataset
    dataset, mask_encoder, segmentation_classes, palette = create_dataset_and_encoder(args)
    
    # Create data loader
    drop_last = len(dataset) >= args.batch_size
    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == 'cuda',
        drop_last=drop_last,
    )
    
    print(f"Created dataset with {len(dataset)} samples, {segmentation_classes} classes")
    
    # Prepare diffusion parameters
    image_size = (
        args.image_height or args.image_size,
        args.image_width or args.image_size
    )
    
    conditioning_channels = segmentation_classes + (1 if args.include_edges else 0)
    print(f"Model will use {conditioning_channels} conditioning channels")
    
    diffusion_params = build_diffusion_params(args, segmentation_classes)
    
    # Parse visualization grid
    grid = parse_grid(args.visualization_grid)
    
    # Create training config
    dataset_metadata = {
        'num_samples': len(dataset),
        'label_nc': mask_encoder.label_nc,
        'conditioning_channels': conditioning_channels,
        'condition_with_edges': args.include_edges,
        'image_size': image_size,
        'palette': palette,
        'class_values': mask_encoder.class_values,
        'ignore_label': mask_encoder.ignore_label,
    }
    
    config = TrainingConfig(
        label_nc=segmentation_classes,
        conditioning_channels=conditioning_channels,
        image_size=image_size,
        output_dir=args.output_dir,
        epochs=args.epochs,
        max_steps=args.max_steps,
        lr=args.lr,
        weight_decay=args.weight_decay,
        ema_decay=args.ema_decay,
        log_every=args.log_every,
        save_every=args.save_every,
        sample_every=args.sample_every,
        device=device,
        dataset_metadata=dataset_metadata,
        resume=args.resume,
        diffusion_params=diffusion_params,
        use_fp16=args.use_fp16,
        condition_with_edges=args.include_edges,
        num_visualization_items=min(args.num_visualization_items, grid[0] * grid[1]),
        visualization_grid=grid,
        preview_current_model=not args.no_preview_current,
        preview_ema_model=False,
        schedule_sampler=args.schedule_sampler,
        gradient_clip=args.gradient_clip if args.gradient_clip and args.gradient_clip > 0 else None,
        class_cond=True,
    )
    
    # Save dataset metadata
    utils.ensure_dir(args.output_dir)
    metadata_path = args.output_dir / 'dataset_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump({
            'num_samples': len(dataset),
            'label_nc': segmentation_classes,
            'conditioning_channels': conditioning_channels,
            'condition_with_edges': args.include_edges,
            'image_size': image_size,
            'palette': palette,
            'class_values': mask_encoder.class_values,
            'ignore_label': mask_encoder.ignore_label,
        }, f, indent=2)
    
    # Create trainer
    print("Creating SISDM trainer...")
    trainer = SISDMTrainer(config)
    
    # Start training
    print("Starting training...")
    trainer.train(data_loader)
    
    print("Training completed!")


if __name__ == "__main__":
    main()
