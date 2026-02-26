from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import DataLoader

from scdm import SegmentationDiffusionDataset, SCDMTrainer, TrainingConfig, utils
from scdm.guided_diffusion import script_util


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an SCDM diffusion model on paired masks/images for multi-class segmentation.")
    parser.add_argument('--data-root', type=Path, default=Path('./train_data'),
                        help='Root directory containing "images" and "labels" sub-folders.')
    parser.add_argument('--images-dir', type=Path, default=None,
                        help='Override for the images directory.')
    parser.add_argument('--labels-dir', type=Path, default=None,
                        help='Override for the labels directory.')
    parser.add_argument('--output-dir', type=Path, default=Path('./scdm_runs_multi'),
                        help='Directory to store checkpoints and samples.')
    parser.add_argument('--batch-size', type=int, default=4) # Change in production
    parser.add_argument('--epochs', type=int, default=1000) # Change in production
    parser.add_argument('--max-steps', type=int, default=None,
                        help='Optional cap on the number of optimization steps.')
    parser.add_argument('--image-size', type=int, default=384, # Change in production
                        help='Target square image resolution (supports 64/128/256/384/512).')
    parser.add_argument('--image-height', type=int, default=None,
                        help='Optional specific target height. Overrides --image-size if set.')
    parser.add_argument('--image-width', type=int, default=None,
                        help='Optional specific target width. Overrides --image-size if set.')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight-decay', type=float, default=0.0)
    parser.add_argument('--ema-decay', type=float, default=0.9999)
    parser.add_argument('--save-every', type=int, default=2500, # Change in production
                        help='Save a checkpoint every N steps.')
    parser.add_argument('--sample-every', type=int, default=2500, # Change in production
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
    parser.add_argument('--resume', type=Path, default="/scratch/project_465001696/playground/0.1.scdm/scdm_runs_multi/checkpoints/scdm_step_0338100.pt",  # Change in production when resuming.
                        help='Checkpoint path to resume training.')
    parser.add_argument('--gradient-clip', type=float, default=1.0,
                        help='Gradient clipping value. Set to 0 or negative to disable.')
    parser.add_argument('--no-flip', action='store_true', help='Disable random horizontal flips during training.')
    parser.add_argument('--use-fp16', action='store_true', help='Enable AMP mixed precision training if CUDA is available.')
    parser.add_argument('--diffusion-steps', type=int, default=1000)
    parser.add_argument('--noise-schedule', type=str, default='linear', choices=['linear', 'cosine'])
    parser.add_argument('--timestep-respacing', type=str, default='')
    parser.add_argument('--learn-sigma', action='store_true')
    parser.add_argument('--predict-xstart', action='store_true')
    parser.add_argument('--rescale-timesteps', action='store_true')
    parser.add_argument('--rescale-learned-sigmas', action='store_true')
    parser.add_argument('--num-channels', type=int, default=64)
    parser.add_argument('--num-res-blocks', type=int, default=3)
    parser.add_argument('--channel-mult', type=str, default='1,2,4,8')
    parser.add_argument('--attention-resolutions', type=str, default='32,16,8')
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--num-heads', type=int, default=4)
    parser.add_argument('--num-head-channels', type=int, default=-1)
    parser.add_argument('--num-heads-ups', type=int, default=-1,
                        help='Number of heads in upsampling layers. Use -1 to inherit from --num-heads.')
    parser.add_argument('--resblock-updown', action='store_true')
    parser.add_argument('--use-checkpoint', action='store_true')
    parser.add_argument('--cond-diffuse', action='store_true')
    parser.add_argument('--cond-opt', type=str, default='')
    parser.add_argument('--dataset-mode', type=str, default='custom')
    parser.add_argument('--visualization-grid', type=str, default='2x4',
                        help='Grid layout for sample mosaics, e.g. 2x4.')
    parser.add_argument('--num-visualization-items', type=int, default=8)
    parser.add_argument('--no-edges', dest='include_edges', action='store_false',
                        help='Disable instance-edge conditioning channel.')
    parser.add_argument('--schedule-sampler', choices=['uniform', 'loss-second-moment'], default='uniform',
                        help='Timestep sampling strategy for diffusion training.')
    parser.add_argument('--no-preview-current', action='store_true',
                        help='Disable sample mosaics produced by the non-EMA model.')
    parser.set_defaults(include_edges=True)
    return parser.parse_args()


def resolve_paths(args: argparse.Namespace) -> Tuple[Path, Path]:
    images_dir = Path(args.images_dir) if args.images_dir is not None else Path(args.data_root) / 'images'
    labels_dir = Path(args.labels_dir) if args.labels_dir is not None else Path(args.data_root) / 'labels'

    if not images_dir.is_dir():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    if not labels_dir.is_dir():
        raise FileNotFoundError(f"Labels directory not found: {labels_dir}")

    return images_dir, labels_dir


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


def build_diffusion_params(args: argparse.Namespace, label_nc: int) -> dict:
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
        'num_heads': args.num_heads,
        'num_head_channels': args.num_head_channels,
        'num_heads_upsample': args.num_heads_ups if args.num_heads_ups >= 0 else args.num_heads,
        'attention_resolutions': args.attention_resolutions,
        'dropout': args.dropout,
        'diffusion_steps': args.diffusion_steps,
        'noise_schedule': args.noise_schedule,
        'timestep_respacing': args.timestep_respacing,
        'use_kl': False,
        'predict_xstart': args.predict_xstart,
        'rescale_timesteps': args.rescale_timesteps,
        'rescale_learned_sigmas': args.rescale_learned_sigmas,
        'use_checkpoint': args.use_checkpoint,
        'use_scale_shift_norm': True,
        'resblock_updown': args.resblock_updown,
        'use_fp16': args.use_fp16,
        'use_new_attention_order': False,
        'cond_diffuse': args.cond_diffuse,
        'cond_opt': args.cond_opt,
        'dataset_mode': args.dataset_mode,
    })
    return params


def main() -> None:
    args = parse_args()

    if args.image_height and args.image_width:
        if args.image_height != args.image_width:
            raise ValueError("SCDM supports square inputs; height and width must match.")
        args.image_size = args.image_height

    if args.image_size not in {64, 128, 256, 384, 512}:
        raise ValueError("image_size must be one of {64, 128, 256, 384, 512} for the current architecture.")

    utils.set_seed(args.seed)

    device = utils.resolve_device(args.device)

    torch.backends.cudnn.benchmark = device.type == 'cuda'

    images_dir, labels_dir = resolve_paths(args)
    height = args.image_height or args.image_size
    width = args.image_width or args.image_size
    image_size = (int(height), int(width))

    train_dataset = SegmentationDiffusionDataset(
        images_dir=images_dir,
        masks_dir=labels_dir,
        image_size=image_size,
        random_flip=not args.no_flip,
        label_nc=args.label_nc,
        ignore_label=args.ignore_label,
    )
    eval_dataset = SegmentationDiffusionDataset(
        images_dir=images_dir,
        masks_dir=labels_dir,
        image_size=image_size,
        random_flip=False,
        label_nc=train_dataset.label_nc,
        ignore_label=args.ignore_label,
    )

    drop_last = len(train_dataset) >= args.batch_size
    dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == 'cuda'),
        drop_last=drop_last,
    )

    metadata = train_dataset.metadata()
    utils.ensure_dir(args.output_dir)

    print(f"Dataset contains {len(train_dataset)} samples with {train_dataset.label_nc} classes.")

    diffusion_params = build_diffusion_params(args, train_dataset.label_nc)
    grid = parse_grid(args.visualization_grid)

    cfg = TrainingConfig(
        label_nc=train_dataset.label_nc,
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
        dataset_metadata=metadata,
        resume=args.resume,
        diffusion_params=diffusion_params,
        schedule_sampler=args.schedule_sampler,
        gradient_clip=args.gradient_clip if args.gradient_clip and args.gradient_clip > 0 else None,
        use_fp16=args.use_fp16,
        condition_with_edges=args.include_edges,
        num_visualization_items=min(args.num_visualization_items, grid[0] * grid[1]),
        visualization_grid=grid,
        preview_current_model=not args.no_preview_current,
        preview_ema_model=False,
    )

    trainer = SCDMTrainer(cfg)
    trainer.attach_dataset(eval_dataset)
    trainer.train(dataloader)


if __name__ == '__main__':
    main()
