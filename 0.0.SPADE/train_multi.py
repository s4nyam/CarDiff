from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Tuple

import torch
from torch.utils.data import DataLoader

from spade import SegmentationImageDataset, SPADETrainer, TrainingConfig, utils


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a SPADE generator on paired masks/images.")
    parser.add_argument('--data-root', type=Path, default=Path('/mnt/d/playground/DC1000_dataset/train_multi'),
                        help='Root directory containing "images" and "labels" sub-folders.')
    parser.add_argument('--images-dir', type=Path, default=None,
                        help='Override for the images directory.')
    parser.add_argument('--labels-dir', type=Path, default=None,
                        help='Override for the labels directory.')
    parser.add_argument('--output-dir', type=Path, default=Path('./spade_runs_multi'),
                        help='Directory to store checkpoints and samples.')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--max-steps', type=int, default=None,
                        help='Optional cap on the number of optimization steps.')
    parser.add_argument('--image-size', type=int, default=384,
                        help='Target square image resolution (both height and width).')
    parser.add_argument('--image-height', type=int, default=None,
                        help='Optional specific target height. Overrides --image-size if set.')
    parser.add_argument('--image-width', type=int, default=None,
                        help='Optional specific target width. Overrides --image-size if set.')
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--beta1', type=float, default=0.0)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--lambda-l1', type=float, default=10.0)
    parser.add_argument('--save-every', type=int, default=2000,
                        help='Save a checkpoint every N steps.')
    parser.add_argument('--sample-every', type=int, default=500,
                        help='Write sample images every N steps.')
    parser.add_argument('--log-every', type=int, default=100,
                        help='Update the progress bar with losses every N steps.')
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--label-nc', type=int, default=None,
                        help='Number of semantic classes. If omitted, inferred from the masks.')
    parser.add_argument('--ignore-label', type=int, default=0,
                        help='Mask value that should be treated as ignore/background.')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default="cuda:1",
                        help='Choose "cpu" or "cuda". Auto-selects if omitted.')
    parser.add_argument('--resume', type=Path, default=None,
                        help='Checkpoint path to resume training.')
    parser.add_argument('--ngf', type=int, default=64)
    parser.add_argument('--ndf', type=int, default=64)
    parser.add_argument('--z-dim', type=int, default=256)
    parser.add_argument('--num-upsampling-layers', choices=['normal', 'more', 'most'], default='normal')
    parser.add_argument('--param-free-norm', choices=['batch', 'instance'], default='instance')
    parser.add_argument('--no-flip', action='store_true', help='Disable random horizontal flips.')
    return parser.parse_args()


def resolve_paths(args: argparse.Namespace) -> Tuple[Path, Path]:
    images_dir = args.images_dir if args.images_dir is not None else args.data_root / 'images'
    labels_dir = args.labels_dir if args.labels_dir is not None else args.data_root / 'labels'
    return Path(images_dir), Path(labels_dir)


def main() -> None:
    args = parse_args()

    height = args.image_height or args.image_size
    width = args.image_width or args.image_size
    image_size = (int(height), int(width))

    device_str = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device(device_str)

    utils.set_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    images_dir, labels_dir = resolve_paths(args)
    dataset = SegmentationImageDataset(
        images_dir=images_dir,
        masks_dir=labels_dir,
        image_size=image_size,
        random_flip=not args.no_flip,
        label_nc=args.label_nc,
        ignore_label=args.ignore_label,
    )

    drop_last = len(dataset) >= args.batch_size
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == 'cuda'),
        drop_last=drop_last,
    )

    utils.ensure_dir(args.output_dir)

    metadata = dataset.metadata()
    metadata_path = args.output_dir / 'dataset_metadata.json'
    with metadata_path.open('w') as f:
        json.dump(metadata, f, indent=2)

    print(f"Dataset contains {len(dataset)} samples with {dataset.label_nc} classes.")

    class_values = None
    if isinstance(metadata, dict):
        class_values = metadata.get('class_values')
        if class_values is None:
            encoder_meta = metadata.get('encoder')
            if isinstance(encoder_meta, dict):
                class_values = encoder_meta.get('class_values')
    if class_values is None:
        class_values = list(range(dataset.label_nc))
    class_values = [int(v) for v in class_values]
    print(f"Discovered mask pixel values (including background): {class_values}")

    config = TrainingConfig(
        label_nc=dataset.label_nc,
        image_size=image_size,
        output_dir=args.output_dir,
        epochs=args.epochs,
        max_steps=args.max_steps,
        lr=args.lr,
        beta1=args.beta1,
        beta2=args.beta2,
        lambda_l1=args.lambda_l1,
        save_every=args.save_every,
        sample_every=args.sample_every,
        log_every=args.log_every,
        z_dim=args.z_dim,
        ngf=args.ngf,
        ndf=args.ndf,
        num_upsampling_layers=args.num_upsampling_layers,
        param_free_norm=args.param_free_norm,
        device=device,
        dataset_metadata=metadata,
        resume=args.resume,
    )

    trainer = SPADETrainer(config, dataset)
    trainer.train(dataloader)


if __name__ == '__main__':
    main()
