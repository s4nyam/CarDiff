#!/usr/bin/env python3
"""Quick utility to convert diffusion mosaics to grayscale."""
from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image


SUPPORTED_SUFFIXES = {".png", ".jpg", ".jpeg"}


def convert(path: Path, overwrite: bool = True) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    if path.suffix.lower() not in SUPPORTED_SUFFIXES:
        raise ValueError(f"Unsupported file extension for {path.name}")

    with Image.open(path) as img:
        grayscale = img.convert("L")
        result = grayscale.convert("RGB")  # keep 3 channels for downstream tools

    if overwrite:
        result_path = path
    else:
        result_path = path.with_name(f"{path.stem}_gray{path.suffix}")

    result.save(result_path)
    return result_path


def iter_image_paths(root: Path, recursive: bool = True) -> list[Path]:
    pattern = "**/*" if recursive else "*"
    # Sort to make the execution order deterministic across runs.
    return sorted(
        path
        for path in root.glob(pattern)
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert mosaic images to grayscale.")
    parser.add_argument("path", type=Path, help="Path to an image file or a directory of images.")
    parser.add_argument(
        "--keep-original",
        action="store_true",
        help="Write *_gray copies instead of overwriting the source files.",
    )
    parser.add_argument(
        "--non-recursive",
        action="store_true",
        help="Only convert images directly under the directory (skip subdirectories).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.path.exists():
        raise FileNotFoundError(f"Input path not found: {args.path}")

    overwrite = not args.keep_original

    if args.path.is_dir():
        images = iter_image_paths(args.path, recursive=not args.non_recursive)
        if not images:
            print("No images found to convert.")
            return

        for image in images:
            output = convert(image, overwrite=overwrite)
            print(f"Grayscale image written to {output}")

        print(f"Converted {len(images)} image(s).")
    else:
        output = convert(args.path, overwrite=overwrite)
        print(f"Grayscale image written to {output}")


if __name__ == "__main__":
    main()
