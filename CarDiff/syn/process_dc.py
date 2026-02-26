#!/usr/bin/env python3
"""
DC1000 tiling utility

Reads train/test images and masks from DC1000_dataset, crops 384x384 non-overlapping
tiles by sliding a fixed window left-to-right, top-to-bottom, ignoring pixels on
the right/bottom edges that don't fit a full tile, and writes aligned tiles to:

  - <out_root>/images       (image tiles)
  - <out_root>/annotations  (mask tiles)

Defaults are set for your provided dataset layout:
  dataset_root = /home/sj/working_dir/ssd-multi/DC1000_dataset
  train images = org_train_dataset/images
  train masks  = org_train_dataset/colors_clean
  test  images = org_test_dataset/images
  test  masks  = org_test_dataset/colors

Usage (default paths, tile size 384):
  python process_dc.py

Or specify options:
  python process_dc.py \
    --dataset-root /home/sj/working_dir/ssd-multi/DC1000_dataset \
    --out-root /home/sj/working_dir/ssd-multi/dc \
    --tile-size 384

Add --dry-run to only report counts without writing files.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

from PIL import Image
from tqdm import tqdm


TILE_SIZE_DEFAULT = 384


@dataclass
class SplitPaths:
    name: str
    images_dir: Path
    masks_dir: Path
    fallback_masks_dir: Path | None = None


def list_image_files(folder: Path) -> List[Path]:
    exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
    return sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in exts])


def intersect_by_stem(
    images: Iterable[Path],
    masks: Iterable[Path],
    fallback_masks_dir: Path | None = None,
) -> Tuple[List[Tuple[Path, Path]], List[Path], List[Tuple[Path, Path]]]:
    """
    Returns (pairs, still_missing_images, recovered_pairs_from_fallback).
    Tries to match images to masks by filename stem in the primary mask dir and,
    if missing, attempts recovery from an optional fallback mask directory.
    """
    mask_by_stem = {p.stem: p for p in masks}
    pairs: List[Tuple[Path, Path]] = []
    missing: List[Path] = []
    recovered: List[Tuple[Path, Path]] = []

    # Primary pass
    for img in images:
        m = mask_by_stem.get(img.stem)
        if m is not None:
            pairs.append((img, m))
        else:
            missing.append(img)

    # Fallback recovery
    if missing and fallback_masks_dir and fallback_masks_dir.exists():
        # Build quick index for fallback by stem
        try:
            fb_files = [p for p in fallback_masks_dir.iterdir() if p.is_file()]
        except Exception:
            fb_files = []
        fb_by_stem = {p.stem: p for p in fb_files}
        still_missing: List[Path] = []
        for img in missing:
            fb = fb_by_stem.get(img.stem)
            if fb is not None:
                recovered.append((img, fb))
            else:
                still_missing.append(img)
        missing = still_missing

    return pairs + recovered, missing, recovered


def ensure_out_dirs(out_images: Path, out_masks: Path) -> None:
    out_images.mkdir(parents=True, exist_ok=True)
    out_masks.mkdir(parents=True, exist_ok=True)


def tile_image_and_mask(
    img: Image.Image,
    msk: Image.Image,
    tile: int,
) -> Iterable[Tuple[int, int, Image.Image, Image.Image]]:
    """Yield (x, y, img_tile, mask_tile) for non-overlapping tiles within full multiples."""
    if img.size != msk.size:
        raise ValueError(f"Image/Mask size mismatch: {img.size} vs {msk.size}")
    w, h = img.size
    max_w = (w // tile) * tile
    max_h = (h // tile) * tile
    for top in range(0, max_h, tile):
        for left in range(0, max_w, tile):
            box = (left, top, left + tile, top + tile)
            yield left, top, img.crop(box), msk.crop(box)


def mask_has_foreground(msk: Image.Image) -> bool:
    """Return True if mask contains any non-zero pixel (not completely black).

    Works across modes (L, P, RGB, RGBA) using extrema. For multi-band images,
    any band having max > 0 is considered foreground present.
    """
    ext = msk.getextrema()
    # Multi-band returns tuple of (min,max) per band
    if isinstance(ext[0], tuple):
        return any(band_max > 0 for (_min, band_max) in ext)  # type: ignore[arg-type]
    # Single-band: ext is (min, max)
    return ext[1] > 0  # type: ignore[index]


def process_pairs(
    pairs: List[Tuple[Path, Path]],
    out_images: Path,
    out_masks: Path,
    tile: int,
    split_name: str,
    dry_run: bool = False,
) -> Tuple[int, int]:
    """Process aligned (image, mask) pairs. Returns (files_processed, tiles_written)."""
    files = 0
    tiles = 0
    skipped_empty = 0
    for img_path, msk_path in tqdm(pairs, desc=f"{split_name}: tiling", unit="file"):
        try:
            with Image.open(img_path) as img, Image.open(msk_path) as msk:
                img.load()
                msk.load()
                if img.mode not in ("RGB", "L"):
                    img = img.convert("RGB")
                for left, top, ti, tm in tile_image_and_mask(img, msk, tile):
                    # Skip tiles where mask has no foreground
                    if not mask_has_foreground(tm):
                        skipped_empty += 1
                        continue
                    tiles += 1
                    if dry_run:
                        continue
                    stem = img_path.stem
                    tile_tag = f"{split_name}_{stem}_x{left}_y{top}"
                    ti.save(out_images / f"{tile_tag}.png")
                    tm.save(out_masks / f"{tile_tag}.png")
            files += 1
        except Exception as e:
            print(f"[error] Failed {img_path.name}: {e}")
    if skipped_empty:
        print(f"[info] {split_name}: skipped {skipped_empty} tile(s) with empty masks")
    return files, tiles


def build_splits(dataset_root: Path) -> List[SplitPaths]:
    return [
        SplitPaths(
            name="train",
            images_dir=dataset_root / "org_train_dataset" / "images",
            masks_dir=dataset_root / "org_train_dataset" / "colors_clean",
            fallback_masks_dir=dataset_root / "org_train_dataset" / "colors_bydoctors",
        ),
        SplitPaths(
            name="test",
            images_dir=dataset_root / "org_test_dataset" / "images",
            masks_dir=dataset_root / "org_test_dataset" / "colors",
            fallback_masks_dir=dataset_root / "org_test_dataset" / "colors_origin_bydoctor",
        ),
    ]


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Crop 384x384 tiles from DC1000 images and masks.")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/home/sj/working_dir/ssd-multi/DC1000_dataset"),
        help="Root of DC1000_dataset",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path("/home/sj/working_dir/ssd-multi/dc"),
        help="Output root containing 'images' and 'annotations'",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=TILE_SIZE_DEFAULT,
        help="Tile edge size (pixels). Default 384",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute counts without writing files.",
    )
    return parser.parse_args(argv)


def main(argv: List[str]) -> int:
    args = parse_args(argv)
    dataset_root: Path = args.dataset_root
    out_root: Path = args.out_root
    tile: int = args.tile_size
    dry_run: bool = args.dry_run

    splits = build_splits(dataset_root)

    out_images = out_root / "images"
    out_masks = out_root / "annotations"
    if not dry_run:
        ensure_out_dirs(out_images, out_masks)

    grand_files = 0
    grand_tiles = 0

    for split in splits:
        if not split.images_dir.exists():
            print(f"[warn] Missing images dir: {split.images_dir} — skipping {split.name}")
            continue
        if not split.masks_dir.exists():
            print(f"[warn] Missing masks dir: {split.masks_dir} — skipping {split.name}")
            continue

        imgs = list_image_files(split.images_dir)
        msks = list_image_files(split.masks_dir)
        pairs, missing, recovered = intersect_by_stem(imgs, msks, split.fallback_masks_dir)
        if recovered:
            print(f"[info] {split.name}: recovered {len(recovered)} missing mask(s) from fallback: {split.fallback_masks_dir}")
        if missing:
            print(f"[warn] {len(missing)} image(s) had no matching mask by stem; they were skipped.")
            print("[warn] Missing mask details (image file -> stem):")
            for p in missing:
                print(f"  - {p.name} -> '{p.stem}'")
        if not pairs:
            print(f"[warn] No matching (image,mask) pairs found in {split.name}; skipping.")
            continue

        files, tiles = process_pairs(
            pairs=pairs,
            out_images=out_images,
            out_masks=out_masks,
            tile=tile,
            split_name=split.name,
            dry_run=dry_run,
        )
        print(f"[info] {split.name}: processed {files} files -> {tiles} tiles")
        grand_files += files
        grand_tiles += tiles

    print(f"[done] Total: {grand_files} files -> {grand_tiles} tiles" + (" (dry-run)" if dry_run else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
