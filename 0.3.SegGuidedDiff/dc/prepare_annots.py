#!/usr/bin/env python3
"""
Prepare derived segmentation masks for single and pairwise class combinations.

Source masks contain three classes with fixed grayscale values:
- SC: 102
- MC: 153
- DC: 255

This script creates 7 folders under a destination root:
- SC, MC, DC, SC+DC, SC+MC, MC+DC, SC+MC+DC

Behavior:
- For each input mask, we generate versions that keep only the pixels
  belonging to the target class set; all other pixels are set to 0 (black).
- For SC+MC+DC, the input mask is copied as-is.

Usage example:
	python prepare_annots.py \
	  --src /home/sj/working_dir/ssd-multi/dc/annotations \
	  --dst /home/sj/working_dir/ssd-multi/dc \
	  --overwrite

Dependencies: numpy, opencv-python, tqdm
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple
import shutil

import cv2
import numpy as np
from tqdm import tqdm


CLASS_VALUES: Dict[str, int] = {
	"SC": 102,
	"MC": 153,
	"DC": 255,
}

# Folder names to generate with their included classes
TARGET_SETS: Dict[str, Tuple[str, ...]] = {
	"SC": ("SC",),
	"MC": ("MC",),
	"DC": ("DC",),
	"SC+DC": ("SC", "DC"),
	"SC+MC": ("SC", "MC"),
	"MC+DC": ("MC", "DC"),
	# This one is copied as-is
	"SC+MC+DC": ("SC", "MC", "DC"),
}


def find_images(src: Path, exts: Sequence[str]) -> List[Path]:
	exts_norm = {e.lower().lstrip(".") for e in exts}
	files = [
		p for p in sorted(src.iterdir())
		if p.is_file() and p.suffix.lower().lstrip(".") in exts_norm
	]
	return files


def ensure_dirs(dst_root: Path, folder_names: Iterable[str]) -> Dict[str, Path]:
	paths: Dict[str, Path] = {}
	for name in folder_names:
		out_dir = dst_root / name
		out_dir.mkdir(parents=True, exist_ok=True)
		paths[name] = out_dir
	return paths


def filter_mask(img: np.ndarray, allowed_values: Iterable[int]) -> np.ndarray:
	"""Return an image where only allowed_values are kept; others set to 0.

	img: uint8 grayscale mask
	allowed_values: collection of grayscale values to retain
	"""
	if img.dtype != np.uint8:
		img = img.astype(np.uint8)
	allowed = np.array(list(allowed_values), dtype=np.uint8)
	# np.isin over uint8 is fine; result is boolean mask
	keep = np.isin(img, allowed)
	out = np.where(keep, img, 0).astype(np.uint8)
	return out


def process(
	src_dir: Path,
	dst_root: Path,
	overwrite: bool = False,
	exts: Sequence[str] = ("png",),
	dry_run: bool = False,
) -> None:
	assert src_dir.is_dir(), f"Source directory does not exist: {src_dir}"
	dst_root.mkdir(parents=True, exist_ok=True)

	files = find_images(src_dir, exts)
	if not files:
		raise SystemExit(f"No images found in {src_dir} with extensions {list(exts)}")

	out_dirs = ensure_dirs(dst_root, TARGET_SETS.keys())

	# Precompute numeric value lists per target (except the all-classes copy)
	allowed_by_target: Dict[str, Tuple[int, ...]] = {
		name: tuple(CLASS_VALUES[c] for c in classes)
		for name, classes in TARGET_SETS.items()
		if name != "SC+MC+DC"
	}

	# Progress bars by overall files
	pbar = tqdm(files, desc="Processing masks", unit="img")
	for in_path in pbar:
		rel_name = in_path.name

		if dry_run:
			# Check readability only
			img = cv2.imread(str(in_path), cv2.IMREAD_GRAYSCALE)
			if img is None:
				pbar.write(f"[WARN] Cannot read image: {in_path}")
			# Skip writes in dry-run
			continue

		# Read once, reuse for all outputs
		img = cv2.imread(str(in_path), cv2.IMREAD_GRAYSCALE)
		if img is None:
			pbar.write(f"[WARN] Skipping unreadable image: {in_path}")
			continue

		# 1) Copy original to SC+MC+DC
		dst_all = out_dirs["SC+MC+DC"] / rel_name
		if overwrite or not dst_all.exists():
			# Copy bytes to preserve exact content
			dst_all.parent.mkdir(parents=True, exist_ok=True)
			shutil.copy2(in_path, dst_all)

		# 2) Generate filtered versions for other targets
		for target, allowed_vals in allowed_by_target.items():
			out_path = out_dirs[target] / rel_name
			if not overwrite and out_path.exists():
				continue
			out_img = filter_mask(img, allowed_vals)
			ok = cv2.imwrite(str(out_path), out_img)
			if not ok:
				pbar.write(f"[WARN] Failed to write: {out_path}")

	# Final summary
	print("Done. Output folders:")
	for name, d in out_dirs.items():
		print(f"- {name}: {d}")


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Prepare derived segmentation masks for class subsets",
		formatter_class=argparse.ArgumentDefaultsHelpFormatter,
	)
	parser.add_argument(
		"--src",
		type=Path,
		default=Path("/home/sj/working_dir/ssd-multi/dc/annotations"),
		help="Source directory with original masks",
	)
	parser.add_argument(
		"--dst",
		type=Path,
		default=Path("/home/sj/working_dir/ssd-multi/dc"),
		help="Destination root directory where target folders will be created",
	)
	parser.add_argument(
		"--exts",
		type=str,
		default="png",
		help="Comma-separated list of file extensions to include",
	)
	parser.add_argument(
		"--overwrite",
		action="store_true",
		help="Overwrite outputs if files already exist",
	)
	parser.add_argument(
		"--dry-run",
		action="store_true",
		help="Scan and validate input images but do not write outputs",
	)
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	exts = [e.strip() for e in args.exts.split(",") if e.strip()]
	process(
		src_dir=args.src,
		dst_root=args.dst,
		overwrite=bool(args.overwrite),
		exts=exts,
		dry_run=bool(args.dry_run),
	)


if __name__ == "__main__":
	main()

