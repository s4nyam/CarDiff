#!/usr/bin/env python3
"""
Select top-N segmentation masks by covered area and save masks, images and
contour-only overlays into a `top30/` folder.

This script reuses helper routines from `create_mosaic_with_grid.py` to keep
color mapping and contour behavior consistent (contour lines, not filled
regions). It writes the following structure under the `dc` folder:

  top30/
    images/           -- copies of the corresponding radiographs
    masks/            -- original mask files (L)
    colored_masks/    -- filled, colored RGBA masks (for reference)
    contours/         -- contour-only RGBA images (contour pixels only)
    overlays/         -- radiograph + contour overlay (RGBA)

The ranking is by the number of non-zero pixels in the mask (multi-class
counts as covered area). Filenames are prefixed with a 2-digit rank and
contain the pixel area for convenience.
"""

import os
from pathlib import Path
import shutil
from PIL import Image

# Import helpers from create_mosaic_with_grid which lives in the same folder
from create_mosaic_with_grid import (
    Path as _Path,
    list_annotation_files,
    compute_caries_area,
    load_image_for_mask,
    multiclass_color_mask,
    multiclass_contour_mask,
)


ROOT = Path(__file__).resolve().parent
ANN_DIR = ROOT / 'annotations'
OUT_ROOT = ROOT / 'top30'
TOP_K = 30


def ensure_dirs():
    (OUT_ROOT).mkdir(parents=True, exist_ok=True)
    for d in ('images', 'masks', 'colored_masks', 'contours', 'overlays'):
        (OUT_ROOT / d).mkdir(parents=True, exist_ok=True)


def pick_top_k(annotation_files, k=TOP_K):
    areas = []
    for p in annotation_files:
        a = compute_caries_area(p)
        areas.append((p, a))
    areas.sort(key=lambda x: x[1], reverse=True)
    return areas[:k]


def save_ranked_items(top_list):
    for idx, (mask_path, area) in enumerate(top_list, start=1):
        stem = Path(mask_path).stem
        rank_prefix = f"{idx:02d}"

        # Save original mask (L)
        mask_dest = OUT_ROOT / 'masks' / f"{rank_prefix}_{stem}_mask.png"
        shutil.copy(mask_path, mask_dest)

        # Create colored mask (RGBA) using helper (filled colors)
        mask_img = Image.open(mask_path).convert('L')
        colored = multiclass_color_mask(mask_img)
        colored_dest = OUT_ROOT / 'colored_masks' / f"{rank_prefix}_{stem}_colored_mask.png"
        colored.convert('RGBA').save(colored_dest)

        # Create contour-only image (RGBA) -> not filled, only contour lines
        contour = multiclass_contour_mask(mask_img)
        contour_dest = OUT_ROOT / 'contours' / f"{rank_prefix}_{stem}_contour.png"
        contour.save(contour_dest)

        # Copy corresponding radiograph image
        try:
            img = load_image_for_mask(mask_path)
        except FileNotFoundError as e:
            print(f"Warning: {e}")
            continue
        img_dest = OUT_ROOT / 'images' / f"{rank_prefix}_{stem}_image.png"
        img.convert('RGB').save(img_dest)

        # Create overlay: radiograph + contour (contours drawn, not filled)
        base = img.convert('RGBA')
        # Ensure contour is same size as base; if not resize using NEAREST to preserve lines
        if contour.size != base.size:
            contour_resized = contour.resize(base.size, resample=Image.NEAREST)
        else:
            contour_resized = contour

        try:
            overlay = Image.alpha_composite(base, contour_resized)
        except Exception:
            # fallback: paste with mask
            overlay = base.copy()
            overlay.paste(contour_resized, (0, 0), contour_resized)

        overlay_dest = OUT_ROOT / 'overlays' / f"{rank_prefix}_{stem}_overlay_area{area}.png"
        overlay.save(overlay_dest)

        print(f"Saved rank {idx:02d}: {Path(mask_path).name} (area={area})")


def main():
    ensure_dirs()

    ann_files = list_annotation_files()
    if not ann_files:
        print(f'No annotation files found in {ANN_DIR}')
        return

    top = pick_top_k(ann_files, k=TOP_K)
    print(f"Selected top {len(top)} masks by covered area")

    save_ranked_items(top)

    print(f"All outputs saved to: {OUT_ROOT}")


if __name__ == '__main__':
    main()
#!/usr/bin/env python3
"""
Select top-N annotation masks by total non-zero area and extract the largest ROI per mask.

Outputs saved into a `top30/` folder next to this script. For each selected mask the
script will produce:
 - cropped ROI from the original radiograph (largest connected component)
 - cropped ROI mask (corresponding mask patch)
 - overlay image (radiograph ROI + colored mask overlay)
 - a CSV summary listing mask name, total mask area, largest component area, class value and bbox

This mirrors conventions used in `create_mosaic_with_grid.py` (color map and image lookup).
"""

import os
from pathlib import Path
from PIL import Image
import numpy as np
import glob
import csv
from collections import deque

# Configuration (adapted from create_mosaic_with_grid.py)
ROOT = Path(__file__).resolve().parent
ANNOTATIONS_DIR = ROOT / 'annotations'
IMAGES_DIR = ROOT / 'images'
OUT_DIR = ROOT / 'top30'
TOP_K = 30
IMG_EXTS = ('.png', '.jpg', '.jpeg')
CONTOUR_WIDTH = 1


def ensure_out_dir():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / 'masks').mkdir(exist_ok=True)
    (OUT_DIR / 'rois').mkdir(exist_ok=True)
    (OUT_DIR / 'overlays').mkdir(exist_ok=True)


def list_annotation_files():
    return sorted(glob.glob(str(ANNOTATIONS_DIR / '*.png')))


def compute_caries_area(mask_path):
    m = Image.open(mask_path).convert('L')
    arr = np.array(m)
    return int((arr != 0).sum())


def pick_top_k(annotation_files, k=TOP_K):
    areas = []
    for p in annotation_files:
        a = compute_caries_area(p)
        areas.append((p, a))
    areas.sort(key=lambda x: x[1], reverse=True)
    return areas[:k]


def load_image_for_mask(mask_path):
    name = Path(mask_path).name
    candidates = [IMAGES_DIR / name, IMAGES_DIR / (Path(name).stem + '.png'), IMAGES_DIR / (Path(name).stem + '.jpg')]
    for c in candidates:
        if c.exists():
            return Image.open(c).convert('RGB')
    # fallback search in IMAGES_DIR
    for ext in IMG_EXTS:
        c = IMAGES_DIR / (Path(name).stem + ext)
        if c.exists():
            return Image.open(c).convert('RGB')
    raise FileNotFoundError(f'No corresponding image found in {IMAGES_DIR} for mask {mask_path}')


def multiclass_color_mask(mask_img):
    # same color mapping as create_mosaic_with_grid.py
    arr = np.array(mask_img.convert('L'))
    h, w = arr.shape
    out_arr = np.zeros((h, w, 4), dtype=np.uint8)
    color_map = {
        102: (46, 134, 193, 180),
        153: (40, 180, 99, 180),
        255: (231, 76, 60, 180)
    }
    for v, col in color_map.items():
        mask_v = (arr == v)
        out_arr[mask_v] = col
    return Image.fromarray(out_arr, mode='RGBA')


def multiclass_contour_mask(mask_img, contour_width=CONTOUR_WIDTH):
    """Return an RGBA image containing the contour of each class region.

    Implemented similarly to create_mosaic_with_grid.py: compute 1-px boundaries
    using a 4-neighborhood and dilate to achieve thicker lines when
    contour_width>1.
    """
    arr = np.array(mask_img.convert('L'))
    h, w = arr.shape
    out_arr = np.zeros((h, w, 4), dtype=np.uint8)
    color_map = {
        102: (46, 134, 193, 255),  # blue-ish
        153: (40, 180, 99, 255),   # green-ish
        255: (231, 76, 60, 255)    # red-ish
    }

    # Compute 1-px wide boundaries first (4-neighborhood)
    for v, col in color_map.items():
        binary = (arr == v).astype(np.uint8)
        if binary.sum() == 0:
            continue
        pad = np.pad(binary, pad_width=1, mode='constant', constant_values=0)
        center = pad[1:-1, 1:-1].astype(bool)
        up = pad[:-2, 1:-1].astype(bool)
        down = pad[2:, 1:-1].astype(bool)
        left = pad[1:-1, :-2].astype(bool)
        right = pad[1:-1, 2:].astype(bool)

        boundary = center & (~(up & down & left & right))

        if contour_width <= 1:
            out_arr[boundary] = col
        else:
            dilated = np.zeros_like(boundary, dtype=bool)
            radius = contour_width // 2
            for dy in range(-radius, radius + (1 if contour_width % 2 else 0)):
                for dx in range(-radius, radius + (1 if contour_width % 2 else 0)):
                    shifted = np.roll(boundary, shift=(dy, dx), axis=(0, 1))
                    if dy > 0:
                        shifted[:dy, :] = False
                    elif dy < 0:
                        shifted[dy:, :] = False
                    if dx > 0:
                        shifted[:, :dx] = False
                    elif dx < 0:
                        shifted[:, dx:] = False
                    dilated |= shifted

            out_arr[dilated] = col

    return Image.fromarray(out_arr, mode='RGBA')


def find_connected_components(binary_array, connectivity=8):
    """Label connected components in a binary numpy array (bool or 0/1).

    Returns:
      labels: int array same shape as input with 0 meaning background and 1..N component ids
      components: list of dicts {id, area, bbox}
    """
    arr = (binary_array != 0).astype(np.uint8)
    h, w = arr.shape
    labels = np.zeros((h, w), dtype=np.int32)
    comp_id = 0
    comps = []
    # neighbors offsets for 8-connectivity
    if connectivity == 8:
        neigh = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    else:
        neigh = [(-1, 0), (0, -1), (0, 1), (1, 0)]

    for y in range(h):
        for x in range(w):
            if arr[y, x] and labels[y, x] == 0:
                comp_id += 1
                # BFS flood fill
                q = deque()
                q.append((y, x))
                labels[y, x] = comp_id
                min_y, max_y = y, y
                min_x, max_x = x, x
                area = 0
                while q:
                    yy, xx = q.popleft()
                    area += 1
                    if yy < min_y: min_y = yy
                    if yy > max_y: max_y = yy
                    if xx < min_x: min_x = xx
                    if xx > max_x: max_x = xx
                    for dy, dx in neigh:
                        ny, nx = yy + dy, xx + dx
                        if 0 <= ny < h and 0 <= nx < w and arr[ny, nx] and labels[ny, nx] == 0:
                            labels[ny, nx] = comp_id
                            q.append((ny, nx))
                comps.append({'id': comp_id, 'area': area, 'bbox': (min_x, min_y, max_x + 1, max_y + 1)})
    return labels, comps


def save_roi_images(out_prefix, img, mask, bbox):
    x0, y0, x1, y1 = bbox
    roi_img = img.crop((x0, y0, x1, y1))
    roi_mask = mask.crop((x0, y0, x1, y1))
    # create contour overlay (contours drawn with opaque colors)
    contour = multiclass_contour_mask(roi_mask, contour_width=CONTOUR_WIDTH)
    roi_img_rgba = roi_img.convert('RGBA')
    try:
        overlay = Image.alpha_composite(roi_img_rgba, contour)
    except Exception:
        # fallback: paste with mask
        overlay = roi_img_rgba.copy()
        overlay.paste(contour, (0, 0), contour)

    roi_img.save(out_prefix + '_pr.png')
    # save mask as RGB to preserve viewing
    roi_mask.convert('L').save(out_prefix + '_mask.png')
    overlay.convert('RGB').save(out_prefix + '_overlay.png')


def process_top_k(top_list, out_csv_path):
    rows = []
    for rank, (mask_path, total_area) in enumerate(top_list, start=1):
        stem = Path(mask_path).stem
        try:
            img = load_image_for_mask(mask_path)
        except FileNotFoundError as e:
            print('Warning:', e)
            continue
        mask_img = Image.open(mask_path).convert('L')
        arr = np.array(mask_img)

        # For multiclass masks, treat non-zero as foreground but keep class value for reporting
        labels, comps = find_connected_components(arr != 0, connectivity=8)

        if not comps:
            print(f'No connected components found in {stem}')
            continue

        # find largest component by area
        comps.sort(key=lambda c: c['area'], reverse=True)
        largest = comps[0]
        bbox = largest['bbox']
        comp_area = largest['area']

        # determine the dominant class value inside this bbox (most frequent non-zero)
        x0, y0, x1, y1 = bbox
        sub = arr[y0:y1, x0:x1]
        vals, counts = np.unique(sub[sub != 0], return_counts=True)
        dom_class = int(vals[np.argmax(counts)]) if vals.size > 0 else 0

        # save full mask copy and ROI images
        out_mask_path = OUT_DIR / 'masks' / f'{rank:02d}_{stem}_mask.png'
        mask_img.save(out_mask_path)

        out_prefix = str(OUT_DIR / 'rois' / f'{rank:02d}_{stem}')
        save_roi_images(out_prefix, img, mask_img, bbox)

        # also save overlay full-size for convenience
        colored_full = multiclass_color_mask(mask_img)
        try:
            full_overlay = Image.alpha_composite(img.convert('RGBA'), colored_full).convert('RGB')
            full_overlay.save(OUT_DIR / 'overlays' / f'{rank:02d}_{stem}_overlay.png')
        except Exception:
            pass

        rows.append({
            'rank': rank,
            'mask_file': Path(mask_path).name,
            'total_area': int(total_area),
            'largest_comp_area': int(comp_area),
            'dominant_class_in_roi': int(dom_class),
            'bbox': bbox,
            'roi_pr': f'rois/{rank:02d}_{stem}_pr.png',
            'roi_mask': f'rois/{rank:02d}_{stem}_mask.png',
            'roi_overlay': f'rois/{rank:02d}_{stem}_overlay.png',
        })

    # write CSV
    with open(out_csv_path, 'w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=['rank', 'mask_file', 'total_area', 'largest_comp_area', 'dominant_class_in_roi', 'bbox', 'roi_pr', 'roi_mask', 'roi_overlay'])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(f'Saved summary to {out_csv_path}')


def main():
    ensure_out_dir()
    ann_files = list_annotation_files()
    if not ann_files:
        print(f'No annotation files found in {ANNOTATIONS_DIR}')
        return

    top = pick_top_k(ann_files, k=TOP_K)
    print('Top files by caries area:')
    for i, (p, a) in enumerate(top, 1):
        print(f'{i}. {Path(p).name} - caries pixels: {a}')

    out_csv = OUT_DIR / 'top30_summary.csv'
    process_top_k(top, out_csv)
    print(f'Outputs saved to: {OUT_DIR}')


if __name__ == '__main__':
    main()
