#!/usr/bin/env python3
"""
Create mosaic-like outputs with grid overlays for top-covered masks.

This script searches the `annotations/` folder for segmentation masks, computes
the total caries pixel area (all non-zero pixels), selects the top-N images by
area (default 5), and creates three images per selected pair:
 - PR ROI with grid (grid drawn in black with shadow on the radiograph)
 - Mask with grid (grid drawn in white with shadow on the mask)
 - PR ROI + Multi-class overlay + grid (yellow grid)

Grid layout: for a 384x384 input and p=24 the grid is 16x16 (256 patches).

Outputs are saved into a `with_grid/` directory created next to this script.

This file intentionally keeps dependencies minimal (Pillow, numpy).
"""

import os
from pathlib import Path
from PIL import Image, ImageDraw, ImageFilter
import numpy as np
import glob
import random
random.seed(88)
# Configuration
ROOT = Path(__file__).resolve().parent
ANNOTATIONS_DIR = ROOT / 'annotations'
IMAGES_DIR = ROOT / 'images'
OUT_DIR = ROOT / 'with_grid'
TOP_K = 5
IMG_SIZE = (384, 384)
PATCH_P = 24  # patch size in pixels
CONTOUR_WIDTH = 2  # width of contour lines in pixels; increase to thicken contours
# neighborhood radius: nh=1 -> 3x3 (8 neighbors), nh=2 -> 5x5 (24 neighbors), etc.
NEIGHBORHOOD_NH = 2


def ensure_out_dir():
    OUT_DIR.mkdir(parents=True, exist_ok=True)


def list_annotation_files():
    return sorted(glob.glob(str(ANNOTATIONS_DIR / '*.png')))


def compute_caries_area(mask_path):
    # caries = any non-zero pixel (multi-class: 102,153,255)
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
    # Try same basename in IMAGES_DIR
    candidates = [IMAGES_DIR / name, IMAGES_DIR / (Path(name).stem + '.png'), IMAGES_DIR / (Path(name).stem + '.jpg')]
    for c in candidates:
        if c.exists():
            return Image.open(c).convert('RGB')
    # Fallback: look for image with same stem anywhere in IMAGES_DIR
    for ext in ('png', 'jpg', 'jpeg'):
        c = IMAGES_DIR / (Path(name).stem + '.' + ext)
        if c.exists():
            return Image.open(c).convert('RGB')
    raise FileNotFoundError(f'No corresponding image found in {IMAGES_DIR} for mask {mask_path}')


def multiclass_color_mask(mask_img):
    # mask_img: PIL Image L (values 0,102,153,255)
    arr = np.array(mask_img.convert('L'))
    h, w = arr.shape
    out = Image.new('RGBA', (w, h), (0, 0, 0, 0))
    # color mapping matches analyze_multiclass_components file usage
    color_map = {
        102: (46, 134, 193, 180),  # blue-ish
        153: (40, 180, 99, 180),   # green-ish
        255: (231, 76, 60, 180)    # red-ish
    }
    out_arr = np.zeros((h, w, 4), dtype=np.uint8)
    for v, col in color_map.items():
        mask_v = (arr == v)
        out_arr[mask_v] = col
    return Image.fromarray(out_arr, mode='RGBA')


def multiclass_contour_mask(mask_img, contour_width=CONTOUR_WIDTH):
    """Return an RGBA image containing the contour of each class region.

    Args:
        mask_img: PIL Image (L) with class values (0,102,153,255).
        contour_width: integer pixel width for the contour lines (>=1).

    The contour is computed on 4-neighborhood (up/down/left/right). For
    contour_width > 1 the 1-pixel-wide boundary is dilated using a simple
    square kernel to achieve thicker lines without adding a heavy dependency.
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
            # Dilate boundary using a square kernel of size contour_width
            # Create an empty mask for dilated boundary
            dilated = np.zeros_like(boundary, dtype=bool)
            radius = contour_width // 2
            # For each offset within the square kernel, shift and OR
            for dy in range(-radius, radius + (1 if contour_width % 2 else 0)):
                for dx in range(-radius, radius + (1 if contour_width % 2 else 0)):
                    # roll and handle edges by zeroing wrapped regions
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


def draw_grid(img, p=PATCH_P, line_color=(0, 0, 0, 255), shadow_color=(0, 0, 0, 200), shadow_offset=(2,2)):
    # img: PIL Image (RGB or RGBA)
    w, h = img.size
    draw = ImageDraw.Draw(img)

    # Make a shadow layer
    shadow = Image.new('RGBA', img.size, (0, 0, 0, 0))
    sd = ImageDraw.Draw(shadow)

    # Vertical lines
    for x in range(p, w, p):
        sd.line([(x + shadow_offset[0], 0 + shadow_offset[1]), (x + shadow_offset[0], h + shadow_offset[1])], fill=shadow_color, width=1)
    # Horizontal lines
    for y in range(p, h, p):
        sd.line([(0 + shadow_offset[0], y + shadow_offset[1]), (w + shadow_offset[0], y + shadow_offset[1])], fill=shadow_color, width=1)

    # Blur shadow slightly to make it softer
    shadow = shadow.filter(ImageFilter.GaussianBlur(radius=1))

    # Composite shadow onto image
    base = img.convert('RGBA')
    base = Image.alpha_composite(base, shadow)

    # Draw main grid lines on top
    top = Image.new('RGBA', img.size, (0, 0, 0, 0))
    td = ImageDraw.Draw(top)
    for x in range(p, w, p):
        td.line([(x, 0), (x, h)], fill=line_color, width=1)
    for y in range(p, h, p):
        td.line([(0, y), (w, y)], fill=line_color, width=1)

    out = Image.alpha_composite(base, top)
    return out


def save_with_grid(mask_path, img, mask, overlay, idx):
    stem = Path(mask_path).stem
    # PR ROI with black grid (shadow)
    img_resized = img.resize(IMG_SIZE, resample=Image.BILINEAR)
    img_grid = draw_grid(img_resized.copy(), p=PATCH_P, line_color=(0,0,0,255), shadow_color=(0,0,0,180))
    img_grid.save(OUT_DIR / f'{idx:02d}_{stem}_pr_with_grid.png')

    # Mask with white grid
    mask_rgb = mask.convert('RGB').resize(IMG_SIZE, resample=Image.NEAREST)
    # Ensure multiclass colors preserved: create colored mask separately
    colored_mask = multiclass_color_mask(mask).resize(IMG_SIZE, resample=Image.NEAREST)
    mask_grid = draw_grid(colored_mask.copy(), p=PATCH_P, line_color=(255,255,255,255), shadow_color=(0,0,0,160))
    mask_grid.save(OUT_DIR / f'{idx:02d}_{stem}_mask_with_grid.png')

    # PR ROI + multi-class contour overlay + yellow grid
    pr = img_resized.convert('RGBA')
    contour_mask = multiclass_contour_mask(mask, contour_width=CONTOUR_WIDTH).resize(IMG_SIZE, resample=Image.NEAREST)
    combined = Image.alpha_composite(pr, contour_mask)
    combined_grid = draw_grid(combined.copy(), p=PATCH_P, line_color=(255, 204, 0, 220), shadow_color=(0,0,0,160))
    combined_grid.save(OUT_DIR / f'{idx:02d}_{stem}_overlay_with_grid.png')

    # --- New: create tree diagrams from a single (2*nh+1)x(2*nh+1) neighborhood ---
    def pick_random_center_patch(img_size=IMG_SIZE, p=PATCH_P, nh=NEIGHBORHOOD_NH):
        # number of patches per axis
        nx = img_size[0] // p
        ny = img_size[1] // p
        # choose center such that a (2*nh+1) neighborhood fits
        if nx <= 2 * nh or ny <= 2 * nh:
            raise ValueError(f'image too small for neighborhood nh={nh} (nx={nx}, ny={ny})')
        cx = random.randint(nh, nx - nh - 1)
        cy = random.randint(nh, ny - nh - 1)
        return cx, cy

    def extract_neighborhood_patches(source_img, center_patch, p=PATCH_P, nh=NEIGHBORHOOD_NH):
        # source_img expected to be same size as IMG_SIZE
        cx, cy = center_patch
        patches = []
        for ry in range(-nh, nh + 1):
            for rx in range(-nh, nh + 1):
                x0 = (cx + rx) * p
                y0 = (cy + ry) * p
                box = (x0, y0, x0 + p, y0 + p)
                patches.append(source_img.crop(box))
        return patches

    def make_tree_diagram(patches, nh=NEIGHBORHOOD_NH, display_scale=4, line_color=(0, 0, 0, 255), bg=(255,255,255,255)):
        # patches: list of (2*nh+1)^2 PIL images in row-major
        grid_size = 2 * nh + 1
        p = patches[0].size[0]
        ds = display_scale
        tile = p * ds
        spacing = max(8, ds * 2)

        canvas_w = tile * grid_size + spacing * (grid_size - 1)
        canvas_h = tile * grid_size + spacing * (grid_size - 1)
        canvas = Image.new('RGBA', (canvas_w, canvas_h), bg)
        draw = ImageDraw.Draw(canvas)

        # positions for each patch
        coords = []
        for i in range(grid_size):
            for j in range(grid_size):
                x = j * (tile + spacing)
                y = i * (tile + spacing)
                coords.append((x, y))

        # paste thumbnails and record centers
        centers = []
        for (patch, (x, y)) in zip(patches, coords):
            thumb = patch.resize((tile, tile), resample=Image.NEAREST)
            canvas.paste(thumb, (x, y))
            centers.append((x + tile // 2, y + tile // 2))

        # draw lines from center to each neighbor
        center_idx = (grid_size // 2) * grid_size + (grid_size // 2)
        root_center = centers[center_idx]
        for i, c in enumerate(centers):
            if i == center_idx:
                # draw a ring around root
                rx = max(4, ds)
                draw.ellipse([root_center[0]-rx, root_center[1]-rx, root_center[0]+rx, root_center[1]+rx], outline=line_color, width=2)
                continue
            draw.line([root_center, c], fill=line_color, width=max(2, ds // 2))

        return canvas

    # Ensure we operate on the same grid-aligned resized images
    img_res = img_resized.convert('RGBA')
    mask_colored_res = colored_mask.convert('RGBA')
    combined_res = combined.convert('RGBA')

    # pick a random neighborhood (same for all three diagrams)
    center_patch = pick_random_center_patch(IMG_SIZE, PATCH_P, nh=NEIGHBORHOOD_NH)

    # use the colored mask (preserves multiclass colors) for the mask tree
    patches_mask = extract_neighborhood_patches(colored_mask, center_patch, p=PATCH_P, nh=NEIGHBORHOOD_NH)
    patches_img = extract_neighborhood_patches(img_res, center_patch, p=PATCH_P, nh=NEIGHBORHOOD_NH)
    patches_overlay = extract_neighborhood_patches(combined_res, center_patch, p=PATCH_P, nh=NEIGHBORHOOD_NH)

    tree_mask = make_tree_diagram(patches_mask, nh=NEIGHBORHOOD_NH, display_scale=4, line_color=(0,0,0,255))
    tree_pr = make_tree_diagram(patches_img, nh=NEIGHBORHOOD_NH, display_scale=4, line_color=(0,0,0,255))
    tree_overlay = make_tree_diagram(patches_overlay, nh=NEIGHBORHOOD_NH, display_scale=4, line_color=(255,204,0,220))

    tree_mask.save(OUT_DIR / f'{idx:02d}_{stem}_tree_nh{NEIGHBORHOOD_NH}_mask.png')
    tree_pr.save(OUT_DIR / f'{idx:02d}_{stem}_tree_nh{NEIGHBORHOOD_NH}_pr.png')
    tree_overlay.save(OUT_DIR / f'{idx:02d}_{stem}_tree_nh{NEIGHBORHOOD_NH}_overlay.png')

    # combine the three tree diagrams into a single-row mosaic (fixed positions)
    try:
        a = tree_mask.convert('RGBA')
        b = tree_pr.convert('RGBA')
        c = tree_overlay.convert('RGBA')

        w, h = a.size
        spacing = 8
        mosaic_w = w * 3 + spacing * 2
        mosaic_h = h
        mosaic = Image.new('RGBA', (mosaic_w, mosaic_h), (255, 255, 255, 255))

        mosaic.paste(a, (0, 0), a)
        mosaic.paste(b, (w + spacing, 0), b)
        mosaic.paste(c, ((w + spacing) * 2, 0), c)

        mosaic_path = OUT_DIR / f'{idx:02d}_{stem}_tree_mosaic_row.png'
        mosaic.convert('RGB').save(mosaic_path)
    except Exception as e:
        print(f'Warning: failed to create tree mosaic for {stem}: {e}')

    # --- New: annotate the full-size grid images to highlight the neighborhood and tree lines ---
    def annotate_full_image(base_img, center_patch, nh=NEIGHBORHOOD_NH, p=PATCH_P,
                            neigh_fill=(255, 204, 0, 90), center_outline=(255, 0, 0, 220), line_color=(255,204,0,220),
                            mask_overlay=None, mask_alpha=25):
        """Return an RGBA image with neighborhood patches highlighted, center outlined, and tree lines drawn.

        base_img should be an RGBA image of size IMG_SIZE.
        """
        canvas = base_img.convert('RGBA').copy()

        # If a mask overlay is provided, composite a translucent version on top
        if mask_overlay is not None:
            mo = mask_overlay.convert('RGBA').resize(IMG_SIZE, resample=Image.NEAREST)
            arr = np.array(mo)
            # set mask alpha to desired mask_alpha where mask exists, else 0
            alpha = (arr[..., 3] > 0).astype(np.uint8) * int(mask_alpha)
            arr[..., 3] = alpha
            mo2 = Image.fromarray(arr, mode='RGBA')
            try:
                canvas = Image.alpha_composite(canvas, mo2)
            except Exception:
                # fallback to paste
                canvas.paste(mo2, (0, 0), mo2)

        draw = ImageDraw.Draw(canvas, 'RGBA')

        cx, cy = center_patch
        grid_size = 2 * nh + 1

        # compute centers for all patches and draw fills/outlines
        centers = []
        idxs = []
        for ry in range(-nh, nh + 1):
            for rx in range(-nh, nh + 1):
                x0 = (cx + rx) * p
                y0 = (cy + ry) * p
                x1 = x0 + p
                y1 = y0 + p
                centers.append(((x0 + x1)//2, (y0 + y1)//2))
                idxs.append((x0, y0, x1, y1))

        # Do not fill neighborhood patches; keep background visible.
        # (previously we filled patches here, which added a grey block.)
        # If a fill is desired later, we can draw it here using `neigh_fill`.

        # outline center patch thicker
        center_idx = (grid_size // 2) * grid_size + (grid_size // 2)
        cx0, cy0, cx1, cy1 = idxs[center_idx]
        draw.rectangle([cx0, cy0, cx1, cy1], outline=center_outline, width=max(3, nh+1))

        # draw tree lines from center to each neighbor
        root = centers[center_idx]
        for i, c in enumerate(centers):
            if i == center_idx:
                continue
            draw.line([root, c], fill=line_color, width=max(2, nh))

        return canvas

    try:
        # apply translucent colored mask overlay (90% transparency -> 10% opacity ~= alpha=25)
        mask_alpha = int(255 * 0.10)
        annotated_mask_full = annotate_full_image(mask_rgb.convert('RGBA'), center_patch, nh=NEIGHBORHOOD_NH,
                                                  p=PATCH_P, neigh_fill=(0,0,0,100), center_outline=(255,0,0,220), line_color=(0,0,0,220),
                                                  mask_overlay=colored_mask, mask_alpha=mask_alpha)
        annotated_pr_full = annotate_full_image(img_grid.convert('RGBA'), center_patch, nh=NEIGHBORHOOD_NH,
                                                p=PATCH_P, neigh_fill=(0,0,0,100), center_outline=(255,0,0,220), line_color=(0,0,0,220),
                                                mask_overlay=colored_mask, mask_alpha=mask_alpha)
        annotated_overlay_full = annotate_full_image(combined_grid.convert('RGBA'), center_patch, nh=NEIGHBORHOOD_NH,
                                                     p=PATCH_P, neigh_fill=(0,0,0,100), center_outline=(255,0,0,220), line_color=(255,204,0,220),
                                                     mask_overlay=colored_mask, mask_alpha=mask_alpha)

        annotated_mask_full.save(OUT_DIR / f'{idx:02d}_{stem}_annotated_nh{NEIGHBORHOOD_NH}_mask.png')
        annotated_pr_full.save(OUT_DIR / f'{idx:02d}_{stem}_annotated_nh{NEIGHBORHOOD_NH}_pr.png')
        annotated_overlay_full.save(OUT_DIR / f'{idx:02d}_{stem}_annotated_nh{NEIGHBORHOOD_NH}_overlay.png')

        # create a 3-image mosaic of the annotated full images
        a = annotated_mask_full.convert('RGBA')
        b = annotated_pr_full.convert('RGBA')
        c = annotated_overlay_full.convert('RGBA')
        w, h = a.size
        spacing = 4
        mosaic_w = w * 3 + spacing * 2
        mosaic = Image.new('RGBA', (mosaic_w, h), (255,255,255,255))
        mosaic.paste(a, (0,0), a)
        mosaic.paste(b, (w + spacing, 0), b)
        mosaic.paste(c, ((w + spacing) * 2, 0), c)
        mosaic.convert('RGB').save(OUT_DIR / f'{idx:02d}_{stem}_annotated_full_mosaic_nh{NEIGHBORHOOD_NH}.png')
    except Exception as e:
        print(f'Warning: failed to create annotated full images for {stem}: {e}')

    # Create a single-row mosaic: [mask_with_grid | pr_with_grid | overlay_with_grid]
    try:
        # Ensure all three are RGBA and same size
        a = mask_grid.convert('RGBA')
        b = img_grid.convert('RGBA')
        c = combined_grid.convert('RGBA')

        w, h = a.size
        spacing = 4
        mosaic_w = w * 3 + spacing * 2
        mosaic_h = h
        mosaic = Image.new('RGBA', (mosaic_w, mosaic_h), (255, 255, 255, 255))

        # Make mask tile have a black background where mask is transparent
        mask_tile = Image.new('RGBA', (w, h), (0, 0, 0, 255))
        mask_tile.paste(a, (0, 0), a)

        mosaic.paste(mask_tile, (0, 0), mask_tile)
        mosaic.paste(b, (w + spacing, 0), b)
        mosaic.paste(c, ((w + spacing) * 2, 0), c)

        mosaic_path = OUT_DIR / f'{idx:02d}_{stem}_mosaic_row.png'
        mosaic.convert('RGB').save(mosaic_path)
    except Exception as e:
        print(f'Warning: failed to create mosaic for {stem}: {e}')


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

    for idx, (mask_path, area) in enumerate(top, start=1):
        try:
            img = load_image_for_mask(mask_path)
        except FileNotFoundError as e:
            print('Warning:', e)
            continue
        mask_img = Image.open(mask_path).convert('L')
        save_with_grid(mask_path, img, mask_img, None, idx)

    print(f'Outputs saved to: {OUT_DIR}')


if __name__ == '__main__':
    main()
