"""
Create a mosaic of images from each model folder.
Columns: Label, Original, CarDiff, SCDM, SegDiff, SPADE
Rows: one per image (same filenames across folders).
Overlays coloured class-contours from the label masks onto every image.
Saves output as both PNG and PDF.
"""

import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from pathlib import Path
from PIL import Image
from scipy.ndimage import binary_erosion

# ── Configuration ────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent

# Contour thickness in pixels (change this value to make contours thicker/thinner)
CONTOUR_THICKNESS = 1

# Class pixel values → contour colours (R, G, B) in [0, 1]
#   0   = background  (no contour)
#   102 = Superficial Caries (SC) → Blue
#   153 = Medium Caries (MC)     → Green
#   255 = Deep Caries (DC)       → Red
CLASS_CONTOURS = {
    102: (0.0, 0.0, 1.0),   # Blue  – SC
    153: (0.0, 1.0, 0.0),   # Green – MC
    255: (1.0, 0.0, 0.0),   # Red   – DC
}

# (column_display_name, folder_name)
# Add ("SISDM", "sisdm") when the folder is available.
COLUMNS = [
    ("Label",    "labels"),
    ("Original", "org"),
    ("CarDiff",  "cardiff"),
    ("SCDM",    "scdm"),
    ("SISDM",   "sisdm"),   # uncomment when folder exists
    ("SegDiff",  "segd"),
    ("SPADE",    "spade"),
]

# Sorted so rows appear in a deterministic order
IMAGE_NAMES = sorted(os.listdir(BASE_DIR / COLUMNS[0][1]))


# ── Helper functions ─────────────────────────────────────────────────────────
def load_label_mask(path):
    """Load a label/mask image as a uint8 greyscale array."""
    return np.array(Image.open(path).convert("L"))


def compute_contours(label_mask, thickness=CONTOUR_THICKNESS):
    """Return an RGBA overlay image with coloured contours for each class."""
    h, w = label_mask.shape
    overlay = np.zeros((h, w, 4), dtype=np.float32)
    struct = np.ones((3, 3), dtype=bool)

    for class_val, colour in CLASS_CONTOURS.items():
        region = label_mask == class_val
        if not region.any():
            continue
        eroded = binary_erosion(region, structure=struct, iterations=thickness)
        boundary = region & ~eroded
        overlay[boundary, :3] = colour
        overlay[boundary, 3] = 1.0  # fully opaque on contour pixels

    return overlay


def to_greyscale_rgb(img):
    """Convert any image array to a 3-channel greyscale float32 in [0, 1]."""
    if img.dtype == np.uint8:
        img = img.astype(np.float32) / 255.0
    if img.ndim == 3 and img.shape[2] >= 3:
        grey = np.dot(img[..., :3], [0.2989, 0.5870, 0.1140])
    else:
        grey = img if img.ndim == 2 else img[..., 0]
    return np.stack([grey, grey, grey], axis=-1)


def overlay_contours(base_rgb, contour_rgba):
    """Alpha-composite the contour RGBA overlay onto an RGB base."""
    alpha = contour_rgba[..., 3:4]
    blended = base_rgb * (1 - alpha) + contour_rgba[..., :3] * alpha
    return np.clip(blended, 0.0, 1.0)


# ── Build mosaic ─────────────────────────────────────────────────────────────
n_rows = len(IMAGE_NAMES)
n_cols = len(COLUMNS)

fig, axes = plt.subplots(
    n_rows, n_cols,
    figsize=(3 * n_cols, 3 * n_rows),
)
fig.subplots_adjust(wspace=0.03, hspace=0.03)

# Ensure axes is always 2-D
if n_rows == 1:
    axes = axes[np.newaxis, :]
if n_cols == 1:
    axes = axes[:, np.newaxis]

# Column headers
for col_idx, (col_name, _) in enumerate(COLUMNS):
    axes[0, col_idx].set_title(col_name, fontsize=16, fontweight="bold", pad=10)

# Fill in images
for row_idx, img_name in enumerate(IMAGE_NAMES):
    # Load label mask once per row → derive contour overlay
    label_path = BASE_DIR / "labels" / img_name
    label_mask = load_label_mask(label_path)
    contour_overlay = compute_contours(label_mask)

    for col_idx, (col_name, folder) in enumerate(COLUMNS):
        ax = axes[row_idx, col_idx]
        img_path = BASE_DIR / folder / img_name

        if img_path.exists():
            img = mpimg.imread(str(img_path))
            rgb = to_greyscale_rgb(img)
            composited = overlay_contours(rgb, contour_overlay)
            ax.imshow(composited)
        else:
            ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                    fontsize=14, transform=ax.transAxes)

        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines[:].set_visible(False)

# ── Save ─────────────────────────────────────────────────────────────────────
out_png = BASE_DIR / "mosaic.png"
out_pdf = BASE_DIR / "mosaic.pdf"

fig.savefig(out_png, dpi=200, bbox_inches="tight")
fig.savefig(out_pdf, bbox_inches="tight")
plt.close(fig)

print(f"Saved  PNG → {out_png}")
print(f"Saved  PDF → {out_pdf}")
