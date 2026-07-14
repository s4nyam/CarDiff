import os
import cv2
import numpy as np
import random
from tqdm import tqdm

# ================= CONFIG =================
INPUT_FOLDER = "all_masks"
OUTPUT_FOLDER = "all_masks_processed"

NUM_OUTPUTS_PER_IMAGE = 5
MAX_PATCHES_PER_MASK = 5
MASK_SIZE = 384  # original mask size (384x384)

ALLOWED_EXTENSIONS = (".png", ".jpg", ".jpeg")
# ==========================================

os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# --------------------------------------------------
# STEP 1: Extract square patches from segments
# --------------------------------------------------

def extract_segment_patches(mask):
    patches = []

    binary = (mask > 0).astype(np.uint8)
    num_labels, labels = cv2.connectedComponents(binary)

    for label_id in range(1, num_labels):
        ys, xs = np.where(labels == label_id)

        if len(xs) == 0:
            continue

        xmin, xmax = xs.min(), xs.max()
        ymin, ymax = ys.min(), ys.max()

        width = xmax - xmin + 1
        height = ymax - ymin + 1
        side = max(width, height)

        cx = (xmin + xmax) // 2
        cy = (ymin + ymax) // 2

        half = side // 2
        x1 = max(0, cx - half)
        y1 = max(0, cy - half)
        x2 = min(mask.shape[1], x1 + side)
        y2 = min(mask.shape[0], y1 + side)

        square_patch = mask[y1:y2, x1:x2]

        h, w = square_patch.shape
        if h != side or w != side:
            padded = np.zeros((side, side), dtype=mask.dtype)
            padded[:h, :w] = square_patch
            square_patch = padded

        patches.append({
            "patch": square_patch,
            "size": side
        })

    return patches


# --------------------------------------------------
# STEP 2: Generate synthetic mask
# --------------------------------------------------

def generate_synthetic_mask(patch_pool):
    new_mask = np.zeros((MASK_SIZE, MASK_SIZE), dtype=np.uint8)
    occupied = np.zeros((MASK_SIZE, MASK_SIZE), dtype=np.uint8)

    num_patches = random.randint(1, MAX_PATCHES_PER_MASK)

    for _ in range(num_patches):

        valid_patches = [p for p in patch_pool if p["size"] <= MASK_SIZE]
        if not valid_patches:
            break

        chosen = random.choice(valid_patches)
        patch = chosen["patch"]
        size = chosen["size"]

        for _ in range(50):
            x = random.randint(0, MASK_SIZE - size)
            y = random.randint(0, MASK_SIZE - size)

            region = occupied[y:y+size, x:x+size]
            if np.any(region > 0):
                continue

            new_mask[y:y+size, x:x+size] = patch
            occupied[y:y+size, x:x+size] = (patch > 0).astype(np.uint8)
            break

    return new_mask


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():
    all_patches = []

    files = [
        f for f in os.listdir(INPUT_FOLDER)
        if f.lower().endswith(ALLOWED_EXTENSIONS)
    ]

    print("Extracting segment patches...")

    for file in tqdm(files):
        path = os.path.join(INPUT_FOLDER, file)
        mask = cv2.imread(path, cv2.IMREAD_UNCHANGED)

        if mask is None:
            continue

        if len(mask.shape) == 3:
            mask = mask[:, :, 0]

        patches = extract_segment_patches(mask)
        all_patches.extend(patches)

    print(f"Total patches collected: {len(all_patches)}")
    print("Generating synthetic masks...")

    counter = 1  # Global sequential counter

    for _ in tqdm(range(len(files) * NUM_OUTPUTS_PER_IMAGE)):
        synthetic_mask = generate_synthetic_mask(all_patches)

        out_name = f"aug_{counter:06d}.png"
        cv2.imwrite(
            os.path.join(OUTPUT_FOLDER, out_name),
            synthetic_mask
        )

        counter += 1


if __name__ == "__main__":
    main()