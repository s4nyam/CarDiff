#!/usr/bin/env python3
import os
from PIL import Image
import numpy as np

# ------------------ CONFIG ------------------ #

# Input directory with original label masks
LABEL_DIR = "./test_data/labels"

# Root directory where the processed masks will be saved
OUT_ROOT = "./test_data/processed_labels"

# Image extensions to consider
EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

# Pixel values for each class
SC_VAL = 102  # Superficial Caries
MC_VAL = 153  # Medium Caries
DC_VAL = 255  # Deep Caries

# Combinations we want to generate:
#   - single classes
#   - pairwise combinations
#   - all three together
COMBINATIONS = {
    "SC": {SC_VAL},
    "MC": {MC_VAL},
    "DC": {DC_VAL},
    "SC+MC": {SC_VAL, MC_VAL},
    "SC+DC": {SC_VAL, DC_VAL},
    "MC+DC": {MC_VAL, DC_VAL},
    "SC+MC+DC": {SC_VAL, MC_VAL, DC_VAL},
}

# -------------------------------------------- #


def ensure_output_dirs():
    """
    Create the output root and all subfolders for each combination.
    """
    print(f"[INFO] Ensuring output root directory exists at: {OUT_ROOT}")
    os.makedirs(OUT_ROOT, exist_ok=True)

    for combo_name in COMBINATIONS.keys():
        combo_dir = os.path.join(OUT_ROOT, combo_name)
        os.makedirs(combo_dir, exist_ok=True)
        print(f"[INFO]   Ensured directory for combination '{combo_name}': {combo_dir}")


def is_image_file(filename: str) -> bool:
    """
    Check if the file has a valid image extension.
    """
    ext = os.path.splitext(filename)[1].lower()
    return ext in EXTS


def process_labels():
    """
    Main processing function:
    For each label image:
      - detect which of SC/MC/DC are present
      - for each of the 7 combinations, if the required labels are present,
        create a new mask keeping only those labels (others set to 0).
    """
    print(f"[INFO] Starting processing of label images in: {LABEL_DIR}")

    if not os.path.isdir(LABEL_DIR):
        print(f"[ERROR] Label directory does not exist: {LABEL_DIR}")
        return

    ensure_output_dirs()

    all_files = [f for f in os.listdir(LABEL_DIR) if is_image_file(f)]
    print(f"[INFO] Found {len(all_files)} image file(s) to process.")

    # For summary statistics
    combo_counts = {name: 0 for name in COMBINATIONS.keys()}
    processed_files = 0
    skipped_files = 0

    for filename in sorted(all_files):
        in_path = os.path.join(LABEL_DIR, filename)
        print(f"\n[INFO] Processing file: {filename}")

        try:
            img = Image.open(in_path).convert("L")  # force grayscale
            arr = np.array(img)
        except Exception as e:
            print(f"[ERROR]   Failed to open/convert '{filename}': {e}")
            skipped_files += 1
            continue

        # Find which labeled pixels are present in this image
        unique_vals = set(np.unique(arr).tolist())
        present_labels = {
            v for v in unique_vals if v in {SC_VAL, MC_VAL, DC_VAL}
        }

        print(f"[DEBUG]   Unique pixel values in image: {sorted(unique_vals)}")
        print(f"[DEBUG]   Present label values (SC/MC/DC only): {sorted(present_labels)}")

        if not present_labels:
            print(f"[WARN]    No SC/MC/DC labels found in '{filename}'. Skipping combinations.")
            processed_files += 1
            continue

        # For each combination, check if its labels are a subset of the present labels
        for combo_name, combo_labels in COMBINATIONS.items():
            if combo_labels.issubset(present_labels):
                # Create mask that keeps only labels in combo_labels, rest -> 0
                # We keep the actual label values (102/153/255) as-is.
                mask_arr = np.where(np.isin(arr, list(combo_labels)), arr, 0).astype(np.uint8)

                out_dir = os.path.join(OUT_ROOT, combo_name)
                out_path = os.path.join(out_dir, filename)

                try:
                    out_img = Image.fromarray(mask_arr, mode="L")
                    out_img.save(out_path)
                    combo_counts[combo_name] += 1
                    print(
                        f"[INFO]   Saved combination '{combo_name}' for '{filename}' "
                        f"-> {out_path}"
                    )
                except Exception as e:
                    print(
                        f"[ERROR]   Failed to save '{combo_name}' mask for '{filename}': {e}"
                    )

        processed_files += 1

    # Summary
    print("\n[SUMMARY] Processing complete.")
    print(f"[SUMMARY] Total files processed: {processed_files}")
    print(f"[SUMMARY] Total files skipped due to errors: {skipped_files}")
    print("[SUMMARY] Per-combination counts:")
    for combo_name, count in combo_counts.items():
        print(f"  {combo_name}: {count} file(s) generated")


if __name__ == "__main__":
    process_labels()
