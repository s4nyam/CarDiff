import argparse
from torch_fidelity import calculate_metrics
from PIL import Image
import os
from pathlib import Path
import shutil
from tqdm import tqdm
import sys
import csv


IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}


def resize_images(input_folder, output_folder, size=(224, 224)):
    os.makedirs(output_folder, exist_ok=True)

    image_files = [
        f for f in Path(input_folder).iterdir()
        if f.suffix.lower() in IMAGE_EXTENSIONS
    ]

    for img_path in tqdm(image_files, desc=f"Resizing {input_folder}", unit="img", file=sys.stderr):
        try:
            img = Image.open(img_path)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img_resized = img.resize(size, Image.LANCZOS)
            img_resized.save(os.path.join(output_folder, img_path.name))
        except Exception as e:
            print(f"Error processing {img_path.name}: {e}", file=sys.stderr)

    return len(image_files)


def extract_epoch(folder_name):
    if "-syn" in folder_name:
        return int(folder_name.split("-syn")[-1])
    return -1


def compute_is_for_class(synthetic_parent, device, batch_size, results_dir):

    synthetic_name = Path(synthetic_parent).name

    print(f"\n==============================")
    print(f"Processing Synthetic Class: {synthetic_name}")
    print(f"Path: {synthetic_parent}")
    print(f"==============================\n")

    temp_dir = Path("temp_is_resized")

    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True)

    csv_rows = []

    # Collect and naturally sort epoch folders
    syn_folders = [
        f for f in Path(synthetic_parent).iterdir()
        if f.is_dir() and "-syn" in f.name
    ]

    syn_folders = sorted(
        syn_folders,
        key=lambda x: extract_epoch(x.name)
    )

    for syn_folder in syn_folders:

        epoch = extract_epoch(syn_folder.name)
        temp_syn = temp_dir / f"{syn_folder.name}_resized"

        print(f"\nComputing IS for {syn_folder.name}")

        num_generated = resize_images(syn_folder, temp_syn)

        if num_generated == 0:
            print(f"Skipping epoch {epoch} (no images found)")
            continue

        metrics = calculate_metrics(
            input1=str(temp_syn),
            cuda=(device == 'cuda'),
            batch_size=batch_size,
            isc=True,
            fid=False,
            kid=False,
            verbose=False
        )

        is_mean = metrics['inception_score_mean']
        is_std = metrics['inception_score_std']

        print(f"Epoch {epoch}: IS mean={is_mean:.6f}, std={is_std:.6f}")

        csv_rows.append([
            synthetic_name,
            syn_folder.name,
            epoch,
            is_mean,
            is_std
        ])

        shutil.rmtree(temp_syn)

    # Save CSV
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    csv_path = results_dir / f"is_{synthetic_name}.csv"

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["synthetic_class", "epoch_folder", "epoch", "is_mean", "is_std"])
        writer.writerows(csv_rows)

    print(f"\nSaved: {csv_path}")

    shutil.rmtree(temp_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_path', type=str, required=True,
                        help="Path containing DC, DCs, MC, MCs, etc.")
    parser.add_argument('--results_dir', type=str, required=True,
                        help="Directory where CSVs will be saved.")
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--batch_size', type=int, default=16)
    args = parser.parse_args()

    base_path = Path(args.base_path)

    # Only synthetic folders (ending with 's')
    synthetic_folders = [
        f for f in base_path.iterdir()
        if f.is_dir() and f.name.endswith('s')
    ]

    for synthetic_folder in synthetic_folders:
        compute_is_for_class(
            synthetic_parent=str(synthetic_folder),
            device=args.device,
            batch_size=args.batch_size,
            results_dir=args.results_dir
        )


if __name__ == "__main__":
    main()
