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


def compute_kid_for_class(real_path, synthetic_parent, device, batch_size, results_dir):

    real_name = Path(real_path).name
    synthetic_name = Path(synthetic_parent).name

    print(f"\n==============================")
    print(f"Processing Class: {real_name}")
    print(f"Real: {real_path}")
    print(f"Synthetic Parent: {synthetic_parent}")
    print(f"==============================\n")

    temp_dir = Path("temp_kid_resized")
    temp_real = temp_dir / f"{real_name}_real"

    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True)

    # Resize real once
    num_real = resize_images(real_path, temp_real)

    csv_rows = []

    # Collect and naturally sort synthetic folders
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

        print(f"\nComputing KID for {syn_folder.name}")

        num_generated = resize_images(syn_folder, temp_syn)

        subset_size = min(num_real, num_generated)

        metrics = calculate_metrics(
            input1=str(temp_real),
            input2=str(temp_syn),
            cuda=(device == 'cuda'),
            batch_size=batch_size,
            fid=False,
            kid=True,
            kid_subset_size=subset_size,
            verbose=False,
            input1_cache_name=f"{real_name}_kid_cache"
        )

        kid_mean = metrics['kernel_inception_distance_mean']
        kid_std = metrics['kernel_inception_distance_std']

        print(f"Epoch {epoch}: KID mean={kid_mean:.6f}, std={kid_std:.6f}")

        csv_rows.append([
            real_name,
            syn_folder.name,
            epoch,
            kid_mean,
            kid_std
        ])

        shutil.rmtree(temp_syn)

    # Save CSV to results directory
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    csv_path = results_dir / f"kid_{synthetic_name}.csv"

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["real_class", "synthetic_folder", "epoch", "kid_mean", "kid_std"])
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

    folders = [f for f in base_path.iterdir() if f.is_dir()]
    real_folders = [f for f in folders if not f.name.endswith('s')]

    for real_folder in real_folders:

        synthetic_folder_name = real_folder.name + "s"
        synthetic_folder = base_path / synthetic_folder_name

        if synthetic_folder.exists():
            compute_kid_for_class(
                real_path=str(real_folder),
                synthetic_parent=str(synthetic_folder),
                device=args.device,
                batch_size=args.batch_size,
                results_dir=args.results_dir
            )
        else:
            print(f"Skipping {real_folder.name} (No matching synthetic folder found)")


if __name__ == "__main__":
    main()
