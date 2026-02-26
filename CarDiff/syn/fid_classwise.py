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


def extract_epoch(folder_name):
    if "-syn" in folder_name:
        return folder_name.split("-syn")[-1]
    return "NA"


def compute_fid_for_class(real_path, synthetic_parent, device, batch_size, results_dir):
    real_name = Path(real_path).name
    synthetic_name = Path(synthetic_parent).name

    print(f"\n==============================")
    print(f"Processing Class: {real_name}")
    print(f"Real: {real_path}")
    print(f"Synthetic Parent: {synthetic_parent}")
    print(f"==============================\n")

    temp_dir = Path("temp_fid_resized")
    temp_real = temp_dir / f"{real_name}_real"

    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True)

    # Resize real once
    resize_images(real_path, temp_real)

    csv_rows = []

    # Collect synthetic folders
    syn_folders = [
        f for f in Path(synthetic_parent).iterdir()
        if f.is_dir() and "-syn" in f.name
    ]

    # Natural numeric sort
    syn_folders = sorted(
        syn_folders,
        key=lambda x: int(extract_epoch(x.name))
    )

    for syn_folder in syn_folders:

        epoch = extract_epoch(syn_folder.name)
        temp_syn = temp_dir / f"{syn_folder.name}_resized"

        print(f"\nComputing FID for {syn_folder.name}")

        resize_images(syn_folder, temp_syn)

        metrics = calculate_metrics(
            input1=str(temp_real),
            input2=str(temp_syn),
            cuda=(device == 'cuda'),
            batch_size=batch_size,
            fid=True,
            kid=False,
            verbose=False,
            input1_cache_name=f"{real_name}_cache"
        )

        fid_value = metrics['frechet_inception_distance']

        print(f"Epoch {epoch}: FID = {fid_value:.6f}")

        csv_rows.append([
            real_name,
            syn_folder.name,
            epoch,
            fid_value
        ])

        shutil.rmtree(temp_syn)


    # ✅ Create results directory if needed
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # ✅ Save CSV inside target directory
    csv_path = results_dir / f"fid_{synthetic_name}.csv"

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["real_class", "synthetic_folder", "epoch", "fid"])
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
            compute_fid_for_class(
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
