import argparse
import os
from pathlib import Path
import shutil
import sys
import csv
import torch
import lpips
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm


IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}


def load_image(path, device):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])

    img = Image.open(path).convert("RGB")
    img = transform(img).unsqueeze(0).to(device)
    return img


def extract_epoch(folder_name):
    if "-syn" in folder_name:
        return folder_name.split("-syn")[-1]
    return "NA"


def build_real_dict(real_folder):
    real_dict = {}
    for img_path in Path(real_folder).iterdir():
        if img_path.suffix.lower() in IMAGE_EXTENSIONS:
            real_dict[img_path.name] = str(img_path)
    return real_dict


def compute_lpips_for_class(real_path, synthetic_parent, device, results_dir):

    real_name = Path(real_path).name
    synthetic_name = Path(synthetic_parent).name

    print(f"\n==============================")
    print(f"Processing Class: {real_name}")
    print(f"Real: {real_path}")
    print(f"Synthetic Parent: {synthetic_parent}")
    print(f"==============================\n")

    device = torch.device(device if torch.cuda.is_available() else 'cpu')

    print("Loading LPIPS model...")
    loss_fn = lpips.LPIPS(net='alex').to(device)
    loss_fn.eval()

    real_dict = build_real_dict(real_path)

    csv_rows = []

    syn_folders = [
        f for f in Path(synthetic_parent).iterdir()
        if f.is_dir() and "-syn" in f.name
    ]

    syn_folders = sorted(
        syn_folders,
        key=lambda x: int(extract_epoch(x.name))
    )

    for syn_folder in syn_folders:

        epoch = extract_epoch(syn_folder.name)
        print(f"\nComputing LPIPS for {syn_folder.name}")

        scores = []

        for gen_img_path in tqdm(
            list(Path(syn_folder).iterdir()),
            desc=f"{real_name} Epoch {epoch}",
            file=sys.stderr
        ):

            if gen_img_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            filename = gen_img_path.name

            if filename not in real_dict:
                continue

            real_img_path = real_dict[filename]

            try:
                img_real = load_image(real_img_path, device)
                img_fake = load_image(str(gen_img_path), device)

                with torch.no_grad():
                    score = loss_fn(img_real, img_fake)

                scores.append(score.item())

            except Exception as e:
                print(f"Error processing {filename}: {e}", file=sys.stderr)

        if len(scores) == 0:
            print(f"No matching pairs found for epoch {epoch}")
            continue

        mean_score = sum(scores) / len(scores)

        print(f"Epoch {epoch}: LPIPS = {mean_score:.6f}")

        csv_rows.append([
            real_name,
            syn_folder.name,
            epoch,
            mean_score,
            len(scores)
        ])

    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    csv_path = results_dir / f"lpips_{synthetic_name}.csv"

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["real_class", "synthetic_folder", "epoch", "lpips_mean", "num_pairs"])
        writer.writerows(csv_rows)

    print(f"\nSaved: {csv_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_path', type=str, required=True,
                        help="Path containing DC, DCs, MC, MCs, etc.")
    parser.add_argument('--results_dir', type=str, required=True,
                        help="Directory where CSVs will be saved.")
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])
    args = parser.parse_args()

    base_path = Path(args.base_path)

    folders = [f for f in base_path.iterdir() if f.is_dir()]
    real_folders = [f for f in folders if not f.name.endswith('s')]

    for real_folder in real_folders:

        synthetic_folder_name = real_folder.name + "s"
        synthetic_folder = base_path / synthetic_folder_name

        if synthetic_folder.exists():
            compute_lpips_for_class(
                real_path=str(real_folder),
                synthetic_parent=str(synthetic_folder),
                device=args.device,
                results_dir=args.results_dir
            )
        else:
            print(f"Skipping {real_folder.name} (No matching synthetic folder found)")


if __name__ == "__main__":
    main()