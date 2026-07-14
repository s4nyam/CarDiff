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
from torch_fidelity import calculate_metrics


# -------------------------------------------------
# Resize images
# -------------------------------------------------
def resize_images(input_folder, output_folder, size=(224, 224)):
    os.makedirs(output_folder, exist_ok=True)

    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}

    image_files = [
        f for f in Path(input_folder).iterdir()
        if f.suffix.lower() in image_extensions
    ]

    for img_path in tqdm(image_files, desc=f"Resizing {input_folder}", unit="img", file=sys.stderr):
        try:
            img = Image.open(img_path)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img = img.resize(size, Image.LANCZOS)
            img.save(os.path.join(output_folder, img_path.name))
        except Exception as e:
            print(f"Error processing {img_path.name}: {e}", file=sys.stderr)

    return len(image_files)


# -------------------------------------------------
# LPIPS helpers
# -------------------------------------------------
def load_image_lpips(path, device):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])

    img = Image.open(path).convert("RGB")
    img = transform(img).unsqueeze(0).to(device)
    return img


def build_real_dict(real_root):
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    real_dict = {}

    for img_path in Path(real_root).rglob("*"):
        if img_path.suffix.lower() in image_extensions:
            real_dict[img_path.name] = str(img_path)

    return real_dict


# -------------------------------------------------
# Main
# -------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Compute FID, KID, IS and LPIPS between two folders.")
    parser.add_argument('--real_dir', type=str, default="/scratch/project_465002351/cardiff/compare_other_models/cardiff/images", help='Path to real images')
    parser.add_argument('--syn_dir', type=str, default="/scratch/project_465002351/cardiff/compare_other_models/cardiff/syn", help='Path to synthetic images')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    temp_dir = "cardiff_resized_temp"
    temp_real = os.path.join(temp_dir, "real")
    temp_syn = os.path.join(temp_dir, "syn")

    os.makedirs(temp_dir, exist_ok=True)

    try:
        print("\nResizing images...", file=sys.stderr)

        num_real = resize_images(args.real_dir, temp_real)
        num_syn = resize_images(args.syn_dir, temp_syn)

        if num_real == 0 or num_syn == 0:
            print("One of the folders has no images.", file=sys.stderr)
            return

        subset_size = min(num_real, num_syn)

        # -------------------------------------------------
        # FID + KID
        # -------------------------------------------------
        metrics = calculate_metrics(
            input1=temp_real,
            input2=temp_syn,
            cuda=(device.type == 'cuda'),
            batch_size=args.batch_size,
            fid=True,
            kid=True,
            kid_subset_size=subset_size,
            isc=False,
            verbose=False,
            input1_cache_name="cache_real_cardiff"
        )

        fid_value = metrics['frechet_inception_distance']
        kid_mean = metrics['kernel_inception_distance_mean']
        kid_std = metrics['kernel_inception_distance_std']

        # -------------------------------------------------
        # Inception Score (only synthetic)
        # -------------------------------------------------
        metrics_is = calculate_metrics(
            input1=temp_syn,
            cuda=(device.type == 'cuda'),
            batch_size=args.batch_size,
            isc=True,
            verbose=False
        )

        is_mean = metrics_is['inception_score_mean']
        is_std = metrics_is['inception_score_std']

        # -------------------------------------------------
        # LPIPS (paired by filename)
        # -------------------------------------------------
        print("Computing LPIPS...", file=sys.stderr)

        loss_fn = lpips.LPIPS(net='alex').to(device)
        loss_fn.eval()

        real_dict = build_real_dict(args.real_dir)

        scores = []

        for syn_img_path in tqdm(Path(args.syn_dir).glob("*"), desc="LPIPS", file=sys.stderr):
            filename = syn_img_path.name
            if filename not in real_dict:
                continue

            try:
                img_real = load_image_lpips(real_dict[filename], device)
                img_syn = load_image_lpips(str(syn_img_path), device)

                with torch.no_grad():
                    score = loss_fn(img_real, img_syn)

                scores.append(score.item())

            except Exception as e:
                print(f"LPIPS error {filename}: {e}", file=sys.stderr)

        lpips_mean = sum(scores) / len(scores) if len(scores) > 0 else None

        # -------------------------------------------------
        # Print results
        # -------------------------------------------------
        print("\n========== RESULTS ==========")
        print(f"FID:     {fid_value:.6f}")
        print(f"KID:     {kid_mean:.6f} ± {kid_std:.6f}")
        print(f"IS:      {is_mean:.6f} ± {is_std:.6f}")
        print(f"LPIPS:   {lpips_mean:.6f}" if lpips_mean else "LPIPS:   N/A")
        print("=============================\n")

        # -------------------------------------------------
        # Save CSV
        # -------------------------------------------------
        with open("cardiff.csv", mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                "real_path",
                "synthetic_path",
                "fid",
                "kid_mean",
                "kid_std",
                "is_mean",
                "is_std",
                "lpips_mean",
                "num_lpips_pairs"
            ])
            writer.writerow([
                Path(args.real_dir).name,
                Path(args.syn_dir).name,
                fid_value,
                kid_mean,
                kid_std,
                is_mean,
                is_std,
                lpips_mean,
                len(scores)
            ])

        print("Saved results to cardiff.csv", file=sys.stderr)

        shutil.rmtree(temp_dir)

    except Exception as e:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()