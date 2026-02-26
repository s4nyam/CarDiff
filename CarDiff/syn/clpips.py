import argparse
import os
from pathlib import Path
import csv
import torch
import lpips
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm
import sys


def load_image(path, device):
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


def main():
    parser = argparse.ArgumentParser(description="Compute LPIPS epoch-wise (paired).")
    parser.add_argument('--real_root', type=str, required=True)
    parser.add_argument('--generated_base', type=str, required=True)
    parser.add_argument('--epochs', type=int, nargs='+', required=True)
    parser.add_argument('--prefix', type=str, default='')
    parser.add_argument('--suffix', type=str, default='-ddpm')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    print("Loading LPIPS model...", file=sys.stderr)
    loss_fn = lpips.LPIPS(net='alex').to(device)
    loss_fn.eval()

    print("Building real image dictionary...", file=sys.stderr)
    real_dict = build_real_dict(args.real_root)

    csv_rows = []

    for epoch in args.epochs:

        gen_folder_name = f"{args.prefix}{epoch}{args.suffix}"
        gen_path = os.path.join(args.generated_base, gen_folder_name)

        if not os.path.exists(gen_path):
            print(f"Skipping epoch {epoch}", file=sys.stderr)
            continue

        print(f"\nProcessing Epoch {epoch}", file=sys.stderr)

        scores = []

        for gen_img_path in tqdm(list(Path(gen_path).glob("*")), desc=f"Epoch {epoch}", file=sys.stderr):
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
            continue

        mean_score = sum(scores) / len(scores)

        print(f"Epoch {epoch}: LPIPS={mean_score:.6f}")

        csv_rows.append([
            Path(args.real_root).name,
            gen_folder_name,
            epoch,
            mean_score,
            len(scores)
        ])

    with open("lpips.csv", mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["real_path", "generated_path", "epoch", "lpips_mean", "num_pairs"])
        writer.writerows(csv_rows)

    print("\nSaved results to lpips.csv", file=sys.stderr)


if __name__ == '__main__':
    main()