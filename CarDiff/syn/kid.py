import argparse
from torch_fidelity import calculate_metrics
from PIL import Image
import os
from pathlib import Path
import shutil
from tqdm import tqdm
import sys
import csv


def resize_images(input_folder, output_folder, size=(224, 224)):
    os.makedirs(output_folder, exist_ok=True)

    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}

    image_files = [f for f in Path(input_folder).iterdir()
                   if f.suffix.lower() in image_extensions]

    for img_path in tqdm(image_files, desc=f"Resizing {input_folder}", unit="img", file=sys.stderr):
        try:
            img = Image.open(img_path)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img_resized = img.resize(size, Image.LANCZOS)
            img_resized.save(os.path.join(output_folder, img_path.name))
        except Exception as e:
            print(f"\nError processing {img_path.name}: {e}", file=sys.stderr)

    return len(image_files)


def main():
    parser = argparse.ArgumentParser(description="Calculate KID score: real vs multiple generated folders.")
    parser.add_argument('--real', type=str, required=True)
    parser.add_argument('--generated_base', type=str, required=True)
    parser.add_argument('--epochs', type=int, nargs='+', required=True)
    parser.add_argument('--prefix', type=str, default='')
    parser.add_argument('--suffix', type=str, default='-ddpm')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])
    args = parser.parse_args()

    temp_dir = 'resized_temp_kid'
    temp_real = os.path.join(temp_dir, 'real_resized')
    os.makedirs(temp_dir, exist_ok=True)

    try:
        print("=" * 60, file=sys.stderr)
        print("Step 1: Resizing real images (only once)...", file=sys.stderr)
        print("=" * 60, file=sys.stderr)

        num_real = resize_images(args.real, temp_real)

        csv_rows = []

        print("\n" + "=" * 60, file=sys.stderr)
        print("Step 2: Computing KID epoch-wise...", file=sys.stderr)
        print("=" * 60, file=sys.stderr)

        real_folder_name = Path(args.real).name

        for epoch in args.epochs:

            gen_folder_name = f"{args.prefix}{epoch}{args.suffix}"
            gen_path = os.path.join(args.generated_base, gen_folder_name)

            if not os.path.exists(gen_path):
                print(f"Skipping epoch {epoch} (folder not found)", file=sys.stderr)
                continue

            temp_generated = os.path.join(temp_dir, f'generated_resized_{epoch}')

            print(f"\nProcessing Epoch {epoch}", file=sys.stderr)
            print(f"Generated path: {gen_path}", file=sys.stderr)

            num_generated = resize_images(gen_path, temp_generated)

            # Safe subset size
            subset_size = min(num_real, num_generated)

            metrics = calculate_metrics(
                input1=temp_real,
                input2=temp_generated,
                cuda=(args.device == 'cuda'),
                batch_size=args.batch_size,
                kid=True,
                fid=False,
                kid_subset_size=subset_size,   # 🔥 FIX
                verbose=False,
                input1_cache_name="kid_cache_22"
            )

            kid_mean = metrics['kernel_inception_distance_mean']
            kid_std = metrics['kernel_inception_distance_std']

            gen_folder_end = Path(gen_path).name

            print(f"Epoch {epoch}: KID mean={kid_mean:.6f}, std={kid_std:.6f}")

            csv_rows.append([
                real_folder_name,
                gen_folder_end,
                epoch,
                kid_mean,
                kid_std
            ])

            shutil.rmtree(temp_generated)

        with open("kid.csv", mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["real_path", "generated_path", "epoch", "kid_mean", "kid_std"])
            writer.writerows(csv_rows)

        print("\nSaved results to kid.csv", file=sys.stderr)

        shutil.rmtree(temp_dir)

    except Exception as e:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
