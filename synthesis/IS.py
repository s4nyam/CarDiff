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
    """Resize all images in input_folder to the specified size and save to output_folder."""
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
    parser = argparse.ArgumentParser(description="Calculate Inception Score epoch-wise.")
    parser.add_argument('--generated_base', type=str, required=True, help='Base path containing generated epoch folders.')
    parser.add_argument('--epochs', type=int, nargs='+', required=True, help='List of epoch numbers.')
    parser.add_argument('--prefix', type=str, default='', help='Prefix before epoch number.')
    parser.add_argument('--suffix', type=str, default='-ddpm', help='Suffix after epoch number.')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])
    args = parser.parse_args()

    temp_dir = 'resized_temp_IS'
    os.makedirs(temp_dir, exist_ok=True)

    try:
        print("=" * 60, file=sys.stderr)
        print("Computing Inception Score epoch-wise...", file=sys.stderr)
        print("=" * 60, file=sys.stderr)

        csv_rows = []

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

            if num_generated == 0:
                print(f"Skipping epoch {epoch} (no images found)", file=sys.stderr)
                continue

            metrics = calculate_metrics(
                input1=temp_generated,
                cuda=(args.device == 'cuda'),
                batch_size=args.batch_size,
                isc=True,
                verbose=False
            )

            is_mean = metrics['inception_score_mean']
            is_std = metrics['inception_score_std']

            gen_folder_end = Path(gen_path).name

            print(f"Epoch {epoch}: IS mean={is_mean:.6f}, std={is_std:.6f}")

            csv_rows.append([
                gen_folder_end,
                epoch,
                is_mean,
                is_std
            ])

            shutil.rmtree(temp_generated)

        # Save CSV
        with open("is.csv", mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["generated_path", "epoch", "is_mean", "is_std"])
            writer.writerows(csv_rows)

        print("\nSaved results to is.csv", file=sys.stderr)

        shutil.rmtree(temp_dir)

    except Exception as e:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
