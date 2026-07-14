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
            output_path = os.path.join(output_folder, img_path.name)
            img_resized.save(output_path)
        except Exception as e:
            print(f"\nError processing {img_path.name}: {e}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description="Calculate FID score: real vs multiple generated folders.")
    parser.add_argument('--real', type=str, required=True, help='Path to real images folder.')
    parser.add_argument('--generated_base', type=str, required=True, help='Base path containing generated epoch folders.')
    parser.add_argument('--epochs', type=int, nargs='+', required=True, help='List of epoch numbers.')
    parser.add_argument('--prefix', type=str, default='', help='Prefix before epoch number.')
    parser.add_argument('--suffix', type=str, default='-ddpm', help='Suffix after epoch number.')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])
    args = parser.parse_args()

    temp_dir = 'resized_temp_fid'
    temp_real = os.path.join(temp_dir, 'real_resized')

    os.makedirs(temp_dir, exist_ok=True)

    try:
        print("=" * 60, file=sys.stderr)
        print("Step 1: Resizing real images (only once)...", file=sys.stderr)
        print("=" * 60, file=sys.stderr)

        resize_images(args.real, temp_real, size=(224, 224))

        csv_rows = []

        print("\n" + "=" * 60, file=sys.stderr)
        print("Step 2: Computing FID epoch-wise...", file=sys.stderr)
        print("=" * 60, file=sys.stderr)

        # Extract only folder names (end nodes)
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

            resize_images(gen_path, temp_generated, size=(224, 224))

            metrics = calculate_metrics(
                input1=temp_real,
                input2=temp_generated,
                cuda=(args.device == 'cuda'),
                batch_size=args.batch_size,
                fid=True,
                kid=False,
                verbose=False,
                input1_cache_name="fid_cache_22"
            )

            fid_value = metrics['frechet_inception_distance']

            # Extract only generated folder name
            gen_folder_end = Path(gen_path).name

            # Print epoch-wise FID to stdout
            print(f"Epoch {epoch}: {fid_value:.6f}")

            csv_rows.append([
                real_folder_name,
                gen_folder_end,
                epoch,
                fid_value
            ])

            shutil.rmtree(temp_generated)

        # Save CSV
        csv_path = "fid.csv"
        with open(csv_path, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["real_path", "generated_path", "epoch", "fid"])
            writer.writerows(csv_rows)

        print("\nSaved results to fid.csv", file=sys.stderr)

        shutil.rmtree(temp_dir)

    except Exception as e:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
