import os
import shutil
import random

def split_dataset(sourcedir, datadir, maskdir, train_ratio=0.7, val_ratio=0.2):
    # Ensure the source directory exists
    if not os.path.exists(sourcedir):
        raise ValueError(f"Source directory {sourcedir} does not exist.")
    
    # Define paths for images and masks
    image_dir = os.path.join(sourcedir, "images")
    mask_dir = os.path.join(sourcedir, "annotations")

    # Ensure the image and mask directories exist
    if not os.path.exists(image_dir) or not os.path.exists(mask_dir):
        raise ValueError("Image or mask directories do not exist in the source directory.")

    # Create target directories
    os.makedirs(os.path.join(datadir, "train"), exist_ok=True)
    os.makedirs(os.path.join(datadir, "val"), exist_ok=True)
    os.makedirs(os.path.join(datadir, "test"), exist_ok=True)
    os.makedirs(os.path.join(maskdir, "all", "train"), exist_ok=True)
    os.makedirs(os.path.join(maskdir, "all", "val"), exist_ok=True)
    os.makedirs(os.path.join(maskdir, "all", "test"), exist_ok=True)

    # Get all image file names
    images = [f for f in os.listdir(image_dir) if f.endswith(".png")]
    images.sort()  # Sort to ensure consistent splitting

    # Shuffle images for random split
    random.shuffle(images)

    # Calculate split indices
    total_images = len(images)
    train_end = int(train_ratio * total_images)
    val_end = train_end + int(val_ratio * total_images)

    # Split images into train, val, test
    train_files = images[:train_end]
    val_files = images[train_end:val_end]
    test_files = images[val_end:]

    # Function to copy files
    def copy_files(file_list, dest_data, dest_mask, prefix):
        for i, file_name in enumerate(file_list, 1):
            img_src = os.path.join(image_dir, file_name)
            mask_src = os.path.join(mask_dir, file_name)
            img_dest = os.path.join(dest_data, f"{prefix}_{i}.png")
            mask_dest = os.path.join(dest_mask, f"{prefix}_{i}.png")

            shutil.copy(img_src, img_dest)
            shutil.copy(mask_src, mask_dest)

    # Copy files to respective directories
    copy_files(train_files, os.path.join(datadir, "train"), os.path.join(maskdir, "all", "train"), "tr")
    copy_files(val_files, os.path.join(datadir, "val"), os.path.join(maskdir, "all", "val"), "val")
    copy_files(test_files, os.path.join(datadir, "test"), os.path.join(maskdir, "all", "test"), "ts")

    print("Dataset successfully split and organized.")

# Example usage
sourcedir = "dc"
datadir = "DATA_FOLDER"
maskdir = "MASK_FOLDER"
split_dataset(sourcedir, datadir, maskdir)
