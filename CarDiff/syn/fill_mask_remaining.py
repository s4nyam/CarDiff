#!/usr/bin/env python3
"""
Fill Mask Script for Segmentation-Guided Diffusion Model
Generates images conditioned on input masks using a trained diffusion model.
"""

import os
import argparse
import torch
from torch import nn
from torchvision import transforms
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from pathlib import Path
import random
import cv2

import diffusers
from diffusers import UNet2DModel, DDPMScheduler, DDIMScheduler
import datasets
from PIL import Image

from eval import SegGuidedDDPMPipeline, SegGuidedDDIMPipeline, add_segmentations_to_noise
from training import TrainingConfig

import re

def extract_epoch_from_folder(folder_name):
    """
    Extract epoch number from folder name.
    Example: syntrain-ep180-ddpm -> 180
    """
    match = re.search(r"ep(\d+)", folder_name)
    if match:
        return int(match.group(1))
    return None

def build_mask_catalog(masks_dir):
    """
    Scan masks_dir and return dictionary:
    {
        "file1.png": "/full/path/file1.png",
        ...
    }
    """
    mask_catalog = {}

    for ext in ["*.png", "*.jpg", "*.jpeg"]:
        for path in Path(masks_dir).glob(ext):
            mask_catalog[path.name] = str(path)

    print(f"\n[INFO] Total masks found in source: {len(mask_catalog)}")
    return mask_catalog
def scan_existing_outputs(epoch_output_dir):
    """
    Returns set of filenames already synthesized inside given epoch folder.
    """
    existing = set()

    if not os.path.exists(epoch_output_dir):
        return existing

    for ext in ["*.png", "*.jpg", "*.jpeg"]:
        for path in Path(epoch_output_dir).glob(ext):
            existing.add(path.name)

    return existing

def load_model_and_scheduler(model_dir):
    """Load the trained UNet model and scheduler from checkpoint directory."""
    
    # Check if model directory exists
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
    
    # Load UNet model
    unet_path = os.path.join(model_dir, "unet")
    if not os.path.exists(unet_path):
        raise FileNotFoundError(f"UNet model not found in: {unet_path}")
    
    unet = UNet2DModel.from_pretrained(unet_path, use_safetensors=True)
    
    # Load scheduler
    scheduler_path = os.path.join(model_dir, "scheduler")
    if not os.path.exists(scheduler_path):
        raise FileNotFoundError(f"Scheduler not found in: {scheduler_path}")
    
    # Try loading as DDPM scheduler first, fallback to DDIM
    try:
        scheduler = DDPMScheduler.from_pretrained(scheduler_path)
        model_type = "DDPM"
    except:
        scheduler = DDIMScheduler.from_pretrained(scheduler_path)
        model_type = "DDIM"
    
    return unet, scheduler, model_type


def load_masks_dataset(masks_dir):
    """Load mask images from directory and create dataset."""
    
    if not os.path.exists(masks_dir):
        raise FileNotFoundError(f"Masks directory not found: {masks_dir}")
    
    # Get all PNG files in the directory
    mask_files = []
    for file_ext in ['*.png', '*.jpg', '*.jpeg']:
        mask_files.extend(list(Path(masks_dir).glob(file_ext)))
    
    if not mask_files:
        raise ValueError(f"No mask images found in: {masks_dir}")
    
    # Create paths list
    mask_paths = [str(path) for path in mask_files]
    filenames = [path.name for path in mask_files]
    
    print(f"Found {len(mask_paths)} mask files")
    
    # Create dataset dictionary
    # Using 'seg_all' as the segmentation type based on the existing code structure
    dset_dict = {
        "seg_all": mask_paths,
        "image_filename": filenames
    }
    
    # Create dataset and cast column to Image type
    dataset = datasets.Dataset.from_dict(dset_dict)
    dataset = dataset.cast_column("seg_all", datasets.Image())
    
    return dataset


def create_transform_function(config):
    """Create transform function for preprocessing masks."""
    
    preprocess_segmentation = transforms.Compose([
        transforms.Resize((config.image_size, config.image_size), 
                         interpolation=transforms.InterpolationMode.NEAREST),
        transforms.ToTensor(),
    ])
    
    def transform(examples):
        segs = {}
        segs["seg_all"] = [preprocess_segmentation(image.convert("L")) 
                          for image in examples["seg_all"]]
        
        image_filenames = examples["image_filename"]
        
        return {**segs, **{"image_filenames": image_filenames}}
    
    return transform




def generate_images_from_masks(model_dir, masks_dir, output_dir="generated_images", 
                             batch_size=8, num_inference_steps=50, image_size=384, mosaic_count=10, 
                             generate_variations=True, num_variations=10):
    """Main function to generate images from masks."""
    
    print(f"Loading model from: {model_dir}")
    print(f"Loading masks from: {masks_dir}")
    print(f"Output directory: {output_dir}")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model and scheduler
    unet, scheduler, model_type = load_model_and_scheduler(model_dir)
    print(f"Loaded {model_type} model")
    
    # Wrap model with DataParallel and move to device
    unet = nn.DataParallel(unet)
    unet.to(device)
    unet.eval()
    
    # Detect whether the loaded UNet was trained with segmentation guidance
    try:
        in_ch = int(getattr(unet.module.config, "in_channels"))
        out_ch = int(getattr(unet.module.config, "out_channels", 1))
    except Exception:
        # Fallback to sane defaults
        in_ch, out_ch = 2, 1

    seg_guided_detected = in_ch > out_ch  # e.g., 2 in, 1 out => seg-guided single-channel masks
    if not seg_guided_detected:
        print(
            f"Warning: Loaded UNet appears NON-seg-guided (in_channels={in_ch}, out_channels={out_ch}).\n"
            "Proceeding without mask conditioning to avoid shape errors."
        )

    # Create configuration object based on model settings
    config = TrainingConfig(
        image_size=image_size,
        model_type=model_type,
        segmentation_guided=seg_guided_detected,
        segmentation_channel_mode="single" if seg_guided_detected else "none",
        num_segmentation_classes=4,  # 3 caries types + 1 background
        eval_batch_size=batch_size,
        dataset="dc"  # Default dataset name
    )
    
    # Load masks dataset
    dataset = load_masks_dataset(masks_dir)
    
    # Set transform
    transform_fn = create_transform_function(config)
    dataset.set_transform(transform_fn)
    
    # Create dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False
    )
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create pipeline based on model type and whether seg-guided is supported by the checkpoint
    if model_type == "DDPM":
        if seg_guided_detected:
            pipeline = SegGuidedDDPMPipeline(
                unet=unet.module, 
                scheduler=scheduler, 
                eval_dataloader=iter(dataloader),  # Dummy iterator
                external_config=config
            )
        else:
            # Fall back to vanilla DDPM pipeline (no mask conditioning)
            pipeline = diffusers.DDPMPipeline(unet=unet.module, scheduler=scheduler)
            pipeline = pipeline.to(device)
    else:  # DDIM
        if seg_guided_detected:
            pipeline = SegGuidedDDIMPipeline(
                unet=unet.module, 
                scheduler=scheduler, 
                eval_dataloader=iter(dataloader),  # Dummy iterator
                external_config=config
            )
        else:
            pipeline = diffusers.DDIMPipeline(unet=unet.module, scheduler=scheduler)
            pipeline = pipeline.to(device)
    
    print(f"Starting image generation...")
    
    # Generate images
    total_generated = 0
    all_generated_images = []
    all_mask_images = []
    all_filenames = []
    
    with torch.no_grad():
        for batch_idx, seg_batch in enumerate(tqdm(dataloader, desc="Generating images")):
            current_batch_size = seg_batch["seg_all"].shape[0]
            
            # Generate images (conditionally if supported)
            if seg_guided_detected:
                images = pipeline(
                    batch_size=current_batch_size,
                    seg_batch=seg_batch,
                    num_inference_steps=num_inference_steps
                ).images
            else:
                images = pipeline(
                    batch_size=current_batch_size,
                    num_inference_steps=num_inference_steps
                ).images
            
            # Save generated images and collect for mosaic
            for i, (img, filename) in enumerate(zip(images, seg_batch["image_filenames"])):
                # Create output filename
                base_name = os.path.splitext(filename)[0]
                output_filename = f"{base_name}.png"
                output_path = os.path.join(output_dir, output_filename)
                
                # Save image
                img.save(output_path)
                
                # Collect for mosaic (convert to numpy array)
                all_generated_images.append(np.array(img))
                
                # Get corresponding mask and convert to numpy array
                mask_tensor = seg_batch["seg_all"][i]
                # Convert tensor to PIL Image then to numpy
                mask_pil = transforms.ToPILImage()(mask_tensor.squeeze())
                all_mask_images.append(np.array(mask_pil))
                all_filenames.append(filename)
                
                total_generated += 1
    
    print(f"Generated {total_generated} images and saved to: {output_dir}")
    
    return total_generated

# Superficial caries (102) will have blue edges.
# Medium caries (153) will have green edges.
# Deep caries (255) will have red edges.

def main():
    parser = argparse.ArgumentParser(description="Resume synthesis of missing masks")

    parser.add_argument("--model_dir", type=str,
                        default="ddpm-dc-384-segguided",
                        help="Base directory containing checkpoint_epoch_* folders")

    parser.add_argument("--masks_dir", type=str,
                        default="train_data/trainannot",
                        help="Directory containing annotation masks")

    parser.add_argument("--output_dir", type=str,
                        default="epoch_wise_train_data",
                        help="Directory containing epoch-wise synthesized folders")

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_inference_steps", type=int, default=1000)
    parser.add_argument("--image_size", type=int, default=384)

    args = parser.parse_args()

    print("\n==============================")
    print(" RESUME MASK SYNTHESIS SCRIPT ")
    print("==============================\n")

    # --------------------------------------------------
    # 1️⃣ Build source annotation catalog
    # --------------------------------------------------
    mask_catalog = build_mask_catalog(args.masks_dir)
    mask_filenames = set(mask_catalog.keys())

    # --------------------------------------------------
    # 2️⃣ Iterate epoch-wise output folders
    # --------------------------------------------------
    epoch_folders = [
        f for f in os.listdir(args.output_dir)
        if os.path.isdir(os.path.join(args.output_dir, f))
    ]

    if not epoch_folders:
        print("[WARNING] No epoch subfolders found inside output_dir.")
        return

    from natsort import natsorted

    epoch_folders = natsorted(epoch_folders)

    for folder in epoch_folders:
        epoch_number = extract_epoch_from_folder(folder)
        if epoch_number is None:
            print(f"[SKIP] Could not extract epoch from folder: {folder}")
            continue

        print(f"\n======================================")
        print(f"[EPOCH {epoch_number}] Processing folder: {folder}")
        print(f"======================================")

        epoch_output_path = os.path.join(args.output_dir, folder)

        # --------------------------------------------------
        # 3️⃣ Detect already synthesized images
        # --------------------------------------------------
        existing_files = scan_existing_outputs(epoch_output_path)
        missing_files = mask_filenames - existing_files

        print(f"[INFO] Total masks available      : {len(mask_filenames)}")
        print(f"[INFO] Already synthesized        : {len(existing_files)}")
        print(f"[INFO] Remaining to synthesize    : {len(missing_files)}")

        if len(missing_files) == 0:
            print("[SKIP] Nothing to synthesize for this epoch.")
            continue

        print("\n[DETAIL] Files to synthesize:")
        for f in sorted(missing_files):
            print(f"  - {f}")

        # --------------------------------------------------
        # 4️⃣ Load correct checkpoint automatically
        # --------------------------------------------------
        model_checkpoint_path = os.path.join(
            args.model_dir,
            f"checkpoint_epoch_{epoch_number}"
        )

        if not os.path.exists(model_checkpoint_path):
            print(f"[ERROR] Checkpoint not found: {model_checkpoint_path}")
            continue

        print(f"\n[MODEL] Loading checkpoint: {model_checkpoint_path}")

        # --------------------------------------------------
        # 5️⃣ Create temporary subset masks directory
        # --------------------------------------------------
        temp_subset_dir = os.path.join(args.output_dir, "__temp_missing_masks__")
        os.makedirs(temp_subset_dir, exist_ok=True)

        # Copy missing masks into temp folder
        for fname in missing_files:
            src = mask_catalog[fname]
            dst = os.path.join(temp_subset_dir, fname)
            if not os.path.exists(dst):
                import shutil
                shutil.copy(src, dst)

        # --------------------------------------------------
        # 6️⃣ Call your existing generator function
        # --------------------------------------------------
        generate_images_from_masks(
            model_dir=model_checkpoint_path,
            masks_dir=temp_subset_dir,
            output_dir=epoch_output_path,
            batch_size=args.batch_size,
            num_inference_steps=args.num_inference_steps,
            image_size=args.image_size,
            mosaic_count=0,
            generate_variations=False
        )

        # Clean temp
        import shutil
        shutil.rmtree(temp_subset_dir)

    print("\n[COMPLETE] Resume synthesis finished.")



if __name__ == "__main__":
    exit(main())
