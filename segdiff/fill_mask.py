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


def overlay_mask_on_image_array(image_array, mask_array):
    """
    Overlay mask edges on image using arrays. Different colors for different classes.
    Returns the overlayed image as a numpy array.
    """
    # Define colors for each class value (BGR format for OpenCV)
    class_colors = {
        102: [255, 0, 0],   # Blue for superficial caries
        153: [0, 255, 0],   # Green for medium caries
        255: [0, 0, 255],   # Red for deep caries
    }

    # Convert PIL image to numpy array if needed
    if isinstance(image_array, Image.Image):
        image_array = np.array(image_array)
    
    # Convert mask to numpy array if needed
    if isinstance(mask_array, Image.Image):
        mask_array = np.array(mask_array)
    
    # Ensure image is 3-channel (RGB)
    if len(image_array.shape) == 3 and image_array.shape[2] == 3:
        image = image_array
    else:
        # Convert grayscale to RGB
        image = cv2.cvtColor(image_array, cv2.COLOR_GRAY2RGB)
    
    # Ensure mask is grayscale
    if len(mask_array.shape) == 3:
        mask = cv2.cvtColor(mask_array, cv2.COLOR_RGB2GRAY)
    else:
        mask = mask_array
    
    # Resize mask to match image dimensions
    mask_resized = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
    
    # Create a blended image to draw on
    blended_image = image.copy()

    # Find edges for each class and overlay them
    for class_value, color in class_colors.items():
        # Create a binary mask for the current class
        class_mask = np.zeros_like(mask_resized)
        class_mask[mask_resized == class_value] = 255
        
        # Detect edges of the class mask
        edges = cv2.Canny(class_mask, 100, 200)
        
        # Create a colored overlay for the edges
        overlay = np.zeros_like(image)
        overlay[edges > 0] = color
        
        # Blend the overlay with the image
        blended_image = cv2.addWeighted(blended_image, 1.0, overlay, 0.7, 0)
    
    return blended_image


def create_mosaic_grid(images, output_path, num_images=10, rows=2, prefix=""):
    """
    Create a mosaic grid from a list of images.
    """
    # Randomly select images
    total_images = len(images)
    if total_images < num_images:
        selected_indices = list(range(total_images))
        print(f"Only {total_images} images available for {prefix}mosaic, using all of them")
    else:
        selected_indices = random.sample(range(total_images), num_images)
    
    selected_images = [images[idx] for idx in selected_indices]
    
    # Determine grid dimensions
    cols = min(5, len(selected_images))
    if len(selected_images) <= 5:
        rows = 1
        cols = len(selected_images)
    
    # Get image dimensions (assume all images are same size)
    if len(selected_images[0].shape) == 3:
        img_height, img_width = selected_images[0].shape[:2]
        channels = selected_images[0].shape[2]
    else:
        img_height, img_width = selected_images[0].shape[:2]
        channels = 1
    
    # Create mosaic canvas
    mosaic_height = rows * img_height
    mosaic_width = cols * img_width
    
    if channels == 3:
        mosaic = np.zeros((mosaic_height, mosaic_width, 3), dtype=np.uint8)
    else:
        mosaic = np.zeros((mosaic_height, mosaic_width), dtype=np.uint8)
    
    # Place images in the mosaic
    for i, img in enumerate(selected_images):
        row = i // cols
        col = i % cols
        
        y_start = row * img_height
        y_end = y_start + img_height
        x_start = col * img_width
        x_end = x_start + img_width
        
        if channels == 3:
            mosaic[y_start:y_end, x_start:x_end] = img
        else:
            mosaic[y_start:y_end, x_start:x_end] = img
    
    # Save mosaic
    if channels == 3:
        cv2.imwrite(output_path, cv2.cvtColor(mosaic, cv2.COLOR_RGB2BGR))
    else:
        cv2.imwrite(output_path, mosaic)
    
    print(f"{prefix}Mosaic saved to: {output_path}")
    return output_path


def create_mosaic_with_overlays(generated_images, mask_images, filenames, output_dir, num_images=10):
    """
    Create a mosaic of generated images with mask overlays.
    Select random images and create a 2x5 grid.
    Returns the selected indices for consistency with other mosaics.
    """
    print(f"Creating overlay mosaic with {num_images} random images...")
    
    # Randomly select images
    total_images = len(generated_images)
    if total_images < num_images:
        selected_indices = list(range(total_images))
        print(f"Only {total_images} images available, using all of them")
    else:
        selected_indices = random.sample(range(total_images), num_images)
    
    # Create overlayed images
    overlayed_images = []
    for idx in selected_indices:
        overlayed_img = overlay_mask_on_image_array(generated_images[idx], mask_images[idx])
        overlayed_images.append(overlayed_img)
    
    # Create mosaic
    mosaic_path = os.path.join(output_dir, "mosaic_with_overlays.png")
    create_mosaic_grid_from_selected(overlayed_images, mosaic_path, prefix="Overlay ")
    
    return mosaic_path, selected_indices


def create_separate_mosaics(generated_images, mask_images, output_dir, num_images=10):
    """
    Create separate mosaics for generated images and masks.
    Uses the same random selection for both to ensure they correspond.
    """
    print(f"Creating separate mosaics for images and masks...")
    
    # Randomly select images - use same indices for both images and masks
    total_images = len(generated_images)
    if total_images < num_images:
        selected_indices = list(range(total_images))
        print(f"Only {total_images} images available, using all of them")
    else:
        selected_indices = random.sample(range(total_images), num_images)
    
    # Select corresponding images and masks using the same indices
    selected_generated_images = [generated_images[idx] for idx in selected_indices]
    selected_mask_images = [mask_images[idx] for idx in selected_indices]
    
    # Create images mosaic
    images_mosaic_path = os.path.join(output_dir, "mosaic_images.png")
    create_mosaic_grid_from_selected(selected_generated_images, images_mosaic_path, prefix="Images ")
    
    # Create masks mosaic
    masks_mosaic_path = os.path.join(output_dir, "mosaic_masks.png")
    create_mosaic_grid_from_selected(selected_mask_images, masks_mosaic_path, prefix="Masks ")
    
    return images_mosaic_path, masks_mosaic_path


def create_mosaic_grid_from_selected(selected_images, output_path, prefix=""):
    """
    Create a mosaic grid from already selected images (no random selection).
    """
    # Determine grid dimensions
    num_images = len(selected_images)
    cols = min(5, num_images)
    rows = 2
    if num_images <= 5:
        rows = 1
        cols = num_images
    
    # Get image dimensions (assume all images are same size)
    if len(selected_images[0].shape) == 3:
        img_height, img_width = selected_images[0].shape[:2]
        channels = selected_images[0].shape[2]
    else:
        img_height, img_width = selected_images[0].shape[:2]
        channels = 1
    
    # Create mosaic canvas
    mosaic_height = rows * img_height
    mosaic_width = cols * img_width
    
    if channels == 3:
        mosaic = np.zeros((mosaic_height, mosaic_width, 3), dtype=np.uint8)
    else:
        mosaic = np.zeros((mosaic_height, mosaic_width), dtype=np.uint8)
    
    # Place images in the mosaic
    for i, img in enumerate(selected_images):
        row = i // cols
        col = i % cols
        
        y_start = row * img_height
        y_end = y_start + img_height
        x_start = col * img_width
        x_end = x_start + img_width
        
        if channels == 3:
            mosaic[y_start:y_end, x_start:x_end] = img
        else:
            mosaic[y_start:y_end, x_start:x_end] = img
    
    # Save mosaic
    if channels == 3:
        cv2.imwrite(output_path, cv2.cvtColor(mosaic, cv2.COLOR_RGB2BGR))
    else:
        cv2.imwrite(output_path, mosaic)
    
    print(f"{prefix}Mosaic saved to: {output_path}")
    return output_path


def generate_multiple_from_same_mask(pipeline, mask_tensor, filename, output_dir, num_generations=10, num_inference_steps=50):
    """
    Generate multiple images from the same mask and create a mosaic with overlays.
    """
    print(f"Generating {num_generations} images from mask: {filename}")
    
    # Prepare batch with the same mask repeated
    seg_batch = {
        "seg_all": mask_tensor.unsqueeze(0).repeat(num_generations, 1, 1, 1),
        "image_filenames": [filename] * num_generations
    }
    
    # Generate multiple images
    images = pipeline(
        batch_size=num_generations,
        seg_batch=seg_batch,
        num_inference_steps=num_inference_steps
    ).images
    
    # Convert to numpy arrays
    generated_arrays = [np.array(img) for img in images]
    
    # Convert mask tensor to numpy array for overlay
    mask_pil = transforms.ToPILImage()(mask_tensor.squeeze())
    mask_array = np.array(mask_pil)
    
    # Create overlayed images with mask boundaries
    overlayed_images = []
    for img_array in generated_arrays:
        overlayed_img = overlay_mask_on_image_array(img_array, mask_array)
        overlayed_images.append(overlayed_img)
    
    # Create mosaic for this mask with overlays
    base_name = os.path.splitext(filename)[0]
    mosaic_path = os.path.join(output_dir, f"mosaic_variations_{base_name}.png")
    
    # Use 2 rows, 5 cols for 10 images
    rows = 2
    cols = 5
    if num_generations <= 5:
        rows = 1
        cols = num_generations
    
    create_mosaic_grid(overlayed_images, mosaic_path, num_generations, rows, f"Variations for {base_name} ")
    
    return images, mosaic_path


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
    
    # Create various mosaics if we have generated images
    if total_generated > 0:
        # 1. Create mosaic with overlays (original functionality) and get selected indices
        overlay_mosaic_path, selected_indices = create_mosaic_with_overlays(
            all_generated_images, all_mask_images, all_filenames, output_dir, mosaic_count
        )
        
        # 2. Create separate mosaics for images and masks using the same indices
        print(f"Using the same {len(selected_indices)} images for separate mosaics...")
        selected_generated_images = [all_generated_images[idx] for idx in selected_indices]
        selected_mask_images = [all_mask_images[idx] for idx in selected_indices]
        
        # Create images mosaic
        images_mosaic_path = os.path.join(output_dir, "mosaic_images.png")
        create_mosaic_grid_from_selected(selected_generated_images, images_mosaic_path, prefix="Images ")
        
        # Create masks mosaic
        masks_mosaic_path = os.path.join(output_dir, "mosaic_masks.png")
        create_mosaic_grid_from_selected(selected_mask_images, masks_mosaic_path, prefix="Masks ")
        
        # 3. Generate variations for the first mask (or a random mask) and create mosaic
        if generate_variations and total_generated > 0:
            # Use the first mask for variations
            first_batch = next(iter(dataloader))
            first_mask = first_batch["seg_all"][0]
            first_filename = first_batch["image_filenames"][0]

            print(f"\nGenerating {num_variations} variations for mask: {first_filename}")
            if seg_guided_detected:
                variation_images, variation_mosaic_path = generate_multiple_from_same_mask(
                    pipeline, first_mask, first_filename, output_dir, num_variations, num_inference_steps
                )
            else:
                # Non seg-guided model: generate unconditional variations and still overlay the same mask
                images = pipeline(
                    batch_size=num_variations,
                    num_inference_steps=num_inference_steps
                ).images
                generated_arrays = [np.array(img) for img in images]
                mask_pil = transforms.ToPILImage()(first_mask.squeeze())
                mask_array = np.array(mask_pil)
                overlayed_images = [overlay_mask_on_image_array(img_array, mask_array) for img_array in generated_arrays]

                base_name = os.path.splitext(first_filename)[0]
                variation_mosaic_path = os.path.join(output_dir, f"mosaic_variations_{base_name}.png")
                rows = 2 if num_variations > 5 else 1
                create_mosaic_grid(overlayed_images, variation_mosaic_path, num_variations, rows, f"Variations for {base_name} ")
            
            print(f"Created variation mosaic for {first_filename} at: {variation_mosaic_path}")
    
    return total_generated

# Superficial caries (102) will have blue edges.
# Medium caries (153) will have green edges.
# Deep caries (255) will have red edges.


def main():
    parser = argparse.ArgumentParser(description="Generate images from masks using trained diffusion model")
    
    parser.add_argument("--model_dir", type=str, default="ddim-dc-384-segguided/checkpoint_epoch_980",
                       help="Path to model checkpoint directory")
    
    parser.add_argument("--masks_dir", type=str, default="/scratch/project_465001696/playground/0.3.segguideddiff/DATA_FOLDER/testannot",
                       help="Path to directory containing mask images (e.g., /path/to/MASK_FOLDER/all/test)")
    
    parser.add_argument("--output_dir", type=str, default="/scratch/project_465001696/playground/0.3.segguideddiff/DATA_FOLDER/syntest-ep980-ddim",
                       help="Output directory for generated images (default: wild_images/)")
    
    parser.add_argument("--batch_size", type=int, default=16,
                       help="Batch size for generation (default: 16)")
    
    parser.add_argument("--num_inference_steps", type=int, default=1000,
                       help="Number of denoising steps (default: 1000)")
    
    parser.add_argument("--image_size", type=int, default=384,
                       help="Image size (default: 384)")
    
    parser.add_argument("--mosaic_count", type=int, default=10,
                       help="Number of images to include in mosaic (default: 10)")

    parser.add_argument("--generate_variations", action="store_true", default=False,
                       help="Generate multiple variations from the same mask (default: False)")

    parser.add_argument("--num_variations", type=int, default=1,
                       help="Number of variations to generate from the same mask (default: 10)")
    
    args = parser.parse_args()
    
    try:
        total_generated = generate_images_from_masks(
            model_dir=args.model_dir,
            masks_dir=args.masks_dir,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            num_inference_steps=args.num_inference_steps,
            image_size=args.image_size,
            mosaic_count=args.mosaic_count,
            generate_variations=args.generate_variations,
            num_variations=args.num_variations
        )
        print(f"Successfully completed! Generated {total_generated} images with mosaic.")
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
