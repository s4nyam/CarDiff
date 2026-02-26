#!/usr/bin/env python3
import os
import sys
import numpy as np
from PIL import Image
from collections import defaultdict

def process_folder(folder_path):
    """Process all grayscale images and count which pixel values appear in each image"""
    # Check if folder exists
    if not os.path.isdir(folder_path):
        print(f"Error: '{folder_path}' is not a valid directory")
        return
    
    # Get folder name
    folder_name = os.path.basename(os.path.abspath(folder_path))
    
    # Supported image extensions
    image_extensions = ('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.webp')
    
    # Get all image files in the folder
    images = []
    for file in os.listdir(folder_path):
        if file.lower().endswith(image_extensions):
            full_path = os.path.join(folder_path, file)
            if os.path.isfile(full_path):
                images.append(full_path)
    
    if not images:
        print(f"No images found in '{folder_path}'")
        return
    
    print(f"Processing folder: {folder_name}")
    print(f"Found {len(images)} images")
    print("-" * 50)
    
    # Dictionary to count how many images contain each pixel value
    pixel_value_counts = defaultdict(int)
    
    # Process each image
    for i, img_path in enumerate(images, 1):
        try:
            # Open image and convert to grayscale
            with Image.open(img_path) as img:
                # Convert to grayscale if not already
                if img.mode != 'L':
                    img = img.convert('L')
                
                # Convert to numpy array for faster processing
                img_array = np.array(img)
                
                # Get unique pixel values in this image
                unique_values = np.unique(img_array)
                
                # Increment count for each unique pixel value found
                for value in unique_values:
                    pixel_value_counts[value] += 1
                
                # Progress indicator
                if i % 10 == 0:
                    print(f"Processed {i}/{len(images)} images...")
                    
        except Exception as e:
            print(f"Error processing {os.path.basename(img_path)}: {e}")
    
    # Print results
    print("\n" + "=" * 60)
    print(f"RESULTS for folder: {folder_name}")
    print("Pixel value : Number of images containing this value")
    print("-" * 60)
    
    for pixel_value in range(256):
        count = pixel_value_counts.get(pixel_value, 0)
        if count > 0:  # Only show values that appear in at least one image
            bar = "█" * int((count / len(images)) * 50)
            print(f"{pixel_value:3d} : {count:4d} {bar}")
    
    # Summary statistics
    print("\n" + "=" * 60)
    print("SUMMARY:")
    print(f"Total images processed: {len(images)}")
    
    # Find most common pixel values
    if pixel_value_counts:
        max_count = max(pixel_value_counts.values())
        most_common = [v for v, c in pixel_value_counts.items() if c == max_count]
        print(f"Most frequent pixel value(s): {most_common} (appears in {max_count} images)")
        
        min_count = min(pixel_value_counts.values())
        least_common = [v for v, c in pixel_value_counts.items() if c == min_count]
        print(f"Least frequent pixel value(s): {least_common} (appears in {min_count} images)")
        
        print(f"Total unique pixel values found across all images: {len(pixel_value_counts)}")

def main():
    # Check if folder path is provided as argument
    if len(sys.argv) > 1:
        folder_path = sys.argv[1]
    else:
        # If no argument, prompt user
        folder_path = input("Enter folder path (or press Enter for current directory): ").strip()
        if not folder_path:
            folder_path = "."
    
    process_folder(folder_path)

if __name__ == "__main__":
    main()