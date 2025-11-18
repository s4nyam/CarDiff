#!/usr/bin/env python3
"""
Creates a mosaic of dental caries images showing original, mask, and overlay.
Selects images based on highest caries coverage (pixels with value 255).
"""

import os
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import random
from pathlib import Path
import argparse

def calculate_caries_area(mask_path):
    """Calculate the area covered by different caries classes in the mask."""
    try:
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            return {}, None
        
        total_pixels = mask.shape[0] * mask.shape[1]
        
        # Count pixels for each class
        superficial_pixels = np.sum(mask == 102)  # Superficial caries
        medium_pixels = np.sum(mask == 153)       # Medium caries
        deep_pixels = np.sum(mask == 255)         # Deep caries
        total_caries_pixels = superficial_pixels + medium_pixels + deep_pixels
        
        caries_data = {
            'superficial': superficial_pixels,
            'medium': medium_pixels,
            'deep': deep_pixels,
            'total': total_caries_pixels,
            'superficial_pct': (superficial_pixels / total_pixels) * 100,
            'medium_pct': (medium_pixels / total_pixels) * 100,
            'deep_pct': (deep_pixels / total_pixels) * 100,
            'total_pct': (total_caries_pixels / total_pixels) * 100
        }
        
        return caries_data, total_caries_pixels
    except Exception as e:
        print(f"Error processing {mask_path}: {e}")
        return {}, None

def calculate_class_balance_score(caries_info):
    """Calculate a balance score based on equal representation of caries classes."""
    try:
        total = caries_info['total']
        if total == 0:
            return 0  # No caries present
        
        superficial = caries_info['superficial']
        medium = caries_info['medium'] 
        deep = caries_info['deep']
        
        # Count how many classes are present
        classes_present = sum([superficial > 0, medium > 0, deep > 0])
        
        if classes_present < 2:
            # Penalize masks with only one class
            return total * 0.1
        
        # Calculate relative proportions
        proportions = []
        if superficial > 0:
            proportions.append(superficial / total)
        if medium > 0:
            proportions.append(medium / total)
        if deep > 0:
            proportions.append(deep / total)
        
        # Calculate balance score - prefer more equal distributions
        # Use inverse of coefficient of variation (lower CV = more balanced)
        if len(proportions) > 1:
            mean_prop = np.mean(proportions)
            std_prop = np.std(proportions)
            cv = std_prop / mean_prop if mean_prop > 0 else float('inf')
            # Balance score increases with total pixels and decreases with CV
            balance_score = total * (1 / (1 + cv)) * classes_present
        else:
            balance_score = total * 0.5  # Single class gets lower score
            
        return balance_score
    except Exception as e:
        print(f"Error calculating balance score: {e}")
        return 0

def get_corresponding_image_path(mask_filename, images_dir):
    """Get the corresponding image path for a given mask filename."""
    image_path = os.path.join(images_dir, mask_filename)
    return image_path if os.path.exists(image_path) else None

def create_overlay(image, mask, alpha=0.7):
    """Create an overlay of image and mask with colored contours for different caries types."""
    try:
        # Ensure image is in color
        if len(image.shape) == 2:
            image_color = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif len(image.shape) == 3 and image.shape[2] == 3:
            image_color = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_color = image.copy()
        
        overlay = image_color.copy()
        
        # Define colors for each caries type (RGB format)
        colors = {
            102: (0, 0, 255),    # Blue for superficial caries
            153: (0, 255, 0),    # Green for medium caries  
            255: (255, 0, 0)     # Red for deep caries
        }
        
        labels = {
            102: 'SC',  # Superficial Caries
            153: 'MC',  # Medium Caries
            255: 'DC'   # Deep Caries
        }
        
        # Process each caries type
        for pixel_value, color in colors.items():
            # Create binary mask for this caries type
            binary_mask = (mask == pixel_value).astype(np.uint8) * 255
            
            if np.any(binary_mask):
                # Find contours
                contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                # Draw contours
                cv2.drawContours(overlay, contours, -1, color, 2)
                
                # Add text labels for larger contours
                for contour in contours:
                    if cv2.contourArea(contour) > 100:  # Only label larger regions
                        # Get centroid of contour
                        M = cv2.moments(contour)
                        if M["m00"] != 0:
                            cx = int(M["m10"] / M["m00"])
                            cy = int(M["m01"] / M["m00"])
                            
                            # Add text
                            cv2.putText(overlay, labels[pixel_value], (cx-10, cy+5), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        
        return overlay
    except Exception as e:
        print(f"Error creating overlay: {e}")
        return image

def main():
    # Set up paths - using annotations for labels
    current_dir = os.path.dirname(os.path.abspath(__file__))
    images_dir = os.path.join(current_dir, 'images')
    labels_dir = os.path.join(current_dir, 'annotations')  # Changed to annotations
    
    print(f"Images directory: {images_dir}")
    print(f"Labels directory: {labels_dir}")
    
    # Verify directories exist
    if not os.path.exists(images_dir) or not os.path.exists(labels_dir):
        print("Error: images or annotations directory not found!")
        return
    
    # Get all mask files
    mask_files = [f for f in os.listdir(labels_dir) if f.endswith('.png')]
    print(f"Found {len(mask_files)} mask files")
    
    if not mask_files:
        print("No mask files found!")
        return
    
    # Calculate caries area for each mask
    print("Analyzing multi-class caries coverage and class balance in masks...")
    caries_data = []
    
    for mask_file in mask_files:
        mask_path = os.path.join(labels_dir, mask_file)
        image_path = get_corresponding_image_path(mask_file, images_dir)
        
        if image_path and os.path.exists(image_path):
            caries_info, total_caries_pixels = calculate_caries_area(mask_path)
            if total_caries_pixels is not None:
                # Calculate balance score for class representation
                balance_score = calculate_class_balance_score(caries_info)
                
                caries_data.append({
                    'filename': mask_file,
                    'caries_info': caries_info,
                    'total_caries_pixels': total_caries_pixels,
                    'balance_score': balance_score,
                    'mask_path': mask_path,
                    'image_path': image_path
                })
    
    print(f"Successfully processed {len(caries_data)} image-mask pairs")
    
    if len(caries_data) < 3:
        print("Not enough valid image-mask pairs found!")
        return
    
    # Sort by balance score (descending) - prefer masks with better class representation
    caries_data.sort(key=lambda x: x['balance_score'], reverse=True)
    
    # Show top 10 with best class balance and caries coverage
    print("\nTop 10 masks with best class balance and caries coverage:")
    for i, data in enumerate(caries_data[:10]):
        info = data['caries_info']
        classes_present = sum([info['superficial'] > 0, info['medium'] > 0, info['deep'] > 0])
        print(f"{i+1}. {data['filename']}: {data['total_caries_pixels']} pixels ({info['total_pct']:.2f}%) "
              f"[SC: {info['superficial']}, MC: {info['medium']}, DC: {info['deep']}] "
              f"Classes: {classes_present}/3, Balance Score: {data['balance_score']:.1f}")
    
    # Select 3 random files from top 30 (best balanced)
    top_30 = caries_data[:min(30, len(caries_data))]
    selected_files = random.sample(top_30, min(3, len(top_30)))
    
    print(f"\nSelected files for mosaic (from top 30 most balanced):")
    for i, data in enumerate(selected_files):
        info = data['caries_info']
        classes_present = sum([info['superficial'] > 0, info['medium'] > 0, info['deep'] > 0])
        print(f"{i+1}. {data['filename']}: {data['total_caries_pixels']} pixels ({info['total_pct']:.2f}%) "
              f"[SC: {info['superficial']}, MC: {info['medium']}, DC: {info['deep']}] "
              f"Classes: {classes_present}/3, Balance Score: {data['balance_score']:.1f}")
    
    # Create the mosaic
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    # fig.suptitle('Multi-Class Dental Caries Analysis Mosaic', fontsize=16, fontweight='bold')
    
    # Set column titles only for the first row
    column_titles = ['PR ROI', 'Multi-Class Mask', 'PR ROI + Multi-Class Overlay']
    
    for row, data in enumerate(selected_files):
        # Load images
        original_image = cv2.imread(data['image_path'])
        mask = cv2.imread(data['mask_path'], cv2.IMREAD_GRAYSCALE)
        
        if original_image is None or mask is None:
            print(f"Failed to load images for {data['filename']}")
            continue
        
        # Convert BGR to RGB for matplotlib
        original_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        
        # Create colored mask for display
        colored_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
        colored_mask[mask == 102] = [0, 0, 255]    # Blue for superficial
        colored_mask[mask == 153] = [0, 255, 0]    # Green for medium
        colored_mask[mask == 255] = [255, 0, 0]    # Red for deep
        
        # Create overlay
        overlay = create_overlay(original_image, mask)
        
        # Column 1: Original PR ROI
        axes[row, 0].imshow(original_rgb)
        if row == 0:  # Only set title for first row
            axes[row, 0].set_title(column_titles[0], fontsize=20, fontweight='bold')
        axes[row, 0].axis('off')
        
        # Column 2: Multi-class Mask
        axes[row, 1].imshow(colored_mask)
        if row == 0:  # Only set title for first row
            axes[row, 1].set_title(column_titles[1], fontsize=20, fontweight='bold')
        axes[row, 1].axis('off')
        
        # Column 3: Overlay
        axes[row, 2].imshow(overlay)
        if row == 0:  # Only set title for first row
            axes[row, 2].set_title(column_titles[2], fontsize=20, fontweight='bold')
        axes[row, 2].axis('off')
    
    # Add legend for color coding
    legend_elements = [
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='blue', markersize=10, label='SC - Superficial Caries'),
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='green', markersize=10, label='MC - Medium Caries'),
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='red', markersize=10, label='DC - Deep Caries')
    ]
    # fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.02), ncol=3, fontsize=12)
    
    # Adjust layout - reduce gaps between columns and rows
    plt.tight_layout()
    plt.subplots_adjust(top=0.95, bottom=0.08, wspace=0.005, hspace=0.01)
    
    # Save as PDF
    output_path = os.path.join(current_dir, 'multiclass_caries_mosaic.pdf')
    with PdfPages(output_path) as pdf:
        pdf.savefig(fig, bbox_inches='tight', dpi=300)
    
    # Also save as PNG
    output_png_path = os.path.join(current_dir, 'multiclass_caries_mosaic.png')
    plt.savefig(output_png_path, bbox_inches='tight', dpi=300)
    
    # Don't show in headless environment
    # plt.show()
    
    print(f"\nMosaic saved as:")
    print(f"- PDF: {output_path}")
    print(f"- PNG: {output_png_path}")
    
    # Print summary statistics
    print(f"\nMulti-Class Caries Summary Statistics:")
    print(f"Total masks processed: {len(caries_data)}")
    
    # Count masks with different numbers of classes
    single_class = sum([1 for d in caries_data if sum([d['caries_info']['superficial'] > 0, d['caries_info']['medium'] > 0, d['caries_info']['deep'] > 0]) == 1])
    two_class = sum([1 for d in caries_data if sum([d['caries_info']['superficial'] > 0, d['caries_info']['medium'] > 0, d['caries_info']['deep'] > 0]) == 2])
    three_class = sum([1 for d in caries_data if sum([d['caries_info']['superficial'] > 0, d['caries_info']['medium'] > 0, d['caries_info']['deep'] > 0]) == 3])
    no_caries = sum([1 for d in caries_data if d['total_caries_pixels'] == 0])
    
    print(f"Class distribution in dataset:")
    print(f"  No caries: {no_caries} masks ({(no_caries/len(caries_data))*100:.1f}%)")
    print(f"  Single class: {single_class} masks ({(single_class/len(caries_data))*100:.1f}%)")
    print(f"  Two classes: {two_class} masks ({(two_class/len(caries_data))*100:.1f}%)")
    print(f"  Three classes: {three_class} masks ({(three_class/len(caries_data))*100:.1f}%)")
    
    total_superficial = sum([d['caries_info']['superficial'] for d in caries_data])
    total_medium = sum([d['caries_info']['medium'] for d in caries_data])
    total_deep = sum([d['caries_info']['deep'] for d in caries_data])
    total_all_caries = total_superficial + total_medium + total_deep
    
    if total_all_caries > 0:
        print(f"\nTotal caries pixels across all images:")
        print(f"  Superficial (SC): {total_superficial} pixels ({(total_superficial/total_all_caries)*100:.1f}%)")
        print(f"  Medium (MC): {total_medium} pixels ({(total_medium/total_all_caries)*100:.1f}%)")
        print(f"  Deep (DC): {total_deep} pixels ({(total_deep/total_all_caries)*100:.1f}%)")
    
    avg_total_coverage = np.mean([d['caries_info']['total_pct'] for d in caries_data])
    print(f"\nAverage total caries coverage: {avg_total_coverage:.2f}%")
    print(f"Best balanced mask: {caries_data[0]['filename']} (Balance Score: {caries_data[0]['balance_score']:.1f})")
    if len(caries_data) > 0:
        print(f"Lowest balanced mask: {caries_data[-1]['filename']} (Balance Score: {caries_data[-1]['balance_score']:.1f})")

if __name__ == "__main__":
    main()
