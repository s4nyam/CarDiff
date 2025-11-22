#!/usr/bin/env python3
"""
Multi-Class Connected Components Analysis for Dental Dataset
Analyzes pixel distribution and connected components for classes: 0, 102, 153, 255
"""

import os
import numpy as np
from PIL import Image
import glob
import matplotlib.pyplot as plt
from scipy.ndimage import label
from collections import Counter, defaultdict
import seaborn as sns

def analyze_multiclass_components(annotations_dir, output_dir):
    """
    Analyze multi-class segmentation masks for pixel distribution and connected components.
    
    Args:
        annotations_dir (str): Directory containing annotation images
        output_dir (str): Directory to save output files
    """
    annotation_files = glob.glob(os.path.join(annotations_dir, '*.png'))
    
    print(f'Found {len(annotation_files)} annotation files')
    print('Analyzing multi-class pixel distribution and connected components...')
    
    # Initialize counters
    total_pixels_per_class = {0: 0, 102: 0, 153: 0, 255: 0}
    total_pixels = 0
    component_counts_per_class = {0: [], 102: [], 153: [], 255: []}
    
    # Track images with each class
    images_with_class = {0: 0, 102: 0, 153: 0, 255: 0}
    
    for i, annotation_file in enumerate(annotation_files):
        # Load the annotation image
        img = Image.open(annotation_file)
        img_array = np.array(img)
        
        # Count pixels for each class
        unique_values, counts = np.unique(img_array, return_counts=True)
        pixel_counts = dict(zip(unique_values, counts))
        
        # Update total pixel counts
        for class_val in [0, 102, 153, 255]:
            class_pixels = pixel_counts.get(class_val, 0)
            total_pixels_per_class[class_val] += class_pixels
            
            if class_pixels > 0:
                images_with_class[class_val] += 1
        
        total_pixels += img_array.size
        
        # Analyze connected components for each class
        for class_val in [102, 153, 255]:  # Skip background (0) for connected components
            # Create binary mask for current class
            class_mask = (img_array == class_val).astype(int)
            
            # Find connected components
            labeled_array, num_components = label(class_mask)
            component_counts_per_class[class_val].append(num_components)
        
        # Print progress every 50 files
        if (i + 1) % 50 == 0:
            print(f'Processed {i + 1}/{len(annotation_files)} files...')
    
    # Calculate percentages
    print(f'\n=== PIXEL DISTRIBUTION ANALYSIS ===')
    print(f'Total pixels analyzed: {total_pixels:,}')
    
    class_names = {0: 'Background', 102: 'Superficial-caries', 153: 'Medium-caries', 255: 'Deep-caries'}
    
    for class_val, class_name in class_names.items():
        pixels = total_pixels_per_class[class_val]
        percentage = (pixels / total_pixels) * 100
        print(f'{class_name} (value {class_val}): {pixels:,} pixels ({percentage:.4f}%)')
        print(f'  - Present in {images_with_class[class_val]} images ({(images_with_class[class_val]/len(annotation_files))*100:.2f}%)')
    
    # Analyze connected components
    print(f'\n=== CONNECTED COMPONENTS ANALYSIS ===')
    
    component_frequencies = {}
    for class_val in [102, 153, 255]:
        component_freq = Counter(component_counts_per_class[class_val])
        component_frequencies[class_val] = component_freq
        
        class_name = class_names[class_val]
        print(f'\n{class_name} (value {class_val}) component distribution:')
        
        sorted_components = sorted(component_freq.items())
        for comp_count, freq in sorted_components:
            percentage = (freq / len(annotation_files)) * 100
            print(f'  {comp_count} components: {freq} images ({percentage:.2f}%)')
    
    # Create comprehensive visualizations
    create_pixel_distribution_plot(total_pixels_per_class, class_names, output_dir)
    create_grouped_components_plot(component_frequencies, class_names, output_dir, len(annotation_files))
    create_class_presence_plot(images_with_class, class_names, len(annotation_files), output_dir)
    create_class_imbalance_analysis_plot(total_pixels_per_class, class_names, output_dir)
    
    # Save standalone pie chart as PDF
    create_standalone_pie_chart_pdf(total_pixels_per_class, component_frequencies, class_names, output_dir)
    
    # Create class combination analysis table
    create_class_combination_table(annotation_files, class_names, output_dir)
    
    # Save detailed results
    save_detailed_results(total_pixels_per_class, component_frequencies, images_with_class, 
                         class_names, len(annotation_files), total_pixels, output_dir)
    
    return total_pixels_per_class, component_frequencies, images_with_class

def create_pixel_distribution_plot(total_pixels_per_class, class_names, output_dir):
    """Create pie chart and bar plot for pixel distribution with enhanced class imbalance visualization."""
    
    # Create figure with 3 subplots: original + enhanced imbalance view
    plt.figure(figsize=(18, 6))
    
    # Subplot 1: Original Pie Chart (All classes including background)
    plt.subplot(1, 3, 1)
    values = list(total_pixels_per_class.values())
    labels = [f'{class_names[k]}\n({v:,} pixels)' for k, v in total_pixels_per_class.items()]
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    
    plt.pie(values, labels=labels, autopct='%1.2f%%', colors=colors, startangle=90)
    plt.title('All Classes Pixel Distribution\n(Including Background)', fontweight='bold', fontsize=12)
    
    # Subplot 2: Original Bar Plot (All classes)
    plt.subplot(1, 3, 2)
    classes = list(total_pixels_per_class.keys())
    pixels = list(total_pixels_per_class.values())
    
    bars = plt.bar([class_names[c] for c in classes], pixels, color=colors, alpha=0.8, edgecolor='black')
    plt.title('Pixel Count by Class\n(Linear Scale)', fontweight='bold', fontsize=12)
    plt.ylabel('Number of Pixels', fontsize=10)
    plt.xticks(rotation=45)
    
    # Add value labels on bars
    for bar, pixel_count in zip(bars, pixels):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(pixels)*0.01,
                f'{pixel_count:,}', ha='center', va='bottom', fontsize=8)
    
    # Subplot 3: Caries-only distribution (enhanced for imbalance analysis)
    plt.subplot(1, 3, 3)
    caries_classes = [k for k in classes if k != 0]
    caries_pixels = [total_pixels_per_class[k] for k in caries_classes]
    caries_colors = ['#4ECDC4', '#45B7D1', '#96CEB4']  # Skip background color
    
    # Calculate percentages among caries classes only
    total_caries = sum(caries_pixels)
    caries_percentages = [(p/total_caries)*100 for p in caries_pixels]
    
    bars = plt.bar([class_names[c] for c in caries_classes], caries_pixels, 
                   color=caries_colors, alpha=0.8, edgecolor='black')
    plt.title('Caries Classes Only\n(Shows Class Imbalance)', fontweight='bold', fontsize=12)
    plt.ylabel('Number of Pixels', fontsize=10)
    plt.xticks(rotation=45)
    
    # Add value labels with percentages
    for bar, pixel_count, percentage in zip(bars, caries_pixels, caries_percentages):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(caries_pixels)*0.02,
                f'{pixel_count:,}\n({percentage:.3f}%)', ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    plt.tight_layout()
    
    # Save plots
    pdf_path = os.path.join(output_dir, 'pixel_distribution_analysis.pdf')
    png_path = os.path.join(output_dir, 'pixel_distribution_analysis.png')
    plt.savefig(pdf_path, format='pdf', dpi=300, bbox_inches='tight')
    plt.savefig(png_path, format='png', dpi=300, bbox_inches='tight')
    print(f'Pixel distribution plots saved to: {pdf_path} and {png_path}')
    plt.show()
    
    # Print caries-only distribution for clarity
    print(f'\n=== CARIES-ONLY DISTRIBUTION (Class Imbalance Analysis) ===')
    for i, class_val in enumerate(caries_classes):
        print(f'{class_names[class_val]}: {caries_pixels[i]:,} pixels ({caries_percentages[i]:.3f}%)')
        
    # Calculate and print imbalance ratios
    max_caries_pixels = max(caries_pixels)
    print(f'\nImbalance Ratios (relative to majority class):')
    for i, class_val in enumerate(caries_classes):
        ratio = max_caries_pixels / caries_pixels[i]
        print(f'{class_names[class_val]}: {ratio:.1f}x less frequent')

def create_standalone_pie_chart_pdf(total_pixels_per_class, component_frequencies, class_names, output_dir):
    """Create and save a standalone pie chart as PDF for caries classes distribution (without background)."""
    
    plt.figure(figsize=(12, 8))
    
    # Filter out background class (0) - only include caries classes
    caries_classes = [k for k in total_pixels_per_class.keys() if k != 0]
    caries_pixels = [total_pixels_per_class[k] for k in caries_classes]
    
    # Calculate total caries pixels for percentage calculation
    total_caries_pixels = sum(caries_pixels)
    caries_percentages = [(pixels/total_caries_pixels)*100 for pixels in caries_pixels]
    
    # Calculate total components for each class
    total_components_per_class = {}
    for class_val in caries_classes:
        component_freq = component_frequencies[class_val]
        # Sum up all components: for each number of components, multiply by frequency
        total_components = sum(comp_count * freq for comp_count, freq in component_freq.items())
        total_components_per_class[class_val] = total_components
    
    # Create enhanced labels with class names, percentages, and component counts
    labels = []
    for k, percentage in zip(caries_classes, caries_percentages):
        component_count = total_components_per_class[k]
        labels.append(f'{class_names[k]}\n{percentage:.3f}%\n({component_count} segments)')
    
    # Use specific colors for caries classes (excluding background color)
    colors = ['#2E86C1', '#28B463', '#E74C3C']  # Blue, Green, Red for superficial, medium, deep
    
    # Create pie chart
    plt.pie(caries_percentages, labels=labels, colors=colors, autopct='%1.3f%%', 
            startangle=90, textprops={'fontsize': 11})
    plt.title('Class Distribution Among Caries Types\n(Percentage of Total Caries Pixels & Component Counts)', 
              fontweight='bold', fontsize=16)
    
    # Save as PDF
    pdf_path = os.path.join(output_dir, 'standalone_pie_chart_caries_distribution.pdf')
    plt.savefig(pdf_path, format='pdf', dpi=300, bbox_inches='tight')
    print(f'Standalone caries distribution pie chart saved as PDF to: {pdf_path}')
    
    # Print component summary
    print(f'\nComponent counts summary:')
    for class_val in caries_classes:
        print(f'{class_names[class_val]}: {total_components_per_class[class_val]} total segments/components')
    
    plt.close()  # Close the figure to free memory

def create_class_combination_table(annotation_files, class_names, output_dir):
    """Create a comprehensive table analyzing class combinations in each image."""
    
    print(f'\nAnalyzing class combinations across {len(annotation_files)} images...')
    
    # Initialize counters for different combinations
    combination_counts = {
        'background_only': 0,           # Only background (0)
        'superficial_only': 0,          # Only 102
        'medium_only': 0,               # Only 153  
        'deep_only': 0,                 # Only 255
        'superficial_medium': 0,        # 102 + 153
        'medium_deep': 0,               # 153 + 255
        'superficial_deep': 0,          # 102 + 255
        'all_three': 0,                 # 102 + 153 + 255
        'other_combinations': 0         # Any unexpected combinations
    }
    
    # Initialize pixel area counters for each combination
    combination_pixel_areas = {
        'background_only': 0,
        'superficial_only': 0,
        'medium_only': 0,
        'deep_only': 0,
        'superficial_medium': 0,
        'medium_deep': 0,
        'superficial_deep': 0,
        'all_three': 0,
        'other_combinations': 0
    }
    
    # Track detailed information for each image
    image_details = []
    
    for i, annotation_file in enumerate(annotation_files):
        # Load the annotation image
        img = Image.open(annotation_file)
        img_array = np.array(img)
        
        # Find unique classes in this image
        unique_classes = set(np.unique(img_array))
        
        # Count pixels for each class in this image
        unique_values, counts = np.unique(img_array, return_counts=True)
        pixel_counts = dict(zip(unique_values, counts))
        
        # Calculate total caries pixels (excluding background)
        caries_pixels_in_image = sum(count for class_val, count in pixel_counts.items() if class_val != 0)
        
        # Remove background class to check caries combinations
        caries_classes = unique_classes - {0}
        
        # Check for unexpected classes (not 0, 102, 153, 255)
        expected_classes = {0, 102, 153, 255}
        unexpected_classes = unique_classes - expected_classes
        
        # Get image name for detailed tracking
        img_name = os.path.basename(annotation_file)
        
        # Classify based on combinations and add pixel areas
        if unexpected_classes:
            # Handle images with unexpected class values
            combination_counts['other_combinations'] += 1
            combination_pixel_areas['other_combinations'] += caries_pixels_in_image
            category = f'Other combination (unexpected classes: {sorted(list(unexpected_classes))})'
            area_info = f"{caries_pixels_in_image} caries pixels"
        elif len(caries_classes) == 0:
            # Only background
            combination_counts['background_only'] += 1
            combination_pixel_areas['background_only'] += pixel_counts.get(0, 0)
            category = 'Background only'
            area_info = f"{pixel_counts.get(0, 0)} background pixels"
        elif caries_classes == {102}:
            # Only superficial caries
            combination_counts['superficial_only'] += 1
            combination_pixel_areas['superficial_only'] += caries_pixels_in_image
            category = 'Superficial-caries only'
            area_info = f"{caries_pixels_in_image} caries pixels"
        elif caries_classes == {153}:
            # Only medium caries
            combination_counts['medium_only'] += 1
            combination_pixel_areas['medium_only'] += caries_pixels_in_image
            category = 'Medium-caries only'
            area_info = f"{caries_pixels_in_image} caries pixels"
        elif caries_classes == {255}:
            # Only deep caries
            combination_counts['deep_only'] += 1
            combination_pixel_areas['deep_only'] += caries_pixels_in_image
            category = 'Deep-caries only'
            area_info = f"{caries_pixels_in_image} caries pixels"
        elif caries_classes == {102, 153}:
            # Superficial + medium
            combination_counts['superficial_medium'] += 1
            combination_pixel_areas['superficial_medium'] += caries_pixels_in_image
            category = 'Superficial + Medium caries'
            area_info = f"{caries_pixels_in_image} caries pixels"
        elif caries_classes == {153, 255}:
            # Medium + deep
            combination_counts['medium_deep'] += 1
            combination_pixel_areas['medium_deep'] += caries_pixels_in_image
            category = 'Medium + Deep caries'
            area_info = f"{caries_pixels_in_image} caries pixels"
        elif caries_classes == {102, 255}:
            # Superficial + deep
            combination_counts['superficial_deep'] += 1
            combination_pixel_areas['superficial_deep'] += caries_pixels_in_image
            category = 'Superficial + Deep caries'
            area_info = f"{caries_pixels_in_image} caries pixels"
        elif caries_classes == {102, 153, 255}:
            # All three caries types
            combination_counts['all_three'] += 1
            combination_pixel_areas['all_three'] += caries_pixels_in_image
            category = 'All three caries types'
            area_info = f"{caries_pixels_in_image} caries pixels"
        else:
            # Any other unexpected caries combinations
            combination_counts['other_combinations'] += 1
            combination_pixel_areas['other_combinations'] += caries_pixels_in_image
            category = f'Other caries combination: {sorted(list(caries_classes))}'
            area_info = f"{caries_pixels_in_image} caries pixels"
        
        image_details.append({
            'filename': img_name,
            'classes_present': sorted(list(unique_classes)),
            'caries_classes': sorted(list(caries_classes)),
            'category': category,
            'caries_pixel_area': caries_pixels_in_image,
            'area_info': area_info
        })
        
        # Print progress
        if (i + 1) % 100 == 0:
            print(f'Processed {i + 1}/{len(annotation_files)} images for combination analysis...')
    
    # Create comprehensive table
    total_images = len(annotation_files)
    total_caries_pixels_all = sum(combination_pixel_areas[key] for key in combination_pixel_areas.keys() if key != 'background_only')
    
    # Prepare data for the table with pixel area information and percentages
    table_data = [
        ['Background only (no caries)', 
         combination_counts['background_only'], 
         f"{(combination_counts['background_only']/total_images)*100:.2f}%",
         f"{combination_pixel_areas['background_only']:,}",
         "N/A (background)"],
        ['Superficial-caries only (102)', 
         combination_counts['superficial_only'], 
         f"{(combination_counts['superficial_only']/total_images)*100:.2f}%",
         f"{combination_pixel_areas['superficial_only']:,}",
         f"{(combination_pixel_areas['superficial_only']/total_caries_pixels_all)*100:.2f}%" if total_caries_pixels_all > 0 else "0.00%"],
        ['Medium-caries only (153)', 
         combination_counts['medium_only'], 
         f"{(combination_counts['medium_only']/total_images)*100:.2f}%",
         f"{combination_pixel_areas['medium_only']:,}",
         f"{(combination_pixel_areas['medium_only']/total_caries_pixels_all)*100:.2f}%" if total_caries_pixels_all > 0 else "0.00%"],
        ['Deep-caries only (255)', 
         combination_counts['deep_only'], 
         f"{(combination_counts['deep_only']/total_images)*100:.2f}%",
         f"{combination_pixel_areas['deep_only']:,}",
         f"{(combination_pixel_areas['deep_only']/total_caries_pixels_all)*100:.2f}%" if total_caries_pixels_all > 0 else "0.00%"],
        ['Superficial + Medium (102 + 153)', 
         combination_counts['superficial_medium'], 
         f"{(combination_counts['superficial_medium']/total_images)*100:.2f}%",
         f"{combination_pixel_areas['superficial_medium']:,}",
         f"{(combination_pixel_areas['superficial_medium']/total_caries_pixels_all)*100:.2f}%" if total_caries_pixels_all > 0 else "0.00%"],
        ['Medium + Deep (153 + 255)', 
         combination_counts['medium_deep'], 
         f"{(combination_counts['medium_deep']/total_images)*100:.2f}%",
         f"{combination_pixel_areas['medium_deep']:,}",
         f"{(combination_pixel_areas['medium_deep']/total_caries_pixels_all)*100:.2f}%" if total_caries_pixels_all > 0 else "0.00%"],
        ['Superficial + Deep (102 + 255)', 
         combination_counts['superficial_deep'], 
         f"{(combination_counts['superficial_deep']/total_images)*100:.2f}%",
         f"{combination_pixel_areas['superficial_deep']:,}",
         f"{(combination_pixel_areas['superficial_deep']/total_caries_pixels_all)*100:.2f}%" if total_caries_pixels_all > 0 else "0.00%"],
        ['All three types (102 + 153 + 255)', 
         combination_counts['all_three'], 
         f"{(combination_counts['all_three']/total_images)*100:.2f}%",
         f"{combination_pixel_areas['all_three']:,}",
         f"{(combination_pixel_areas['all_three']/total_caries_pixels_all)*100:.2f}%" if total_caries_pixels_all > 0 else "0.00%"],
        ['Other/Unexpected combinations', 
         combination_counts['other_combinations'], 
         f"{(combination_counts['other_combinations']/total_images)*100:.2f}%",
         f"{combination_pixel_areas['other_combinations']:,}",
         f"{(combination_pixel_areas['other_combinations']/total_caries_pixels_all)*100:.2f}%" if total_caries_pixels_all > 0 else "0.00%"]
    ]
    
    # Verify total count
    total_counted = sum(combination_counts.values())
    print(f'\n=== VERIFICATION ===')
    print(f'Total images processed: {total_images}')
    print(f'Total images counted in categories: {total_counted}')
    if total_counted != total_images:
        print(f'⚠️  MISMATCH: {total_images - total_counted} images are missing from categorization!')
    else:
        print('✅ All images successfully categorized!')
    
    # Create and display the table
    fig, ax = plt.subplots(figsize=(18, 8))
    ax.axis('tight')
    ax.axis('off')
    
    headers = ['Class Combination', 'Image Count', 'Image %', 'Total Pixel Area', 'Pixel Area %']
    
    # Create table
    table = ax.table(cellText=table_data, colLabels=headers, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 2)
    
    # Style the table
    # Header styling
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#2E86C1')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Row styling with alternating colors
    colors = ['#F8F9FA', '#E9ECEF']
    for i in range(len(table_data)):
        for j in range(len(headers)):
            table[(i+1, j)].set_facecolor(colors[i % 2])
            if j == 1 or j == 3 or j == 4:  # Count, pixel area, and pixel percentage columns - make bold
                table[(i+1, j)].set_text_props(weight='bold')
    
    plt.title('Class Combination Analysis\nDistribution of Caries Type Combinations Across Images with Pixel Areas', 
              fontweight='bold', fontsize=14, pad=20)
    
    # Save table as PDF and PNG
    pdf_path = os.path.join(output_dir, 'class_combination_analysis_table.pdf')
    png_path = os.path.join(output_dir, 'class_combination_analysis_table.png')
    plt.savefig(pdf_path, format='pdf', dpi=300, bbox_inches='tight')
    plt.savefig(png_path, format='png', dpi=300, bbox_inches='tight')
    print(f'Class combination table saved to: {pdf_path} and {png_path}')
    plt.show()
    
    # Print detailed results to console
    print(f'\n=== CLASS COMBINATION ANALYSIS ===')
    print(f'Total images analyzed: {total_images}')
    print(f'Total caries pixels across all images: {total_caries_pixels_all:,}')
    print('-' * 90)
    
    for category, count, img_percentage, pixel_area, pixel_percentage in table_data:
        print(f'{category:<35} {count:>6} ({img_percentage:>6}) {pixel_area:>12} ({pixel_percentage:>7})')
    
    # Save detailed results to CSV and text file
    save_combination_analysis_results(table_data, image_details, total_images, combination_pixel_areas, output_dir)
    
    return combination_counts, image_details, combination_pixel_areas

def save_combination_analysis_results(table_data, image_details, total_images, combination_pixel_areas, output_dir):
    """Save detailed combination analysis results to files."""
    
    # Save summary table to CSV
    import csv
    csv_path = os.path.join(output_dir, 'class_combination_summary.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Class Combination', 'Image Count', 'Image Percentage', 'Total Pixel Area', 'Pixel Area Percentage'])
        writer.writerows(table_data)
    
    # Save detailed results to text file
    txt_path = os.path.join(output_dir, 'class_combination_detailed_results.txt')
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("Class Combination Analysis - Detailed Results\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Total images analyzed: {total_images}\n")
        total_caries_pixels = sum(area for key, area in combination_pixel_areas.items() if key != 'background_only')
        f.write(f"Total caries pixels across all images: {total_caries_pixels:,}\n\n")
        
        f.write("SUMMARY TABLE:\n")
        f.write("-" * 90 + "\n")
        f.write(f"{'Class Combination':<35} {'Count':>6} {'Img %':>8} {'Pixel Area':>15} {'Pixel %':>8}\n")
        f.write("-" * 90 + "\n")
        
        for category, count, img_percentage, pixel_area, pixel_percentage in table_data:
            f.write(f"{category:<35} {count:>6} {img_percentage:>8} {pixel_area:>15} {pixel_percentage:>8}\n")
        
        f.write(f"\n\nDETAILED IMAGE BREAKDOWN:\n")
        f.write("-" * 60 + "\n")
        
        # Group images by category
        by_category = {}
        for detail in image_details:
            category = detail['category']
            if category not in by_category:
                by_category[category] = []
            by_category[category].append(detail)
        
        for category, images in by_category.items():
            f.write(f"\n{category} ({len(images)} images):\n")
            total_pixels_category = sum(img['caries_pixel_area'] for img in images)
            f.write(f"  Total caries pixels in category: {total_pixels_category:,}\n")
            if len(images) > 0:
                f.write(f"  Average caries pixels per image: {total_pixels_category/len(images):.1f}\n")
            if total_caries_pixels > 0:
                category_percentage = (total_pixels_category/total_caries_pixels)*100
                f.write(f"  Percentage of total caries pixels: {category_percentage:.2f}%\n")
            f.write("  Individual images:\n")
            for img in sorted(images, key=lambda x: x['caries_pixel_area'], reverse=True):
                f.write(f"    {img['filename']} - Classes: {img['classes_present']} - Caries pixels: {img['caries_pixel_area']:,}\n")
    
    print(f'Detailed combination analysis saved to: {csv_path} and {txt_path}')

def create_grouped_components_plot(component_frequencies, class_names, output_dir, total_images):
    """Create grouped bar plot for connected components analysis."""
    
    # Find the maximum number of components across all classes
    max_components = 0
    for class_val in [102, 153, 255]:
        if component_frequencies[class_val]:
            max_comp = max(component_frequencies[class_val].keys())
            max_components = max(max_components, max_comp)
    
    # Prepare data for grouped bar plot
    component_range = list(range(0, max_components + 1))
    
    class_102_counts = []
    class_153_counts = []
    class_255_counts = []
    
    for comp_count in component_range:
        class_102_counts.append(component_frequencies[102].get(comp_count, 0))
        class_153_counts.append(component_frequencies[153].get(comp_count, 0))
        class_255_counts.append(component_frequencies[255].get(comp_count, 0))
    
    # Create grouped bar plot
    plt.figure(figsize=(16, 8))
    
    bar_width = 0.29
    x = np.arange(len(component_range))
    
    # Define specific colors: Blue for superficial (102), Green for medium (153), Red for deep (255)
    blue_color = '#2E86C1'    # Blue for superficial-caries (102)
    green_color = '#28B463'   # Green for medium-caries (153)
    red_color = '#E74C3C'     # Red for deep-caries (255)
    
    bars1 = plt.bar(x - bar_width, class_102_counts, bar_width, 
                   label='Superficial-caries (value 102)', color=blue_color, alpha=0.8, edgecolor='black')
    bars2 = plt.bar(x, class_153_counts, bar_width, 
                   label='Medium-caries (value 153)', color=green_color, alpha=0.8, edgecolor='black')
    bars3 = plt.bar(x + bar_width, class_255_counts, bar_width, 
                   label='Deep-caries (value 255)', color=red_color, alpha=0.8, edgecolor='black')
    
    # Customize the plot
    plt.xlabel('Number of Separate Caries Lesions (Multi class)', fontsize=20)
    plt.ylabel('Number of Multi Class Masks', fontsize=20)
    plt.title('Distribution of Unique Caries Masks in Multi-Class DC1000 Dataset', 
              fontsize=24, fontweight='bold')
    plt.xticks(x, component_range, fontsize=17)
    plt.yticks(fontsize=17)
    plt.legend(fontsize=17)
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars with corresponding colors
    def add_value_labels(bars, values, color):
        for bar, value in zip(bars, values):
            if value > 0:
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(max(class_102_counts), max(class_153_counts), max(class_255_counts))*0.01,
                        str(value), ha='center', va='bottom', fontsize=17, fontweight='bold', color=color)
    
    add_value_labels(bars1, class_102_counts, blue_color)   # Blue labels for superficial-caries
    add_value_labels(bars2, class_153_counts, green_color)  # Green labels for medium-caries  
    add_value_labels(bars3, class_255_counts, red_color)    # Red labels for deep-caries
    
    plt.tight_layout()
    
    # Save plots
    pdf_path = os.path.join(output_dir, 'multiclass_connected_components_distribution.pdf')
    png_path = os.path.join(output_dir, 'multiclass_connected_components_distribution.png')
    plt.savefig(pdf_path, format='pdf', dpi=300, bbox_inches='tight')
    plt.savefig(png_path, format='png', dpi=300, bbox_inches='tight')
    print(f'Connected components plots saved to: {pdf_path} and {png_path}')
    plt.show()

def create_class_imbalance_analysis_plot(total_pixels_per_class, class_names, output_dir):
    """Create comprehensive class imbalance analysis visualization."""
    
    # Remove background class for caries-only analysis
    caries_pixels = {k: v for k, v in total_pixels_per_class.items() if k != 0}
    caries_names = {k: v for k, v in class_names.items() if k != 0}
    
    # Calculate total caries pixels
    total_caries_pixels = sum(caries_pixels.values())
    
    # Calculate percentages and imbalance ratios
    percentages = {}
    for class_val, pixels in caries_pixels.items():
        percentages[class_val] = (pixels / total_caries_pixels) * 100
    
    # Find the most frequent class for ratio calculation
    max_pixels_class = max(caries_pixels.keys(), key=lambda x: caries_pixels[x])
    max_pixels = caries_pixels[max_pixels_class]
    
    # Calculate imbalance ratios (how many times more frequent is the majority class)
    imbalance_ratios = {}
    for class_val, pixels in caries_pixels.items():
        imbalance_ratios[class_val] = max_pixels / pixels if pixels > 0 else 0
    
    # Create figure with multiple subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Colors for each class
    class_colors = {102: '#2E86C1', 153: '#28B463', 255: '#E74C3C'}
    
    # 1. Percentage Distribution (Pie Chart)
    ax1.pie(list(percentages.values()), 
            labels=[f'{caries_names[k]}\n{v:.3f}%' for k, v in percentages.items()],
            colors=[class_colors[k] for k in percentages.keys()],
            autopct='%1.3f%%', startangle=90, textprops={'fontsize': 10})
    ax1.set_title('Class Distribution Among Caries Types\n(Percentage of Total Caries Pixels)', 
                  fontweight='bold', fontsize=12)
    
    # 2. Absolute Pixel Counts (Log Scale)
    classes = list(caries_pixels.keys())
    pixels = list(caries_pixels.values())
    colors = [class_colors[c] for c in classes]
    
    bars = ax2.bar([caries_names[c] for c in classes], pixels, color=colors, alpha=0.8, edgecolor='black')
    ax2.set_title('Absolute Pixel Counts per Class\n(Log Scale)', fontweight='bold', fontsize=12)
    ax2.set_ylabel('Number of Pixels (Log Scale)', fontsize=10)
    ax2.set_yscale('log')
    ax2.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, pixel_count in zip(bars, pixels):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.1,
                f'{pixel_count:,}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # 3. Imbalance Ratios
    ratios = list(imbalance_ratios.values())
    bars3 = ax3.bar([caries_names[c] for c in classes], ratios, color=colors, alpha=0.8, edgecolor='black')
    ax3.set_title(f'Class Imbalance Ratios\n(Relative to {caries_names[max_pixels_class]})', 
                  fontweight='bold', fontsize=12)
    ax3.set_ylabel('Imbalance Ratio (x times less frequent)', fontsize=10)
    ax3.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, ratio in zip(bars3, ratios):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(ratios)*0.02,
                f'{ratio:.1f}x', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 4. Class Statistics Table
    ax4.axis('off')
    
    # Prepare table data
    table_data = []
    headers = ['Class', 'Pixels', 'Percentage', 'Imbalance Ratio']
    
    for class_val in sorted(classes):
        table_data.append([
            caries_names[class_val],
            f'{caries_pixels[class_val]:,}',
            f'{percentages[class_val]:.3f}%',
            f'{imbalance_ratios[class_val]:.1f}x'
        ])
    
    table = ax4.table(cellText=table_data, colLabels=headers, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 2)
    
    # Color table headers
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#E8E8E8')
        table[(0, i)].set_text_props(weight='bold')
    
    # Color table rows according to class colors
    for i, class_val in enumerate(sorted(classes)):
        for j in range(len(headers)):
            table[(i+1, j)].set_facecolor(class_colors[class_val] + '30')  # Add transparency
    
    ax4.set_title('Class Imbalance Summary Statistics', fontweight='bold', fontsize=12)
    
    plt.tight_layout()
    
    # Save plots
    pdf_path = os.path.join(output_dir, 'class_imbalance_analysis.pdf')
    png_path = os.path.join(output_dir, 'class_imbalance_analysis.png')
    plt.savefig(pdf_path, format='pdf', dpi=300, bbox_inches='tight')
    plt.savefig(png_path, format='png', dpi=300, bbox_inches='tight')
    print(f'Class imbalance analysis plots saved to: {pdf_path} and {png_path}')
    plt.show()
    
    # Print imbalance analysis
    print(f'\n=== CLASS IMBALANCE ANALYSIS ===')
    print(f'Total caries pixels: {total_caries_pixels:,}')
    print(f'Most frequent class: {caries_names[max_pixels_class]} ({percentages[max_pixels_class]:.3f}%)')
    
    for class_val in sorted(classes):
        print(f'{caries_names[class_val]}:')
        print(f'  - {percentages[class_val]:.3f}% of total caries pixels')
        print(f'  - {imbalance_ratios[class_val]:.1f}x less frequent than majority class')
        
    return percentages, imbalance_ratios

def create_class_presence_plot(images_with_class, class_names, total_images, output_dir):
    """Create plot showing class presence across images."""
    
    plt.figure(figsize=(10, 6))
    
    classes = list(images_with_class.keys())
    counts = list(images_with_class.values())
    percentages = [(count/total_images)*100 for count in counts]
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    
    bars = plt.bar([class_names[c] for c in classes], counts, color=colors, alpha=0.8, edgecolor='black')
    
    plt.title('Class Presence Across Images', fontweight='bold', fontsize=14)
    plt.ylabel('Number of Images', fontsize=12)
    plt.xlabel('Classes', fontsize=12)
    
    # Add value labels and percentages on bars
    for bar, count, percentage in zip(bars, counts, percentages):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(counts)*0.01,
                f'{count}\n({percentage:.1f}%)', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    
    # Save plots
    pdf_path = os.path.join(output_dir, 'class_presence_analysis.pdf')
    png_path = os.path.join(output_dir, 'class_presence_analysis.png')
    plt.savefig(pdf_path, format='pdf', dpi=300, bbox_inches='tight')
    plt.savefig(png_path, format='png', dpi=300, bbox_inches='tight')
    print(f'Class presence plots saved to: {pdf_path} and {png_path}')
    plt.show()

def save_detailed_results(total_pixels_per_class, component_frequencies, images_with_class, 
                         class_names, total_images, total_pixels, output_dir):
    """Save detailed analysis results to text file."""
    
    results_path = os.path.join(output_dir, 'multiclass_analysis_results.txt')
    
    with open(results_path, 'w') as f:
        f.write("Multi-Class Analysis Results\n")
        f.write("===========================\n\n")
        
        f.write(f"Total images analyzed: {total_images}\n")
        f.write(f"Total pixels analyzed: {total_pixels:,}\n\n")
        
        f.write("PIXEL DISTRIBUTION:\n")
        f.write("-" * 40 + "\n")
        for class_val, class_name in class_names.items():
            pixels = total_pixels_per_class[class_val]
            percentage = (pixels / total_pixels) * 100
            f.write(f"{class_name} (value {class_val}):\n")
            f.write(f"  Pixels: {pixels:,} ({percentage:.4f}%)\n")
            f.write(f"  Present in: {images_with_class[class_val]} images ({(images_with_class[class_val]/total_images)*100:.2f}%)\n\n")
        
        f.write("CONNECTED COMPONENTS ANALYSIS:\n")
        f.write("-" * 40 + "\n")
        for class_val in [102, 153, 255]:
            class_name = class_names[class_val]
            f.write(f"\n{class_name} (value {class_val}) component distribution:\n")
            
            component_freq = component_frequencies[class_val]
            sorted_components = sorted(component_freq.items())
            
            for comp_count, freq in sorted_components:
                percentage = (freq / total_images) * 100
                f.write(f"  {comp_count} components: {freq} images ({percentage:.2f}%)\n")
            
            # Calculate statistics
            if component_freq:
                total_components = sum(comp_count * freq for comp_count, freq in component_freq.items())
                avg_components = total_components / total_images
                max_components = max(component_freq.keys()) if component_freq.keys() else 0
                min_components = min(component_freq.keys()) if component_freq.keys() else 0
                
                f.write(f"  Average components per image: {avg_components:.2f}\n")
                f.write(f"  Maximum components: {max_components}\n")
                f.write(f"  Minimum components: {min_components}\n")
                f.write(f"  Total components across all images: {total_components}\n")
    
    print(f'Detailed results saved to: {results_path}')

def main():
    """Main function to run the multi-class analysis."""
    # Define paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    annotations_dir = os.path.join(current_dir, 'annotations')
    output_dir = current_dir
    
    # Check if annotations directory exists
    if not os.path.exists(annotations_dir):
        print(f"Error: Annotations directory not found at {annotations_dir}")
        return
    
    # Run analysis
    total_pixels_per_class, component_frequencies, images_with_class = analyze_multiclass_components(annotations_dir, output_dir)
    
    print(f"\nMulti-class analysis completed successfully!")
    print(f"Results saved in: {output_dir}")

if __name__ == "__main__":
    main()
