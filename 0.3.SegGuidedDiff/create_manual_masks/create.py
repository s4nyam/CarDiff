from time import time
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import cv2
import numpy as np
import os
from pathlib import Path
from PIL import Image

# To run use command - streamlit run create.py
# Streamlit app title
st.title("Multi-class Segmentation Mask Editor (384x384)")

# Instructions
st.write("Load and edit segmentation masks with multiple classes.")
st.write("- Load an existing mask or start fresh")
st.write("- Select drawing mode: Freehand, Rectangle, or Eraser")
st.write("- Define custom categories and colors")
st.write("- Click 'Save Mask' to save your edits")

# Initialize session state
if "canvas_image" not in st.session_state:
    st.session_state.canvas_image = None
if "canvas_image_pil" not in st.session_state:
    st.session_state.canvas_image_pil = None
if "loaded_filename" not in st.session_state:
    st.session_state.loaded_filename = None

# Get available masks in current directory
current_dir = Path(".")
png_files = sorted([f.name for f in current_dir.glob("*.png")])

# Load the first PNG file by default
if png_files and st.session_state.canvas_image is None:
    try:
        mask_img = cv2.imread(png_files[0], cv2.IMREAD_GRAYSCALE)
        if mask_img is not None:
            # Resize to 384x384 if needed
            if mask_img.shape != (384, 384):
                mask_img = cv2.resize(mask_img, (384, 384))
            # Convert grayscale to RGBA for canvas
            rgba_img = cv2.cvtColor(cv2.cvtColor(mask_img, cv2.COLOR_GRAY2BGR), cv2.COLOR_BGR2RGBA)
            st.session_state.canvas_image = rgba_img
            # Convert to PIL Image for st_canvas
            st.session_state.canvas_image_pil = Image.fromarray(rgba_img)
            st.session_state.loaded_filename = png_files[0]
    except Exception as e:
        st.error(f"Error loading mask: {e}")
elif st.session_state.canvas_image is None:
    # If no PNG files found, create blank canvas
    st.session_state.canvas_image = np.zeros((384, 384, 4), dtype=np.uint8)
    st.session_state.canvas_image_pil = Image.fromarray(st.session_state.canvas_image)

# Sidebar for configuration
st.sidebar.header("Configuration")
if st.session_state.loaded_filename:
    st.sidebar.info(f"Editing: {st.session_state.loaded_filename}")

# Category and color configuration
st.sidebar.header("Categories & Colors")

num_categories = st.sidebar.slider("Number of Categories:", 1, 10, 3)

# Create default category config
categories = {}
colors = {}
rgb_values = {}

col1, col2, col3 = st.sidebar.columns(3)

for i in range(num_categories):
    with col1:
        if i < 3:
            default_names = ['Superficial Caries', 'Medium Caries', 'Deep Caries']
            cat_name = st.text_input(f"Category {i+1}", value=default_names[i] if i < len(default_names) else f"Class {i+1}", key=f"cat_{i}")
        else:
            cat_name = st.text_input(f"Category {i+1}", value=f"Class {i+1}", key=f"cat_{i}")
    
    with col2:
        default_values = [102, 153, 255]
        pixel_val = st.number_input(f"Pixel Value {i+1}", min_value=0, max_value=255, 
                                     value=default_values[i] if i < len(default_values) else (i+1)*50, 
                                     key=f"val_{i}", step=1)
    
    with col3:
        # Convert pixel value to hex color for the color picker
        hex_color = f"#{pixel_val:02x}{pixel_val:02x}{pixel_val:02x}"
        rgb_val = st.color_picker(f"Color {i+1}", value=hex_color, key=f"col_{i}")
        # Convert hex to RGB
        rgb_values[i] = tuple(int(rgb_val[j:j+2], 16) for j in (1, 3, 5))
    
    categories[i] = cat_name
    colors[i] = pixel_val

st.sidebar.header("Drawing Settings")

# Drawing mode selection
drawing_mode = st.sidebar.radio(
    "Drawing Mode:",
    ("Freehand", "Rectangle", "Eraser")
)

# Brush/stroke width
stroke_width = st.sidebar.slider("Brush Size:", 2, 50, 10)

# Select current category for drawing
current_category = st.sidebar.selectbox("Draw with Category:", 
                                        [f"{categories[i]} (value: {colors[i]})" for i in range(num_categories)])
category_idx = int(current_category.split()[0]) if current_category[0].isdigit() else 0
for i in range(num_categories):
    if f"{categories[i]}" in current_category:
        category_idx = i
        break

pixel_value = colors[category_idx]
rgb_tuple = rgb_values[category_idx]
drawing_color = f"rgba({rgb_tuple[0]}, {rgb_tuple[1]}, {rgb_tuple[2]}, 1)"

# Drawing mode mapping
mode_map = {
    "Freehand": "freedraw",
    "Rectangle": "rect",
    "Eraser": "freedraw"  # We'll handle eraser separately
}

drawing_mode_canvas = mode_map[drawing_mode]

# Create canvas with loaded or new image
canvas_result = st_canvas(
    fill_color=drawing_color,
    stroke_width=stroke_width,
    stroke_color=drawing_color,
    background_color="rgba(0, 0, 0, 1)",
    background_image=st.session_state.canvas_image_pil,
    width=384,
    height=384,
    drawing_mode=drawing_mode_canvas,
    key="canvas",
)

# Main area buttons
st.write("---")
col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.button("Save Mask", key="save_btn"):
        if canvas_result.image_data is not None:
            img = canvas_result.image_data.astype(np.uint8)
            gray_mask = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
            
            # Generate filename
            if st.session_state.loaded_filename:
                # Save as updated version
                base_name = st.session_state.loaded_filename.rsplit(".", 1)[0]
                timestamp = str(time()).replace(".", "_")
                filename = f"{base_name}_edited_{timestamp}.png"
            else:
                timestamp = str(time()).replace(".", "_")
                filename = f"mask_{timestamp}.png"
            
            cv2.imwrite(filename, gray_mask)
            st.success(f"Mask saved as '{filename}'")
        else:
            st.error("No drawing to save!")

with col2:
    if st.button("Reset Canvas", key="reset_btn"):
        st.session_state.canvas_image = np.zeros((384, 384, 4), dtype=np.uint8)
        st.rerun()

with col3:
    if st.button("Show Current Mask", key="show_btn"):
        if canvas_result.image_data is not None:
            img = canvas_result.image_data.astype(np.uint8)
            gray_mask = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
            st.image(gray_mask, caption="Current Mask (Grayscale)", width=400)

with col4:
    if st.button("Export Stats", key="stats_btn"):
        if canvas_result.image_data is not None:
            img = canvas_result.image_data.astype(np.uint8)
            gray_mask = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
            
            st.write("**Mask Statistics:**")
            for i in range(num_categories):
                pixels = np.sum(gray_mask == colors[i])
                percentage = (pixels / (384*384)) * 100
                st.write(f"- {categories[i]}: {pixels} pixels ({percentage:.2f}%)")