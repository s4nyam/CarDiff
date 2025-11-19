from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

from . import utils


@dataclass
class MaskEncoderMetadata:
    label_nc: int
    ignore_label: Optional[int]
    color_to_index: Optional[Dict[int, int]]
    value_to_index: Optional[Dict[int, int]]
    class_values: List[int]


class MaskEncoder:
    """Utility for converting raw segmentation masks to class index tensors."""

    def __init__(self, label_nc: int, ignore_label: Optional[int] = 255,
                 color_to_index: Optional[Dict[int, int]] = None,
                 value_to_index: Optional[Dict[int, int]] = None,
                 class_values: Optional[Sequence[int]] = None) -> None:
        self.label_nc = int(label_nc)
        self.ignore_label = ignore_label
        if color_to_index:
            self.color_to_index = {int(k): int(v) for k, v in color_to_index.items()}
            self.use_color_map = True
        else:
            self.color_to_index = None
            self.use_color_map = False
        if value_to_index:
            self.value_to_index = {int(k): int(v) for k, v in value_to_index.items()}
        else:
            self.value_to_index = None
        self.class_values = list(int(v) for v in class_values) if class_values is not None else list(range(self.label_nc))

    @staticmethod
    def _mask_array(image: Image.Image) -> np.ndarray:
        array = np.array(image)
        if array.ndim == 2:
            return array
        if array.ndim == 3 and array.shape[2] == 1:
            return array[:, :, 0]
        return array

    @classmethod
    def analyze(cls, mask_paths: Sequence[Path], label_nc: Optional[int] = None,
                ignore_label: Optional[int] = 255) -> Tuple['MaskEncoder', Dict[int, Tuple[int, int, int]]]:
        color_values: set[int] = set()
        gray_values: set[int] = set()
        uses_color = False

        for path in mask_paths:
            with Image.open(path) as img:
                array = cls._mask_array(img)

            if array.ndim == 2:
                gray_values.update(int(v) for v in np.unique(array))
            else:
                uses_color = True
                packed = utils.rgb_to_int(array)
                color_values.update(int(v) for v in np.unique(packed))

        if uses_color:
            sorted_colors = sorted(color_values)
            mapping = {color: idx for idx, color in enumerate(sorted_colors)}
            inferred_nc = len(mapping)
            target_nc = label_nc if label_nc is not None else inferred_nc
            if target_nc < inferred_nc:
                raise ValueError(
                    f"label_nc ({target_nc}) is smaller than the number of discovered colors ({inferred_nc})."
                )
            encoder = cls(label_nc=target_nc, ignore_label=ignore_label, color_to_index=mapping)
            palette = {idx: rgb for idx, rgb in enumerate(utils.ints_to_rgb(sorted_colors))}
            return encoder, palette

        unique_values = sorted(gray_values) if gray_values else [0]
        inferred_nc = len(unique_values)
        if label_nc is not None and label_nc != inferred_nc:
            raise ValueError(
                f"label_nc ({label_nc}) does not match the number of discovered mask values ({inferred_nc})."
            )
        target_nc = inferred_nc if label_nc is None else label_nc
        value_mapping = {value: idx for idx, value in enumerate(unique_values)}
        encoder = cls(label_nc=target_nc, ignore_label=ignore_label,
                      color_to_index=None, value_to_index=value_mapping,
                      class_values=unique_values)
        palette = {value_mapping[value]: (value, value, value) for value in unique_values}
        return encoder, palette

    def encode(self, mask: Image.Image) -> torch.Tensor:
        """Convert a PIL mask image to a class index tensor."""
        array = self._mask_array(mask)
        
        if self.use_color_map:
            # Handle color masks
            if array.ndim == 3:
                packed = utils.rgb_to_int(array)
                result = np.full(packed.shape, self.ignore_label or 0, dtype=np.int64)
                for color, idx in self.color_to_index.items():
                    result[packed == color] = idx
            else:
                raise ValueError("Expected color image but got grayscale")
        else:
            # Handle grayscale masks
            if self.value_to_index:
                result = np.full(array.shape, self.ignore_label or 0, dtype=np.int64)
                for value, idx in self.value_to_index.items():
                    result[array == value] = idx
            else:
                result = array.astype(np.int64)

        return torch.from_numpy(result).long()


class SegmentationGuidedDataset(Dataset):
    """Dataset for segmentation-guided diffusion following SISDM approach."""
    
    def __init__(
        self,
        images_dir: Path,
        labels_dir: Path,
        mask_encoder: MaskEncoder,
        image_size: Tuple[int, int] = (384, 384),
        random_flip: bool = True,
        random_crop: bool = True,
        is_train: bool = True,
    ) -> None:
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.mask_encoder = mask_encoder
        self.image_size = image_size
        self.random_flip = random_flip
        self.random_crop = random_crop
        self.is_train = is_train
        
        # Find paired image and mask files
        self.image_paths = []
        self.label_paths = []
        
        image_files = sorted([f for f in self.images_dir.iterdir() if f.suffix.lower() in ['.jpg', '.jpeg', '.png']])
        
        for img_path in image_files:
            # Try to find corresponding mask
            mask_candidates = [
                self.labels_dir / f"{img_path.stem}.png",
                self.labels_dir / f"{img_path.stem}.jpg", 
                self.labels_dir / f"{img_path.stem}.jpeg",
                self.labels_dir / img_path.name,
            ]
            
            for mask_path in mask_candidates:
                if mask_path.exists():
                    self.image_paths.append(img_path)
                    self.label_paths.append(mask_path)
                    break
        
        if len(self.image_paths) == 0:
            raise ValueError(f"No paired images found in {images_dir} and {labels_dir}")
            
        print(f"Found {len(self.image_paths)} paired images and masks")

    def __len__(self) -> int:
        return len(self.image_paths)

    def _preprocess_image(self, image: Image.Image, mask: Image.Image) -> Tuple[torch.Tensor, torch.Tensor]:
        """Preprocess image and mask pair with augmentations."""
        # Ensure RGB for image and L for mask
        if image.mode != 'RGB':
            image = image.convert('RGB')
        if mask.mode != 'L' and mask.mode != 'RGB':
            mask = mask.convert('L')
            
        # Resize maintaining aspect ratio if needed
        img_w, img_h = image.size
        target_h, target_w = self.image_size
        
        if self.is_train and self.random_crop:
            # Random crop (zoom in)
            crop_size = min(img_w, img_h)
            if crop_size > min(target_w, target_h):
                # Random crop to a square
                left = random.randint(0, img_w - crop_size)
                top = random.randint(0, img_h - crop_size)
                image = image.crop((left, top, left + crop_size, top + crop_size))
                mask = mask.crop((left, top, left + crop_size, top + crop_size))
        
        # Resize to target size
        image = image.resize((target_w, target_h), Image.Resampling.LANCZOS)
        mask = mask.resize((target_w, target_h), Image.Resampling.NEAREST)
        
        # Random horizontal flip
        if self.is_train and self.random_flip and random.random() < 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)
        
        # Convert to tensors
        image_tensor = TF.to_tensor(image)  # [0, 1]
        image_tensor = TF.normalize(image_tensor, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # [-1, 1]
        
        mask_tensor = self.mask_encoder.encode(mask)
        
        return image_tensor, mask_tensor

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Get a sample from the dataset."""
        image_path = self.image_paths[idx]
        label_path = self.label_paths[idx]
        
        # Load images
        with Image.open(image_path) as img:
            image = img.copy()
        with Image.open(label_path) as mask_img:
            mask = mask_img.copy()
        
        # Preprocess
        image_tensor, mask_tensor = self._preprocess_image(image, mask)
        
        # Prepare output dict following SISDM format
        out_dict = {
            'label': mask_tensor.unsqueeze(0),  # Add channel dimension
            'label_ori': mask_tensor.clone(),
            'path': str(image_path),
        }
        
        # Add instance map as edge detection (following SISDM approach)
        # For now, we'll use the same mask as instance, but this can be enhanced
        edges = self._compute_edges(mask_tensor)
        out_dict['instance'] = edges.unsqueeze(0)
        
        return image_tensor, out_dict
    
    def _compute_edges(self, mask: torch.Tensor) -> torch.Tensor:
        """Compute edge map from segmentation mask."""
        # Simple edge detection using gradient
        mask_float = mask.float().unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
        
        # Sobel edge detection
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        
        edges_x = torch.nn.functional.conv2d(mask_float, sobel_x, padding=1)
        edges_y = torch.nn.functional.conv2d(mask_float, sobel_y, padding=1)
        
        edges = torch.sqrt(edges_x**2 + edges_y**2).squeeze()
        edges = (edges > 0.1).float()  # Threshold to binary
        
        return edges.long()