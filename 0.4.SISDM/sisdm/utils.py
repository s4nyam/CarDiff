from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple, Union, List

import numpy as np
import torch
from PIL import Image


def set_seed(seed: int) -> None:
    """Seed random number generators for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: Path) -> None:
    """Create a directory (and parents) if it does not exist."""
    Path(path).mkdir(parents=True, exist_ok=True)


def resolve_device(device_str: Optional[str] = None) -> torch.device:
    """Resolve a torch.device from a user-supplied string with validation."""
    if device_str is None:
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    normalized = device_str.strip().lower()
    if normalized in {'', 'auto'}:
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    try:
        device = torch.device(device_str)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise ValueError(f"Invalid device specifier '{device_str}'.") from exc

    if device.type == 'cuda':
        if not torch.cuda.is_available():
            raise ValueError("CUDA device requested but no CUDA runtime is available.")
        if device.index is not None:
            device_count = torch.cuda.device_count()
            if device_count == 0 or device.index >= device_count:
                raise ValueError(
                    f"Requested CUDA device index {device.index} but only {device_count} device(s) detected."
                )

    return device


def masks_to_onehot(masks: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Convert integer segmentation masks to one-hot tensors."""
    if masks.dim() == 3:
        batch, height, width = masks.shape
    elif masks.dim() == 4 and masks.size(1) == 1:
        batch, _, height, width = masks.shape
        masks = masks.squeeze(1)
    else:
        raise ValueError("Expected mask tensor of shape (B, H, W) or (B, 1, H, W)")

    one_hot = torch.nn.functional.one_hot(masks.long(), num_classes=num_classes)
    one_hot = one_hot.permute(0, 3, 1, 2).float()
    return one_hot


def mask_to_edge_map(masks: torch.Tensor) -> torch.Tensor:
    """Convert segmentation masks to edge maps using simple gradient-based edge detection."""
    # Ensure masks are float for gradient computation
    if masks.dim() == 3:
        masks = masks.unsqueeze(1)  # Add channel dim: (B, H, W) -> (B, 1, H, W)
    
    masks_float = masks.float()
    
    # Sobel-style edge detection 
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
    
    sobel_x = sobel_x.to(masks.device)
    sobel_y = sobel_y.to(masks.device)
    
    # Apply convolution with padding
    edge_x = torch.nn.functional.conv2d(masks_float, sobel_x, padding=1)
    edge_y = torch.nn.functional.conv2d(masks_float, sobel_y, padding=1)
    
    # Compute edge magnitude
    edges = torch.sqrt(edge_x**2 + edge_y**2)
    
    # Normalize edges to [0, 1]
    edges = torch.clamp(edges / edges.max() if edges.max() > 0 else edges, 0, 1)
    
    return edges


def compute_diffusion_input_conditioning(masks: torch.Tensor, config) -> torch.Tensor:
    """Compute conditioning tensor following SCDM pattern.
    
    Args:
        masks: Segmentation masks tensor of shape [B, H, W] with integer labels
        config: TrainingConfig with segmentation class count and edge usage
        
    Returns:
        Conditioning tensor of shape [B, C, H, W]
    """
    # Convert masks to one-hot encoding using segmentation classes only
    onehot = masks_to_onehot(masks, config.label_nc)
    
    conditioning_tensors = [onehot]
    
    if config.condition_with_edges:
        edges = mask_to_edge_map(masks)
        conditioning_tensors.append(edges)
    
    # Concatenate along channel dimension
    conditioning = torch.cat(conditioning_tensors, dim=1)
    if conditioning.shape[1] != config.conditioning_channels:
        raise ValueError(
            "Conditioning channel mismatch: got"
            f" {conditioning.shape[1]} but expected {config.conditioning_channels}."
        )
    
    return conditioning


def _denormalize_image(tensor: torch.Tensor) -> torch.Tensor:
    """Convert tensor in [-1, 1] to [0, 1] range."""
    tensor = tensor.detach().cpu()
    return tensor.clamp(-1.0, 1.0).add(1.0).div(2.0)


def tensor_to_pil_image(tensor: torch.Tensor, denormalize: bool = True) -> Image.Image:
    """Convert a CHW tensor to a PIL RGB image."""
    if tensor.dim() == 4:
        if tensor.size(0) != 1:
            raise ValueError("Expected single image tensor, got batch of size > 1")
        tensor = tensor.squeeze(0)

    if tensor.dim() != 3 or tensor.size(0) not in (1, 3):
        raise ValueError("Expected image tensor with shape (3,H,W) or (1,H,W)")

    if denormalize:
        tensor = _denormalize_image(tensor)
    else:
        tensor = tensor.detach().cpu()

    if tensor.size(0) == 1:
        tensor = tensor.repeat(3, 1, 1)

    array = (tensor.clamp(0.0, 1.0).permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
    return Image.fromarray(array)


def mask_to_color_image(
    mask: torch.Tensor,
    size: Optional[Tuple[int, int]] = None,
    palette: Optional[Union[Dict[int, Sequence[int]], Sequence[Sequence[int]], Sequence[int]]] = None,
    default_color: Tuple[int, int, int] = (127, 127, 127),
) -> Image.Image:
    """Map mask tensor to a color image for visualization."""
    mask_np = mask.detach().cpu().numpy()
    if mask_np.ndim == 3:
        if mask_np.shape[0] == 1:
            mask_np = mask_np[0]
        else:
            mask_np = mask_np.argmax(axis=0)

    mask_np = mask_np.astype(np.int32)
    unique_vals = sorted(int(v) for v in np.unique(mask_np))

    color_image = np.zeros((*mask_np.shape, 3), dtype=np.uint8)
    color_image[:, :] = np.array(default_color, dtype=np.uint8)

    base_palette = [
        (0, 0, 0),
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
        (255, 255, 0),
        (255, 0, 255),
        (0, 255, 255),
        (255, 128, 0),
        (128, 0, 255),
        (0, 128, 255),
    ]

    if palette is None:
        palette_mapping = {value: base_palette[idx % len(base_palette)] for idx, value in enumerate(unique_vals)}
    elif isinstance(palette, dict):
        palette_mapping = {
            int(key): tuple(int(c) for c in value)
            for key, value in palette.items()
        }
    else:
        palette_list = list(palette)
        palette_mapping = {}
        for idx, value in enumerate(unique_vals):
            color = palette_list[idx % len(palette_list)]
            if isinstance(color, Sequence) and len(color) >= 3:
                palette_mapping[value] = tuple(int(c) for c in color[:3])
            else:
                palette_mapping[value] = base_palette[idx % len(base_palette)]

    for value in unique_vals:
        color = palette_mapping.get(value)
        if color is not None:
            color_image[mask_np == value] = color

    pil_img = Image.fromarray(color_image, mode='RGB')
    if size is not None and pil_img.size != size:
        pil_img = pil_img.resize(size, Image.NEAREST)
    return pil_img


def create_visualization_grid(
    images: torch.Tensor,
    masks: Optional[torch.Tensor] = None,
    grid_size: Tuple[int, int] = (2, 4),
    overlay_alpha: float = 0.4,
) -> torch.Tensor:
    """Create a tiled grid of generated images with optional mask overlays."""
    if images.dim() != 4:
        raise ValueError("Expected images tensor with shape (N, C, H, W)")

    total_slots = grid_size[0] * grid_size[1]
    count = min(images.size(0), total_slots)
    if count == 0:
        raise ValueError("No images available to create a visualization grid")

    pil_images = []
    for idx in range(count):
        pil_img = tensor_to_pil_image(images[idx], denormalize=True)
        if masks is not None:
            mask_tensor = masks[idx]
            mask_pil = mask_to_color_image(mask_tensor, size=pil_img.size)
            pil_img = Image.blend(pil_img, mask_pil, alpha=overlay_alpha)
        pil_images.append(pil_img)

    tile_w, tile_h = pil_images[0].size
    grid_width = grid_size[1] * tile_w
    grid_height = grid_size[0] * tile_h
    canvas = Image.new('RGB', (grid_width, grid_height))

    for idx, img in enumerate(pil_images):
        row = idx // grid_size[1]
        col = idx % grid_size[1]
        canvas.paste(img, (col * tile_w, row * tile_h))

    canvas_np = np.array(canvas).astype(np.float32) / 255.0
    canvas_tensor = torch.from_numpy(canvas_np).permute(2, 0, 1)
    return canvas_tensor


def save_mosaic(
    images: Sequence[Image.Image],
    grid_size: Tuple[int, int],
    path: Path,
    background_color: Tuple[int, int, int] = (0, 0, 0),
) -> None:
    """Save images arranged in a grid mosaic similar to SCDM output."""
    if not images:
        raise ValueError("No images provided for mosaic generation")

    rows, cols = grid_size
    max_items = rows * cols
    tiles = list(images)[:max_items]

    tile_width, tile_height = tiles[0].size
    canvas = Image.new('RGB', (cols * tile_width, rows * tile_height), color=background_color)

    for idx, img in enumerate(tiles):
        if img.size != (tile_width, tile_height):
            img = img.resize((tile_width, tile_height), Image.BILINEAR)
        row = idx // cols
        col = idx % cols
        canvas.paste(img, (col * tile_width, row * tile_height))

    ensure_dir(path.parent)
    canvas.save(path)


def tensor_to_pil_image(tensor: torch.Tensor, denormalize: bool = True) -> Image.Image:
    """Convert a CHW tensor to a PIL RGB image."""
    if tensor.dim() == 4:
        if tensor.size(0) != 1:
            raise ValueError("Expected single image tensor, got batch of size > 1")
        tensor = tensor.squeeze(0)

    if tensor.dim() != 3 or tensor.size(0) not in (1, 3):
        raise ValueError("Expected image tensor with shape (3,H,W) or (1,H,W)")

    if denormalize:
        tensor = _denormalize_image(tensor)
    else:
        tensor = tensor.detach().cpu()

    if tensor.size(0) == 1:
        tensor = tensor.repeat(3, 1, 1)

    array = (tensor.clamp(0.0, 1.0).permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
    return Image.fromarray(array)


alias_tensor_to_pil = tensor_to_pil_image  # Alias for compatibility


def rgb_to_int(rgb_array: np.ndarray) -> np.ndarray:
    """Convert RGB array to packed integer representation."""
    if rgb_array.ndim == 3 and rgb_array.shape[2] == 3:
        return (rgb_array[:, :, 0].astype(np.int32) << 16) + \
               (rgb_array[:, :, 1].astype(np.int32) << 8) + \
               rgb_array[:, :, 2].astype(np.int32)
    else:
        raise ValueError("Expected RGB array with shape (H, W, 3)")


def ints_to_rgb(packed_ints: Sequence[int]) -> List[Tuple[int, int, int]]:
    """Convert packed integers back to RGB tuples."""
    rgb_list = []
    for packed in packed_ints:
        r = (packed >> 16) & 0xFF
        g = (packed >> 8) & 0xFF
        b = packed & 0xFF
        rgb_list.append((r, g, b))
    return rgb_list
    """Save a tensor or PIL image to disk."""
    ensure_dir(path.parent)

    if isinstance(image, Image.Image):
        pil_image = image
    elif isinstance(image, torch.Tensor):
        pil_image = tensor_to_pil_image(image, denormalize=denormalize)
    else:
        raise TypeError("Expected image to be a torch.Tensor or PIL.Image.Image")

    pil_image.save(path)
