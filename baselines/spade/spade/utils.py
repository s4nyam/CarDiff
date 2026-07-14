import random
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple

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


def save_image(tensor: torch.Tensor, path: Path) -> None:
    """Save a tensor image in the range [-1, 1] to disk."""
    tensor = tensor.detach().cpu().clamp(-1.0, 1.0)
    tensor = (tensor + 1.0) / 2.0
    array = (tensor.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
    image = Image.fromarray(array)
    image.save(str(path))


def rgb_to_int(rgb: np.ndarray) -> np.ndarray:
    """Pack an HxWx3 RGB array into a single 32-bit integer array."""
    if rgb.ndim != 3 or rgb.shape[2] < 3:
        raise ValueError("Expected an RGB array with shape (H, W, 3)")
    rgb = rgb[:, :, :3].astype(np.int64)
    value = (rgb[:, :, 0] << 16) | (rgb[:, :, 1] << 8) | rgb[:, :, 2]
    return value


def ints_to_rgb(values: Iterable[int]) -> Iterable[tuple[int, int, int]]:
    """Convert packed color integers back to RGB tuples."""
    for value in values:
        r = (value >> 16) & 255
        g = (value >> 8) & 255
        b = value & 255
        yield (r, g, b)


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Convert a tensor in [-1, 1] to a PIL image."""
    tensor = tensor.detach().cpu().clamp(-1.0, 1.0)
    tensor = (tensor + 1.0) / 2.0
    array = (tensor.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
    return Image.fromarray(array)


def mask_to_color_image(mask: torch.Tensor,
                        palette: Dict[int, Tuple[int, int, int]],
                        default_color: Tuple[int, int, int] = (127, 127, 127)) -> Image.Image:
    """Convert a class-index mask tensor to a colorized PIL image using the provided palette."""
    if mask.dim() == 3 and mask.size(0) == 1:
        mask = mask.squeeze(0)
    array = mask.detach().cpu().numpy().astype(np.int64)

    height, width = array.shape
    color_image = np.zeros((height, width, 3), dtype=np.uint8)
    color_image[:, :] = np.array(default_color, dtype=np.uint8)

    for idx, color in palette.items():
        idx = int(idx)
        color = tuple(int(c) for c in color)
        mask_region = array == idx
        if np.any(mask_region):
            color_image[mask_region] = color

    return Image.fromarray(color_image)


def save_mosaic(images: Sequence[Image.Image], grid_size: Tuple[int, int], path: Path,
                background_color: Tuple[int, int, int] = (0, 0, 0)) -> None:
    """Save a list of PIL images arranged into a grid mosaic."""
    if not images:
        raise ValueError("No images provided for mosaic generation.")

    rows, cols = grid_size
    max_items = rows * cols
    tiles = list(images)[:max_items]

    tile_width, tile_height = tiles[0].size
    canvas = Image.new('RGB', (cols * tile_width, rows * tile_height), color=background_color)

    for idx, image in enumerate(tiles):
        if image.size != (tile_width, tile_height):
            image = image.resize((tile_width, tile_height), Image.BILINEAR)
        row = idx // cols
        col = idx % cols
        canvas.paste(image, (col * tile_width, row * tile_height))

    canvas.save(str(path))
