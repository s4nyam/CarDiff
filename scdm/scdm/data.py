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
            # ensure deterministic ordering
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

    def encode(self, mask_image: Image.Image) -> torch.Tensor:
        array = self._mask_array(mask_image)

        if self.use_color_map:
            packed = utils.rgb_to_int(array)
            flat = packed.reshape(-1)
            try:
                mapped = np.array([self.color_to_index[int(v)] for v in flat], dtype=np.int64)
            except KeyError as exc:
                missing = int(exc.args[0])
                rgb = next(utils.ints_to_rgb([missing]))
                raise KeyError(
                    f"Encountered an unknown color {rgb} (packed={missing}) in mask."
                ) from exc
            mapped = mapped.reshape(packed.shape[:2])
        else:
            if array.ndim == 3:
                array = array[:, :, 0]
            if self.value_to_index is not None:
                flat = array.reshape(-1)
                try:
                    mapped_flat = np.array([self.value_to_index[int(v)] for v in flat], dtype=np.int64)
                except KeyError as exc:
                    missing = int(exc.args[0])
                    raise KeyError(
                        f"Encountered an unknown grayscale value {missing} in mask."
                    ) from exc
                mapped = mapped_flat.reshape(array.shape)
            else:
                mapped = array.astype(np.int64)

            if self.ignore_label is not None:
                ignore_mask = (array == self.ignore_label)
                mapped[ignore_mask] = 0

        return torch.from_numpy(mapped)

    def to_metadata(self) -> MaskEncoderMetadata:
        return MaskEncoderMetadata(
            label_nc=self.label_nc,
            ignore_label=self.ignore_label,
            color_to_index=self.color_to_index.copy() if self.color_to_index is not None else None,
            value_to_index=self.value_to_index.copy() if self.value_to_index is not None else None,
            class_values=list(self.class_values),
        )

    @classmethod
    def from_metadata(cls, metadata: MaskEncoderMetadata) -> 'MaskEncoder':
        return cls(
            label_nc=metadata.label_nc,
            ignore_label=metadata.ignore_label,
            color_to_index=metadata.color_to_index,
            value_to_index=metadata.value_to_index,
            class_values=metadata.class_values,
        )


class SegmentationDiffusionDataset(Dataset):
    """Dataset yielding paired segmentation masks and target RGB images."""

    def __init__(self,
                 images_dir: Path,
                 masks_dir: Path,
                 image_size: Tuple[int, int],
                 random_flip: bool = True,
                 label_nc: Optional[int] = None,
                 ignore_label: Optional[int] = 255) -> None:
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.image_size = (int(image_size[0]), int(image_size[1]))
        self.random_flip = random_flip
        self.ignore_label = ignore_label

        self.pairs = self._collect_pairs()
        mask_paths = [mask for _, mask in self.pairs]
        self.encoder, self.palette = MaskEncoder.analyze(mask_paths, label_nc=label_nc, ignore_label=ignore_label)
        self.label_nc = self.encoder.label_nc
        self.class_values = list(self.encoder.class_values)

    def _collect_pairs(self) -> List[Tuple[Path, Path]]:
        image_files = {path.stem: path for path in self._list_files(self.images_dir)}
        mask_files = {path.stem: path for path in self._list_files(self.masks_dir)}

        common_keys = sorted(set(image_files.keys()) & set(mask_files.keys()))
        if not common_keys:
            raise RuntimeError(
                f"No matching image/mask pairs found between {self.images_dir} and {self.masks_dir}."
            )

        missing_images = sorted(set(mask_files.keys()) - set(image_files.keys()))
        missing_masks = sorted(set(image_files.keys()) - set(mask_files.keys()))
        if missing_images:
            raise RuntimeError(f"Missing images for masks: {missing_images[:10]}...")
        if missing_masks:
            raise RuntimeError(f"Missing masks for images: {missing_masks[:10]}...")

        return [(image_files[key], mask_files[key]) for key in common_keys]

    @staticmethod
    def _list_files(directory: Path) -> List[Path]:
        exts = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}
        files = [path for path in directory.glob('*') if path.suffix.lower() in exts]
        return sorted(files)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        image_path, mask_path = self.pairs[index]

        with Image.open(image_path) as img:
            image = img.convert('RGB')
        with Image.open(mask_path) as mask_img:
            mask = mask_img.convert('RGB') if self.encoder.use_color_map else mask_img.copy()

        if self.image_size is not None:
            image = image.resize(self.image_size[::-1], Image.BICUBIC)
            mask = mask.resize(self.image_size[::-1], Image.NEAREST)

        if self.random_flip and random.random() < 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        image_tensor = TF.to_tensor(image) * 2.0 - 1.0
        mask_tensor = self.encoder.encode(mask)

        sample = {
            'image': image_tensor,
            'mask': mask_tensor,
            'image_path': str(image_path),
            'mask_path': str(mask_path),
        }
        return sample

    def metadata(self) -> Dict[str, object]:
        encoder_meta = self.encoder.to_metadata()
        data = {
            'encoder': {
                'label_nc': encoder_meta.label_nc,
                'ignore_label': encoder_meta.ignore_label,
                'color_to_index': encoder_meta.color_to_index,
                'value_to_index': encoder_meta.value_to_index,
                'class_values': encoder_meta.class_values,
            },
            'image_size': self.image_size,
            'palette': {int(k): tuple(map(int, v)) for k, v in self.palette.items()},
            'num_samples': len(self.pairs),
            'class_values': self.class_values,
        }
        return data
