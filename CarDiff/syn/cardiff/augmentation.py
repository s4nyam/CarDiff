"""
Mask Augmentation  A(·)  for Self-Supervised Path B
=====================================================

Generates novel masks  M_tilde = A(M)  using morphological and
geometric transformations:

- Dilation / Erosion  (change caries extent)
- Elastic deformation (change caries shape)
- CutMix             (recombine caries patches across masks)
- Random placement    (translate / mirror lesion patches)

These create variations in caries shape, severity, and position
while preserving anatomical plausibility.
"""

from __future__ import annotations

import random
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F


# ======================================================================
# Individual transforms
# ======================================================================

def _morphological_op(
    mask: np.ndarray, op: str = "dilate", kernel_size: int = 3
) -> np.ndarray:
    """Dilate or erode *each* nonzero class independently."""
    import cv2

    out = mask.copy()
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    for cls_val in np.unique(mask):
        if cls_val == 0:
            continue
        binary = (mask == cls_val).astype(np.uint8)
        if op == "dilate":
            binary = cv2.dilate(binary, kernel, iterations=1)
        elif op == "erode":
            binary = cv2.erode(binary, kernel, iterations=1)
        # Write back, but don't overwrite other classes
        out[binary == 1] = cls_val
    return out


def _elastic_deform(
    mask: np.ndarray,
    alpha: float = 8.0,
    sigma: float = 3.0,
) -> np.ndarray:
    """Apply random elastic deformation (nearest-interp to keep labels)."""
    import cv2

    H, W = mask.shape[:2]
    dx = cv2.GaussianBlur((np.random.rand(H, W).astype(np.float32) * 2 - 1), (0, 0), sigma) * alpha
    dy = cv2.GaussianBlur((np.random.rand(H, W).astype(np.float32) * 2 - 1), (0, 0), sigma) * alpha
    x, y = np.meshgrid(np.arange(W), np.arange(H))
    mapx = (x + dx).astype(np.float32)
    mapy = (y + dy).astype(np.float32)
    return cv2.remap(mask, mapx, mapy, interpolation=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT)


def _cutmix(
    mask: np.ndarray,
    donor_mask: Optional[np.ndarray] = None,
    num_patches: int = 1,
) -> np.ndarray:
    """Paste random rectangular lesion patches from `donor_mask`."""
    if donor_mask is None:
        donor_mask = mask
    H, W = mask.shape[:2]
    out = mask.copy()
    for _ in range(num_patches):
        ph = random.randint(H // 8, H // 3)
        pw = random.randint(W // 8, W // 3)
        sy = random.randint(0, H - ph)
        sx = random.randint(0, W - pw)
        dy = random.randint(0, H - ph)
        dx = random.randint(0, W - pw)
        patch = donor_mask[sy : sy + ph, sx : sx + pw]
        if patch.any():
            out[dy : dy + ph, dx : dx + pw] = patch
    return out


def _random_placement(mask: np.ndarray) -> np.ndarray:
    """Extract connected components and randomly re-place them."""
    import cv2

    H, W = mask.shape[:2]
    out = np.zeros_like(mask)
    for cls_val in np.unique(mask):
        if cls_val == 0:
            continue
        binary = (mask == cls_val).astype(np.uint8)
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary)
        for lbl in range(1, n_labels):
            comp = (labels == lbl).astype(np.uint8)
            y, x = np.where(comp)
            if len(y) == 0:
                continue
            ch, cw = y.max() - y.min() + 1, x.max() - x.min() + 1
            patch = comp[y.min() : y.min() + ch, x.min() : x.min() + cw]
            ny = random.randint(0, max(0, H - ch))
            nx = random.randint(0, max(0, W - cw))
            region = out[ny : ny + ch, nx : nx + cw]
            region[patch == 1] = cls_val
    return out


# ======================================================================
# Public API
# ======================================================================

class MaskAugmentation:
    """
    Applies a random subset of morphological and geometric
    augmentations to produce A(M).

    Parameters
    ----------
    p_morph : float
        Probability of applying a morphological op (dilate or erode).
    p_elastic : float
        Probability of applying elastic deformation.
    p_cutmix : float
        Probability of applying CutMix.
    p_place : float
        Probability of random re-placement.
    """

    def __init__(
        self,
        p_morph: float = 0.5,
        p_elastic: float = 0.5,
        p_cutmix: float = 0.3,
        p_place: float = 0.3,
    ):
        self.p_morph = p_morph
        self.p_elastic = p_elastic
        self.p_cutmix = p_cutmix
        self.p_place = p_place

    def __call__(self, mask_tensor: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        mask_tensor : (1, H, W) float tensor, values in [0, 1].

        Returns
        -------
        augmented : same shape and dtype.
        """
        # Work in uint8 numpy (class values 0-255)
        mask_np = (mask_tensor.squeeze(0).cpu().numpy() * 255).astype(np.uint8)

        if random.random() < self.p_morph:
            op = random.choice(["dilate", "erode"])
            ks = random.choice([3, 5])
            mask_np = _morphological_op(mask_np, op, ks)

        if random.random() < self.p_elastic:
            mask_np = _elastic_deform(mask_np, alpha=random.uniform(4, 12), sigma=random.uniform(2, 5))

        if random.random() < self.p_cutmix:
            mask_np = _cutmix(mask_np)

        if random.random() < self.p_place:
            mask_np = _random_placement(mask_np)

        out = torch.from_numpy(mask_np.astype(np.float32) / 255.0).unsqueeze(0)
        return out.to(mask_tensor.device)

    def augment_batch(self, masks: torch.Tensor) -> torch.Tensor:
        """Apply augmentation to a batch (B, 1, H, W)."""
        return torch.stack([self(m) for m in masks])
