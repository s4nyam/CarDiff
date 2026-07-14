"""
CarDiff Loss Functions
=======================

Collects all loss terms used during dual-path training.

Path A (paired):
    L_diff       – standard diffusion MSE on noise prediction
    L_seg        – segmentation consistency  Dice(S(X_hat), M)
    L_adv_I      – image-realism adversarial hinge loss (D_i)
    L_adv_P      – pair-correspondence adversarial hinge loss (D_p)

Path B (self-supervised):
    L_diff_B     – diffusion loss (same formulation, new mask)
    L_self_seg   – self-consistency  Dice(S(X_tilde), M_tilde)
    L_adv_I_B    – image-realism for self-supervised path
    L_adv_P_B    – pair-realism for self-supervised path

Regularizers (causal module):
    L_sparse     – ℓ1 penalty on GNN attention coefficients
    L_smooth     – spatial smoothness on neighbouring node embeddings
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ======================================================================
# Dice Loss (used for segmentation consistency)
# ======================================================================

def dice_loss(logits: torch.Tensor, target: torch.Tensor, num_classes: int = 4, smooth: float = 1.0) -> torch.Tensor:
    """
    Soft Dice loss.

    Parameters
    ----------
    logits : (B, C, H, W) – raw segmentation logits.
    target : (B, 1, H, W) – mask with class indices in [0, C-1]
             (float, typically normalised as val/255).
    """
    # Convert target to class indices
    target_idx = (target.squeeze(1) * 255).long()  # (B, H, W)
    target_idx = target_idx.clamp(0, num_classes - 1)
    target_onehot = F.one_hot(target_idx, num_classes).permute(0, 3, 1, 2).float()  # (B, C, H, W)

    probs = F.softmax(logits, dim=1)
    dims = (0, 2, 3)
    intersection = (probs * target_onehot).sum(dim=dims)
    union = probs.sum(dim=dims) + target_onehot.sum(dim=dims)
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return 1.0 - dice.mean()


# ======================================================================
# Adversarial Hinge Losses
# ======================================================================

def hinge_loss_d(real_logits: torch.Tensor, fake_logits: torch.Tensor) -> torch.Tensor:
    """Discriminator hinge loss."""
    return (F.relu(1.0 - real_logits).mean() + F.relu(1.0 + fake_logits).mean()) / 2.0


def hinge_loss_g(fake_logits: torch.Tensor) -> torch.Tensor:
    """Generator hinge loss (maximise discriminator score)."""
    return -fake_logits.mean()


# ======================================================================
# GNN Regularizers
# ======================================================================

def sparsity_loss(alphas: list) -> torch.Tensor:
    """ℓ1 penalty on attention coefficients."""
    total = torch.tensor(0.0, device=alphas[0].device)
    for a in alphas:
        total = total + a.abs().sum()
    return total / len(alphas)


def smoothness_loss(h_locals: list, grid_shapes: list) -> torch.Tensor:
    """Squared ℓ2 distance between neighbouring node embeddings on the grid."""
    total = torch.tensor(0.0, device=h_locals[0].device)
    count = 0
    for h, (gH, gW) in zip(h_locals, grid_shapes):
        # h: (N, D) with N = gH * gW
        h_grid = h.view(gH, gW, -1)
        # Horizontal neighbours
        if gW > 1:
            diff_h = (h_grid[:, :-1] - h_grid[:, 1:]).pow(2).sum()
            total = total + diff_h
            count += (gH * (gW - 1))
        # Vertical neighbours
        if gH > 1:
            diff_v = (h_grid[:-1, :] - h_grid[1:, :]).pow(2).sum()
            total = total + diff_v
            count += ((gH - 1) * gW)
    return total / max(count, 1)


# ======================================================================
# Combined Loss
# ======================================================================

class CarDiffLoss(nn.Module):
    """
    Aggregates all CarDiff losses with configurable weights.
    """

    def __init__(
        self,
        num_classes: int = 4,
        w_diff: float = 1.0,
        w_seg: float = 1.0,
        w_adv_i: float = 0.1,
        w_adv_p: float = 0.1,
        w_self_seg: float = 1.0,
        w_sparse: float = 0.01,
        w_smooth: float = 0.01,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.w_diff = w_diff
        self.w_seg = w_seg
        self.w_adv_i = w_adv_i
        self.w_adv_p = w_adv_p
        self.w_self_seg = w_self_seg
        self.w_sparse = w_sparse
        self.w_smooth = w_smooth

    def generator_loss(
        self,
        # Path A
        noise: torch.Tensor,
        noise_pred: torch.Tensor,
        seg_logits_a: torch.Tensor,
        mask_a: torch.Tensor,
        fake_logits_di_a: torch.Tensor,
        fake_logits_dp_a: torch.Tensor,
        # Path B
        noise_b: torch.Tensor | None = None,
        noise_pred_b: torch.Tensor | None = None,
        seg_logits_b: torch.Tensor | None = None,
        mask_b: torch.Tensor | None = None,
        fake_logits_di_b: torch.Tensor | None = None,
        fake_logits_dp_b: torch.Tensor | None = None,
        # Causal regularizers
        alphas: list | None = None,
        h_locals: list | None = None,
        grid_shapes: list | None = None,
    ) -> dict:
        """
        Compute total generator loss.

        Returns dict with individual loss terms and 'total'.
        """
        losses = {}

        # Path A: Diffusion
        l_diff = F.mse_loss(noise_pred, noise)
        losses["diff_a"] = l_diff

        # Path A: Segmentation consistency
        l_seg = dice_loss(seg_logits_a, mask_a, self.num_classes)
        losses["seg_a"] = l_seg

        # Path A: Adversarial (generator side)
        l_adv_i_a = hinge_loss_g(fake_logits_di_a)
        l_adv_p_a = hinge_loss_g(fake_logits_dp_a)
        losses["adv_i_a"] = l_adv_i_a
        losses["adv_p_a"] = l_adv_p_a

        total = (
            self.w_diff * l_diff
            + self.w_seg * l_seg
            + self.w_adv_i * l_adv_i_a
            + self.w_adv_p * l_adv_p_a
        )

        # Path B (if available)
        if noise_b is not None and noise_pred_b is not None:
            l_diff_b = F.mse_loss(noise_pred_b, noise_b)
            losses["diff_b"] = l_diff_b
            total = total + self.w_diff * l_diff_b

        if seg_logits_b is not None and mask_b is not None:
            l_self_seg = dice_loss(seg_logits_b, mask_b, self.num_classes)
            losses["self_seg_b"] = l_self_seg
            total = total + self.w_self_seg * l_self_seg

        if fake_logits_di_b is not None:
            l_adv_i_b = hinge_loss_g(fake_logits_di_b)
            losses["adv_i_b"] = l_adv_i_b
            total = total + self.w_adv_i * l_adv_i_b

        if fake_logits_dp_b is not None:
            l_adv_p_b = hinge_loss_g(fake_logits_dp_b)
            losses["adv_p_b"] = l_adv_p_b
            total = total + self.w_adv_p * l_adv_p_b

        # Causal regularizers
        if alphas is not None:
            l_sparse = sparsity_loss(alphas)
            losses["sparse"] = l_sparse
            total = total + self.w_sparse * l_sparse

        if h_locals is not None and grid_shapes is not None:
            l_smooth = smoothness_loss(h_locals, grid_shapes)
            losses["smooth"] = l_smooth
            total = total + self.w_smooth * l_smooth

        losses["total"] = total
        return losses

    def discriminator_loss(
        self,
        real_logits_di: torch.Tensor,
        fake_logits_di: torch.Tensor,
        real_logits_dp: torch.Tensor,
        fake_logits_dp: torch.Tensor,
    ) -> dict:
        """
        Compute discriminator losses for D_i and D_p.
        """
        l_di = hinge_loss_d(real_logits_di, fake_logits_di)
        l_dp = hinge_loss_d(real_logits_dp, fake_logits_dp)
        return {
            "d_i": l_di,
            "d_p": l_dp,
            "total": l_di + l_dp,
        }
