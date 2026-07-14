"""
Causally-Aware Learning Module
================================

Implements the two-world causal system described in the paper:

1. **Patch-Graph GNN** – The feature map F = C_m(M) is divided into
   non-overlapping patches treated as graph nodes.  Attention-based
   message-passing (L=2 layers) captures local structural cause-effect
   interactions (e.g., enamel mineral loss → reduced radiodensity).

2. **Global Attention Block (GAB)** – A single self-attention layer
   over all node embeddings provides long-range anatomical context.

3. **Hybrid Fusion & Attention Pooling** – Per-node local and global
   features are fused and attention-pooled into a single causal context
   vector  C_hyb.

4. **FiLM Conditioning** – C_hyb (concatenated with the raw mask
   encoding) modulates U-Net intermediate features via per-channel
   scale γ and shift β.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ======================================================================
# 1.  Patch-Graph GNN
# ======================================================================

class PatchGraphGNN(nn.Module):
    """
    Builds a patch graph from the mask feature map and performs
    attention-based message passing.

    Each non-overlapping patch becomes a node with initial features:
        x_i = [ mean(F_{P_i}) || ρ_i || c_i ]
    where ρ_i is the lesion-pixel fraction and c_i ∈ {0,1}^3 is a
    one-hot encoding over {SC, MC, DC}.

    Edges connect 8-neighbours on the grid **plus** k-nearest neighbours
    in feature space.  For lesion-overlapping nodes, edges are directed
    outward to model causal radiopathological flow.

    Parameters
    ----------
    feat_dim : int
        Dimension of per-patch mean-pooled features from C_m output.
    num_classes : int
        Number of lesion classes (excluding background).  Default 3 for
        {SC, MC, DC}.
    hidden_dim : int
        Hidden dimension of the GNN.
    num_layers : int
        Number of message-passing layers (L in the paper; default 2).
    patch_size : int
        Side length p of each square patch.
    knn_k : int
        Number of nearest neighbours in feature space to add edges.
    """

    def __init__(
        self,
        feat_dim: int = 256,
        num_classes: int = 3,
        hidden_dim: int = 256,
        num_layers: int = 2,
        patch_size: int = 8,
        knn_k: int = 4,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.knn_k = knn_k
        self.num_classes = num_classes
        self.num_layers = num_layers

        # Node input: mean-pooled features + lesion fraction + one-hot class
        node_in_dim = feat_dim + 1 + num_classes
        self.mlp_in = nn.Sequential(
            nn.Linear(node_in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Per-layer attention + message modules
        self.attn_mlps = nn.ModuleList()
        self.msg_projs = nn.ModuleList()
        self.update_mlps = nn.ModuleList()
        edge_feat_dim = 2  # spatial distance + lesion transition flag

        for _ in range(num_layers):
            # ψ(h_i, h_j, e_{ij})  →  attention logit
            self.attn_mlps.append(nn.Sequential(
                nn.Linear(2 * hidden_dim + edge_feat_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, 1),
            ))
            # W_m
            self.msg_projs.append(nn.Linear(hidden_dim, hidden_dim, bias=False))
            # φ(h_i, aggregated messages)
            self.update_mlps.append(nn.Sequential(
                nn.Linear(2 * hidden_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, hidden_dim),
            ))

        self.hidden_dim = hidden_dim

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _grid_neighbours(H: int, W: int, device: torch.device):
        """Return edge index (2, E) for 8-connected grid of size H×W."""
        idx = torch.arange(H * W, device=device).view(H, W)
        edges_src, edges_dst = [], []
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                if di == 0 and dj == 0:
                    continue
                si = max(0, -di); ei = H - max(0, di)
                sj = max(0, -dj); ej = W - max(0, dj)
                src = idx[si:ei, sj:ej].reshape(-1)
                dst = idx[si + di:ei + di, sj + dj:ej + dj].reshape(-1)
                edges_src.append(src)
                edges_dst.append(dst)
        return torch.stack([torch.cat(edges_src), torch.cat(edges_dst)])

    def _knn_edges(self, feats: torch.Tensor, grid_H: int, grid_W: int):
        """Add k-nearest-neighbour edges in feature space."""
        N = feats.shape[0]
        if N <= self.knn_k:
            return torch.empty(2, 0, device=feats.device, dtype=torch.long)
        dists = torch.cdist(feats.unsqueeze(0), feats.unsqueeze(0)).squeeze(0)
        # exclude self
        dists.fill_diagonal_(float("inf"))
        _, knn_idx = dists.topk(self.knn_k, largest=False)
        src = torch.arange(N, device=feats.device).unsqueeze(1).expand_as(knn_idx).reshape(-1)
        dst = knn_idx.reshape(-1)
        return torch.stack([src, dst])

    # ------------------------------------------------------------------
    def _build_graph(self, F_map: torch.Tensor, mask: torch.Tensor):
        """
        Parameters
        ----------
        F_map : (B, C, H', W')  output of mask encoder.
        mask  : (B, 1, H, W)    raw mask (class values in [0,1]).

        Returns per-sample: node_feats (N, d_in), edge_index (2, E),
                            edge_attr (E, 2), grid_shape (gH, gW).
        """
        B, C, Hf, Wf = F_map.shape
        p = min(self.patch_size, Hf, Wf)  # clamp to feature map size
        gH, gW = max(1, Hf // p), max(1, Wf // p)

        # Resize mask to feature-map resolution for ρ and class extraction
        mask_resized = F.interpolate(mask, size=(Hf, Wf), mode="nearest")

        graphs = []
        for b in range(B):
            f = F_map[b]  # (C, Hf, Wf)
            m = mask_resized[b, 0]  # (Hf, Wf)

            # Patch mean-pool features
            # Handle case where feature map is smaller than or equal to patch size
            if Hf <= p and Wf <= p:
                # Single patch covering the entire feature map
                node_mean = f.mean(dim=[1, 2]).unsqueeze(0)  # (1, C)
            else:
                actual_pH = min(p, Hf)
                actual_pW = min(p, Wf)
                patches_f = f.unfold(1, actual_pH, actual_pH).unfold(2, actual_pW, actual_pW)  # (C, gH, gW, pH, pW)
                gH_actual, gW_actual = patches_f.shape[1], patches_f.shape[2]
                patches_f = patches_f.contiguous().view(C, gH_actual * gW_actual, actual_pH * actual_pW)
                node_mean = patches_f.mean(dim=2).T  # (N, C)
                gH, gW = gH_actual, gW_actual

            # Lesion fraction ρ_i per patch
            N = gH * gW
            if Hf <= p and Wf <= p:
                rho = (m > 0).float().mean().unsqueeze(0).unsqueeze(1)  # (1, 1)
                patches_m_flat = m.reshape(1, -1)
            else:
                actual_pH = min(p, Hf)
                actual_pW = min(p, Wf)
                patches_m = m.unfold(0, actual_pH, actual_pH).unfold(1, actual_pW, actual_pW)  # (gH, gW, pH, pW)
                patches_m_flat = patches_m.contiguous().view(N, actual_pH * actual_pW)
                rho = (patches_m_flat > 0).float().mean(dim=1, keepdim=True)  # (N, 1)

            # One-hot class vector c_i (dominant nonzero class)
            # Classes encoded as SC≈0.4(102/255), MC≈0.6(153/255), DC≈1.0(255/255)
            c_onehot = torch.zeros(N, self.num_classes, device=f.device)
            for ni in range(N):
                vals = patches_m_flat[ni]
                nonzero = vals[vals > 0]
                if len(nonzero) > 0:
                    dominant = nonzero.median()
                    if dominant < 0.45:
                        c_onehot[ni, 0] = 1.0  # SC
                    elif dominant < 0.75:
                        c_onehot[ni, 1] = 1.0  # MC
                    else:
                        c_onehot[ni, 2] = 1.0  # DC

            # Concatenate node input
            x_i = torch.cat([node_mean, rho, c_onehot], dim=1)  # (N, C+1+3)

            # Edges: 8-grid + KNN
            grid_edges = self._grid_neighbours(gH, gW, f.device)
            knn_edges = self._knn_edges(node_mean, gH, gW)

            if grid_edges.shape[1] > 0 or knn_edges.shape[1] > 0:
                edge_index = torch.cat([grid_edges, knn_edges], dim=1)
                # Remove duplicate edges
                edge_set = set()
                keep = []
                for ei in range(edge_index.shape[1]):
                    key = (edge_index[0, ei].item(), edge_index[1, ei].item())
                    if key not in edge_set:
                        edge_set.add(key)
                        keep.append(ei)
                edge_index = edge_index[:, keep]
            else:
                edge_index = torch.empty(2, 0, device=f.device, dtype=torch.long)

            # Edge attributes: spatial distance + lesion transition flag
            N = gH * gW
            coords = torch.stack([
                torch.arange(gH, device=f.device).unsqueeze(1).expand(gH, gW).reshape(-1).float(),
                torch.arange(gW, device=f.device).unsqueeze(0).expand(gH, gW).reshape(-1).float(),
            ], dim=1)  # (N, 2)

            if edge_index.shape[1] > 0:
                src, dst = edge_index
                spatial_dist = (coords[src] - coords[dst]).norm(dim=1, keepdim=True)
                has_lesion = (rho.squeeze(-1).view(-1) > 0).float()  # ensure 1D
                transition = (has_lesion[src] - has_lesion[dst]).abs().unsqueeze(1)
                edge_attr = torch.cat([spatial_dist, transition], dim=1)
            else:
                edge_attr = torch.empty(0, 2, device=f.device)

            graphs.append((x_i, edge_index, edge_attr, (gH, gW)))

        return graphs

    # ------------------------------------------------------------------
    def forward(self, F_map: torch.Tensor, mask: torch.Tensor):
        """
        Returns
        -------
        h_local : list[Tensor]  – per-sample local embeddings (N_b, hidden_dim).
        alpha_all : list[Tensor] – attention coefficients for sparsity loss.
        grid_shapes : list[tuple] – (gH, gW) per sample.
        """
        graphs = self._build_graph(F_map, mask)
        h_local_all = []
        alpha_all = []
        grid_shapes = []

        for x_i, edge_index, edge_attr, (gH, gW) in graphs:
            h = self.mlp_in(x_i)  # (N, hidden)
            sample_alphas = []

            if edge_index.shape[1] == 0:
                # No edges: skip message passing, keep initial embeddings
                for _ in range(self.num_layers):
                    sample_alphas.append(torch.zeros(0, device=h.device))
            else:
                for layer_idx in range(self.num_layers):
                    src, dst = edge_index
                    # Attention logits
                    cat_inp = torch.cat([h[src], h[dst], edge_attr], dim=1)
                    logits = self.attn_mlps[layer_idx](cat_inp).squeeze(-1)  # (E,)

                    # Softmax per destination node
                    alpha = torch.zeros_like(logits)
                    for ni in range(h.shape[0]):
                        mask_ni = (dst == ni)
                        if mask_ni.any():
                            alpha[mask_ni] = F.softmax(logits[mask_ni], dim=0)
                    sample_alphas.append(alpha)

                    # Messages
                    msg = alpha.unsqueeze(1) * self.msg_projs[layer_idx](h[src])  # (E, hidden)
                    agg = torch.zeros_like(h)
                    agg.index_add_(0, dst, msg)

                    # Node update
                    h = self.update_mlps[layer_idx](torch.cat([h, agg], dim=1))

            h_local_all.append(h)
            alpha_all.append(torch.cat(sample_alphas) if sample_alphas and sample_alphas[0].numel() > 0 else torch.zeros(1, device=h.device))
            grid_shapes.append((gH, gW))

        return h_local_all, alpha_all, grid_shapes


# ======================================================================
# 2.  Global Attention Block
# ======================================================================

class GlobalAttentionBlock(nn.Module):
    """
    Single multi-head self-attention layer applied over all node
    embeddings to capture long-range contextual relations (tone
    gradients, anatomical continuity).
    """

    def __init__(self, embed_dim: int = 256, num_heads: int = 4):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, h_local: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        h_local : (N, D) – node embeddings from GNN for a single sample.

        Returns
        -------
        h_global : (N, D)
        """
        x = h_local.unsqueeze(0)  # (1, N, D)
        out, _ = self.attn(x, x, x)
        return self.norm(out.squeeze(0) + h_local)


# ======================================================================
# 3.  Hybrid Fusion & Attention Pooling
# ======================================================================

class HybridFusion(nn.Module):
    """
    Fuses local GNN embeddings and global attention embeddings per node,
    then attention-pools into a single causal context vector C_hyb.
    """

    def __init__(self, hidden_dim: int = 256, out_dim: int = 256):
        super().__init__()
        self.mlp_f = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )
        # Learnable global query vector u
        self.u = nn.Parameter(torch.randn(out_dim))

    def forward(
        self, h_local: torch.Tensor, h_global: torch.Tensor
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        h_local  : (N, D)
        h_global : (N, D)

        Returns
        -------
        C_hyb : (D,) causal context vector (single sample).
        """
        g = self.mlp_f(torch.cat([h_local, h_global], dim=1))  # (N, D)
        omega = F.softmax(g @ self.u, dim=0)  # (N,)
        C_hyb = (omega.unsqueeze(1) * g).sum(dim=0)  # (D,)
        return C_hyb


# ======================================================================
# 4.  FiLM Conditioning Layers
# ======================================================================

class FiLMLayer(nn.Module):
    """
    Feature-wise Linear Modulation layer.

    Given conditioning vector  c,  produces per-channel scale γ and
    shift β to modulate a U-Net feature map:
        U'_ℓ = γ(c) ⊙ U_ℓ + β(c)
    """

    def __init__(self, cond_dim: int, num_channels: int):
        super().__init__()
        self.gamma_net = nn.Sequential(
            nn.Linear(cond_dim, cond_dim),
            nn.ReLU(inplace=True),
            nn.Linear(cond_dim, num_channels),
        )
        self.beta_net = nn.Sequential(
            nn.Linear(cond_dim, cond_dim),
            nn.ReLU(inplace=True),
            nn.Linear(cond_dim, num_channels),
        )

    def forward(self, U: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        U    : (B, C, H, W) – U-Net feature map at some layer.
        cond : (B, cond_dim) – conditioning vector.
        """
        gamma = self.gamma_net(cond).unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)
        beta = self.beta_net(cond).unsqueeze(-1).unsqueeze(-1)
        return gamma * U + beta


# ======================================================================
# 5.  Full Causal Module
# ======================================================================

class CausalModule(nn.Module):
    """
    End-to-end causal conditioning module.

    Given F_map = C_m(M) and the raw mask M, produces:
    - C_hyb  (B, hidden_dim)  -- hybrid causal context vector
    - alpha  list of attention coefficients for sparsity regularization
    - h_local list for smoothness regularization
    - grid_shapes for smoothness loss
    """

    def __init__(
        self,
        feat_dim: int = 256,
        num_classes: int = 3,
        hidden_dim: int = 256,
        num_gnn_layers: int = 2,
        patch_size: int = 8,
        knn_k: int = 4,
        num_attn_heads: int = 4,
    ):
        super().__init__()
        self.gnn = PatchGraphGNN(
            feat_dim=feat_dim,
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            num_layers=num_gnn_layers,
            patch_size=patch_size,
            knn_k=knn_k,
        )
        self.gab = GlobalAttentionBlock(hidden_dim, num_attn_heads)
        self.fusion = HybridFusion(hidden_dim, hidden_dim)
        self.hidden_dim = hidden_dim

    def forward(self, F_map: torch.Tensor, mask: torch.Tensor):
        """
        Returns
        -------
        C_hyb : (B, hidden_dim)
        alphas : list of attention coefficient tensors (for sparsity loss)
        h_locals : list of local embedding tensors (for smoothness loss)
        grid_shapes : list of (gH, gW) per sample
        """
        h_locals, alphas, grid_shapes = self.gnn(F_map, mask)

        C_hyb_list = []
        for b in range(len(h_locals)):
            h_global = self.gab(h_locals[b])
            c = self.fusion(h_locals[b], h_global)
            C_hyb_list.append(c)

        C_hyb = torch.stack(C_hyb_list, dim=0)  # (B, hidden_dim)
        return C_hyb, alphas, h_locals, grid_shapes
