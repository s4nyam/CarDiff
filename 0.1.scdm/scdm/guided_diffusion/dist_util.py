"""Minimal single-process distributed utilities for local training."""

from __future__ import annotations

from typing import Iterable

import torch as th


def setup_dist(gpus_per_node: int = 1) -> None:  # noqa: ARG001 - kept for API compatibility
    """No-op setup helper retained for API compatibility."""
    return


def dev() -> th.device:
    """Return the preferred device for training."""
    return th.device("cuda" if th.cuda.is_available() else "cpu")


def load_state_dict(path, **kwargs):  # type: ignore[override]
    """Thin wrapper around :func:`torch.load` for single-process training."""
    return th.load(path, **kwargs)


def sync_params(params: Iterable[th.Tensor]) -> None:
    """Single-process variant – parameters are already in sync."""
    return
