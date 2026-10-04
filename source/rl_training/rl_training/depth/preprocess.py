# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause

"""Depth preprocessing shared by the simulator and the robot. Pure torch -- keep it that way.

    metres (..., H, W)  ->  preprocess()  ->  policy image (..., h, w) in [0, 1], 0 = invalid

1. invalidate non-finite, <= 0 (the camera's "no data") and closer than `near` (below Min-Z);
   clamp anything beyond `far` to `far`;
2. pool H x W -> h x w with a median over each block that ignores invalid pixels. A median returns a
   depth that exists in the block, so a block straddling a step edge gets one side or the other,
   never an in-between "ghost" depth (which an average would give). A block with fewer than
   `min_valid_frac` valid pixels becomes invalid. When H x W == h x w (the simulator casts at the
   policy resolution) pooling is the identity;
3. map near..far -> [1e-3, 1] and keep invalid at exactly 0, so the CNN can tell "no data" from "near".
"""

from __future__ import annotations

import torch

_EPS = 1e-3


def _block_indices(n_in: int, n_out: int, device) -> torch.Tensor:
    """(n_out, k) input indices of each output block; k = n_in // n_out, blocks centred on the output cell."""
    k = n_in // n_out
    centres = (torch.arange(n_out, device=device, dtype=torch.float32) + 0.5) * n_in / n_out
    starts = torch.clamp((centres - k / 2).floor().long(), 0, n_in - k)
    return starts[:, None] + torch.arange(k, device=device)[None, :]


def sanitize(depth: torch.Tensor, near: float, far: float) -> torch.Tensor:
    """Metres -> metres with 0 = invalid (non-finite, <= 0, < near); > far clamped to far."""
    d = torch.nan_to_num(depth, nan=0.0, posinf=far, neginf=0.0)
    d = torch.clamp(d, max=far)
    return torch.where(d >= near, d, torch.zeros_like(d))


def pool_depth(depth: torch.Tensor, out_h: int, out_w: int, min_valid_frac: float = 0.5) -> torch.Tensor:
    """Zero-ignoring block median, (..., H, W) -> (..., out_h, out_w). Input must already be sanitized."""
    H, W = depth.shape[-2:]
    if (H, W) == (out_h, out_w):
        return depth
    if H < out_h or W < out_w:
        raise ValueError(f"cannot pool {H}x{W} up to {out_h}x{out_w}")
    rows = _block_indices(H, out_h, depth.device)  # (out_h, kh)
    cols = _block_indices(W, out_w, depth.device)  # (out_w, kw)
    blocks = depth[..., rows[:, :, None, None], cols[None, None, :, :]]  # (..., out_h, kh, out_w, kw)
    blocks = blocks.movedim(-3, -2).flatten(-2)  # (..., out_h, out_w, kh*kw)
    valid = blocks > 0
    med = torch.nanmedian(torch.where(valid, blocks, torch.full_like(blocks, float("nan"))), dim=-1).values
    keep = valid.float().mean(dim=-1) >= min_valid_frac
    return torch.where(keep, torch.nan_to_num(med, nan=0.0), torch.zeros_like(med))


def normalize(depth: torch.Tensor, near: float, far: float) -> torch.Tensor:
    """Sanitized metres -> [0, 1]: invalid stays 0, near..far -> [1e-3, 1]."""
    img = (depth - near) / (far - near) * (1.0 - _EPS) + _EPS
    return torch.where(depth > 0, img, torch.zeros_like(depth))


def preprocess(
    depth: torch.Tensor, out_h: int, out_w: int, near: float, far: float, min_valid_frac: float = 0.5
) -> torch.Tensor:
    """Full pipeline, (..., H, W) metres -> (..., out_h, out_w) policy image."""
    d = sanitize(depth.float(), near, far)
    d = pool_depth(d, out_h, out_w, min_valid_frac)
    return normalize(d, near, far)
