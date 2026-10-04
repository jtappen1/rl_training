# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause

"""Simulator-only stereo-depth artefacts for a D435i-like camera. Pure torch.

All functions take a clean depth image `z` (N, H, W) in metres with 0 = invalid, and are applied
BEFORE `preprocess()` so the robot and the simulator share everything downstream of the sensor.

- `disparity_noise`: noise added in disparity (d = f*B/z) space, which gives the z^2 growth of stereo
  depth error for free, plus a smooth low-frequency field (the "wavy plane" of real stereo), subpixel
  quantisation, and a per-env depth scale error (calibration).
- `stereo_invalid_mask`: deterministic geometry of a horizontal-baseline stereo pair with depth
  in the left imager's frame: (a) pixels the right imager cannot see because a nearer surface to
  their right hides them (the invalid band on the LEFT side of near objects), and (b) the left image
  border that falls outside the right imager (width = disparity). The baseline is horizontal, so
  this shows up beside vertical edges (walls, step ends, stairs approached at an angle), not along
  head-on stair nosings.
- `random_invalid_mask`: random pixel dropout, rectangular holes (dark / shiny / sunlit surfaces),
  and dropped pixels at depth discontinuities (flying pixels).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def disparity_noise(
    z: torch.Tensor,
    focal_px: float,
    baseline_m: float,
    sigma_px: torch.Tensor,
    lowfreq_px: torch.Tensor,
    scale: torch.Tensor,
    quant_px: float = 1.0 / 32.0,
    lowfreq_grid: tuple[int, int] = (3, 5),
) -> torch.Tensor:
    """z (N, H, W) metres -> noisy metres; 0 stays 0, pixels whose disparity goes <= 0 become 0.

    `focal_px` is the NATIVE focal length (noise is specified in native disparity pixels, as the
    effective noise after pooling). `sigma_px`, `lowfreq_px`, `scale` are per-env (N,) tensors.
    """
    n, h, w = z.shape
    valid = z > 0
    fb = focal_px * baseline_m
    disp = fb / torch.clamp(z, min=1e-6)
    disp = disp + torch.randn_like(z) * sigma_px.view(n, 1, 1)
    if torch.any(lowfreq_px > 0):
        field = torch.randn(n, 1, *lowfreq_grid, device=z.device)
        field = F.interpolate(field, size=(h, w), mode="bilinear", align_corners=True)[:, 0]
        disp = disp + field * lowfreq_px.view(n, 1, 1)
    if quant_px > 0:
        disp = torch.round(disp / quant_px) * quant_px
    out = torch.where(disp > 0, fb / torch.clamp(disp, min=1e-6), torch.zeros_like(disp)) * scale.view(n, 1, 1)
    return torch.where(valid, out, torch.zeros_like(out))


def stereo_invalid_mask(z: torch.Tensor, focal_px: float, baseline_m: float) -> torch.Tensor:
    """(N, H, W) bool: pixels a left-referenced horizontal stereo pair cannot measure.

    `focal_px` is the focal length at the RESOLUTION OF `z` (geometry is in z's pixel grid).
    A left-image pixel u at disparity d lands at column q = u - d in the right image. It is hidden if
    some pixel further right lands at or left of it (min_{v>u} q_v < q_u), and unseen if q < -0.5.
    """
    w = z.shape[-1]
    disp = torch.where(z > 0, focal_px * baseline_m / torch.clamp(z, min=1e-6), torch.zeros_like(z))
    u = torch.arange(w, device=z.device, dtype=z.dtype)
    q = u - disp
    suffix_min = torch.flip(torch.cummin(torch.flip(q, dims=[-1]), dim=-1).values, dims=[-1])
    right_min = torch.cat([suffix_min[..., 1:], torch.full_like(q[..., :1], float("inf"))], dim=-1)
    return ((right_min < q) | (q < -0.5)) & (z > 0)


def random_invalid_mask(
    z: torch.Tensor,
    dropout_rate: torch.Tensor,
    hole_prob: float,
    hole_max_frac: float,
    edge_threshold: float,
    edge_drop_prob: float,
) -> torch.Tensor:
    """(N, H, W) bool of randomly invalidated pixels. `dropout_rate` is per-env (N,)."""
    n, h, w = z.shape
    dev = z.device
    invalid = torch.rand_like(z) < dropout_rate.view(n, 1, 1)

    # flying pixels at depth discontinuities
    gx = torch.zeros_like(z)
    gy = torch.zeros_like(z)
    gx[:, :, 1:] = torch.abs(z[:, :, 1:] - z[:, :, :-1])
    gy[:, 1:, :] = torch.abs(z[:, 1:, :] - z[:, :-1, :])
    edge = torch.maximum(gx, gy) > edge_threshold
    invalid |= edge & (torch.rand_like(z) < edge_drop_prob)

    # random rectangular holes
    hole_env = torch.rand(n, device=dev) < hole_prob
    k = int(hole_env.sum())
    if k > 0:
        hh = (torch.rand(k, device=dev) * hole_max_frac * h).long() + 1
        ww = (torch.rand(k, device=dev) * hole_max_frac * w).long() + 1
        y0 = (torch.rand(k, device=dev) * (h - hh)).long()
        x0 = (torch.rand(k, device=dev) * (w - ww)).long()
        ys = torch.arange(h, device=dev).view(1, -1, 1)
        xs = torch.arange(w, device=dev).view(1, 1, -1)
        box = (
            (ys >= y0.view(-1, 1, 1))
            & (ys < (y0 + hh).view(-1, 1, 1))
            & (xs >= x0.view(-1, 1, 1))
            & (xs < (x0 + ww).view(-1, 1, 1))
        )
        invalid[hole_env] |= box
    return invalid
