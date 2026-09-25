# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause

"""Depth-image observation for the M20 stairs student (command.md Phase 6/7.1).

Reads a `RayCasterCamera` ("distance_to_image_plane"), clips it to [near, far], optionally applies
the depth noise model below, and returns a (N, 1, H, W) image in [0, 1] (0 = invalid pixel, then
near..far mapped to (0, 1]). Channels-first, so RSL-RL's CNN models pick it up as a 2D group.

Noise model (only while the observation group's `enable_corruption` is True, so Play/eval configs
that turn corruption off get the clean image):
- Gaussian noise with std proportional to depth (`noise_rel_std` x depth), per step;
- random pixel dropout (per-env rate drawn from `dropout_range` at reset) -> invalid (0);
- random rectangular holes (`hole_prob` per env per step, up to `hole_max_frac` of each side);
- "flying pixels" at depth discontinuities (step edges): pixels whose local depth gradient
  exceeds `edge_threshold` m are invalidated with probability `edge_drop_prob`;
- latency: each env delays the image by a fixed number of policy steps drawn from
  `latency_steps_range` at reset (on top of the camera's own `update_period` frame hold);
- frame drops: with probability `frame_drop_prob` per step the previous delivered image is repeated.

Not modelled yet (see command.md 7.1): self-occlusion by the legs/wheels (ray casting only sees the
terrain), camera extrinsics jitter, blur. Whatever preprocessing the real robot does must match this
pipeline (document it in docs/depth_pipeline.md once the real sensor is chosen).
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import ManagerTermBase, SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import ObservationTermCfg
    from isaaclab.sensors import RayCasterCamera

__all__ = ["depth_image"]


class depth_image(ManagerTermBase):
    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        p = cfg.params
        self.camera: RayCasterCamera = env.scene.sensors[p["sensor_cfg"].name]
        h, w = self.camera.image_shape
        self.h, self.w = h, w
        lat_lo, lat_hi = p.get("latency_steps_range", (0, 2))
        self.max_latency = int(lat_hi)
        n = env.num_envs
        # ring buffer of the last (max_latency + 1) processed images, fp16 to keep memory down
        self.buffer = torch.zeros(self.max_latency + 1, n, 1, h, w, device=env.device, dtype=torch.float16)
        self.buf_idx = 0
        self.latency = torch.zeros(n, dtype=torch.long, device=env.device)
        self.dropout = torch.zeros(n, device=env.device)
        self.last_out = torch.zeros(n, 1, h, w, device=env.device)
        self.lat_range = (int(lat_lo), int(lat_hi))
        self.dropout_range = p.get("dropout_range", (0.0, 0.05))
        self.reset(None)

    def reset(self, env_ids=None) -> None:
        ids = slice(None) if env_ids is None else env_ids
        n = self.latency[ids].numel()
        self.latency[ids] = torch.randint(self.lat_range[0], self.lat_range[1] + 1, (n,), device=self.latency.device)
        lo, hi = self.dropout_range
        self.dropout[ids] = torch.rand(n, device=self.dropout.device) * (hi - lo) + lo
        # a reset env should not see the previous episode's frames
        self.buffer[:, ids] = 0.0
        self.last_out[ids] = 0.0

    def _noisy(
        self,
        d: torch.Tensor,
        near: float,
        far: float,
        noise_rel_std: float,
        hole_prob: float,
        hole_max_frac: float,
        edge_threshold: float,
        edge_drop_prob: float,
    ) -> torch.Tensor:
        """d: (N, H, W) metres, already clipped to [near, far]. Returns metres with 0 = invalid."""
        n = d.shape[0]
        dev = d.device
        d = d + torch.randn_like(d) * noise_rel_std * d
        d = torch.clamp(d, near, far)
        invalid = torch.rand_like(d) < self.dropout.view(n, 1, 1)

        # flying pixels at depth discontinuities
        gx = torch.zeros_like(d)
        gy = torch.zeros_like(d)
        gx[:, :, 1:] = torch.abs(d[:, :, 1:] - d[:, :, :-1])
        gy[:, 1:, :] = torch.abs(d[:, 1:, :] - d[:, :-1, :])
        edge = torch.maximum(gx, gy) > edge_threshold
        invalid |= edge & (torch.rand_like(d) < edge_drop_prob)

        # random rectangular holes
        hole_env = torch.rand(n, device=dev) < hole_prob
        if hole_env.any():
            max_frac = hole_max_frac
            k = int(hole_env.sum())
            hh = (torch.rand(k, device=dev) * max_frac * self.h).long() + 1
            ww = (torch.rand(k, device=dev) * max_frac * self.w).long() + 1
            y0 = (torch.rand(k, device=dev) * (self.h - hh)).long()
            x0 = (torch.rand(k, device=dev) * (self.w - ww)).long()
            ys = torch.arange(self.h, device=dev).view(1, -1, 1)
            xs = torch.arange(self.w, device=dev).view(1, 1, -1)
            box = ((ys >= y0.view(-1, 1, 1)) & (ys < (y0 + hh).view(-1, 1, 1))
                   & (xs >= x0.view(-1, 1, 1)) & (xs < (x0 + ww).view(-1, 1, 1)))
            invalid[hole_env] |= box
        return torch.where(invalid, torch.zeros_like(d), d)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        sensor_cfg: SceneEntityCfg,
        near: float = 0.1,
        far: float = 2.5,
        group_name: str = "depth",
        noise_rel_std: float = 0.01,
        dropout_range: tuple[float, float] = (0.0, 0.05),
        hole_prob: float = 0.1,
        hole_max_frac: float = 0.25,
        edge_threshold: float = 0.1,
        edge_drop_prob: float = 0.3,
        latency_steps_range: tuple[int, int] = (0, 2),
        frame_drop_prob: float = 0.05,
    ) -> torch.Tensor:
        # dropout_range / latency_steps_range are consumed at reset (see __init__/reset)
        d = self.camera.data.output["distance_to_image_plane"][..., 0]  # (N, H, W)
        d = torch.nan_to_num(d, nan=far, posinf=far, neginf=near)
        d = torch.clamp(d, near, far)

        noisy = getattr(env.cfg.observations, group_name).enable_corruption
        if noisy:
            d = self._noisy(
                d, near, far, noise_rel_std, hole_prob, hole_max_frac, edge_threshold, edge_drop_prob
            )
        # map near..far -> (0, 1], keep invalid (0 m) at exactly 0
        img = torch.where(d > 0, (d - near) / (far - near) * (1.0 - 1e-3) + 1e-3, torch.zeros_like(d))
        img = img.unsqueeze(1)  # (N, 1, H, W)

        if not noisy:
            return img

        # latency: push this frame, read the one `latency` steps back
        self.buf_idx = (self.buf_idx + 1) % (self.max_latency + 1)
        self.buffer[self.buf_idx] = img.to(self.buffer.dtype)
        read_idx = (self.buf_idx - self.latency) % (self.max_latency + 1)
        out = self.buffer[read_idx, torch.arange(img.shape[0], device=img.device)].float()
        # frame drops: repeat the previously delivered image
        drop = torch.rand(img.shape[0], device=img.device) < frame_drop_prob
        out = torch.where(drop.view(-1, 1, 1, 1), self.last_out, out)
        self.last_out = out
        return out
