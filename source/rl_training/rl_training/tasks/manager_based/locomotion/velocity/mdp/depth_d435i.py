# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause

"""RealSense D435i depth observation + camera-mount randomization for the M20 student.

Pipeline per policy step (see docs/depth_pipeline.md and `rl_training.depth`):

    ray-cast depth (m) -> sanitize -> [stereo occlusion mask] -> [disparity noise, random holes]
        -> preprocess() (same code as the robot) -> [latency, frame drops] -> (N, 1, h, w) in [0, 1]

Bracketed steps: the stereo occlusion mask is geometry, so it is on whenever `stereo_occlusion` is
True, even with corruption off; the noise, holes, latency and frame drops only run while the
observation group's `enable_corruption` is True (Play/eval turn it off to get a clean image).

Per-env draws at reset: noise level, low-frequency noise level, depth scale error, pixel dropout
rate, latency. The camera's mount pose is jittered per env at reset by `randomize_camera_offset`.

Replaces `mdp.depth_image` (kept in depth.py for the first student run, which was trained with it).
"""

from __future__ import annotations

import inspect

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.managers import ManagerTermBase, SceneEntityCfg

from rl_training.depth.preprocess import preprocess, sanitize
from rl_training.depth.stereo_noise import disparity_noise, random_invalid_mask, stereo_invalid_mask

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv
    from isaaclab.managers import ObservationTermCfg
    from isaaclab.sensors import RayCasterCamera

__all__ = ["d435i_depth_image", "randomize_camera_offset"]


def _uniform(lo_hi: tuple[float, float], n: int, device) -> torch.Tensor:
    lo, hi = lo_hi
    return torch.rand(n, device=device) * (hi - lo) + lo


class d435i_depth_image(ManagerTermBase):
    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        # params the cfg leaves out fall back to __call__'s defaults; reset() reads the *_range ones
        defaults = {
            k: v.default for k, v in inspect.signature(self.__call__).parameters.items() if v.default is not v.empty
        }
        self.params = {**defaults, **cfg.params}
        p = self.params
        self.camera: RayCasterCamera = env.scene.sensors[p["sensor_cfg"].name]
        self.out_h, self.out_w = p["out_hw"]
        n = env.num_envs
        dev = env.device
        lat_lo, lat_hi = p["latency_steps_range"]
        self.lat_range = (int(lat_lo), int(lat_hi))
        self.max_latency = int(lat_hi)
        # ring buffer of the last (max_latency + 1) processed images, fp16 to keep memory down
        self.buffer = torch.zeros(self.max_latency + 1, n, 1, self.out_h, self.out_w, device=dev, dtype=torch.float16)
        self.buf_idx = 0
        self.last_out = torch.zeros(n, 1, self.out_h, self.out_w, device=dev)
        self.latency = torch.zeros(n, dtype=torch.long, device=dev)
        self.dropout = torch.zeros(n, device=dev)
        self.sigma_px = torch.zeros(n, device=dev)
        self.lowfreq_px = torch.zeros(n, device=dev)
        self.scale = torch.ones(n, device=dev)
        self.reset(None)

    def reset(self, env_ids=None) -> None:
        p = self.params
        ids = slice(None) if env_ids is None else env_ids
        n = self.latency[ids].numel()
        dev = self.latency.device
        self.latency[ids] = torch.randint(self.lat_range[0], self.lat_range[1] + 1, (n,), device=dev)
        self.dropout[ids] = _uniform(p["dropout_range"], n, dev)
        self.sigma_px[ids] = _uniform(p["sigma_px_range"], n, dev)
        self.lowfreq_px[ids] = _uniform(p["lowfreq_px_range"], n, dev)
        self.scale[ids] = _uniform(p["scale_range"], n, dev)
        # a reset env must not see the previous episode's frames
        self.buffer[:, ids] = 0.0
        self.last_out[ids] = 0.0

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        sensor_cfg: SceneEntityCfg,
        out_hw: tuple[int, int],
        near: float,
        far: float,
        min_valid_frac: float,
        baseline_m: float,
        native_focal_px: float,
        sim_focal_px: float,
        group_name: str = "depth",
        stereo_occlusion: bool = True,
        sigma_px_range: tuple[float, float] = (0.02, 0.10),
        lowfreq_px_range: tuple[float, float] = (0.0, 0.05),
        scale_range: tuple[float, float] = (0.98, 1.02),
        dropout_range: tuple[float, float] = (0.0, 0.03),
        hole_prob: float = 0.05,
        hole_max_frac: float = 0.2,
        edge_threshold: float = 0.1,
        edge_drop_prob: float = 0.1,
        latency_steps_range: tuple[int, int] = (1, 3),
        frame_drop_prob: float = 0.02,
    ) -> torch.Tensor:
        # *_range params are consumed at reset (see reset())
        z = sanitize(self.camera.data.output["distance_to_image_plane"][..., 0], near, far)  # (N, H, W) m
        invalid = stereo_invalid_mask(z, sim_focal_px, baseline_m) if stereo_occlusion else None

        noisy = getattr(env.cfg.observations, group_name).enable_corruption
        if noisy:
            z = disparity_noise(z, native_focal_px, baseline_m, self.sigma_px, self.lowfreq_px, self.scale)
            rand_invalid = random_invalid_mask(
                z, self.dropout, hole_prob, hole_max_frac, edge_threshold, edge_drop_prob
            )
            invalid = rand_invalid if invalid is None else invalid | rand_invalid
        if invalid is not None:
            z = torch.where(invalid, torch.zeros_like(z), z)

        img = preprocess(z, self.out_h, self.out_w, near, far, min_valid_frac).unsqueeze(1)  # (N, 1, h, w)
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


def randomize_camera_offset(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    sensor_cfg: SceneEntityCfg,
    pos_range: tuple[float, float, float] = (0.01, 0.01, 0.01),
    rot_range_deg: tuple[float, float, float] = (2.0, 3.0, 2.0),
):
    """Jitter a RayCasterCamera's mount per env around its nominal pose (event, mode="reset").

    Uniform +/- `pos_range` (m, base frame x/y/z) and +/- `rot_range_deg` (roll, pitch, yaw about the
    camera's own forward/left/up axes). Writes the camera's per-env offset buffers, which the sensor
    re-reads every update.
    """
    cam: RayCasterCamera = env.scene.sensors[sensor_cfg.name]
    if not hasattr(cam, "_nominal_offset"):
        cam._nominal_offset = (cam._offset_pos.clone(), cam._offset_quat.clone())
    nom_pos, nom_quat = cam._nominal_offset
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=nom_pos.device)
    n = len(env_ids)
    dev = nom_pos.device

    def sym(r):
        return (torch.rand(n, device=dev) * 2.0 - 1.0) * r

    dpos = torch.stack([sym(r) for r in pos_range], dim=-1)
    rpy = [sym(r * torch.pi / 180.0) for r in rot_range_deg]
    dquat = math_utils.quat_from_euler_xyz(rpy[0], rpy[1], rpy[2])
    cam._offset_pos[env_ids] = nom_pos[env_ids] + dpos
    cam._offset_quat[env_ids] = math_utils.quat_mul(nom_quat[env_ids], dquat)
