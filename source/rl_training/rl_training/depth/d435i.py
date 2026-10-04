# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause

"""Intel RealSense D435i depth-stream spec: the single source of truth for sim and robot.

Pure Python (no torch / Isaac Lab imports) so the robot-side node can import it as-is.
Datasheet values are marked; anything marked VERIFY should be checked against the real unit.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class D435iSpec:
    # --- native depth stream (what the camera delivers) ---
    native_width: int = 848  # best depth quality mode on D4xx
    native_height: int = 480
    fps: int = 60  # 30 / 60 / 90 available at 848x480
    hfov_deg: float = 87.0  # datasheet depth FOV 87 x 58 deg (+/-3 deg); square pixels -> ~56.5 deg vertical
    baseline_m: float = 0.050  # datasheet
    min_z_m: float = 0.2  # VERIFY: datasheet Min-Z ~0.28 m at 1280x720, scales down with resolution
    far_m: float = 3.0  # clip range for the policy image (the camera itself sees further)

    # --- policy image (what the CNN sees) ---
    width: int = 64
    height: int = 36
    min_valid_frac: float = 0.5  # pooled pixel is invalid unless >= this fraction of its block is valid

    # --- timing ---
    latency_s: tuple[float, float] = (0.02, 0.06)  # VERIFY: capture -> policy input, incl. USB + preprocess

    @property
    def native_focal_px(self) -> float:
        """Focal length in native pixels (~447 px at 848 wide / 87 deg)."""
        return self.focal_px(self.native_width)

    def focal_px(self, width: int) -> float:
        return width / 2.0 / math.tan(math.radians(self.hfov_deg) / 2.0)

    @property
    def vfov_deg(self) -> float:
        return math.degrees(2.0 * math.atan(math.tan(math.radians(self.hfov_deg) / 2.0) * self.height / self.width))

    def depth_rms_m(self, z: float, subpixel_rms_px: float = 0.08) -> float:
        """Stereo depth RMS error at distance z: z^2 * subpixel / (f * baseline)."""
        return z * z * subpixel_rms_px / (self.native_focal_px * self.baseline_m)


@dataclass(frozen=True)
class CameraMount:
    """Camera pose in the robot base frame (+X forward, +Z up)."""

    pos: tuple[float, float, float] = (0.38, 0.0, 0.05)  # PLACEHOLDER until the real bracket is known
    pitch_down_deg: float = 40.0  # PLACEHOLDER


D435I = D435iSpec()
M20_D435I_MOUNT = CameraMount()
# Lite3: just ahead of the 0.5 m torso box (front face at x = +0.25), ~0.36 m above the ground when
# standing; at 40 deg down the image covers roughly 0.4-2 m of ground ahead of the base. PLACEHOLDER.
LITE3_D435I_MOUNT = CameraMount(pos=(0.26, 0.0, 0.03), pitch_down_deg=40.0)
