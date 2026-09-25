# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause

"""Height-map visualization for recorded videos (used by `eval_stairs.py --video`).

`HeightmapOverlay` is a gym wrapper that, for one followed env:
- draws the height-scanner ray hits in the 3D scene as spheres colored by terrain height relative to
  the ground under the robot (so they show up in the RTX-rendered video frame), and
- composites a top-down inset of the same grid onto each `render()` frame, oriented forward-up,
  with the robot's base cell marked -- i.e. the height map the policy reads, in its own frame.

Wrap the raw env *before* `gym.wrappers.RecordVideo`, which grabs frames via `env.render()`.
Must be imported after the Isaac Sim app has been launched (it imports `isaaclab.markers`).
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch
from matplotlib import colormaps
from PIL import Image, ImageDraw, ImageFont

import isaaclab.sim as sim_utils
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg

NUM_COLOR_BINS = 16


class HeightmapOverlay(gym.Wrapper):
    def __init__(
        self,
        env: gym.Env,
        env_id: int = 0,
        sensor_name: str = "height_scanner",
        height_range: float = 0.6,
        cell_px: int = 14,
        label_fn=None,
    ):
        """
        Args:
            env_id: env whose height map is drawn (match the camera's followed env).
            height_range: color scale is [-height_range, +height_range] m relative to the ground
                under the robot base.
            cell_px: inset pixels per grid cell.
            label_fn: optional `f(step) -> str` drawn under the inset (e.g. trial combo, time).
        """
        super().__init__(env)
        self.env_id = env_id
        self.height_range = height_range
        self.cell_px = cell_px
        self.label_fn = label_fn
        self._step = 0

        self.sensor = self.env.unwrapped.scene.sensors[sensor_name]
        pattern = self.sensor.cfg.pattern_cfg
        res = pattern.resolution
        self.nx = int(round(pattern.size[0] / res)) + 1
        self.ny = int(round(pattern.size[1] / res)) + 1
        if self.nx * self.ny != self.sensor.num_rays or pattern.ordering != "xy":
            raise ValueError("HeightmapOverlay expects a GridPatternCfg with 'xy' ordering.")
        # grid cell containing the sensor origin (the robot base), in pattern coordinates
        off_x, off_y = self.sensor.cfg.offset.pos[0], self.sensor.cfg.offset.pos[1]
        self.base_ix = int(np.clip(round((pattern.size[0] / 2 - off_x) / res), 0, self.nx - 1))
        self.base_iy = int(np.clip(round((pattern.size[1] / 2 - off_y) / res), 0, self.ny - 1))
        self.x_extent = (-pattern.size[0] / 2 + off_x, pattern.size[0] / 2 + off_x)

        self.cmap = colormaps["turbo"]
        bin_colors = self.cmap(np.linspace(0.0, 1.0, NUM_COLOR_BINS))[:, :3]
        self.markers = VisualizationMarkers(
            VisualizationMarkersCfg(
                prim_path="/Visuals/HeightmapOverlay",
                markers={
                    f"bin_{i}": sim_utils.SphereCfg(
                        radius=0.02,
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=tuple(float(c) for c in color)),
                    )
                    for i, color in enumerate(bin_colors)
                },
            )
        )
        try:
            self.font = ImageFont.truetype("DejaVuSans.ttf", 14)
        except OSError:
            self.font = ImageFont.load_default()

    # ------------------------------------------------------------------ gym API

    def reset(self, **kwargs):
        self._step = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        self._step += 1
        return self.env.step(action)

    def render(self):
        rel, hits = self._relative_heights()
        self._update_markers(rel, hits)
        frame = self.env.render()  # re-renders the sim, so the markers above are in this frame
        if frame is None:
            return frame
        return self._composite(frame, rel)

    # ------------------------------------------------------------------ internals

    def _relative_heights(self) -> tuple[np.ndarray, np.ndarray]:
        """Terrain height of every ray hit relative to the hit under the base, shape (ny, nx)."""
        hits = self.sensor.data.ray_hits_w[self.env_id].detach()
        z = hits[:, 2].clone()
        z[~torch.isfinite(z)] = float("nan")
        grid = z.reshape(self.ny, self.nx).cpu().numpy()  # meshgrid(x, y, indexing="xy") layout
        ref = grid[self.base_iy, self.base_ix]
        if not np.isfinite(ref):
            ref = np.nanmin(grid) if np.isfinite(grid).any() else 0.0
        return grid - ref, hits.cpu().numpy()

    def _normalize(self, rel: np.ndarray) -> np.ndarray:
        return np.clip((rel + self.height_range) / (2 * self.height_range), 0.0, 1.0)

    def _update_markers(self, rel: np.ndarray, hits: np.ndarray) -> None:
        flat = rel.reshape(-1)
        valid = np.isfinite(flat) & np.all(np.isfinite(hits), axis=1)
        if not valid.any():
            return
        bins = np.minimum((self._normalize(flat[valid]) * NUM_COLOR_BINS).astype(int), NUM_COLOR_BINS - 1)
        points = hits[valid].copy()
        points[:, 2] += 0.01  # sit just above the surface so the spheres aren't z-fighting it
        self.markers.visualize(translations=points, marker_indices=bins)

    def _composite(self, frame: np.ndarray, rel: np.ndarray) -> np.ndarray:
        # (ny, nx) with x forward / y left  ->  image rows = forward (top = far ahead), cols = left..right
        img_grid = rel.T[::-1, ::-1]
        rgb = (self.cmap(self._normalize(np.nan_to_num(img_grid, nan=0.0)))[..., :3] * 255).astype(np.uint8)
        rgb[~np.isfinite(img_grid)] = 40
        inset = Image.fromarray(np.kron(rgb, np.ones((self.cell_px, self.cell_px, 1), dtype=np.uint8)))

        draw = ImageDraw.Draw(inset)
        # robot base cell and a forward arrow
        row = self.nx - 1 - self.base_ix
        col = self.ny - 1 - self.base_iy
        c = self.cell_px
        cx, cy = col * c + c // 2, row * c + c // 2
        draw.rectangle([col * c, row * c, col * c + c - 1, row * c + c - 1], outline=(255, 255, 255), width=2)
        draw.line([cx, cy, cx, max(cy - 3 * c, 0)], fill=(255, 255, 255), width=2)

        pad, bar_h, text_h = 6, 10, 18
        lines = [f"policy height map  ({self.x_extent[0]:+.1f}..{self.x_extent[1]:+.1f} m fwd)"]
        if self.label_fn is not None:
            lines.append(self.label_fn(self._step))
        text_w = max(int(self.font.getlength(line)) for line in lines)
        panel_w = max(inset.width, 300, text_w) + 2 * pad
        panel_h = inset.height + bar_h + text_h * (len(lines) + 1) + 3 * pad
        panel = Image.new("RGB", (panel_w, panel_h), (20, 20, 20))
        panel.paste(inset, (pad, pad))

        # color bar
        bar_y = pad * 2 + inset.height
        bar = (self.cmap(np.linspace(0, 1, panel_w - 2 * pad))[:, :3] * 255).astype(np.uint8)
        panel.paste(Image.fromarray(np.repeat(bar[None], bar_h, axis=0)), (pad, bar_y))
        pdraw = ImageDraw.Draw(panel)
        ty = bar_y + bar_h + 2
        pdraw.text((pad, ty), f"-{self.height_range:.1f} m", fill=(230, 230, 230), font=self.font)
        pdraw.text((panel_w // 2 - 20, ty), "0 (under base)", fill=(230, 230, 230), font=self.font)
        pdraw.text((panel_w - pad - 50, ty), f"+{self.height_range:.1f} m", fill=(230, 230, 230), font=self.font)
        for i, line in enumerate(lines):
            pdraw.text((pad, ty + text_h * (i + 1)), line, fill=(230, 230, 230), font=self.font)

        out = frame.copy()
        panel_np = np.asarray(panel)
        h = min(panel_np.shape[0], out.shape[0] - 10)
        w = min(panel_np.shape[1], out.shape[1] - 10)
        out[10 : 10 + h, out.shape[1] - 10 - w : out.shape[1] - 10] = panel_np[:h, :w]
        return out
