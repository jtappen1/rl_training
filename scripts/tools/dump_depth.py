# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause

"""Dump student depth frames to PNG (command.md Phase 6.5) while the frozen teacher drives.

Builds `Stairs-Student-Deeprobotics-M20-v0`, loads the teacher PPO actor strictly from its
checkpoint onto the `teacher` observation group (the same check the distillation runner's teacher
load does), lets it drive for a while so robots are on/near stairs, and writes a grid of depth
images -- the noisy training image (what the student sees) next to the clean one -- plus the obs
group shapes.

Usage:
    python scripts/tools/dump_depth.py --headless
    python scripts/tools/dump_depth.py --headless --teacher <model.pt> --num_envs 16 --steps 150
"""

import argparse
import os
import sys

sys.stdout.reconfigure(line_buffering=True)

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Dump student depth frames while the teacher drives.")
parser.add_argument("--task", type=str, default="Stairs-Student-Deeprobotics-M20-v0")
parser.add_argument(
    "--teacher", type=str,
    default="logs/rsl_rl/deeprobotics_m20_stairs_sighted_v2/2026-09-23_15-04-55/model_5300.pt",
)
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--steps", type=int, default=150, help="Policy steps before dumping (50 Hz).")
parser.add_argument("--out_dir", type=str, default="debug/depth")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app = AppLauncher(args_cli).app

import gymnasium as gym
import numpy as np
import torch
from matplotlib import colormaps
from PIL import Image, ImageDraw

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils import load_cfg_from_registry
from rsl_rl.models import MLPModel

import rl_training.tasks  # noqa: F401


def teacher_model(obs, checkpoint: str) -> MLPModel:
    """The v2 PPO actor, rebuilt as rsl-rl builds it (see cli_args.convert_rsl_rl_cfg_dict)."""
    model = MLPModel(
        obs,
        {"teacher": ["teacher"]},
        "teacher",
        16,
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=False,
        distribution_cfg={"class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "log"},
    )
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)["actor_state_dict"]
    model.load_state_dict(state, strict=True)
    return model


def to_rgb(img: np.ndarray, scale: int) -> Image.Image:
    rgb = (colormaps["viridis"](img)[..., :3] * 255).astype(np.uint8)
    rgb[img <= 0] = (255, 0, 255)  # invalid pixels in magenta
    return Image.fromarray(np.kron(rgb, np.ones((scale, scale, 1), dtype=np.uint8)))


def main():
    env_cfg = load_cfg_from_registry(args_cli.task, "env_cfg_entry_point")
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.scene.terrain.max_init_terrain_level = 9  # spread envs over difficulty rows
    env = RslRlVecEnvWrapper(gym.make(args_cli.task, cfg=env_cfg))
    base = env.unwrapped

    obs = env.get_observations()
    print("[INFO] observation groups:", {k: tuple(v.shape) for k, v in obs.items()})
    model = teacher_model(obs, args_cli.teacher).to(base.device).eval()
    print(f"[INFO] teacher loaded strictly from {args_cli.teacher}")

    for _ in range(args_cli.steps):
        with torch.inference_mode():
            actions = model(obs)
        obs, _, _, _ = env.step(actions)

    noisy = obs["depth"][:, 0].detach().cpu().numpy()
    obs_mgr = base.observation_manager
    base.cfg.observations.depth.enable_corruption = False  # the term reads this flag each call
    clean = obs_mgr.compute_group("depth")[:, 0].detach().cpu().numpy()
    base.cfg.observations.depth.enable_corruption = True

    os.makedirs(args_cli.out_dir, exist_ok=True)
    scale, pad = 4, 6
    h, w = clean.shape[1:]
    cols = 4
    rows = int(np.ceil(len(clean) / cols))
    tile_w, tile_h = 2 * w * scale + pad, h * scale + 18
    sheet = Image.new("RGB", (cols * tile_w + pad, rows * tile_h + pad), (24, 24, 24))
    draw = ImageDraw.Draw(sheet)
    terrain_type = base.scene.terrain.terrain_types.tolist()
    level = base.scene.terrain.terrain_levels.tolist()
    for i in range(len(clean)):
        r, c = divmod(i, cols)
        x, y = pad + c * tile_w, pad + r * tile_h
        sheet.paste(to_rgb(noisy[i], scale), (x, y + 16))
        sheet.paste(to_rgb(clean[i], scale), (x + w * scale, y + 16))
        draw.text((x, y), f"env {i} col {terrain_type[i]} lvl {level[i]}  noisy | clean", fill=(230, 230, 230))
    path = os.path.join(args_cli.out_dir, "depth_grid.png")
    sheet.save(path)
    print(f"[INFO] wrote {path}  (magenta = invalid pixel; yellow = far, purple = near)")
    env.close()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback

        traceback.print_exc()
        raise
    finally:
        app.close()
