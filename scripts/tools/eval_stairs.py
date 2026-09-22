# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause

"""Evaluate a trained M20 checkpoint on a fixed-geometry stairs benchmark.

Milestone A / Phase 1 baseline measurement (see command.md). Builds a stairs-only terrain with
known, fixed step heights and tread widths (a `TerrainGeneratorCfg` with one sub-terrain per
combination, `curriculum=True`, one difficulty row, so step height is not difficulty-interpolated
and each combination maps deterministically to a column -- see `terrain_generator.py` and
`terrain_importer._compute_env_origins_curriculum`). Every environment gets a fixed forward
velocity command (no lateral/yaw) and is scored on whether it crosses to the far side of its
stair tile before a time budget runs out.

Ascent uses `pyramid_stairs_inv` (robot spawns on the low centre platform, walking out climbs);
descent uses `pyramid_stairs` (robot spawns on the high centre platform, walking out descends) --
see command.md's notes on the existing ROUGH_TERRAINS_CFG stair sub-terrains.

Usage:
    python scripts/tools/eval_stairs.py --checkpoint <run_dir_or_model.pt> --headless
    python scripts/tools/eval_stairs.py --checkpoint <run_dir_or_model.pt> --video --repeats 1 \\
        --step_heights 0.08 0.15 0.25 --tread_widths 0.30
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import sys

# Isaac Sim's shutdown path does not reliably flush a block-buffered stdout (the default when
# stdout isn't a TTY, e.g. redirected to a log file), which can silently swallow all `print()`
# output from this script. Force line buffering so progress/results are never lost.
sys.stdout.reconfigure(line_buffering=True)

from isaaclab.app import AppLauncher

# local imports (cli_args lives next to the rsl_rl train/play scripts)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "reinforcement_learning", "rsl_rl")))
import cli_args  # noqa: E402

parser = argparse.ArgumentParser(description="Evaluate an M20 checkpoint on a fixed-geometry stairs benchmark.")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to a model_*.pt file, or a run directory (the highest-iteration checkpoint in it is used).")
parser.add_argument("--task", type=str, default="Rough-Deeprobotics-M20-v0", help="Registered task whose entry_point/robot config to reuse.")
parser.add_argument("--speeds", type=float, nargs="+", default=[0.5, 1.0], help="Fixed forward command speeds to evaluate (m/s).")
parser.add_argument(
    "--step_heights", type=float, nargs="+", default=[0.08, 0.12, 0.15, 0.18, 0.20, 0.23, 0.25], help="Riser heights (m)."
)
parser.add_argument("--tread_widths", type=float, nargs="+", default=[0.25, 0.30, 0.35], help="Tread/step widths (m).")
parser.add_argument("--repeats", type=int, default=8, help="Parallel trials per (direction, step height, tread width) combo.")
parser.add_argument("--time_out_s", type=float, default=12.0, help="Per-trial time budget (s) to reach the far platform.")
parser.add_argument("--out_dir", type=str, default="eval", help="Directory for the CSV/markdown report.")
parser.add_argument("--video", action="store_true", default=False, help="Record a video of the run.")
parser.add_argument("--video_length", type=int, default=300, help="Length of the recorded video (in steps).")
parser.add_argument("--seed", type=int, default=42)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for anything downstream that re-parses it
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import csv
import glob
import itertools
import re
from collections import OrderedDict, defaultdict
from datetime import datetime

import gymnasium as gym
import torch

import isaaclab.terrains as terrain_gen
from isaaclab.terrains import TerrainGeneratorCfg

import importlib.metadata as metadata
from packaging import version

RSL_RL_VERSION = "5.0.1"
installed_version = metadata.version("rsl-rl-lib")
if version.parse(installed_version) < version.parse(RSL_RL_VERSION):
    print(
        f"[WARN] Installed rsl-rl-lib ({installed_version}) is older than the version this repo targets"
        f" ({RSL_RL_VERSION}). Continuing, but behaviour may differ."
    )

from rsl_rl.runners import OnPolicyRunner

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

import rl_training.tasks  # noqa: F401
from rl_training.tasks.manager_based.locomotion.velocity.config.wheeled.deeprobotics_m20.agents.rsl_rl_ppo_cfg import (
    DeeproboticsM20RoughPPORunnerCfg,
)
from rl_training.tasks.manager_based.locomotion.velocity.config.wheeled.deeprobotics_m20.rough_env_cfg import (
    DeeproboticsM20RoughEnvCfg,
)

# ---------------------------------------------------------------------------
# Fixed evaluation-terrain geometry.
#
# These mirror the values ROUGH_TERRAINS_CFG already uses for its pyramid_stairs /
# pyramid_stairs_inv sub-terrains (isaaclab/terrains/config/rough.py), so the eval tiles are the
# same "shape" of stairs the blind policy has already seen during training -- only the step
# height/tread width are pinned to exact values instead of being difficulty-sampled.
# ---------------------------------------------------------------------------
TILE_SIZE = 8.0
PLATFORM_WIDTH = 2.5
SUB_BORDER_WIDTH = 1.0
GEN_BORDER_WIDTH = 4.0
# Extra clearance past the last riser so the whole ~0.82 m-long M20 body (not just the root) has
# cleared the stairs before we call it a crossing.
ROBOT_LENGTH_MARGIN = 0.3

DIRECTIONS = ("ascent", "descent")


def resolve_checkpoint(path: str) -> str:
    """Accept either a direct model_*.pt file or a run directory (picks the highest iteration)."""
    if os.path.isfile(path):
        return path
    candidates = glob.glob(os.path.join(path, "model_*.pt"))
    if not candidates:
        raise FileNotFoundError(f"No model_*.pt checkpoints found under '{path}'.")

    def _iteration(p: str) -> int:
        m = re.search(r"model_(\d+)\.pt$", p)
        return int(m.group(1)) if m else -1

    return max(candidates, key=_iteration)


def num_stairs_steps(step_width: float) -> int:
    """Mirror the step count formula in isaaclab's pyramid_stairs_terrain / inverted_pyramid_stairs_terrain."""
    usable = TILE_SIZE - 2 * SUB_BORDER_WIDTH - PLATFORM_WIDTH
    return int(usable // (2 * step_width) + 1)


def build_combos(step_heights, tread_widths):
    combos = []
    for direction, height, width in itertools.product(DIRECTIONS, step_heights, tread_widths):
        combos.append({"direction": direction, "step_height": height, "tread_width": width})
    return combos


def build_terrain_generator(combos: list[dict]) -> TerrainGeneratorCfg:
    """One sub-terrain per combo, equal proportion, one difficulty row.

    With `curriculum=True` and equal proportions, `TerrainGenerator._generate_curriculum_terrains`
    assigns column `i` to `list(sub_terrains.values())[i]` (see terrain_generator.py) -- so combo
    order here is exactly the column order the terrain importer later hands back as
    `terrain.terrain_types`. `step_height_range=(h, h)` makes the per-row difficulty jitter a
    no-op for pyramid_stairs (only step height depends on difficulty; see mesh_terrains.py), so a
    single row (no promotion to chase) is safe.
    """
    sub_terrains = OrderedDict()
    for i, combo in enumerate(combos):
        cls = (
            terrain_gen.MeshInvertedPyramidStairsTerrainCfg
            if combo["direction"] == "ascent"
            else terrain_gen.MeshPyramidStairsTerrainCfg
        )
        key = f"{i:03d}_{combo['direction']}_h{combo['step_height']:.2f}_w{combo['tread_width']:.2f}"
        sub_terrains[key] = cls(
            proportion=1.0,
            step_height_range=(combo["step_height"], combo["step_height"]),
            step_width=combo["tread_width"],
            platform_width=PLATFORM_WIDTH,
            border_width=SUB_BORDER_WIDTH,
            holes=False,
        )
    return TerrainGeneratorCfg(
        size=(TILE_SIZE, TILE_SIZE),
        border_width=GEN_BORDER_WIDTH,
        num_rows=1,
        num_cols=len(combos),
        horizontal_scale=0.1,
        vertical_scale=0.005,
        slope_threshold=0.75,
        use_cache=False,
        curriculum=True,
        sub_terrains=sub_terrains,
    )


def build_env_cfg(combos: list[dict]) -> DeeproboticsM20RoughEnvCfg:
    """Start from the exact blind-checkpoint task config and swap in the eval terrain + fixed command.

    Deliberately does NOT touch `observations` (besides disabling noise) -- the loaded checkpoint
    is blind (height_scan=None) and any obs shape change would break `ppo_runner.load()`.
    """
    env_cfg = DeeproboticsM20RoughEnvCfg()
    env_cfg.scene.num_envs = len(combos) * args_cli.repeats
    env_cfg.seed = args_cli.seed
    env_cfg.episode_length_s = args_cli.time_out_s

    # ---- fixed evaluation terrain (replaces the training ROUGH_TERRAINS_CFG instance; does not
    # mutate the shared object, per command.md's ground rules) ----
    env_cfg.scene.terrain.terrain_generator = build_terrain_generator(combos)
    env_cfg.scene.terrain.max_init_terrain_level = 0

    # single-row terrain has no difficulty levels to promote/demote through
    env_cfg.curriculum.terrain_levels = None
    env_cfg.curriculum.gait_level = None
    env_cfg.curriculum.command_levels = None

    # deterministic spawn: tile centre, heading straight across the stairs (local +x)
    env_cfg.events.randomize_reset_base.params = {
        "pose_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0), "yaw": (0.0, 0.0)},
        "velocity_range": {
            "x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0),
            "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0),
        },
    }
    # no pushes/disturbances -- a clean read of the checkpoint's stair-climbing ability
    env_cfg.events.randomize_push_robot = None
    env_cfg.events.randomize_apply_external_force_torque = None

    # fixed forward command only: no lateral/yaw, no resampling mid-trial
    cmd = env_cfg.commands.base_velocity
    cmd.heading_command = False
    cmd.rel_standing_envs = 0.0
    cmd.rel_zero_vel_envs = 0.0
    cmd.rel_only_lin_x_envs = 0.0
    cmd.rel_only_lin_y_envs = 0.0
    cmd.rel_only_ang_z_envs = 0.0
    cmd.resampling_time_range = (1.0e6, 1.0e6)
    cmd.ranges.lin_vel_y = (0.0, 0.0)
    cmd.ranges.ang_vel_z = (0.0, 0.0)
    cmd.debug_vis = False

    # match play.py: noise-free observations for evaluation
    env_cfg.observations.policy.enable_corruption = False

    return env_cfg


def write_outputs(rows: list[dict]) -> None:
    os.makedirs(args_cli.out_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    csv_path = os.path.join(args_cli.out_dir, f"eval_stairs_{timestamp}.csv")
    fieldnames = [
        "speed", "direction", "step_height", "tread_width", "env_id",
        "outcome", "time_to_cross_s", "collisions", "stumbles",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[INFO] Wrote {csv_path}")

    groups = defaultdict(list)
    for r in rows:
        groups[(r["speed"], r["direction"], r["step_height"])].append(r)

    lines = [
        "# Stairs evaluation -- blind baseline",
        "",
        f"Checkpoint: `{resolve_checkpoint(args_cli.checkpoint)}`",
        "",
        f"{len(rows)} trials total ({args_cli.repeats} repeats x {len(args_cli.tread_widths)} tread widths per"
        " (speed, direction, step height) cell). Tread widths are aggregated together in this summary;"
        " see the CSV for the per-tread-width breakdown.",
        "",
        "| speed (m/s) | direction | step height (m) | n | success rate | mean time-to-cross (s) |"
        " collisions/trial | stumbles/trial | falls |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for key in sorted(groups.keys()):
        speed, direction, height = key
        g = groups[key]
        n = len(g)
        successes = [r for r in g if r["outcome"] == "success"]
        falls = [r for r in g if r["outcome"] == "fall"]
        success_rate = len(successes) / n
        mean_time = (sum(r["time_to_cross_s"] for r in successes) / len(successes)) if successes else float("nan")
        mean_collisions = sum(r["collisions"] for r in g) / n
        mean_stumbles = sum(r["stumbles"] for r in g) / n
        lines.append(
            f"| {speed:.1f} | {direction} | {height:.2f} | {n} | {success_rate * 100:.0f}% | "
            f"{mean_time:.2f} | {mean_collisions:.2f} | {mean_stumbles:.2f} | {len(falls)} |"
        )
    md_path = os.path.join(args_cli.out_dir, f"eval_stairs_{timestamp}.md")
    with open(md_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[INFO] Wrote {md_path}")


def main():
    combos = build_combos(args_cli.step_heights, args_cli.tread_widths)
    env_cfg = build_env_cfg(combos)

    checkpoint_path = resolve_checkpoint(args_cli.checkpoint)
    print(f"[INFO] Using checkpoint: {checkpoint_path}")
    print(f"[INFO] {len(combos)} combos x {args_cli.repeats} repeats = {env_cfg.scene.num_envs} envs")

    agent_cfg = DeeproboticsM20RoughPPORunnerCfg()

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(os.path.abspath(args_cli.out_dir), "videos"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording video during evaluation.")
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    train_cfg = agent_cfg.to_dict()
    # RslRlOnPolicyRunnerCfg carries both the legacy `policy` field (which
    # DeeproboticsM20RoughPPORunnerCfg actually populates) and newer, unset `actor`/`critic`
    # fields (MISSING). Drop the MISSING ones so convert_rsl_rl_cfg_dict's "already new-format"
    # short-circuit doesn't skip the policy->actor/critic conversion and leave them empty.
    train_cfg.pop("actor", None)
    train_cfg.pop("critic", None)
    if version.parse(installed_version) >= version.parse("5.0.0"):
        train_cfg = cli_args.convert_rsl_rl_cfg_dict(train_cfg)
    ppo_runner = OnPolicyRunner(env, train_cfg, log_dir=None, device=agent_cfg.device)
    ppo_runner.load(checkpoint_path)
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # per-env combo bookkeeping, read back from the terrain importer's own (deterministic) column
    # assignment rather than recomputed independently, so it can never drift out of sync with it.
    terrain_types = env.unwrapped.scene.terrain.terrain_types.clone().tolist()
    combo_of_env = [combos[c] for c in terrain_types]

    device = env.unwrapped.device
    num_envs = env.unwrapped.num_envs
    target_x = torch.zeros(num_envs, device=device)
    for i, combo in enumerate(combo_of_env):
        n_steps = num_stairs_steps(combo["tread_width"])
        target_x[i] = PLATFORM_WIDTH / 2.0 + n_steps * combo["tread_width"] + ROBOT_LENGTH_MARGIN

    # contact sensor body groups for collision / stumble detection (same patterns rough_env_cfg.py
    # wires up for the undesired_contacts / feet_stumble reward terms, just read out directly here
    # instead of through a reward weight).
    contact_sensor = env.unwrapped.scene["contact_forces"]
    wheel_ids, _ = contact_sensor.find_bodies(env_cfg.foot_link_name)
    non_wheel_ids, _ = contact_sensor.find_bodies(f"^(?!.*{env_cfg.foot_link_name}).*")
    wheel_ids_t = torch.tensor(wheel_ids, device=device, dtype=torch.long)
    non_wheel_ids_t = torch.tensor(non_wheel_ids, device=device, dtype=torch.long)

    step_dt = env.unwrapped.step_dt
    horizon_steps = int(round(args_cli.time_out_s / step_dt))

    all_rows = []
    for speed in args_cli.speeds:
        print(f"[INFO] Running speed={speed} m/s over {horizon_steps} steps ({args_cli.time_out_s}s)...")
        cmd_term = env.unwrapped.command_manager.get_term("base_velocity")
        cmd_term.cfg.ranges.lin_vel_x = (speed, speed)

        obs, _ = env.reset()
        robot = env.unwrapped.scene["robot"]
        x0 = robot.data.root_pos_w[:, 0].clone()

        finalized = torch.zeros(num_envs, dtype=torch.bool, device=device)
        outcome = ["timeout"] * num_envs
        time_to_cross = [float("nan")] * num_envs
        collisions = torch.zeros(num_envs, dtype=torch.long, device=device)
        stumbles = torch.zeros(num_envs, dtype=torch.long, device=device)
        prev_collide = torch.zeros(num_envs, dtype=torch.bool, device=device)
        prev_stumble = torch.zeros(num_envs, dtype=torch.bool, device=device)

        for step in range(horizon_steps):
            # Only the policy forward pass needs inference_mode. Stepping the env inside it too
            # (as play.py does, for a single never-reset rollout) taints internal simulation
            # buffers as inference tensors; a later env.reset() for the next speed pass then fails
            # in-place writes to them outside of inference_mode ("Inplace update to inference
            # tensor..."). Keeping env.step() outside avoids that across our multi-pass rollout.
            with torch.inference_mode():
                actions = policy(obs)
            obs, _, dones, extras = env.step(actions)

            time_outs = extras.get("time_outs", torch.zeros_like(dones, dtype=torch.bool))
            terminated = dones.bool() & ~time_outs.bool()

            forces = contact_sensor.data.net_forces_w  # (N, B, 3)
            wheel_fz = forces[:, wheel_ids_t, 2].abs()
            wheel_fxy = torch.linalg.norm(forces[:, wheel_ids_t, :2], dim=-1)
            stumble_now = torch.any(wheel_fxy > 4.0 * wheel_fz, dim=1)  # mirrors mdp.rewards.feet_stumble
            stumbles += (stumble_now & ~prev_stumble & ~finalized).long()
            prev_stumble = stumble_now

            collide_now = torch.any(torch.linalg.norm(forces[:, non_wheel_ids_t, :], dim=-1) > 1.0, dim=1)
            collisions += (collide_now & ~prev_collide & ~finalized).long()
            prev_collide = collide_now

            progress = robot.data.root_pos_w[:, 0] - x0
            newly_success = (progress >= target_x) & ~finalized
            for idx in newly_success.nonzero(as_tuple=True)[0].tolist():
                outcome[idx] = "success"
                time_to_cross[idx] = (step + 1) * step_dt
            finalized |= newly_success

            newly_fall = terminated & ~finalized
            for idx in newly_fall.nonzero(as_tuple=True)[0].tolist():
                outcome[idx] = "fall"
            finalized |= newly_fall

            if bool(finalized.all()):
                break

        for i, combo in enumerate(combo_of_env):
            all_rows.append({
                "speed": speed,
                "direction": combo["direction"],
                "step_height": combo["step_height"],
                "tread_width": combo["tread_width"],
                "env_id": i,
                "outcome": outcome[i],
                "time_to_cross_s": time_to_cross[i],
                "collisions": collisions[i].item(),
                "stumbles": stumbles[i].item(),
            })
        n_success = sum(1 for r in all_rows if r["speed"] == speed and r["outcome"] == "success")
        n_fall = sum(1 for r in all_rows if r["speed"] == speed and r["outcome"] == "fall")
        print(f"[INFO]   speed={speed}: {n_success}/{num_envs} reached far platform, {n_fall} fell")

    env.close()
    write_outputs(all_rows)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback

        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
