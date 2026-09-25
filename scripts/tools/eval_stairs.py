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
parser.add_argument(
    "--follow_env_id", type=int, default=0,
    help="Env index the video camera should follow (dynamic asset-root tracking, re-centers every"
    " frame on that env's robot). Pass -1 to keep Isaac Lab's default fixed world-space viewer"
    " camera instead (the wide/far view of the whole tiled terrain grid).",
)
parser.add_argument(
    "--camera_eye", type=float, nargs=3, default=[-1.5, 2.5, 1.2],
    help="Camera position as an (x,y,z) offset in meters from the followed robot's root (only used"
    " when --follow_env_id >= 0). Default is a 3/4 side-behind angle facing the direction of travel"
    " (robots walk out from their tile centre along local +x).",
)
parser.add_argument(
    "--camera_lookat", type=float, nargs=3, default=[1.0, 0.0, 0.1],
    help="Camera look-at target as an (x,y,z) offset in meters from the followed robot's root (only"
    " used when --follow_env_id >= 0). Default looks slightly ahead of and down at the robot.",
)
parser.add_argument(
    "--no_heightmap_viz", action="store_true", default=False,
    help="With --video, do NOT draw the policy height map (3D ray-hit spheres + top-down inset) in the recording.",
)
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
import math
import glob
import re
from collections import defaultdict
from datetime import datetime

import gymnasium as gym
import torch

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
from isaaclab_tasks.utils import load_cfg_from_registry

import rl_training.tasks  # noqa: F401
from rl_training.tasks.manager_based.locomotion.velocity.config.wheeled.deeprobotics_m20.eval_terrains import (  # noqa: E402
    PLATFORM_WIDTH,
    SUB_BORDER_WIDTH,
    TILE_SIZE,
    build_combos,
    build_stairs_benchmark_generator,
    num_stairs_steps,
)
from rl_training.tasks.manager_based.locomotion.velocity.config.wheeled.deeprobotics_m20.stairs_env_cfg import (  # noqa: E402
    TASK_STAIR_ASCENT,
    TASK_STAIR_DESCENT,
    set_task_ids_per_column,
)

# Extra clearance past the last riser so the whole ~0.82 m-long M20 body (not just the root) has
# cleared the stairs before we call it a crossing.
ROBOT_LENGTH_MARGIN = 0.3

# Front wheel axle is ~0.4 m ahead of the root on the 0.82 m-long M20.
FRONT_WHEEL_OFFSET = 0.4


def _yaw(quat_wxyz: torch.Tensor) -> torch.Tensor:
    w, x, y, z = quat_wxyz.unbind(-1)
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _wrap(angle: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(angle), torch.cos(angle))


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


def build_env_cfg(combos: list[dict]) -> object:
    """Start from `--task`'s own registered env cfg and swap in the eval terrain + fixed command.

    Deliberately does NOT touch `observations` (besides disabling noise) -- whatever the loaded
    checkpoint's obs shape is (blind or sighted), changing it here would break `ppo_runner.load()`.
    Resolved dynamically via the gym registry (`load_cfg_from_registry`) rather than hardcoding a
    single task's config class, so this same fixed-geometry benchmark works for any registered M20
    task (blind `Rough-...`, sighted `Stairs-Sighted-...`, etc.) -- same terrain/commands, whatever
    observations/rewards/architecture that task's own cfg defines.
    """
    env_cfg = load_cfg_from_registry(args_cli.task, "env_cfg_entry_point")
    env_cfg.scene.num_envs = len(combos) * args_cli.repeats
    env_cfg.seed = args_cli.seed
    env_cfg.episode_length_s = args_cli.time_out_s

    # ---- fixed evaluation terrain (replaces the training terrain generator instance; does not
    # mutate any shared object, per command.md's ground rules) ----
    env_cfg.scene.terrain.terrain_generator = build_stairs_benchmark_generator(combos)
    env_cfg.scene.terrain.max_init_terrain_level = 0

    # single-row terrain has no difficulty levels to promote/demote through. Covers both the
    # blind Rough cfg's 3 curriculum terms and the stairs cfgs' extra 3.4 per-task logging terms
    # (`terrain_level_by_task`) -- those bake in a `task_ids_per_column` list sized for the
    # *training* terrain's column count, which would silently index out-of-bounds against this
    # eval terrain's different column layout if left active (hasattr-guarded since not every task
    # cfg has all of these).
    for _name in (
        "terrain_levels",
        "gait_level",
        "command_levels",
        "terrain_level_flat_rough",
        "terrain_level_stair_ascent",
        "terrain_level_stair_descent",
    ):
        if hasattr(env_cfg.curriculum, _name):
            setattr(env_cfg.curriculum, _name, None)

    # Stair-aware reward/termination terms (e.g. `Stairs-Sighted-V2-...`) bake in a raw terrain
    # column -> task ID list sized for the *training* terrain. Re-point them at this eval terrain's
    # columns (one per combo), or the stair-specific termination would silently use the wrong
    # limits (or index out of bounds). No-op for tasks without such terms.
    set_task_ids_per_column(
        env_cfg,
        [TASK_STAIR_ASCENT if c["direction"] == "ascent" else TASK_STAIR_DESCENT for c in combos],
    )

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

    # Video camera: default to dynamically tracking one robot's root pose every frame (see
    # `isaaclab.envs.ui.viewport_camera_controller`'s `origin_type="asset_root"` handling) instead
    # of Isaac Lab's default fixed world-space viewer camera, which frames the whole tiled grid of
    # envs from far away and isn't useful for actually watching one robot climb.
    if args_cli.follow_env_id >= 0:
        env_cfg.viewer.origin_type = "asset_root"
        env_cfg.viewer.asset_name = "robot"
        env_cfg.viewer.env_index = args_cli.follow_env_id
        env_cfg.viewer.eye = tuple(args_cli.camera_eye)
        env_cfg.viewer.lookat = tuple(args_cli.camera_lookat)

    return env_cfg


def write_outputs(rows: list[dict]) -> None:
    os.makedirs(args_cli.out_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    csv_path = os.path.join(args_cli.out_dir, f"eval_stairs_{timestamp}.csv")
    fieldnames = [
        "speed", "direction", "step_height", "tread_width", "env_id",
        "outcome", "time_to_cross_s", "collisions", "stumbles",
        "yaw_drift_deg", "yaw_drift_on_stairs_deg", "lateral_offset_m",
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
        f"# Stairs evaluation -- `{args_cli.task}`",
        "",
        f"Checkpoint: `{resolve_checkpoint(args_cli.checkpoint)}`",
        "",
        f"{len(rows)} trials total ({args_cli.repeats} repeats x {len(args_cli.tread_widths)} tread widths per"
        " (speed, direction, step height) cell). Tread widths are aggregated together in this summary;"
        " see the CSV for the per-tread-width breakdown.",
        "",
        "Yaw drift / lateral offset: heading change and sideways (+ = left) displacement from the start"
        " pose, measured when the trial ends (crossing, fall or timeout); mean +/- std over trials."
        " 'On stairs' counts only the heading change after the front wheels reach the first riser."
        " The command has zero yaw rate and no heading control, so any drift comes from the policy.",
        "",
        "| speed (m/s) | direction | step height (m) | n | success rate | mean time-to-cross (s) |"
        " collisions/trial | stumbles/trial | falls | yaw drift (deg) | yaw drift on stairs (deg) |"
        " lateral offset (m) |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
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

        def _mean_std(field):
            v = torch.tensor([r[field] for r in g], dtype=torch.float64)
            return f"{v.mean().item():+.2f} +/- {v.std(unbiased=False).item():.2f}"

        lines.append(
            f"| {speed:.1f} | {direction} | {height:.2f} | {n} | {success_rate * 100:.0f}% | "
            f"{mean_time:.2f} | {mean_collisions:.2f} | {mean_stumbles:.2f} | {len(falls)} | "
            f"{_mean_std('yaw_drift_deg')} | {_mean_std('yaw_drift_on_stairs_deg')} | {_mean_std('lateral_offset_m')} |"
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

    agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    if args_cli.video and not args_cli.no_heightmap_viz:
        from heightmap_overlay import HeightmapOverlay

        viz_env_id = max(args_cli.follow_env_id, 0)
        viz_combo = combos[int(env.unwrapped.scene.terrain.terrain_types[viz_env_id])]
        viz_dt = env.unwrapped.step_dt
        env = HeightmapOverlay(
            env,
            env_id=viz_env_id,
            label_fn=lambda step: (
                f"env {viz_env_id}: {viz_combo['direction']} h={viz_combo['step_height']:.2f} m"
                f" tread={viz_combo['tread_width']:.2f} m  t={step * viz_dt:.1f} s"
            ),
        )

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
    stair_start_x = torch.zeros(num_envs, device=device)
    for i, combo in enumerate(combo_of_env):
        n_steps = num_stairs_steps(combo["tread_width"])
        target_x[i] = PLATFORM_WIDTH / 2.0 + n_steps * combo["tread_width"] + ROBOT_LENGTH_MARGIN
        # the generator builds steps inward from the border, so the first riser sits where the
        # (actual, smaller-than-`platform_width`) centre platform ends
        stair_start_x[i] = (TILE_SIZE - 2 * SUB_BORDER_WIDTH) / 2.0 - n_steps * combo["tread_width"]

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
        y0 = robot.data.root_pos_w[:, 1].clone()
        yaw0 = _yaw(robot.data.root_quat_w)
        yaw_at_entry = torch.full((num_envs,), float("nan"), device=device)
        yaw_end = yaw0.clone()
        y_end = y0.clone()

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
            # heading / sideways drift, frozen once a trial is finalized (so post-crossing motion
            # doesn't count); stair entry = front wheels (~0.4 m ahead of the root) at the first riser
            yaw_now = _yaw(robot.data.root_quat_w)
            live = ~finalized
            entered = live & torch.isnan(yaw_at_entry) & (progress >= stair_start_x - FRONT_WHEEL_OFFSET)
            yaw_at_entry = torch.where(entered, yaw_now, yaw_at_entry)
            yaw_end = torch.where(live, yaw_now, yaw_end)
            y_end = torch.where(live, robot.data.root_pos_w[:, 1], y_end)
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
                "yaw_drift_deg": math.degrees(_wrap(yaw_end[i] - yaw0[i]).item()),
                "yaw_drift_on_stairs_deg": (
                    math.degrees(_wrap(yaw_end[i] - yaw_at_entry[i]).item())
                    if not torch.isnan(yaw_at_entry[i]) else 0.0
                ),
                "lateral_offset_m": (y_end[i] - y0[i]).item(),
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
