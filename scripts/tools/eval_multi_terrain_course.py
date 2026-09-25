# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause

"""Multi-terrain straight-line course: an M20 evaluation with one robot and one long video.

Builds a single long, straight course (flat start, then stairs up/down at several step heights,
rough patches, ramps, ...; see `DEFAULT_COURSE`) as one height-field terrain, spawns one robot at
the start facing along it, and drives it with a fixed forward velocity command. Reports, per course
segment and overall: whether it got through, time, speed, heading drift, sideways drift, wheel-riser
stumbles, body collisions, peak pitch/roll. Writes a streamed mp4 (no in-RAM frame buffer, so long
runs are fine) with the policy height-map overlay, a trajectory CSV, a markdown report, and a
height / sideways-offset / heading plot along the course.

Steering modes (`--steering`):
- `straight` (default): heading control off, yaw-rate command 0 -- a joystick held straight
  forward. Any heading change comes from the policy itself, which is what "does it go straight"
  asks.
- `heading_hold`: the command generator's heading controller holds heading 0 (as in training /
  a navigation layer), so drift is corrected by the command, not the policy.

Usage:
    python scripts/tools/eval_multi_terrain_course.py --checkpoint <run_dir_or_model.pt> --headless
    python scripts/tools/eval_multi_terrain_course.py --checkpoint <...> --headless --speed 1.0 --steering heading_hold
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import sys

sys.stdout.reconfigure(line_buffering=True)

from isaaclab.app import AppLauncher

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "reinforcement_learning", "rsl_rl")))
import cli_args  # noqa: E402

parser = argparse.ArgumentParser(description="Multi-terrain straight-line course eval for an M20 checkpoint.")
parser.add_argument("--checkpoint", type=str, required=True, help="model_*.pt, or a run dir (latest checkpoint used).")
parser.add_argument("--task", type=str, default="Stairs-Sighted-V2-Deeprobotics-M20-v0")
parser.add_argument("--speed", type=float, default=0.7, help="Fixed forward command (m/s).")
parser.add_argument("--steering", choices=["straight", "heading_hold"], default="straight")
parser.add_argument("--friction", type=float, default=1.0, help="Fixed ground/wheel friction (replaces the DR range).")
parser.add_argument("--max_time_s", type=float, default=None, help="Time budget; default 1.6 x course length / speed.")
parser.add_argument("--stuck_time_s", type=float, default=8.0, help="End the run if < 0.2 m of progress in this long.")
parser.add_argument("--no_video", action="store_true", default=False)
parser.add_argument("--fps", type=int, default=25, help="Video frame rate (policy runs at 50 Hz).")
parser.add_argument("--camera_eye", type=float, nargs=3, default=[-2.0, 3.2, 1.4], help="Camera offset from the robot (world frame).")
parser.add_argument("--camera_lookat", type=float, nargs=3, default=[1.2, 0.0, 0.0])
parser.add_argument("--out_dir", type=str, default="eval")
parser.add_argument("--seed", type=int, default=42)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if not args_cli.no_video:
    args_cli.enable_cameras = True
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import csv
import glob
import math
import re
from datetime import datetime

import gymnasium as gym
import numpy as np
import torch

import importlib.metadata as metadata
from packaging import version

from rsl_rl.runners import OnPolicyRunner

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils import load_cfg_from_registry

import rl_training.tasks  # noqa: F401
from rl_training.tasks.manager_based.locomotion.velocity.config.wheeled.deeprobotics_m20.eval_terrains import (
    COURSE_WIDTH,
    DEFAULT_COURSE,
    START_X,
    build_course_generator,
    layout,
)
from rl_training.tasks.manager_based.locomotion.velocity.config.wheeled.deeprobotics_m20.stairs_env_cfg import (
    TASK_STAIR_ASCENT,
    set_task_ids_per_column,
)

installed_version = metadata.version("rsl-rl-lib")

FINISH_MARGIN = 2.5  # "finished" = root within this of the course end
WHEEL_RADIUS = 0.09  # M20 wheel collision cylinder (urdf)
STOPPED_RIM_SPEED = 0.1  # m/s; below this a planted wheel counts as stopped


def resolve_checkpoint(path: str) -> str:
    if os.path.isfile(path):
        return path
    candidates = glob.glob(os.path.join(path, "model_*.pt"))
    if not candidates:
        raise FileNotFoundError(f"No model_*.pt checkpoints found under '{path}'.")
    return max(candidates, key=lambda p: int(re.search(r"model_(\d+)\.pt$", p).group(1)))


def _yaw(q: torch.Tensor) -> torch.Tensor:
    w, x, y, z = q.unbind(-1)
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _wrap(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


def build_env_cfg(course_len: float, time_budget: float):
    env_cfg = load_cfg_from_registry(args_cli.task, "env_cfg_entry_point")
    env_cfg.scene.num_envs = 1
    env_cfg.seed = args_cli.seed
    env_cfg.episode_length_s = time_budget + 5.0

    env_cfg.scene.terrain.terrain_generator = build_course_generator(DEFAULT_COURSE, args_cli.seed)
    env_cfg.scene.terrain.max_init_terrain_level = 0
    for name in ("terrain_levels", "gait_level", "command_levels", "terrain_level_flat_rough",
                 "terrain_level_stair_ascent", "terrain_level_stair_descent"):
        if hasattr(env_cfg.curriculum, name):
            setattr(env_cfg.curriculum, name, None)
    # One column, mixed terrain: give stair-aware terms the stair limits everywhere (the looser
    # 0.9 pitch termination), so the course never ends early on a pitch the stairs need.
    set_task_ids_per_column(env_cfg, [TASK_STAIR_ASCENT])

    env_cfg.events.randomize_reset_base.params = {
        "pose_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0), "yaw": (0.0, 0.0)},
        "velocity_range": {k: (0.0, 0.0) for k in ("x", "y", "z", "roll", "pitch", "yaw")},
    }
    env_cfg.events.randomize_push_robot = None
    env_cfg.events.randomize_apply_external_force_torque = None
    mat = env_cfg.events.randomize_rigid_body_material.params
    mat["static_friction_range"] = [args_cli.friction, args_cli.friction]
    mat["dynamic_friction_range"] = [args_cli.friction, args_cli.friction]
    mat["restitution_range"] = [0.0, 0.0]

    cmd = env_cfg.commands.base_velocity
    cmd.rel_standing_envs = 0.0
    cmd.rel_zero_vel_envs = 0.0
    cmd.rel_only_lin_x_envs = 0.0
    cmd.rel_only_lin_y_envs = 0.0
    cmd.rel_only_ang_z_envs = 0.0
    cmd.resampling_time_range = (1.0e6, 1.0e6)
    cmd.ranges.lin_vel_x = (args_cli.speed, args_cli.speed)
    cmd.ranges.lin_vel_y = (0.0, 0.0)
    cmd.ranges.ang_vel_z = (-1.0, 1.0) if args_cli.steering == "heading_hold" else (0.0, 0.0)
    cmd.heading_command = args_cli.steering == "heading_hold"
    cmd.rel_heading_envs = 1.0
    cmd.ranges.heading = (0.0, 0.0)
    cmd.debug_vis = False

    env_cfg.observations.policy.enable_corruption = False

    env_cfg.viewer.origin_type = "asset_root"
    env_cfg.viewer.asset_name = "robot"
    env_cfg.viewer.env_index = 0
    env_cfg.viewer.eye = tuple(args_cli.camera_eye)
    env_cfg.viewer.lookat = tuple(args_cli.camera_lookat)
    return env_cfg


def segment_at(segs: list[dict], x: float) -> int:
    for i, s in enumerate(segs):
        if x < s["x1"]:
            return i
    return len(segs) - 1


def write_plot(path: str, segs: list[dict], traj: list[dict], end_note: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, muted, grid_c, surface = "#1f1f1e", "#6b6a64", "#e6e5df", "#fcfcfb"
    series, terrain_c = "#2a78d6", "#c9c8c0"
    x = np.array([r["course_x"] for r in traj])
    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True, gridspec_kw={"height_ratios": [2.2, 1, 1]})
    fig.patch.set_facecolor(surface)

    # terrain centreline profile from the layout (rough shown at its base level)
    tx, tz = [], []
    for s in segs:
        if s["kind"] in ("ascent", "descent"):
            sign = 1 if s["kind"] == "ascent" else -1
            for k in range(s["steps"]):
                z = s["z0"] + sign * s["step_height"] * (k + 1)
                tx += [s["x0"] + k * s["tread"], s["x0"] + (k + 1) * s["tread"]]
                tz += [z, z]
        else:
            tx += [s["x0"], s["x1"]]
            tz += [s["z0"], s["z1"]]
    ax = axes[0]
    ax.fill_between(tx, np.min(tz) - 0.3, tz, step=None, color=terrain_c, linewidth=0)
    ax.plot(x, [r["base_z"] for r in traj], color=series, linewidth=2)
    ax.set_ylabel("height (m)", color=muted)
    ax.set_title(f"Multi-terrain straight-line course: robot base height over the terrain  |  {end_note}", loc="left", color=ink, fontsize=11)
    for s in segs:
        if s["kind"] not in ("flat", "landing"):
            ax.text((s["x0"] + s["x1"]) / 2, max(tz) + 0.9, s["label"], rotation=90, ha="center", va="top",
                    fontsize=7.5, color=muted)
    ax.set_ylim(np.min(tz) - 0.3, max(tz) + 1.0)

    axes[1].plot(x, [r["lateral_m"] for r in traj], color=series, linewidth=2)
    axes[1].axhline(0.0, color=muted, linewidth=1)
    axes[1].set_ylabel("sideways (m)\n+ = left", color=muted)
    axes[1].set_title("Sideways offset from the start line", loc="left", color=ink, fontsize=10)

    axes[2].plot(x, [r["yaw_deg"] for r in traj], color=series, linewidth=2)
    axes[2].axhline(0.0, color=muted, linewidth=1)
    axes[2].set_ylabel("heading (deg)\n+ = left", color=muted)
    axes[2].set_title("Heading relative to the course direction", loc="left", color=ink, fontsize=10)
    axes[2].set_xlabel("distance along course (m)", color=muted)

    for a in axes:
        a.set_facecolor(surface)
        for s in segs:
            a.axvline(s["x0"], color=grid_c, linewidth=0.8, zorder=0)
        a.grid(axis="y", color=grid_c, linewidth=0.8)
        a.tick_params(colors=muted, labelsize=8)
        for sp in a.spines.values():
            sp.set_visible(False)
    axes[2].set_xlim(0, segs[-1]["x1"])
    fig.tight_layout()
    fig.savefig(path, dpi=120, facecolor=surface)
    plt.close(fig)


def main():
    segs = layout(DEFAULT_COURSE)
    course_len = segs[-1]["x1"]
    finish_x = course_len - FINISH_MARGIN
    time_budget = args_cli.max_time_s or 1.6 * (finish_x - START_X) / max(args_cli.speed, 0.1)

    env_cfg = build_env_cfg(course_len, time_budget)
    checkpoint = resolve_checkpoint(args_cli.checkpoint)
    print(f"[INFO] checkpoint: {checkpoint}")
    print(f"[INFO] course: {len(segs)} segments, {course_len:.1f} m; speed {args_cli.speed} m/s;"
          f" steering {args_cli.steering}; budget {time_budget:.0f} s")

    agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None if args_cli.no_video else "rgb_array")

    hud = {"text": ""}
    overlay = None
    if not args_cli.no_video:
        from heightmap_overlay import HeightmapOverlay

        overlay = HeightmapOverlay(env, env_id=0, label_fn=lambda step: hud["text"])
        env = overlay

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    train_cfg = agent_cfg.to_dict()
    train_cfg.pop("actor", None)
    train_cfg.pop("critic", None)
    if version.parse(installed_version) >= version.parse("5.0.0"):
        train_cfg = cli_args.convert_rsl_rl_cfg_dict(train_cfg)
    runner = OnPolicyRunner(env, train_cfg, log_dir=None, device=agent_cfg.device)
    runner.load(checkpoint)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    base_env = env.unwrapped
    robot = base_env.scene["robot"]
    contact = base_env.scene["contact_forces"]
    wheel_ids = torch.tensor(contact.find_bodies(env_cfg.foot_link_name)[0], device=base_env.device)
    body_ids = torch.tensor(contact.find_bodies(f"^(?!.*{env_cfg.foot_link_name}).*")[0], device=base_env.device)
    # per-wheel logging, in a fixed fl/fr/hl/hr order on both the contact sensor and the joints
    wheel_tags = ["fl", "fr", "hl", "hr"]
    wheel_body_ids = torch.tensor(
        contact.find_bodies([f"{w}_wheel" for w in wheel_tags], preserve_order=True)[0], device=base_env.device
    )
    wheel_joint_ids = torch.tensor(
        robot.find_joints([f"{w}_wheel_joint" for w in wheel_tags], preserve_order=True)[0], device=base_env.device
    )
    dt = base_env.step_dt

    obs, _ = env.reset()
    origin = base_env.scene.env_origins[0].clone()
    yaw0 = _yaw(robot.data.root_quat_w)[0].item()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args_cli.out_dir, f"multi_terrain_course_{timestamp}_{args_cli.steering}_v{args_cli.speed:.1f}")
    os.makedirs(run_dir, exist_ok=True)
    writer = None
    if overlay is not None:
        import imageio.v2 as imageio

        writer = imageio.get_writer(os.path.join(run_dir, "course.mp4"), fps=args_cli.fps, quality=7,
                                    macro_block_size=8)
        frame_every = max(1, int(round(1.0 / (dt * args_cli.fps))))

    per_seg = [dict(t_in=None, t_out=None, yaw_in=None, yaw_out=None, y_in=None, y_out=None, stumbles=0,
                    collisions=0, max_pitch=0.0, max_roll=0.0) for _ in segs]
    traj = []
    prev_stumble = prev_collide = False
    progress_ref = (0.0, START_X)
    outcome, end_seg = "timeout", None

    steps = int(time_budget / dt)
    for step in range(steps):
        with torch.inference_mode():
            actions = policy(obs)
        obs, _, dones, extras = env.step(actions)
        t = (step + 1) * dt
        # Any termination (fall, out of bounds -- which Isaac Lab flags as a time-out -- or the
        # episode timer) auto-resets the env inside step(), so the robot state read below would be
        # the respawn pose: stop before recording it and keep the reason.
        if bool(dones[0]):
            tm = base_env.termination_manager
            fired = [n for n in tm.active_terms if bool(tm.get_term(n)[0])]
            outcome = {"bad_orientation_2": "fell", "terrain_out_of_bounds": "left the terrain",
                       "time_out": "timeout"}.get(fired[0] if fired else "", f"terminated ({', '.join(fired)})")
            end_seg = segment_at(segs, traj[-1]["course_x"]) if traj else 0
            break

        pos = robot.data.root_pos_w[0]
        cx = (pos[0] - origin[0]).item() + START_X
        lat = (pos[1] - origin[1]).item()
        yaw = _wrap(_yaw(robot.data.root_quat_w)[0].item() - yaw0)
        g = robot.data.projected_gravity_b[0]
        pitch = math.degrees(math.asin(max(-1.0, min(1.0, -g[0].item()))))
        roll = math.degrees(math.asin(max(-1.0, min(1.0, g[1].item()))))
        si = segment_at(segs, cx)

        f = contact.data.net_forces_w[0]
        stumble = bool(torch.any(torch.linalg.norm(f[wheel_ids, :2], dim=-1) > 4.0 * f[wheel_ids, 2].abs()))
        # rim speed = |wheel angular velocity| x radius; grounded = in contact; on_riser = pressed
        # against a vertical face (same test as the stumble metric)
        rim = (robot.data.joint_vel[0, wheel_joint_ids].abs() * WHEEL_RADIUS).tolist()
        grounded = (contact.data.current_contact_time[0, wheel_body_ids] > 0.0).tolist()
        fw = f[wheel_body_ids]
        on_riser = (torch.linalg.norm(fw[:, :2], dim=-1) > 4.0 * fw[:, 2].abs()).tolist()
        base_speed = torch.linalg.norm(robot.data.root_lin_vel_w[0, :2]).item()
        collide = bool(torch.any(torch.linalg.norm(f[body_ids], dim=-1) > 1.0))

        ps = per_seg[si]
        if ps["t_in"] is None:
            ps.update(t_in=t, yaw_in=yaw, y_in=lat)
        ps.update(t_out=t, yaw_out=yaw, y_out=lat)
        ps["stumbles"] += int(stumble and not prev_stumble)
        ps["collisions"] += int(collide and not prev_collide)
        ps["max_pitch"] = max(ps["max_pitch"], abs(pitch))
        ps["max_roll"] = max(ps["max_roll"], abs(roll))
        prev_stumble, prev_collide = stumble, collide

        traj.append(dict(t=round(t, 3), course_x=cx, lateral_m=lat, base_z=(pos[2] - origin[2]).item(),
                         yaw_deg=math.degrees(yaw), pitch_deg=pitch, roll_deg=roll, segment=segs[si]["label"],
                         stumble=int(stumble), collision=int(collide), base_speed=base_speed,
                         **{f"rim_{w}": rim[k] for k, w in enumerate(wheel_tags)},
                         **{f"ground_{w}": int(grounded[k]) for k, w in enumerate(wheel_tags)},
                         **{f"riser_{w}": int(on_riser[k]) for k, w in enumerate(wheel_tags)}))
        hud["text"] = (f"{segs[si]['label']}  |  t {t:5.1f} s  x {cx:5.1f} m  heading {math.degrees(yaw):+5.1f} deg"
                       f"  sideways {lat:+.2f} m")

        if writer is not None and step % frame_every == 0:
            frame = overlay.render()
            if frame is not None:
                writer.append_data(frame)

        if cx >= finish_x:
            outcome, end_seg = "finished", si
            break
        if abs(lat) > COURSE_WIDTH / 2 - 0.5:
            outcome, end_seg = "drove off the side of the course", si
            break
        if cx > progress_ref[1] + 0.2:
            progress_ref = (t, cx)
        elif t - progress_ref[0] > args_cli.stuck_time_s:
            outcome, end_seg = "stuck", si
            break

    if writer is not None:
        writer.close()
    env.close()

    # ------------------------------ report ------------------------------
    last = traj[-1]
    end_note = (f"{outcome} after {last['t']:.1f} s at x = {last['course_x']:.1f} m"
                + (f" ({segs[end_seg]['label']})" if outcome != "finished" else ""))
    with open(os.path.join(run_dir, "trajectory.csv"), "w", newline="") as fcsv:
        w = csv.DictWriter(fcsv, fieldnames=list(traj[0].keys()))
        w.writeheader()
        w.writerows(traj)
    write_plot(os.path.join(run_dir, "course_plot.png"), segs, traj, end_note)

    lines = [
        f"# Multi-terrain straight-line course -- `{args_cli.task}`",
        "",
        f"- Checkpoint: `{checkpoint}`",
        f"- Command: {args_cli.speed} m/s forward, steering `{args_cli.steering}`"
        + (" (heading control off, yaw-rate command 0)" if args_cli.steering == "straight"
           else " (heading controller holds heading 0)"),
        f"- Friction {args_cli.friction}, seed {args_cli.seed}, course {course_len:.1f} m x {COURSE_WIDTH:.0f} m",
        f"- **Result: {end_note}**",
        f"- Final heading {last['yaw_deg']:+.1f} deg, final sideways offset {last['lateral_m']:+.2f} m"
        f" (max |sideways| {max(abs(r['lateral_m']) for r in traj):.2f} m); + = left",
        "",
        "Per segment (only segments the robot reached). Heading / sideways change is measured across the"
        " segment; speed is along the course.",
        "",
        "| # | segment | time (s) | speed (m/s) | heading change (deg) | sideways change (m) | stumbles |"
        " collisions | max pitch (deg) | max roll (deg) |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for i, (s, ps) in enumerate(zip(segs, per_seg)):
        if ps["t_in"] is None:
            continue
        dur = ps["t_out"] - ps["t_in"] + dt
        seg_x = [r["course_x"] for r in traj if s["x0"] <= r["course_x"] < s["x1"]]
        speed = (max(seg_x) - min(seg_x)) / dur if seg_x else float("nan")
        lines.append(
            f"| {i} | {s['label']} | {dur:.1f} | {speed:.2f} | {math.degrees(_wrap(ps['yaw_out'] - ps['yaw_in'])):+.1f} |"
            f" {ps['y_out'] - ps['y_in']:+.2f} | {ps['stumbles']} | {ps['collisions']} | {ps['max_pitch']:.0f} |"
            f" {ps['max_roll']:.0f} |"
        )
    lines += [
        "",
        "## Wheel use per segment",
        "",
        f"Rim speed = |wheel angular velocity| x {WHEEL_RADIUS} m. 'Planted & stopped' = share of grounded-wheel"
        f" time with rim speed < {STOPPED_RIM_SPEED} m/s. 'On riser' = a wheel pressed against a vertical face"
        " (the stumble test); its rim speed shows whether the wheel keeps driving into the riser or stops."
        " 'Airborne' = wheels off the ground (lifted or stepping).",
        "",
        "| # | segment | base speed (m/s) | grounded rim speed (m/s) | planted & stopped | airborne share |"
        " airborne rim speed (m/s) | on-riser share | on-riser rim speed (m/s) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for i, s in enumerate(segs):
        rows_s = [r for r in traj if s["x0"] <= r["course_x"] < s["x1"]]
        if not rows_s:
            continue
        ground, air, riser = [], [], []
        for r in rows_s:
            for w in wheel_tags:
                (ground if r[f"ground_{w}"] else air).append(r[f"rim_{w}"])
                if r[f"riser_{w}"]:
                    riser.append(r[f"rim_{w}"])
        n_wheel = 4 * len(rows_s)
        mean = lambda v: (sum(v) / len(v)) if v else float("nan")  # noqa: E731
        stopped = (sum(1 for v in ground if v < STOPPED_RIM_SPEED) / len(ground)) if ground else float("nan")
        lines.append(
            f"| {i} | {s['label']} | {mean([r['base_speed'] for r in rows_s]):.2f} | {mean(ground):.2f} |"
            f" {100 * stopped:.0f}% | {100 * len(air) / n_wheel:.0f}% | {mean(air):.2f} |"
            f" {100 * len(riser) / n_wheel:.0f}% | {mean(riser):.2f} |"
        )
    lines += ["", "Files: `course.mp4` (if recorded), `course_plot.png`, `trajectory.csv`."]
    with open(os.path.join(run_dir, "report.md"), "w") as fmd:
        fmd.write("\n".join(lines) + "\n")
    print(f"[INFO] {end_note}")
    print(f"[INFO] wrote {run_dir}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback

        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
