"""Smoke test for the M20 navigation env's frozen low-level policy wrapper (Step 1e).

Drives the nav env with fixed high-level (vx, vy, wz) commands and checks the low-level
policy tracks them without falling.

Tracking test (headless, several commands in parallel, ``--num_per_cmd`` envs each):
    isaaclab.sh -p scripts/tools/nav_smoke_test.py --headless

Video (one robot, scripted command sequence, camera follows it):
    isaaclab.sh -p scripts/tools/nav_smoke_test.py --headless --video --enable_cameras
"""

import argparse
import os
import sys
from datetime import datetime

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_per_cmd", type=int, default=8, help="envs per command in the tracking test")
parser.add_argument("--seconds", type=float, default=10.0, help="tracking test duration")
parser.add_argument("--warmup", type=float, default=2.0, help="seconds ignored at the start of the tracking test")
parser.add_argument("--video", action="store_true", help="record a video of a scripted command sequence instead")
parser.add_argument(
    "--baseline",
    action="store_true",
    help="run the tracking test in the ORIGINAL low-level flat env (no nav wrapper) for comparison",
)
parser.add_argument("--out_dir", type=str, default="eval", help="where videos are written")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

sys.stdout.reconfigure(line_buffering=True)

import cv2  # noqa: E402
import imageio  # noqa: E402
import torch  # noqa: E402

from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402

from rl_training.tasks.manager_based.locomotion.velocity.config.wheeled.deeprobotics_m20.flat_env_cfg import (  # noqa: E402
    DeeproboticsM20FlatEnvCfg,
)
from rl_training.tasks.manager_based.navigation.config.wheeled.deeprobotics_m20.navigation_env_cfg import (  # noqa: E402
    LOW_LEVEL_POLICY_PATH,
    M20NavFlatEnvCfg,
)
from rl_training.tasks.manager_based.navigation.mdp.pretrained_policy_action import _load_actor  # noqa: E402

# (vx, vy, wz) commands for the tracking test. The last one is outside the limits and should be clamped to vx=2.
TRACK_CMDS = [
    (1.0, 0.0, 0.0),
    (2.0, 0.0, 0.0),
    (-1.0, 0.0, 0.0),
    (0.0, 0.5, 0.0),
    (0.5, 0.3, 0.0),
    (0.0, 0.0, 1.0),
    (1.0, 0.0, 1.0),
    (3.0, 0.0, 0.0),
]

# (duration s, (vx, vy, wz), label) for the video
VIDEO_SEQ = [
    (2.0, (0.0, 0.0, 0.0), "stand"),
    (3.0, (1.0, 0.0, 0.0), "forward 1.0 m/s"),
    (3.0, (2.0, 0.0, 0.0), "forward 2.0 m/s"),
    (3.0, (0.0, 0.0, 1.0), "turn in place 1.0 rad/s"),
    (4.0, (1.0, 0.0, 0.8), "arc 1.0 m/s, 0.8 rad/s"),
    (3.0, (0.0, 0.5, 0.0), "strafe 0.5 m/s"),
    (3.0, (-1.0, 0.0, 0.0), "backward 1.0 m/s"),
]
# Commands are blended linearly into each new segment over this long. Stepping from rest straight to
# vx >= 1.8 makes the low-level policy drive BACKWARDS ~15% of the time; ramping avoids it.
VIDEO_BLEND_S = 0.5


def make_env(num_envs: int, video: bool) -> ManagerBasedRLEnv:
    cfg = M20NavFlatEnvCfg()
    cfg.scene.num_envs = num_envs
    cfg.sim.device = args.device
    cfg.episode_length_s = 60.0  # long enough that time_out never fires during the test
    if video:
        cfg.scene.env_spacing = 8.0
        cfg.viewer.origin_type = "asset_root"
        cfg.viewer.asset_name = "robot"
        cfg.viewer.env_index = 0
        cfg.viewer.eye = (-2.5, -2.5, 1.6)
        cfg.viewer.lookat = (0.0, 0.0, 0.2)
        cfg.viewer.resolution = (960, 540)
        cfg.sim.render_interval = cfg.decimation  # render once per env step (= one video frame), not 5x
    return ManagerBasedRLEnv(cfg, render_mode="rgb_array" if video else None)


def measured_vel(env: ManagerBasedRLEnv) -> torch.Tensor:
    robot = env.scene["robot"]
    return torch.cat([robot.data.root_lin_vel_b[:, :2], robot.data.root_ang_vel_b[:, 2:3]], dim=1)


def print_ll_layout(env: ManagerBasedRLEnv):
    term = env.action_manager.get_term("velocity_command")
    om = term._ll_obs_manager
    names = om.active_terms["ll_policy"]
    dims = om.group_obs_term_dim["ll_policy"]
    print("[layout] low-level obs terms:", ", ".join(f"{n}{list(d)}" for n, d in zip(names, dims)))
    print("[layout] low-level obs dim:", om.group_obs_dim["ll_policy"], "| action split:", term._ll_dims)
    print(f"[layout] physics dt {env.physics_dt:.4f} s | env step_dt {env.step_dt:.3f} s "
          f"({1 / env.step_dt:.0f} Hz high level) | low-level every {term.cfg.low_level_decimation} physics steps")


def make_baseline_env(num_envs: int):
    """The original low-level flat env, with the command generator frozen to values we write."""
    cfg = DeeproboticsM20FlatEnvCfg()
    cfg.scene.num_envs = num_envs
    cfg.sim.device = args.device
    cfg.episode_length_s = 60.0
    cfg.observations.policy.enable_corruption = False
    cfg.events.randomize_push_robot = None
    cfg.events.randomize_apply_external_force_torque = None
    cfg.curriculum.command_levels = None
    env = ManagerBasedRLEnv(cfg)
    term = env.command_manager.get_term("base_velocity")
    term._resample_command = lambda env_ids: None  # never resample
    term._update_command = lambda: None  # no heading control / standing-env zeroing
    return env, term


def run_tracking(env, step_fn, n):
    cmd = torch.tensor(TRACK_CMDS, device=env.device).repeat_interleave(n, dim=0)
    steps = int(args.seconds / env.step_dt)
    warm = int(args.warmup / env.step_dt)
    vel_sum = torch.zeros_like(cmd)
    falls = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    for i in range(steps):
        terminated = step_fn(cmd)
        falls += terminated.long()
        if i >= warm:
            vel_sum += measured_vel(env)
    return cmd, vel_sum / (steps - warm), falls


def print_tracking(title, seen, vel, falls, n):
    print(f"\n[tracking] {title}: {n} envs per command, {args.seconds:.0f} s, first {args.warmup:.0f} s ignored")
    print(f"{'command (vx, vy, wz)':>24} | {'seen by low level':>20} | {'measured mean':>22} | {'env std':>18} | falls")
    for k, c in enumerate(TRACK_CMDS):
        s = slice(k * n, (k + 1) * n)
        m, sd, p = vel[s].mean(0), vel[s].std(0), seen[s][0]
        print(f"{str(c):>24} | {fmt(p):>20} | {fmt(m):>22} | {fmt(sd):>18} | {int(falls[s].sum())}")
    print("[tracking] per-env measured vx/vy/wz:")
    for k, c in enumerate(TRACK_CMDS):
        s = slice(k * n, (k + 1) * n)
        print(f"  {str(c):>18}: " + " ".join(fmt(v) for v in vel[s]))


def tracking_test():
    n = args.num_per_cmd
    if args.baseline:
        env, cmd_term = make_baseline_env(n * len(TRACK_CMDS))
        policy = _load_actor(LOW_LEVEL_POLICY_PATH, (512, 256, 128), env.device)
        obs, _ = env.reset()
        state = {"obs": obs}
        lo = torch.tensor([-2.0, -1.0, -2.0], device=env.device)  # same limits as the nav wrapper's clamp

        def step_fn(cmd):
            cmd_term.vel_command_b[:] = torch.clamp(cmd, lo, -lo)
            with torch.no_grad():
                act = policy(state["obs"]["policy"])
            state["obs"], _, terminated, _, _ = env.step(act)
            return terminated

        # the first obs was computed before we wrote the command: write it and recompute once
        cmd_term.vel_command_b[:] = 0.0
        cmd, vel, falls = run_tracking(env, step_fn, n)
        print_tracking("BASELINE (original low-level env)", cmd_term.vel_command_b, vel, falls, n)
    else:
        env = make_env(n * len(TRACK_CMDS), video=False)
        print_ll_layout(env)
        env.reset()
        term = env.action_manager.get_term("velocity_command")

        def step_fn(cmd):
            return env.step(cmd)[2]

        cmd, vel, falls = run_tracking(env, step_fn, n)
        print_tracking("NAV WRAPPER", term.processed_actions, vel, falls, n)
    env.close()


def fmt(t: torch.Tensor) -> str:
    return "(" + ", ".join(f"{x:+.2f}" for x in t.tolist()) + ")"


def video():
    env = make_env(1, video=True)
    print_ll_layout(env)
    env.reset()
    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, f"nav_smoke_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4")
    fps = round(1 / env.step_dt)
    writer = imageio.get_writer(path, fps=fps, codec="libx264", quality=8)

    falls = 0
    prev = torch.zeros(1, 3, device=env.device)
    blend_steps = max(1, round(VIDEO_BLEND_S / env.step_dt))
    for duration, c, label in VIDEO_SEQ:
        target = torch.tensor([c], device=env.device)
        errs = []
        for i in range(int(duration / env.step_dt)):
            cmd = prev + (target - prev) * min(1.0, (i + 1) / blend_steps)
            _, _, terminated, _, _ = env.step(cmd)
            falls += int(terminated.sum())
            v = measured_vel(env)[0]
            errs.append(v)
            frame = env.render()
            if frame is None:
                continue
            frame = frame[..., :3].copy()
            overlay(frame, [f"cmd   {label}  {fmt(cmd[0])}", f"meas  {fmt(v)}"])
            writer.append_data(frame)
        m = torch.stack(errs[len(errs) // 3 :]).mean(0)  # skip the first third of each segment (transient)
        print(f"[video] {label:<26} cmd {fmt(target[0])} measured {fmt(m)}")
        prev = target
    writer.close()
    print(f"[video] falls: {falls} | wrote {path}")
    env.close()


def overlay(frame, lines):
    for i, text in enumerate(lines):
        y = 36 + 34 * i
        cv2.putText(frame, text, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame, text, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)


if __name__ == "__main__":
    try:
        video() if args.video else tracking_test()
    finally:
        app.close()
