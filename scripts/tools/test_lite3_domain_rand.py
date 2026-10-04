"""Check the Lite3 domain randomization (domain_rand.py) on a live Stairs-Sighted-V4 env.

- payload: torso mass minus its nominal value lies in PAYLOAD_KG (beyond the x0.85-1.15 link scale);
- motor strength: per joint, Kp / default, Kd / default and torque limit / nominal share the strength
  factor (limit ratio) inside MOTOR_STRENGTH, and the gain ratios stay inside strength x gain range;
- actuator delay: sampled delays cover ACTUATOR_DELAY_PHYSICS_STEPS;
- proprioception latency: for envs with lag 1, the delayed joint_pos term equals the true value one
  policy step earlier; for lag 0, the current value.

    ./isaaclab.sh -p scripts/tools/test_lite3_domain_rand.py --headless
"""

import argparse
import sys

from isaaclab.app import AppLauncher

sys.stdout.reconfigure(line_buffering=True)

parser = argparse.ArgumentParser(description="Test the Lite3 domain randomization.")
parser.add_argument("--task", type=str, default="Stairs-Sighted-V4-Deeprobotics-Lite3-v0")
parser.add_argument("--num_envs", type=int, default=64)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app = AppLauncher(args_cli).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

from isaaclab_tasks.utils import load_cfg_from_registry  # noqa: E402

import rl_training.tasks  # noqa: F401, E402
from rl_training.assets.deeprobotics import DEEPROBOTICS_LITE3_CFG  # noqa: E402
from rl_training.tasks.manager_based.locomotion.velocity.config.quadruped.deeprobotics_lite3 import (  # noqa: E402
    domain_rand as dr,
)

failures = []


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")
    if not ok:
        failures.append(name)


def main():
    env_cfg = load_cfg_from_registry(args_cli.task, "env_cfg_entry_point")
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.events.randomize_push_robot = None  # keep the latency check free of velocity kicks
    env_cfg.observations.policy.enable_corruption = False  # exact comparison for the latency check
    env = gym.make(args_cli.task, cfg=env_cfg)
    base = env.unwrapped
    env.reset()
    robot = base.scene["robot"]

    # payload: torso mass vs the nominal (unrandomized) torso mass; the x0.85-1.15 link scale runs first
    torso = robot.find_bodies(env_cfg.base_link_name)[0][0]
    nominal = robot.data.default_mass[:, torso]
    mass = robot.root_physx_view.get_masses()[:, torso]
    lo = nominal.min() * 0.85 + dr.PAYLOAD_KG[0] - 1e-3
    hi = nominal.max() * 1.15 + dr.PAYLOAD_KG[1] + 1e-3
    added = mass - nominal
    check("payload: torso mass within scale x payload range", bool(((mass >= lo) & (mass <= hi)).all()),
          f"(added {added.min():+.2f}..{added.max():+.2f} kg)")
    check("payload: spread covers most of the range", float(added.max() - added.min()) > 1.5)

    # motor strength
    strengths, kp_ok, kd_ok = [], True, True
    for name, act in robot.actuators.items():
        nominal_limit = DEEPROBOTICS_LITE3_CFG.actuators[name].effort_limit
        s = act.effort_limit / nominal_limit
        strengths.append(s.flatten())
        cols = slice(None) if isinstance(act.joint_indices, slice) else act.joint_indices
        kp = act.stiffness / robot.data.default_joint_stiffness[:, cols]
        kd = act.damping / robot.data.default_joint_damping[:, cols]
        kp_ok &= bool(((kp / s >= 0.85 - 1e-4) & (kp / s <= 1.15 + 1e-4)).all())
        kd_ok &= bool(((kd / s >= 0.85 - 1e-4) & (kd / s <= 1.15 + 1e-4)).all())
    s = torch.cat(strengths)
    check("motor strength: torque limit factor within range", bool(((s >= dr.MOTOR_STRENGTH[0] - 1e-4) & (s <= dr.MOTOR_STRENGTH[1] + 1e-4)).all()),
          f"({s.min():.3f}..{s.max():.3f})")
    check("motor strength: Kp / strength within the gain range", kp_ok)
    check("motor strength: Kd / strength within the gain range", kd_ok)

    # actuator delay
    delays = torch.cat([act.positions_delay_buffer.time_lags.flatten() for act in robot.actuators.values()])
    check("actuator delay: covers the configured range",
          int(delays.min()) == dr.ACTUATOR_DELAY_PHYSICS_STEPS[0] and int(delays.max()) == dr.ACTUATOR_DELAY_PHYSICS_STEPS[1],
          f"({int(delays.min())}..{int(delays.max())} physics steps)")

    # proprioception latency on joint_pos: obs returned by env.step (one buffer push per step)
    names = base.observation_manager.active_terms["policy"]
    dims = base.observation_manager.group_obs_term_dim["policy"]
    start = sum(d[0] for d in dims[: names.index("joint_pos")])
    term_cfg = base.observation_manager._group_obs_term_cfgs["policy"][names.index("joint_pos")]
    lags = term_cfg.func.buffer.time_lags.clone()
    joint_ids = term_cfg.params["inner_params"]["asset_cfg"].joint_ids
    actions = torch.zeros(base.num_envs, base.action_manager.total_action_dim, device=base.device)
    true_prev, fresh = None, torch.zeros(base.num_envs, dtype=torch.bool, device=base.device)
    ok_lag0 = ok_lag1 = True
    for k in range(8):
        actions.normal_(0.0, 0.3)
        obs, _, terminated, truncated, _ = env.step(actions)
        true_now = (robot.data.joint_pos - robot.data.default_joint_pos)[:, joint_ids]
        seen = obs["policy"][:, start : start + 12]
        reset = (terminated | truncated).bool()
        lags = term_cfg.func.buffer.time_lags.clone()  # resampled for envs that just reset
        if k >= 1:
            valid = ~reset & ~fresh
            m0, m1 = valid & (lags == 0), valid & (lags == 1)
            ok_lag0 &= torch.allclose(seen[m0], true_now[m0], atol=1e-4)
            ok_lag1 &= torch.allclose(seen[m1], true_prev[m1], atol=1e-4)
        fresh = reset
        true_prev = true_now
    check("proprio latency: both lags present", bool((lags == 0).any() and (lags == 1).any()), f"({int((lags == 1).sum())}/{len(lags)} envs lag 1)")
    check("proprio latency: lag-0 envs see the current joint positions", ok_lag0)
    check("proprio latency: lag-1 envs see the previous step's joint positions", ok_lag1)

    print("[RESULT]", "all passed" if not failures else f"{len(failures)} failed: {failures}")
    env.close()


if __name__ == "__main__":
    main()
    app.close()
