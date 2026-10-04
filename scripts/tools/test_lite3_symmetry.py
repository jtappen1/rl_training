"""Check the Lite3 left-right symmetry augmentation on real observations.

- mirroring twice gives back the input (every obs group and the actions);
- the augmented batch keeps the originals first;
- the joint permutation swaps FL<->FR and HL<->HR by name, in the env's actual joint order;
- the height-scan rows are the y axis, so flipping rows maps each ray to y -> -y with x unchanged.

    ./isaaclab.sh -p scripts/tools/test_lite3_symmetry.py --headless
"""

import argparse
import sys

from isaaclab.app import AppLauncher

sys.stdout.reconfigure(line_buffering=True)

parser = argparse.ArgumentParser(description="Test Lite3 left-right symmetry augmentation.")
parser.add_argument("--task", type=str, default="Stairs-Sighted-V2-Deeprobotics-Lite3-v0")
parser.add_argument("--num_envs", type=int, default=16)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app = AppLauncher(args_cli).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import load_cfg_from_registry  # noqa: E402

import rl_training.tasks  # noqa: F401, E402
from rl_training.tasks.manager_based.locomotion.velocity.config.quadruped.deeprobotics_lite3.symmetry import (  # noqa: E402
    _LR_PERM,
    compute_symmetric_states_lr,
)

failures = []


def check(name, ok):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}")
    if not ok:
        failures.append(name)


def main():
    env_cfg = load_cfg_from_registry(args_cli.task, "env_cfg_entry_point")
    env_cfg.scene.num_envs = args_cli.num_envs
    env = RslRlVecEnvWrapper(gym.make(args_cli.task, cfg=env_cfg))
    base = env.unwrapped
    for _ in range(20):
        env.step(torch.randn(base.num_envs, 12, device=base.device) * 0.5)
    obs = env.get_observations()
    actions = torch.randn(base.num_envs, 12, device=base.device)
    n = base.num_envs

    aug, act_aug = compute_symmetric_states_lr(env, obs, actions)
    twice, act_twice = compute_symmetric_states_lr(env, aug[n:], act_aug[n:])
    for group in obs.keys():
        check(f"{group}: originals kept first", torch.equal(aug[group][:n], obs[group]))
        check(f"{group}: mirror twice = identity", torch.allclose(twice[group][n:], obs[group]))
        check(f"{group}: mirror changes something", not torch.allclose(aug[group][n:], obs[group]))
    check("actions: mirror twice = identity", torch.allclose(act_twice[n:], actions))

    names = base.action_manager.get_term("joint_pos")._joint_names
    swap = {"FL": "FR", "FR": "FL", "HL": "HR", "HR": "HL"}
    check(
        f"joint permutation swaps legs by name ({names[:3]} ...)",
        all(names[_LR_PERM[i]] == swap[names[i][:2]] + names[i][2:] for i in range(12)),
    )

    sensor = base.scene.sensors["height_scanner"]
    starts = sensor.ray_starts[0]  # (num_rays, 3), pattern frame
    xy = starts[:, :2]
    # identity of the mirror on positions: apply the same row flip as the obs mirror to the ray positions
    pattern = sensor.cfg.pattern_cfg
    ny = len(torch.arange(-pattern.size[1] / 2, pattern.size[1] / 2 + 1.0e-9, pattern.resolution))
    flipped = xy.view(ny, -1, 2).flip(dims=[0]).reshape(-1, 2)
    check("height scan: flipped rays sit at y -> -y", torch.allclose(flipped[:, 1], -xy[:, 1], atol=1e-5))
    check("height scan: flipped rays keep x", torch.allclose(flipped[:, 0], xy[:, 0], atol=1e-5))

    print("[RESULT]", "all passed" if not failures else f"{len(failures)} failed: {failures}")
    env.close()


if __name__ == "__main__":
    main()
    app.close()
