# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause

"""Left-right symmetry augmentation for Lite3 PPO (rsl-rl `symmetry_cfg`, Mittal et al. ICRA 2024).

Only the left-right mirror: stair tasks are not front-back symmetric (front-biased height scan,
ascent vs descent), so isaaclab's ANYmal front-back/diagonal copies would teach the wrong thing.
Unlike that example, the critic group is mirrored too, so value targets match the mirrored states.

Term slices are looked up by name from the observation manager, and an unknown term raises, so an
observation change can never be silently left unmirrored.
"""

from __future__ import annotations

import torch
from tensordict import TensorDict

# joint order in obs and actions: FL, FR, HL, HR x (HipX, HipY, Knee)
_LR_PERM = [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]
# mirrored HipX (abduction) angles change sign; HipY/Knee pitch joints do not
_LR_SIGN = [-1.0, 1.0, 1.0] * 4

_VECTOR_SIGNS = {
    "base_lin_vel": (1.0, -1.0, 1.0),
    "base_ang_vel": (-1.0, 1.0, -1.0),
    "projected_gravity": (1.0, -1.0, 1.0),
    "velocity_commands": (1.0, -1.0, -1.0),
}
_JOINT_TERMS = ("joint_pos", "joint_vel", "actions")


def _mirror_joints(x: torch.Tensor) -> torch.Tensor:
    return x[:, _LR_PERM] * torch.tensor(_LR_SIGN, device=x.device)


def _mirror_height_scan(env, x: torch.Tensor) -> torch.Tensor:
    pattern = env.scene.sensors["height_scanner"].cfg.pattern_cfg
    ny = len(torch.arange(-pattern.size[1] / 2, pattern.size[1] / 2 + 1.0e-9, pattern.resolution))
    nx = x.shape[1] // ny
    if nx * ny != x.shape[1] or env.scene.sensors["height_scanner"].cfg.offset.pos[1] != 0.0:
        raise ValueError("Height-scan mirroring needs an 'xy' grid centred on the robot's x axis.")
    # 'xy' ordering: rows are y, columns are x; mirroring y reverses the rows
    return x.view(-1, ny, nx).flip(dims=[1]).reshape(-1, ny * nx)


def _mirror_group(env, group: str, obs: torch.Tensor) -> torch.Tensor:
    om = env.observation_manager
    out = obs.clone()
    start = 0
    for name, dims in zip(om.active_terms[group], om.group_obs_term_dim[group]):
        width = int(torch.tensor(dims).prod())
        sl = slice(start, start + width)
        if name in _VECTOR_SIGNS:
            out[:, sl] = obs[:, sl] * torch.tensor(_VECTOR_SIGNS[name], device=obs.device)
        elif name in _JOINT_TERMS:
            out[:, sl] = _mirror_joints(obs[:, sl])
        elif name == "height_scan":
            out[:, sl] = _mirror_height_scan(env, obs[:, sl])
        else:
            raise KeyError(f"No left-right mirror defined for observation term '{group}/{name}'")
        start += width
    if start != obs.shape[1]:
        raise ValueError(f"Observation group '{group}' has {obs.shape[1]} dims, terms cover {start}")
    return out


@torch.no_grad()
def compute_symmetric_states_lr(env, obs: TensorDict | None = None, actions: torch.Tensor | None = None):
    """Original + left-right mirrored copies of observations (every group) and actions."""
    base_env = env.unwrapped
    obs_aug = None
    if obs is not None:
        n = obs.batch_size[0]
        obs_aug = obs.repeat(2)
        for group in obs.keys():
            obs_aug[group][n:] = _mirror_group(base_env, group, obs[group])
    actions_aug = None
    if actions is not None:
        actions_aug = torch.cat((actions, _mirror_joints(actions)), dim=0)
    return obs_aug, actions_aug
