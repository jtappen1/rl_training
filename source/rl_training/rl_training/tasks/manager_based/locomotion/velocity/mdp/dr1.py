# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause

"""MDP terms for M20 domain randomization DR1 (docs/plans/M20_Domain_Randomization_DR1.md).

* IMU bias: :func:`reset_imu_bias` draws a roll/pitch mounting offset per env at reset;
  :func:`projected_gravity_imu_bias` is the policy's gravity observation seen through that offset.
* Action repeat: :func:`sample_action_repeat` (an interval event that runs every policy step) picks envs
  whose next action is dropped; :class:`RepeatableJointPositionAction` and
  :class:`RepeatableJointVelocityAction` then keep the previous targets for one policy step.
* :func:`dr1_state` is a curriculum term that only logs the DR1 values (TensorBoard ``Curriculum/dr1_state/*``),
  so a short run can confirm that every new term is active.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg, JointVelocityActionCfg
from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction, JointVelocityAction
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv


##
# IMU bias
##


def reset_imu_bias(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    roll_range_deg: tuple[float, float] = (-1.0, 1.0),
    pitch_range_deg: tuple[float, float] = (-1.0, 1.0),
):
    """Draw a per-env IMU mounting offset (roll, pitch) and store it as ``env.imu_bias_quat``."""
    if not hasattr(env, "imu_bias_quat"):
        env.imu_bias_quat = torch.zeros(env.num_envs, 4, device=env.device)
        env.imu_bias_quat[:, 0] = 1.0
        env.imu_bias_rp_deg = torch.zeros(env.num_envs, 2, device=env.device)
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    n = len(env_ids)
    roll = math_utils.sample_uniform(*roll_range_deg, (n,), device=env.device)
    pitch = math_utils.sample_uniform(*pitch_range_deg, (n,), device=env.device)
    env.imu_bias_rp_deg[env_ids, 0] = roll
    env.imu_bias_rp_deg[env_ids, 1] = pitch
    env.imu_bias_quat[env_ids] = math_utils.quat_from_euler_xyz(
        torch.deg2rad(roll), torch.deg2rad(pitch), torch.zeros_like(roll)
    )


def projected_gravity_imu_bias(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Gravity in the frame of an IMU mounted with the per-env offset from :func:`reset_imu_bias`.

    Before the first reset (observation-shape probing at start-up) there is no offset yet.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    gravity_b = asset.data.projected_gravity_b
    bias = getattr(env, "imu_bias_quat", None)
    if bias is None:
        return gravity_b
    return math_utils.quat_apply_inverse(bias, gravity_b)


##
# Action repeat (dropped policy packets)
##


def sample_action_repeat(env: ManagerBasedEnv, env_ids: torch.Tensor | None, probability: float = 0.02):
    """Pick the envs whose next policy action is dropped (the previous targets are held for one step).

    Use as an ``interval`` event with ``interval_range_s = (step_dt, step_dt)`` and ``is_global_time=True``:
    it then runs once per policy step, after resets and before the next action is processed.
    """
    if not hasattr(env, "action_repeat_mask"):
        env.action_repeat_mask = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        env.action_repeat_count = torch.zeros((), dtype=torch.long, device=env.device)
        env.action_step_count = torch.zeros((), dtype=torch.long, device=env.device)
    ids = slice(None) if env_ids is None else env_ids
    env.action_repeat_mask[ids] = torch.rand(env.action_repeat_mask[ids].shape, device=env.device) < probability


class _RepeatPreviousMixin:
    """Keeps the previous raw and processed actions for envs flagged in ``env.action_repeat_mask``."""

    def process_actions(self, actions: torch.Tensor):
        mask = getattr(self._env, "action_repeat_mask", None)
        if mask is None or not bool(mask.any()):
            super().process_actions(actions)
        else:
            prev_raw = self._raw_actions.clone()
            prev_processed = self._processed_actions.clone()
            super().process_actions(actions)
            self._raw_actions[mask] = prev_raw[mask]
            self._processed_actions[mask] = prev_processed[mask]
        if mask is not None and getattr(self.cfg, "count_repeats", False):
            self._env.action_repeat_count += mask.sum()
            self._env.action_step_count += mask.numel()

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        super().reset(env_ids)
        mask = getattr(self._env, "action_repeat_mask", None)
        if mask is not None:
            mask[slice(None) if env_ids is None else env_ids] = False


class RepeatableJointPositionAction(_RepeatPreviousMixin, JointPositionAction):
    cfg: RepeatableJointPositionActionCfg


class RepeatableJointVelocityAction(_RepeatPreviousMixin, JointVelocityAction):
    cfg: RepeatableJointVelocityActionCfg


@configclass
class RepeatableJointPositionActionCfg(JointPositionActionCfg):
    class_type: type = RepeatableJointPositionAction
    count_repeats: bool = True
    """Count repeats for the :func:`dr1_state` log (enable on exactly one action term)."""


@configclass
class RepeatableJointVelocityActionCfg(JointVelocityActionCfg):
    class_type: type = RepeatableJointVelocityAction
    count_repeats: bool = False


##
# Logging
##


def dr1_state(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    wheel_joint_names: str = ".*_wheel_joint",
) -> dict[str, float]:
    """Log the DR1 randomization values (means over all envs). Changes nothing."""
    robot: Articulation = env.scene[asset_cfg.name]
    out: dict[str, float] = {}
    for name, act in robot.actuators.items():
        if hasattr(act, "drive_friction"):
            out[f"{name}_drive_friction_Nm"] = act.drive_friction.mean().item()
        if hasattr(act, "positions_delay_buffer"):
            out[f"{name}_delay_steps"] = act.positions_delay_buffer.time_lags.float().mean().item()
    bias = getattr(env, "imu_bias_rp_deg", None)
    if bias is not None:
        out["imu_bias_abs_roll_deg"] = bias[:, 0].abs().mean().item()
        out["imu_bias_abs_pitch_deg"] = bias[:, 1].abs().mean().item()
    steps = getattr(env, "action_step_count", None)
    if steps is not None and int(steps) > 0:
        out["action_repeat_rate"] = (env.action_repeat_count.float() / steps.float()).item()
    wheel_ids, _ = robot.find_joints(wheel_joint_names)
    if wheel_ids:
        out["wheel_armature"] = robot.data.joint_armature[:, wheel_ids].mean().item()
    return out


__all__ = [
    "reset_imu_bias",
    "projected_gravity_imu_bias",
    "sample_action_repeat",
    "RepeatableJointPositionAction",
    "RepeatableJointVelocityAction",
    "RepeatableJointPositionActionCfg",
    "RepeatableJointVelocityActionCfg",
    "dr1_state",
]

