# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause

"""Sim-to-real robustness terms: observation latency and motor strength.

New terms only (the shared reward/event/observation functions are untouched), used by the Lite3
domain-randomization helpers in `config/quadruped/deeprobotics_lite3/domain_rand.py`.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.actuators import ImplicitActuator
from isaaclab.assets import Articulation
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.utils.buffers import DelayBuffer

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    from isaaclab.managers import EventTermCfg, ObservationTermCfg

__all__ = ["delayed_observation", "randomize_actuator_gains_and_strength"]


class delayed_observation(ManagerTermBase):
    """Wrap an observation function with a per-env latency of `min_lag`..`max_lag` policy steps.

    The lag is resampled per env at every reset; until an env has `lag` steps of history the newest
    value is returned (isaaclab `DelayBuffer`). The wrapped function's own parameters go in
    `inner_params`; the managers only resolve top-level `SceneEntityCfg`s (and require the params
    to match the signature exactly), so the ones in `inner_params` are resolved here. Noise/scale/clip
    of the term apply to the delayed value, as they would to a late sensor reading.
    """

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.min_lag = int(cfg.params.get("min_lag", 0))
        self.max_lag = int(cfg.params["max_lag"])
        self.buffer = DelayBuffer(self.max_lag, env.num_envs, env.device)
        self._sample_lags(None)
        for value in cfg.params.get("inner_params", {}).values():
            if isinstance(value, SceneEntityCfg):
                value.resolve(env.scene)

    def _sample_lags(self, env_ids: Sequence[int] | None) -> None:
        n = self.num_envs if env_ids is None else len(env_ids)
        # DelayBuffer stores lags as int32
        lags = torch.randint(self.min_lag, self.max_lag + 1, (n,), device=self.device, dtype=torch.int)
        self.buffer.set_time_lag(lags, env_ids)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        self.buffer.reset(env_ids)
        self._sample_lags(env_ids)

    def __call__(
        self,
        env: ManagerBasedEnv,
        inner_func: Callable[..., torch.Tensor],
        inner_params: dict,
        max_lag: int,
        min_lag: int = 0,
    ) -> torch.Tensor:
        return self.buffer.compute(inner_func(env, **inner_params))


class randomize_actuator_gains_and_strength(ManagerTermBase):
    """Per reset: PD gains and motor strength, both drawn fresh from the asset defaults.

    Per env and joint: Kp = default x kp_scale x strength, Kd = default x kd_scale x strength,
    torque limit = nominal x strength. Scaling Kp, Kd and the torque limit by one factor scales the
    whole torque-speed output of the motor (a weaker or stronger motor), on top of independent gain
    errors. Replaces `randomize_actuator_gains` (same gain ranges), so the two never compound.
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.asset: Articulation = env.scene[cfg.params.get("asset_cfg", SceneEntityCfg("robot")).name]
        self.nominal_effort = {name: act.effort_limit.clone() for name, act in self.asset.actuators.items()}

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | None,
        stiffness_scale_range: tuple[float, float],
        damping_scale_range: tuple[float, float],
        strength_range: tuple[float, float],
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> None:
        if env_ids is None:
            env_ids = torch.arange(env.scene.num_envs, device=self.asset.device)
        env_ids = torch.as_tensor(env_ids, device=self.asset.device)
        data = self.asset.data

        def uniform(shape, bounds):
            return torch.empty(shape, device=self.asset.device).uniform_(*bounds)

        for name, act in self.asset.actuators.items():
            cols = slice(None) if isinstance(act.joint_indices, slice) else act.joint_indices
            shape = (len(env_ids), act.num_joints)
            strength = uniform(shape, strength_range)
            stiffness = data.default_joint_stiffness[env_ids][:, cols] * uniform(shape, stiffness_scale_range) * strength
            damping = data.default_joint_damping[env_ids][:, cols] * uniform(shape, damping_scale_range) * strength
            act.stiffness[env_ids] = stiffness
            act.damping[env_ids] = damping
            act.effort_limit[env_ids] = self.nominal_effort[name][env_ids] * strength
            if isinstance(act, ImplicitActuator):
                self.asset.write_joint_stiffness_to_sim(stiffness, joint_ids=act.joint_indices, env_ids=env_ids)
                self.asset.write_joint_damping_to_sim(damping, joint_ids=act.joint_indices, env_ids=env_ids)
                self.asset.write_joint_effort_limit_to_sim(act.effort_limit[env_ids], joint_ids=act.joint_indices, env_ids=env_ids)
