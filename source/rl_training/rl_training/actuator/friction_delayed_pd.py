# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause

"""Delayed PD actuator with randomized drive (Coulomb) friction, for M20 DR1.

See docs/plans/M20_Domain_Randomization_DR1.md. The PD model already matches the real robot within 2 %,
so only friction and delay are added on top of :class:`DelayedPDActuator`:

    tau = clip(PD(delayed command)) - tau_c * tanh(qdot / v_s)

``tau_c`` is drawn per env and per joint at every reset from ``drive_friction_range``. The friction acts
after the motor torque limit, like friction in the gearbox and bearings: a saturated motor still loses it.
``computed_effort`` stays the motor (PD) torque, so torque-limit statistics are unaffected by friction.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

import isaaclab.utils.math as math_utils
from isaaclab.actuators.actuator_pd import DelayedPDActuator
from isaaclab.actuators.actuator_pd_cfg import DelayedPDActuatorCfg
from isaaclab.utils import configclass
from isaaclab.utils.types import ArticulationActions


class FrictionDelayedPDActuator(DelayedPDActuator):
    """:class:`DelayedPDActuator` minus a smoothed Coulomb drive friction drawn per env at reset."""

    cfg: FrictionDelayedPDActuatorCfg

    def __init__(self, cfg: FrictionDelayedPDActuatorCfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        # per-env, per-joint friction torque (N*m); zero until the first reset
        self.drive_friction = torch.zeros(self._num_envs, self.num_joints, device=self._device)

    def reset(self, env_ids: Sequence[int] | None):
        super().reset(env_ids)
        if env_ids is None or env_ids == slice(None):
            env_ids = self._ALL_INDICES
        lo, hi = self.cfg.drive_friction_range
        self.drive_friction[env_ids] = math_utils.sample_uniform(
            lo, hi, (len(env_ids), self.num_joints), device=self._device
        )

    def compute(
        self, control_action: ArticulationActions, joint_pos: torch.Tensor, joint_vel: torch.Tensor
    ) -> ArticulationActions:
        control_action = super().compute(control_action, joint_pos, joint_vel)
        friction = self.drive_friction * torch.tanh(joint_vel / self.cfg.friction_velocity_scale)
        self.applied_effort = self.applied_effort - friction
        control_action.joint_efforts = self.applied_effort
        return control_action


@configclass
class FrictionDelayedPDActuatorCfg(DelayedPDActuatorCfg):
    """Configuration for :class:`FrictionDelayedPDActuator`."""

    class_type: type = FrictionDelayedPDActuator

    drive_friction_range: tuple[float, float] = (0.0, 0.0)
    """Range of the Coulomb friction torque tau_c (N*m), drawn per env and joint at every reset."""

    friction_velocity_scale: float = 0.1
    """Velocity v_s (rad/s) of the tanh smoothing around zero speed."""
