# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause

"""M20 domain randomization DR1 (docs/plans/M20_Domain_Randomization_DR1.md).

Subclasses the vendor rough config, which stays untouched. Rewards are the vendor rewards unchanged, so the
effect of DR alone is visible. Kept as is: kp/kd +-15 %, ground friction 0.35-1.5, base mass and
centre-of-mass ranges, pushes, external force, observation noise (vendor levels, deliberately not halved
even though real sensor noise is far lower), knee torque limit (76.4 N*m).

| Change                    | Range                                       | Where                                     |
| ------------------------- | ------------------------------------------- | ----------------------------------------- |
| Drive friction            | 0-3 N*m wheels, 0-0.5 N*m legs              | FrictionDelayedPDActuator (asset config)  |
| Latency                   | 1-3 physics steps (5-15 ms)                 | same actuator: min_delay=1, max_delay=3   |
| Dropped actions           | previous action held 1 policy step, 2 %     | events.action_repeat + repeatable actions |
| IMU bias                  | +-1 deg roll and pitch per env              | events.imu_bias + biased gravity obs      |
| Wheel armature            | scale 0.5-1.5 around 0.00243                | events.randomize_wheel_armature           |
| Restitution               | 0-0.2 (was 0-0.7)                           | events.randomize_rigid_body_material      |
| Starting joint positions  | +-0.1 rad                                   | events.randomize_reset_joints (by offset) |
"""

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import rl_training.tasks.manager_based.locomotion.velocity.mdp as mdp

from .rough_env_cfg import DeeproboticsM20RoughEnvCfg

from rl_training.assets.deeprobotics import DEEPROBOTICS_M20_DR1_CFG  # isort: skip

ACTION_REPEAT_PROBABILITY = 0.02
IMU_BIAS_DEG = 1.0
WHEEL_ARMATURE_SCALE = (0.5, 1.5)
RESTITUTION_RANGE = (0.0, 0.2)
START_JOINT_OFFSET = 0.1  # rad


def _as_cfg(cfg, new_cls, **overrides):
    """Copy an action term config into ``new_cls`` (a subclass), keeping every field except class_type."""
    values = {f: getattr(cfg, f) for f in cfg.__dataclass_fields__ if f != "class_type"}
    return new_cls(**values, **overrides)


@configclass
class DeeproboticsM20DR1EnvCfg(DeeproboticsM20RoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # ------------------------------Actuators: drive friction + 1-3 step delay------------------------------
        self.scene.robot = DEEPROBOTICS_M20_DR1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # ------------------------------Observations------------------------------
        # noise stays at the vendor levels (ang vel +-0.2, gravity +-0.05, joint vel +-1.5)
        policy = self.observations.policy
        # the policy sees gravity through the per-env IMU mounting offset; the critic keeps the true value
        policy.projected_gravity.func = mdp.projected_gravity_imu_bias
        policy.projected_gravity.params = {"asset_cfg": SceneEntityCfg("robot")}

        # ------------------------------Actions: dropped packets------------------------------
        self.actions.joint_pos = _as_cfg(self.actions.joint_pos, mdp.RepeatableJointPositionActionCfg, count_repeats=True)
        self.actions.joint_vel = _as_cfg(self.actions.joint_vel, mdp.RepeatableJointVelocityActionCfg, count_repeats=False)

        # ------------------------------Events------------------------------
        step_dt = self.decimation * self.sim.dt
        self.events.action_repeat = EventTerm(
            func=mdp.sample_action_repeat,
            mode="interval",
            interval_range_s=(step_dt, step_dt),
            is_global_time=True,
            params={"probability": ACTION_REPEAT_PROBABILITY},
        )
        self.events.imu_bias = EventTerm(
            func=mdp.reset_imu_bias,
            mode="reset",
            params={
                "roll_range_deg": (-IMU_BIAS_DEG, IMU_BIAS_DEG),
                "pitch_range_deg": (-IMU_BIAS_DEG, IMU_BIAS_DEG),
            },
        )
        self.events.randomize_wheel_armature = EventTerm(
            func=mdp.randomize_joint_parameters,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=self.wheel_joint_names),
                "armature_distribution_params": WHEEL_ARMATURE_SCALE,
                "operation": "scale",
                "distribution": "uniform",
            },
        )
        self.events.randomize_rigid_body_material.params["restitution_range"] = list(RESTITUTION_RANGE)
        # reset_joints_by_scale with (1, 1) left the joints at the default pose
        self.events.randomize_reset_joints = EventTerm(
            func=mdp.reset_joints_by_offset,
            mode="reset",
            params={
                "position_range": (-START_JOINT_OFFSET, START_JOINT_OFFSET),
                "velocity_range": (0.0, 0.0),
            },
        )

        # ------------------------------Logging (confirms each DR1 term is active)------------------------------
        self.curriculum.dr1_state = CurrTerm(func=mdp.dr1_state)

        # zero-weight vendor rewards are only pruned for the exact vendor class name, so prune them here
        if self.__class__.__name__ == "DeeproboticsM20DR1EnvCfg":
            self.disable_zero_weight_rewards()
