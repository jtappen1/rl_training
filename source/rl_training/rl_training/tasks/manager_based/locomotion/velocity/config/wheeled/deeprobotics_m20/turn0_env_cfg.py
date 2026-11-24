# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause

"""M20 Turn0: Cmd1 without the scripted turn-in-place gait, so the turning strategy can emerge.

Cmd1 keeps the vendor's turn-in-place shaping, which prescribes a diagonal stepping gait. Turn0 removes it
and instead makes the two physical costs of turning explicit, so the policy can choose between skid-steer
(wheels slide sideways) and stepping (lift and replace wheels) per situation:

| Change                     | Cmd1                                  | Turn0                                         |
| -------------------------- | ------------------------------------- | --------------------------------------------- |
| feet_air_time_ang_z_M20    | +50, turn in place                    | removed                                       |
| rotation_gait_symmetry     | +15, diagonal 50 % duty               | removed                                       |
| rotation_gait_status       | +2, one diagonal pair lifted 5 cm     | removed                                       |
| feet_slide_ang_z_cmd       | -2, wheel motion rel. to body in turn | removed                                       |
| wheel_lateral_slip (new)   | -                                     | -0.5 x sum |slip along axle| (m/s), always on |
| joint_power                | 0 (vendor note: -2e-5, legs)          | -1e-4 x sum |torque x speed| (W), all joints  |

Both new penalties are scaled by gait_level (0 -> 1 as mean terrain level goes 0 -> 3), like the vendor
gait terms, so early learning is not dominated by them. Rough magnitudes once gait_level = 1: skid-steer at
1 rad/s ~ -0.6 per step from slip; 200 W ~ -0.02 per step from power; tracking terms sum to ~11.
"""

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import rl_training.tasks.manager_based.locomotion.velocity.mdp as mdp

from .cmd1_env_cfg import DeeproboticsM20Cmd1EnvCfg

WHEEL_SLIP_WEIGHT = -0.5
JOINT_POWER_WEIGHT = -1e-4


@configclass
class DeeproboticsM20Turn0EnvCfg(DeeproboticsM20Cmd1EnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # ------------------------------1. remove the scripted turn-in-place gait------------------------------
        # weight 0 (not None): disable_zero_weight_rewards() below removes them and reads .weight on every term
        for name in ("feet_air_time_ang_z_M20", "rotation_gait_symmetry", "rotation_gait_status", "feet_slide_ang_z_cmd"):
            getattr(self.rewards, name).weight = 0.0

        # ------------------------------2. wheel side slip (along the axle), always on------------------------------
        self.rewards.wheel_lateral_slip = RewTerm(
            func=mdp.wheel_lateral_slip,
            weight=WHEEL_SLIP_WEIGHT,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=self.foot_link_name),
                "asset_cfg": SceneEntityCfg("robot", body_names=self.foot_link_name),
                "axle_axis": (0.0, 1.0, 0.0),  # wheel joint axis is local -y in the M20 URDF; sign irrelevant
            },
        )

        # ------------------------------3. energy: mechanical power over all joints------------------------------
        self.rewards.joint_power.weight = JOINT_POWER_WEIGHT
        self.rewards.joint_power.params["asset_cfg"].joint_names = self.leg_joint_names + self.wheel_joint_names

        if self.__class__.__name__ == "DeeproboticsM20Turn0EnvCfg":
            self.disable_zero_weight_rewards()
