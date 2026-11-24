# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause

"""M20 Cmd1: DR1 + command and reward changes aimed at the real-robot issues.

Real robot: slight constant forward roll, poor heading and velocity tracking, jerky turning with large
yaw-rate error, no lateral motion, low responsiveness. All of these also show in the sim eval, so they are
trained behaviour first. DR1 (with drive friction off) is unchanged underneath, so compare against a DR1
run, or the vendor retrain, with scripts/eval/m20_eval_suite.py.

| Change                               | Vendor / DR1                         | Cmd1                                       | Targets                          |
| ------------------------------------ | ------------------------------------ | ------------------------------------------ | -------------------------------- |
| Yaw command source                   | heading controller in 100 % of envs  | constant yaw rate in 80 %, heading in 20 % | heading/yaw tracking, jerky turn |
| Turn-in-place samples                | heading target                      | constant yaw rate from +-2 rad/s           | same                             |
| Command resampling                   | every 10 s                           | every 3-8 s                                | responsiveness                   |
| Lateral-only samples                 | 2 %                                  | 10 %                                       | lateral motion                   |
| Forward/backward-only samples        | 2 %                                  | 15 %                                       | straight driving, fwd and back   |
| Turn-in-place samples share          | 20 %                                 | 10 % (constant wz also in 80 % of the rest)| keeps room for straight driving  |
| hipx posture penalty                 | always on                            | off for lateral-only commands              | lateral motion                   |
| Fine lin. vel. tracking (added)      | -                                    | std 0.25, weight 2.0                       | velocity tracking, forward roll  |
| Fine yaw-rate tracking (added)       | -                                    | std 0.35, weight 1.0                       | yaw tracking                     |
| Wheel speed penalty at zero command  | weight 0                             | weight -0.05 (sum of abs wheel speed)      | forward roll                     |
| Stepping (air-time) reward gate      | any |wz cmd| > 0.1                  | also ||v_xy cmd|| < 0.5 (turn in place)    | no hopping while driving arcs    |
| Terrain curriculum                   | straight-line distance from start    | distance driven vs commanded               | circles no longer get demoted    |

The vendor tracking terms (std 0.707, weights 5 and 3) are kept; the fine terms add on top, so a 0.1 m/s
drift now costs ~15 % of the fine term instead of ~2 % of the vendor one.
"""

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass

import rl_training.tasks.manager_based.locomotion.velocity.mdp as mdp

from .dr1_env_cfg import DeeproboticsM20DR1EnvCfg

REL_HEADING_ENVS = 0.2
RESAMPLING_TIME_S = (3.0, 8.0)
REL_ONLY_LIN_Y_ENVS = 0.10
REL_ONLY_LIN_X_ENVS = 0.15
REL_ONLY_ANG_Z_ENVS = 0.10
STEPPING_LIN_CMD_MAX = 0.5  # m/s, same as the rotation gait terms
FINE_LIN_STD, FINE_LIN_WEIGHT = 0.25, 2.0
FINE_YAW_STD, FINE_YAW_WEIGHT = 0.35, 1.0
WHEEL_STANDSTILL_WEIGHT = -0.05
HIPX_LATERAL_EXEMPT_XZ_MAX = 0.3  # hipx penalty off when |vy| > 0.1 and ||(vx, wz)|| < this


@configclass
class DeeproboticsM20Cmd1EnvCfg(DeeproboticsM20DR1EnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # ------------------------------Commands------------------------------
        cmd = self.commands.base_velocity
        cmd.rel_heading_envs = REL_HEADING_ENVS
        cmd.only_ang_z_heading = False
        cmd.resampling_time_range = RESAMPLING_TIME_S
        cmd.rel_only_lin_y_envs = REL_ONLY_LIN_Y_ENVS
        cmd.rel_only_lin_x_envs = REL_ONLY_LIN_X_ENVS
        cmd.rel_only_ang_z_envs = REL_ONLY_ANG_Z_ENVS

        # ------------------------------Terrain curriculum------------------------------
        # constant yaw rates make robots drive in circles: judge distance driven vs commanded instead of
        # straight-line distance from the start (which promoted nobody and demoted almost everybody)
        self.curriculum.terrain_levels.params["use_path_length"] = True

        # ------------------------------Rewards------------------------------
        self.rewards.track_lin_vel_xy_fine = RewTerm(
            func=mdp.track_lin_vel_xy_exp,
            weight=FINE_LIN_WEIGHT,
            params={"command_name": "base_velocity", "std": FINE_LIN_STD},
        )
        self.rewards.track_ang_vel_z_fine = RewTerm(
            func=mdp.track_ang_vel_z_exp,
            weight=FINE_YAW_WEIGHT,
            params={"command_name": "base_velocity", "std": FINE_YAW_STD},
        )
        # stepping reward only when turning in place (it was also paid while driving arcs)
        self.rewards.feet_air_time_ang_z_M20.params["lin_cmd_max"] = STEPPING_LIN_CMD_MAX

        # sum of |wheel speed| when the command is ~zero and the body is slow (vendor term, weight was 0)
        self.rewards.wheel_vel_penalty.weight = WHEEL_STANDSTILL_WEIGHT

        # lateral stepping needs hipx; exempt it for lateral-only commands like hipy/knee already are.
        # The turn gate is disabled (threshold 1e9) so hipx stays penalised when turning in place.
        hipx = self.rewards.hipx_joint_pos_penalty
        hipx.func = mdp.joint_pos_penalty_except_turn_side_cmd
        hipx.params.update(
            {
                "ang_cmd_threshold": 1e9,
                "y_cmd_threshold": 0.1,
                "xy_norm_max": 0.1,
                "xz_norm_max": HIPX_LATERAL_EXEMPT_XZ_MAX,
            }
        )

        if self.__class__.__name__ == "DeeproboticsM20Cmd1EnvCfg":
            self.disable_zero_weight_rewards()
