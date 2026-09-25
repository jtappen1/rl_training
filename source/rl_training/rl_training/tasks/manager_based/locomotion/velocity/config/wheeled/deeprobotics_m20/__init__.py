# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause
# 
# # Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="Flat-Deeprobotics-M20-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:DeeproboticsM20FlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsM20FlatPPORunnerCfg",
        "cusrl_cfg_entry_point": f"{agents.__name__}.cusrl_ppo_cfg:DeeproboticsM20FlatTrainerCfg",
    },
)

gym.register(
    id="Rough-Deeprobotics-M20-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:DeeproboticsM20RoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsM20RoughPPORunnerCfg",
        "cusrl_cfg_entry_point": f"{agents.__name__}.cusrl_ppo_cfg:DeeproboticsM20RoughTrainerCfg",
    },
)

gym.register(
    id="Skatepark-Deeprobotics-M20-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.skatepark_env_cfg:DeeproboticsM20SkateparkEnvCfg",
        # reuses the Rough runner cfg (same experiment_name/log dir) so `play.py` finds the
        # already-trained checkpoint under logs/rsl_rl/deeprobotics_m20_rough automatically.
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsM20RoughPPORunnerCfg",
        "cusrl_cfg_entry_point": f"{agents.__name__}.cusrl_ppo_cfg:DeeproboticsM20RoughTrainerCfg",
    },
)

gym.register(
    id="Skatepark-Deeprobotics-M20-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        # keyboard-driving variant; see `skatepark_env_cfg.DeeproboticsM20SkateparkEnvCfg_PLAY`
        "env_cfg_entry_point": f"{__name__}.skatepark_env_cfg:DeeproboticsM20SkateparkEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsM20RoughPPORunnerCfg",
        "cusrl_cfg_entry_point": f"{agents.__name__}.cusrl_ppo_cfg:DeeproboticsM20RoughTrainerCfg",
    },
)

gym.register(
    id="Skatepark-V3b-Deeprobotics-M20-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        # keyboard-driving the sighted stairs v3b policy; see
        # `skatepark_env_cfg.DeeproboticsM20SkateparkV3bEnvCfg_PLAY`. Uses the v3b runner cfg so
        # `play.py` finds the latest checkpoint under logs/rsl_rl/deeprobotics_m20_stairs_sighted_v3b.
        "env_cfg_entry_point": f"{__name__}.skatepark_env_cfg:DeeproboticsM20SkateparkV3bEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsM20StairsSightedV3bPPORunnerCfg",
        "cusrl_cfg_entry_point": f"{agents.__name__}.cusrl_ppo_cfg:DeeproboticsM20RoughTrainerCfg",
    },
)

gym.register(
    id="Stairs-Bench-V3b-Deeprobotics-M20-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        # keyboard-driving v3b on the eval_stairs.py benchmark tiles (+ clean depth camera); see
        # `eval_play_env_cfg.DeeproboticsM20StairsBenchV3bEnvCfg_PLAY`
        "env_cfg_entry_point": f"{__name__}.eval_play_env_cfg:DeeproboticsM20StairsBenchV3bEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsM20StairsSightedV3bPPORunnerCfg",
        "cusrl_cfg_entry_point": f"{agents.__name__}.cusrl_ppo_cfg:DeeproboticsM20RoughTrainerCfg",
    },
)

gym.register(
    id="Course-V3b-Deeprobotics-M20-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        # keyboard-driving v3b on the eval_multi_terrain_course.py course (+ clean depth camera); see
        # `eval_play_env_cfg.DeeproboticsM20CourseV3bEnvCfg_PLAY`
        "env_cfg_entry_point": f"{__name__}.eval_play_env_cfg:DeeproboticsM20CourseV3bEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsM20StairsSightedV3bPPORunnerCfg",
        "cusrl_cfg_entry_point": f"{agents.__name__}.cusrl_ppo_cfg:DeeproboticsM20RoughTrainerCfg",
    },
)

gym.register(
    id="Stairs-Teacher-Deeprobotics-M20-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.stairs_env_cfg:DeeproboticsM20StairsTeacherEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsM20StairsTeacherPPORunnerCfg",
        "cusrl_cfg_entry_point": f"{agents.__name__}.cusrl_ppo_cfg:DeeproboticsM20RoughTrainerCfg",
    },
)

gym.register(
    id="Stairs-Teacher-Deeprobotics-M20-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.stairs_env_cfg:DeeproboticsM20StairsTeacherEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsM20StairsTeacherPPORunnerCfg",
        "cusrl_cfg_entry_point": f"{agents.__name__}.cusrl_ppo_cfg:DeeproboticsM20RoughTrainerCfg",
    },
)

gym.register(
    id="Stairs-Sighted-Deeprobotics-M20-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        # Plain single-task/single-critic sighted validation run -- not Milestone A's MoE
        # architecture. See `stairs_env_cfg.DeeproboticsM20StairsSightedEnvCfg`'s docstring.
        "env_cfg_entry_point": f"{__name__}.stairs_env_cfg:DeeproboticsM20StairsSightedEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsM20StairsSightedPPORunnerCfg",
        "cusrl_cfg_entry_point": f"{agents.__name__}.cusrl_ppo_cfg:DeeproboticsM20RoughTrainerCfg",
    },
)

gym.register(
    id="Stairs-Sighted-Deeprobotics-M20-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.stairs_env_cfg:DeeproboticsM20StairsSightedEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsM20StairsSightedPPORunnerCfg",
        "cusrl_cfg_entry_point": f"{agents.__name__}.cusrl_ppo_cfg:DeeproboticsM20RoughTrainerCfg",
    },
)

gym.register(
    id="Stairs-Sighted-V2-Deeprobotics-M20-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        # Sighted spike v2: stair-aware rewards, terrain and curriculum. See
        # `stairs_env_cfg.DeeproboticsM20StairsSightedV2EnvCfg`'s docstring.
        "env_cfg_entry_point": f"{__name__}.stairs_env_cfg:DeeproboticsM20StairsSightedV2EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsM20StairsSightedV2PPORunnerCfg",
        "cusrl_cfg_entry_point": f"{agents.__name__}.cusrl_ppo_cfg:DeeproboticsM20RoughTrainerCfg",
    },
)

gym.register(
    id="Stairs-Sighted-V2-Deeprobotics-M20-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.stairs_env_cfg:DeeproboticsM20StairsSightedV2EnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsM20StairsSightedV2PPORunnerCfg",
        "cusrl_cfg_entry_point": f"{agents.__name__}.cusrl_ppo_cfg:DeeproboticsM20RoughTrainerCfg",
    },
)

gym.register(
    id="Stairs-Sighted-V3-Deeprobotics-M20-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        # Sighted spike v3: v2 + heading-hold reward/commands, no stepping reward. See
        # `stairs_env_cfg.DeeproboticsM20StairsSightedV3EnvCfg`'s docstring.
        "env_cfg_entry_point": f"{__name__}.stairs_env_cfg:DeeproboticsM20StairsSightedV3EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsM20StairsSightedV3PPORunnerCfg",
        "cusrl_cfg_entry_point": f"{agents.__name__}.cusrl_ppo_cfg:DeeproboticsM20RoughTrainerCfg",
    },
)

gym.register(
    id="Stairs-Sighted-V3-Deeprobotics-M20-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.stairs_env_cfg:DeeproboticsM20StairsSightedV3EnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsM20StairsSightedV3PPORunnerCfg",
        "cusrl_cfg_entry_point": f"{agents.__name__}.cusrl_ppo_cfg:DeeproboticsM20RoughTrainerCfg",
    },
)

gym.register(
    id="Stairs-Sighted-V3b-Deeprobotics-M20-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        # v3 + the v2 wheel-clearance (stepping) reward restored. See
        # `stairs_env_cfg.DeeproboticsM20StairsSightedV3bEnvCfg`'s docstring.
        "env_cfg_entry_point": f"{__name__}.stairs_env_cfg:DeeproboticsM20StairsSightedV3bEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsM20StairsSightedV3bPPORunnerCfg",
        "cusrl_cfg_entry_point": f"{agents.__name__}.cusrl_ppo_cfg:DeeproboticsM20RoughTrainerCfg",
    },
)

gym.register(
    id="Stairs-Sighted-V3b-Deeprobotics-M20-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.stairs_env_cfg:DeeproboticsM20StairsSightedV3bEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsM20StairsSightedV3bPPORunnerCfg",
        "cusrl_cfg_entry_point": f"{agents.__name__}.cusrl_ppo_cfg:DeeproboticsM20RoughTrainerCfg",
    },
)

gym.register(
    id="Stairs-Student-Deeprobotics-M20-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        # Depth student for distillation from the frozen v2 teacher. The distillation runner cfg
        # (`agents/rsl_rl_distillation_cfg.py`) is written separately -- see
        # docs/student_distillation_setup.md.
        "env_cfg_entry_point": f"{__name__}.stairs_env_cfg:DeeproboticsM20StairsStudentEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_distillation_cfg:DeeproboticsM20StairsStudentRunnerCfg",
    },
)

gym.register(
    id="Stairs-Student-Deeprobotics-M20-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.stairs_env_cfg:DeeproboticsM20StairsStudentEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_distillation_cfg:DeeproboticsM20StairsStudentRunnerCfg",
    },
)
