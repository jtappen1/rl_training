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

# Domain randomization DR1 (docs/plans/M20_Domain_Randomization_DR1.md): vendor rough config + DR1 terms.
gym.register(
    id="Rough-Deeprobotics-M20-DR1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.dr1_env_cfg:DeeproboticsM20DR1EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsM20DR1PPORunnerCfg",
    },
)

# Cmd1: DR1 + command/reward changes for the real-robot issues (yaw-rate commands, lateral, standstill).
gym.register(
    id="Rough-Deeprobotics-M20-Cmd1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cmd1_env_cfg:DeeproboticsM20Cmd1EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsM20Cmd1PPORunnerCfg",
    },
)

# Evaluation tasks (docs/plans/M20_Evaluation_Suite.md). scripts/eval/m20_eval.py builds its config from
# any M20 rough task instead; these ids are for running the eval config directly (e.g. with play.py).
gym.register(
    id="Eval-Deeprobotics-M20-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.eval_env_cfg:DeeproboticsM20EvalEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsM20RoughPPORunnerCfg",
    },
)

gym.register(
    id="Eval-Deeprobotics-M20-Randomized-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.eval_env_cfg:DeeproboticsM20EvalRandomizedEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsM20RoughPPORunnerCfg",
    },
)
