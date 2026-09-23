# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause

# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class DeeproboticsM20RoughPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 20000
    save_interval = 100
    experiment_name = "deeprobotics_m20_rough"
    empirical_normalization = False
    clip_actions = 100
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        noise_std_type="log",
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.003,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class DeeproboticsM20FlatPPORunnerCfg(DeeproboticsM20RoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 5000
        self.experiment_name = "deeprobotics_m20_flat"


@configclass
class DeeproboticsM20StairsTeacherPPORunnerCfg(DeeproboticsM20RoughPPORunnerCfg):
    """Milestone A runner cfg (command.md Phase 3/4).

    Placeholder plain single-MLP actor-critic so `stairs_env_cfg.py` has something to smoke-test
    against now. Phase 4 replaces `policy`/`algorithm` here with the MoE actor + multi-critic
    architecture (`docs/stairs_research_notes.md` sec 11) -- this class name is expected to be
    rewritten in place when that lands, not kept around as a permanent "simple" alternative.
    """

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "deeprobotics_m20_stairs_teacher"


@configclass
class DeeproboticsM20StairsSightedPPORunnerCfg(DeeproboticsM20StairsTeacherPPORunnerCfg):
    """Runner cfg for the plain sighted single-task validation run (see `stairs_env_cfg.py`).

    Identical plain single-MLP actor-critic, just its own `experiment_name` so its logs don't land
    in the same directory as the blind Phase-3-smoke-test runs of `Stairs-Teacher-...-v0`.
    """

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "deeprobotics_m20_stairs_sighted"
        self.max_iterations = 3000


@configclass
class DeeproboticsM20StairsSightedV2PPORunnerCfg(DeeproboticsM20StairsSightedPPORunnerCfg):
    """Runner cfg for `Stairs-Sighted-V2-Deeprobotics-M20-v0` (stair-aware rewards/curriculum)."""

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "deeprobotics_m20_stairs_sighted_v2"
        self.max_iterations = 6000
