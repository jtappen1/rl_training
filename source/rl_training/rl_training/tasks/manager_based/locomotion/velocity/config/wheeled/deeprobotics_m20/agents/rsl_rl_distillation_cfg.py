# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause

"""Distillation runner config: depth CNN+GRU student <- frozen sighted v3b teacher."""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlCNNModelCfg,
    RslRlDistillationAlgorithmCfg,
    RslRlDistillationRunnerCfg,
    RslRlMLPModelCfg,
)


@configclass
class DepthCNNGRUModelCfg(RslRlCNNModelCfg):
    """Config for `rl_training.models.DepthCNNGRUModel` (CNN on depth + proprio -> GRU -> MLP)."""

    class_name: str = "rl_training.models:DepthCNNGRUModel"
    # Passed to DepthCNNGRUModel.__init__ as **kwargs, so the names must match it exactly.
    rnn_type: str = "gru"
    rnn_num_layers: int = 1
    rnn_hidden_dim: int = 256


@configclass
class DeeproboticsM20StairsStudentRunnerCfg(RslRlDistillationRunnerCfg):
    # Experiment configs
    experiment_name = "deeprobotics_m20_stairs_student"
    num_steps_per_env = 24
    max_iterations = 2000
    save_interval = 100
    empirical_normalization = False
    obs_groups = {"student": ["policy", "depth"], "teacher": ["teacher"]}

    # Teacher checkpoint, via the logs/rsl_rl/deeprobotics_m20_stairs_student/teacher_v3b symlink
    load_run = "teacher_v3b"
    load_checkpoint = "model_5999.pt"

    # Frozen v3b PPO actor.
    teacher = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0, std_type="log"),
    )

    student = DepthCNNGRUModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=0.1, std_type="scalar"),
        cnn_cfg=RslRlCNNModelCfg.CNNCfg(
            output_channels=[16, 32, 32],
            kernel_size=[5, 4, 3],
            stride=[2, 2, 1],
            activation="elu",
        ),
    )

    algorithm = RslRlDistillationAlgorithmCfg(
        num_learning_epochs=2,
        learning_rate=1e-3,
        gradient_length=15,
        max_grad_norm=1.0,
        loss_type="mse",
    )
