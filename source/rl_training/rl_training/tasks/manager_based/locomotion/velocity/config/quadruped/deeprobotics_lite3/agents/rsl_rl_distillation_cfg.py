# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause

"""Distillation runner configs: Lite3 depth CNN+GRU student <- frozen sighted teacher (DAgger).

Same student network as the M20 student (`DepthCNNGRUModelCfg`). The teacher checkpoint is found
through a symlink in the student's experiment dir (train.py only looks there), e.g.
`logs/rsl_rl/deeprobotics_lite3_stairs_student/teacher_v3 -> ../deeprobotics_lite3_stairs_sighted_v3/<run>`.
"""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlCNNModelCfg, RslRlDistillationAlgorithmCfg, RslRlDistillationRunnerCfg, RslRlMLPModelCfg

from rl_training.tasks.manager_based.locomotion.velocity.config.wheeled.deeprobotics_m20.agents.rsl_rl_distillation_cfg import (
    DepthCNNGRUModelCfg,
)


@configclass
class DeeproboticsLite3StairsStudentRunnerCfg(RslRlDistillationRunnerCfg):
    """Stage A: student <- frozen v3 teacher (deeprobotics_lite3_stairs_sighted_v3, model_19999)."""

    experiment_name = "deeprobotics_lite3_stairs_student"
    num_steps_per_env = 24
    max_iterations = 3000
    save_interval = 100
    empirical_normalization = False
    obs_groups = {"student": ["policy", "depth"], "teacher": ["teacher"]}

    load_run = "teacher_v3"
    load_checkpoint = "model_19999.pt"

    # frozen PPO actor, same shape as DeeproboticsLite3RoughPPORunnerCfg's
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


@configclass
class DeeproboticsLite3StairsStudentV4RunnerCfg(DeeproboticsLite3StairsStudentRunnerCfg):
    """Stage B: student <- frozen v4 teacher (full domain randomization). Point `teacher_v4` at the
    v4 run and set `load_checkpoint` once it is trained."""

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "deeprobotics_lite3_stairs_student_v4"
        self.load_run = "teacher_v4"
        self.load_checkpoint = "model_19999.pt"
