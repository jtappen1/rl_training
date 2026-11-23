# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause

"""M20 evaluation task (docs/plans/M20_Evaluation_Suite.md).

The evaluation runs every scripted scenario of a pass in one simulation. Each scenario owns a block of
environments; the driver script (``scripts/eval/m20_eval.py``) sets the commands, spawn poses and
perturbations. This module only provides the environment side:

* :class:`ScriptedVelocityCommand` - a command term that never resamples; the driver writes the commands.
* :func:`reset_root_state_eval` - resets each env to its own origin with a per-env spawn yaw.
* :data:`M20_EVAL_TERRAINS_CFG` - fixed tiles: flat, 15 deg slope, 10 cm stairs, 15 cm stairs, medium rough.
* :func:`configure_m20_eval` - turns any M20 rough config (vendor or DR1) into an eval config.
* :class:`DeeproboticsM20EvalEnvCfg` - the eval task, subclassing :class:`DeeproboticsM20RoughEnvCfg`.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

import isaaclab.terrains as terrain_gen
import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg, SceneEntityCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.terrains import TerrainGeneratorCfg
from isaaclab.utils import configclass

import rl_training.tasks.manager_based.locomotion.velocity.mdp as mdp

from .rough_env_cfg import DeeproboticsM20RoughEnvCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


##
# Terrain: one column per tile type. Every tile in a column is identical (fixed difficulty), so a robot
# walking along +x stays on the same terrain type. The three flat columns give +-12 m of flat ground
# around the centre flat column, enough for lateral and arc commands.
##

EVAL_TILE_SIZE = 8.0
EVAL_NUM_ROWS = 6
# column index of each tile type in M20_EVAL_TERRAINS_CFG (sub_terrains order)
EVAL_TERRAIN_COLUMNS = {
    "flat": 1,
    "slope_15deg": 3,
    "stairs_10cm": 4,
    "stairs_15cm": 5,
    "rough_medium": 6,
}
EVAL_START_ROW = 1

_SLOPE_15DEG = math.tan(math.radians(15.0))

M20_EVAL_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(EVAL_TILE_SIZE, EVAL_TILE_SIZE),
    border_width=20.0,
    num_rows=EVAL_NUM_ROWS,
    num_cols=7,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    # curriculum=True lays sub-terrains out by column in dict order (not random); ranges below are fixed,
    # so the row-dependent difficulty has no effect.
    curriculum=True,
    sub_terrains={
        "flat_a": terrain_gen.MeshPlaneTerrainCfg(proportion=1.0),
        "flat_b": terrain_gen.MeshPlaneTerrainCfg(proportion=1.0),
        "flat_c": terrain_gen.MeshPlaneTerrainCfg(proportion=1.0),
        "slope_15deg": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=1.0, slope_range=(_SLOPE_15DEG, _SLOPE_15DEG), platform_width=2.0, border_width=0.25
        ),
        "stairs_10cm": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=1.0, step_height_range=(0.10, 0.10), step_width=0.3, platform_width=3.0, border_width=1.0
        ),
        "stairs_15cm": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=1.0, step_height_range=(0.15, 0.15), step_width=0.3, platform_width=3.0, border_width=1.0
        ),
        # "medium": half of the vendor training maximum (noise_range up to 0.16 m)
        "rough_medium": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=1.0, noise_range=(0.01, 0.08), noise_step=0.01, border_width=0.25
        ),
    },
)


##
# Scripted command
##


class ScriptedVelocityCommand(CommandTerm):
    """Base-frame (vx, vy, wz) command played back from a schedule set by the evaluation driver.

    ``set_schedule(env_group, schedule)`` gives each env a row of ``schedule`` (shape (groups, T, 3)).
    After a reset the command is ``schedule[group, 0]``; the command manager update inside ``env.step``
    number k (0-based) advances it to ``schedule[group, k + 1]``. Commands update before observations are
    computed, so the action taken in step k always sees ``schedule[group, k]``, as in training.
    Without a schedule the command stays zero.
    """

    cfg: ScriptedVelocityCommandCfg

    def __init__(self, cfg: ScriptedVelocityCommandCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.robot: Articulation = env.scene[cfg.asset_name]
        self.vel_command_b = torch.zeros(self.num_envs, 3, device=self.device)
        self._env_group: torch.Tensor | None = None
        self._schedule: torch.Tensor | None = None
        self.step_index = 0

    def __str__(self) -> str:
        return "ScriptedVelocityCommand: commands are played back from the evaluation schedule."

    @property
    def command(self) -> torch.Tensor:
        return self.vel_command_b

    def set_schedule(self, env_group: torch.Tensor, schedule: torch.Tensor):
        self._env_group = env_group.to(self.device, dtype=torch.long)
        self._schedule = schedule.to(self.device, dtype=torch.float)
        self.step_index = 0
        self.vel_command_b[:] = self._schedule[self._env_group, 0]

    def _scheduled(self, env_ids) -> torch.Tensor:
        k = min(self.step_index, self._schedule.shape[1] - 1)
        return self._schedule[self._env_group[env_ids], k]

    def _update_metrics(self):
        pass

    def _resample_command(self, env_ids: Sequence[int]):
        if self._schedule is not None:
            self.vel_command_b[env_ids] = self._scheduled(env_ids)

    def _update_command(self):
        if self._schedule is not None:
            self.step_index += 1
            self.vel_command_b[:] = self._scheduled(slice(None))


@configclass
class ScriptedVelocityCommandCfg(CommandTermCfg):
    class_type: type = ScriptedVelocityCommand
    asset_name: str = "robot"
    resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9)
    debug_vis: bool = False


##
# Reset event
##


def reset_root_state_eval(env: ManagerBasedEnv, env_ids: torch.Tensor, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    """Reset the root to the env origin with the per-env yaw in ``env.eval_spawn_yaw`` (0 if unset), at rest."""
    asset: Articulation = env.scene[asset_cfg.name]
    root_states = asset.data.default_root_state[env_ids].clone()

    positions = root_states[:, 0:3] + env.scene.env_origins[env_ids]
    spawn_yaw = getattr(env, "eval_spawn_yaw", None)
    yaw = spawn_yaw[env_ids] if spawn_yaw is not None else torch.zeros(len(env_ids), device=asset.device)
    zeros = torch.zeros_like(yaw)
    orientations = math_utils.quat_mul(root_states[:, 3:7], math_utils.quat_from_euler_xyz(zeros, zeros, yaw))
    velocities = torch.zeros_like(root_states[:, 7:13])

    asset.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
    asset.write_root_velocity_to_sim(velocities, env_ids=env_ids)


##
# Config transformation
##

# Event terms that are always kept: the reset of the root (replaced by the eval version) and of the joints.
_RESET_EVENT_NAMES = ("randomize_reset_base", "randomize_reset_joints")
# Interval pushes are off in both passes; the push test applies its own pushes.
_PUSH_EVENT_NAMES = ("randomize_push_robot", "push_robot")
# Physics steps the actuator delay buffers must hold (40 ms latency spikes at dt = 5 ms).
EVAL_MAX_DELAY_STEPS = 8


def configure_m20_eval(cfg: DeeproboticsM20RoughEnvCfg, randomize: bool, episode_length_s: float = 60.0):
    """Turn an M20 rough-terrain config (vendor or DR1) into an evaluation config, in place.

    Args:
        cfg: The config to modify. Its randomization terms are what the randomized pass uses.
        randomize: False for the nominal pass (all randomization and observation noise off),
            True for the randomized pass (the config's randomization terms stay on, except pushes).
        episode_length_s: Long enough that no env times out during a pass.
    """
    # -- terrain: fixed tiles, no curriculum
    cfg.scene.terrain.terrain_type = "generator"
    cfg.scene.terrain.terrain_generator = M20_EVAL_TERRAINS_CFG.copy()
    cfg.scene.terrain.max_init_terrain_level = None
    if cfg.scene.height_scanner is not None:
        cfg.scene.height_scanner.debug_vis = False

    # -- commands: scripted, never resampled
    cfg.commands.base_velocity = ScriptedVelocityCommandCfg(asset_name="robot")

    # -- curriculum and rewards are not needed for evaluation
    for name in list(vars(cfg.curriculum)):
        if not name.startswith("_"):
            setattr(cfg.curriculum, name, None)
    for name in list(vars(cfg.rewards)):
        if not name.startswith("_") and not callable(getattr(cfg.rewards, name)):
            setattr(cfg.rewards, name, None)

    # -- events
    for name in list(vars(cfg.events)):
        if name.startswith("_"):
            continue
        if name in _PUSH_EVENT_NAMES or (not randomize and name not in _RESET_EVENT_NAMES):
            setattr(cfg.events, name, None)
    cfg.events.randomize_reset_base = EventTerm(func=reset_root_state_eval, mode="reset")
    if not randomize:
        cfg.events.randomize_reset_joints = EventTerm(
            func=mdp.reset_joints_by_scale,
            mode="reset",
            params={"position_range": (1.0, 1.0), "velocity_range": (0.0, 0.0)},
        )
        # Fixed mid-range friction instead of the USD default, so the nominal pass does not depend on it.
        cfg.events.nominal_material = EventTerm(
            func=mdp.randomize_rigid_body_material,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
                "static_friction_range": (1.0, 1.0),
                "dynamic_friction_range": (1.0, 1.0),
                "restitution_range": (0.0, 0.0),
                "num_buckets": 1,
            },
        )

    # -- observation noise
    cfg.observations.policy.enable_corruption = randomize

    # -- actuators: delay buffers long enough for the latency test (the driver sets the lags)
    for actuator in cfg.scene.robot.actuators.values():
        if hasattr(actuator, "max_delay"):
            actuator.max_delay = max(actuator.max_delay, EVAL_MAX_DELAY_STEPS)

    # -- episode never times out during a pass
    cfg.episode_length_s = episode_length_s
    cfg.commands.base_velocity.debug_vis = False


@configclass
class DeeproboticsM20EvalEnvCfg(DeeproboticsM20RoughEnvCfg):
    """Nominal-pass evaluation config built on the vendor rough config."""

    randomize: bool = False

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 64
        configure_m20_eval(self, randomize=self.randomize)


@configclass
class DeeproboticsM20EvalRandomizedEnvCfg(DeeproboticsM20EvalEnvCfg):
    """Randomized-pass evaluation config built on the vendor rough config (use the DR1 config once it exists)."""

    randomize: bool = True
