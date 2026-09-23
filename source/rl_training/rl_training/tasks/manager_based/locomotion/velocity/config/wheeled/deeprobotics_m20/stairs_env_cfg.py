# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause

"""Stairs task environment -- Milestone A, part 1 (command.md Phase 3).

Builds the terrain mix (3.1), velocity-command scheme (3.2), and the 3-task definition (3.4) that
Phase 4's multi-critic and MoE router will route on. Observations/rewards/network architecture are
untouched here -- this subclass still trains exactly like the blind `Rough-Deeprobotics-M20-v0`
task apart from terrain/commands, on purpose, so Phase 3's own smoke test isolates terrain/task-ID
plumbing from Phase 4's architecture changes.
"""

from __future__ import annotations

import torch

import isaaclab.terrains as terrain_gen
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.terrains import TerrainGeneratorCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import rl_training.tasks.manager_based.locomotion.velocity.mdp as mdp
from rl_training.tasks.manager_based.locomotion.velocity.config.wheeled.deeprobotics_m20.rough_env_cfg import (
    DeeproboticsM20RoughEnvCfg,
)

##
# Task definition (3.4) -- training-only routing key for Phase 4's multi-critic (4.2) and reward
# manager (4.4). Must NOT be fed to the actor or the MoE router (4.3): the router sees perception
# only, exactly as it will have to at deployment with no privileged task label available.
##

TASK_FLAT_ROUGH = 0
TASK_STAIR_ASCENT = 1
TASK_STAIR_DESCENT = 2

TASK_NAMES = {
    TASK_FLAT_ROUGH: "flat_rough",
    TASK_STAIR_ASCENT: "stair_ascent",
    TASK_STAIR_DESCENT: "stair_descent",
}

# Which task (above) each sub-terrain key in `build_stairs_terrain_generator` belongs to. `flat`,
# `random_rough`, and the two slope terrains are all "drive well and stay upright on everything
# that isn't stairs" per the task table; the tread-width stair variants route to their matching
# ascent/descent task.
SUB_TERRAIN_TASK = {
    "stair_ascent": TASK_STAIR_ASCENT,
    "stair_descent": TASK_STAIR_DESCENT,
    "stair_ascent_narrow_tread": TASK_STAIR_ASCENT,
    "stair_descent_wide_tread": TASK_STAIR_DESCENT,
    "random_rough": TASK_FLAT_ROUGH,
    "hf_pyramid_slope": TASK_FLAT_ROUGH,
    "hf_pyramid_slope_inv": TASK_FLAT_ROUGH,
    "flat": TASK_FLAT_ROUGH,
}


def build_stairs_terrain_generator() -> TerrainGeneratorCfg:
    """3.1: terrain mix for Milestone A.

    Sub-terrain declaration order matters: `TerrainGenerator._generate_curriculum_terrains`
    assigns generator columns to sub-terrains by cumulative proportion over
    `list(sub_terrains.values())`, in declaration order (the same mechanism `eval_stairs.py`
    relies on for its fixed single-row eval terrain). `task_id_per_column` below walks the same
    dict in the same order, so it always matches the live `terrain.terrain_types` column
    assignment -- it is derived from this dict, never hand-kept in sync with it.
    """
    sub_terrains = {
        "stair_ascent": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.30,
            step_height_range=(0.05, 0.26),
            step_width=0.3,
            platform_width=2.5,
            border_width=1.0,
            holes=False,
        ),
        "stair_descent": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.25,
            step_height_range=(0.05, 0.26),
            step_width=0.3,
            platform_width=2.5,
            border_width=1.0,
            holes=False,
        ),
        # 3.1's "stairs with different tread" row (0.15 total), split so both directions get a
        # non-default tread width rather than only one.
        "stair_ascent_narrow_tread": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.075,
            step_height_range=(0.05, 0.26),
            step_width=0.25,
            platform_width=2.5,
            border_width=1.0,
            holes=False,
        ),
        "stair_descent_wide_tread": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.075,
            step_height_range=(0.05, 0.26),
            step_width=0.35,
            platform_width=2.5,
            border_width=1.0,
            holes=False,
        ),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.10, noise_range=(0.02, 0.10), noise_step=0.02, border_width=0.25
        ),
        "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.05, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
        ),
        "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.05, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
        ),
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.10),
    }
    return TerrainGeneratorCfg(
        size=(8.0, 8.0),
        border_width=20.0,
        num_rows=10,
        num_cols=20,
        horizontal_scale=0.1,
        vertical_scale=0.005,
        slope_threshold=0.75,
        use_cache=False,
        curriculum=True,
        sub_terrains=sub_terrains,
    )


def task_id_per_column(terrain_generator: TerrainGeneratorCfg) -> list[int]:
    """Raw grid column (0..num_cols-1) -> task ID.

    `terrain.terrain_types[env_id]` (what `get_task_ids`/`terrain_level_by_task` index with) is a
    raw grid column in `[0, num_cols)`, not an index into `sub_terrains` -- with `num_cols` bigger
    than the number of sub-terrain entries (true here: 20 columns, 8 entries), several columns
    share one sub-terrain type, proportioned by `sub_terrains[...].proportion`. Terrain generation
    doesn't expose that column->sub-terrain assignment anywhere, so this recomputes it, mirroring
    `TerrainGenerator._generate_curriculum_terrains`'s cumulative-proportion formula exactly
    (isaaclab/terrains/terrain_generator.py). `eval_stairs.py` avoids needing this by setting
    `num_cols == len(combos)` (one column per sub-terrain, so raw column == sub-terrain index
    directly) -- that shortcut isn't available here since 3.1's proportions don't split evenly
    into whole columns at a reasonable `num_cols`.
    """
    import numpy as np

    proportions = np.array([cfg.proportion for cfg in terrain_generator.sub_terrains.values()], dtype=float)
    proportions /= proportions.sum()
    cumsum = np.cumsum(proportions)
    keys = list(terrain_generator.sub_terrains.keys())
    num_cols = terrain_generator.num_cols
    task_ids = []
    for col in range(num_cols):
        sub_index = int(np.min(np.where(col / num_cols + 0.001 < cumsum)[0]))
        task_ids.append(SUB_TERRAIN_TASK[keys[sub_index]])
    return task_ids


def get_task_ids(env) -> torch.Tensor:
    """Per-env task ID (3.4), looked up from each env's fixed terrain column (`terrain_types`).

    `terrain_types` is assigned once at terrain generation and never changes after that (only
    `terrain_levels`/`env_origins` move as the curriculum promotes/demotes within a column), so
    this is safe to cache once per episode rather than recomputing every step -- Phase 4's
    multi-critic/reward routing should do so rather than calling this every step.
    """
    terrain = env.scene.terrain
    task_of_column = torch.tensor(
        task_id_per_column(terrain.cfg.terrain_generator), device=env.device, dtype=torch.long
    )
    return task_of_column[terrain.terrain_types]


@configclass
class DeeproboticsM20StairsTeacherEnvCfg(DeeproboticsM20RoughEnvCfg):
    """Milestone A teacher env (Phase 3): terrain + commands + task-ID plumbing only.

    Deliberately does not touch observations/rewards/network architecture yet (that's Phase 4) --
    the policy here is still blind, exactly like `Rough-Deeprobotics-M20-v0`, so this phase's
    smoke test isolates the new terrain/commands/task-ID code from Phase 4's architecture changes.
    """

    def __post_init__(self):
        super().__post_init__()

        # ------------------------------ Terrain (3.1) ------------------------------
        self.scene.terrain.terrain_generator = build_stairs_terrain_generator()
        self.scene.terrain.max_init_terrain_level = 3

        # ------------------------------ Commands (3.2) ------------------------------
        self.commands.base_velocity.ranges.lin_vel_x = (0.3, 1.2)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.3, 0.3)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        self.commands.base_velocity.heading_command = True
        self.commands.base_velocity.rel_heading_envs = 1.0
        self.commands.base_velocity.rel_standing_envs = 0.0
        self.commands.base_velocity.rel_zero_vel_envs = 0.15
        self.commands.base_velocity.rel_only_lin_x_envs = 0.0
        self.commands.base_velocity.rel_only_lin_y_envs = 0.0
        self.commands.base_velocity.rel_only_ang_z_envs = 0.0

        # ------------------------------ Curriculum (3.3) ------------------------------
        # Per-task terrain-level logging, decoupled from `terrain_levels_vel`'s single aggregate
        # mean (see `mdp.terrain_level_by_task`'s docstring and command.md Phase 3.3). Does not
        # replace the base `terrain_levels`/`gait_level`/`command_levels` terms, which still run
        # unchanged and still own the actual promotion/demotion.
        columns = task_id_per_column(self.scene.terrain.terrain_generator)
        self.curriculum.terrain_level_flat_rough = CurrTerm(
            func=mdp.terrain_level_by_task,
            params={"task_ids_per_column": columns, "task_id": TASK_FLAT_ROUGH},
        )
        self.curriculum.terrain_level_stair_ascent = CurrTerm(
            func=mdp.terrain_level_by_task,
            params={"task_ids_per_column": columns, "task_id": TASK_STAIR_ASCENT},
        )
        self.curriculum.terrain_level_stair_descent = CurrTerm(
            func=mdp.terrain_level_by_task,
            params={"task_ids_per_column": columns, "task_id": TASK_STAIR_DESCENT},
        )

        # `DeeproboticsM20RoughEnvCfg.__post_init__` only calls this for its own exact class name
        # (see rough_env_cfg.py), so every subclass -- this one included -- has to call it again
        # itself, or zero-weight reward terms with M20-incompatible default params (e.g.
        # `foot_impact_velocity`'s `.*_foot` pattern; M20 has no `_foot` bodies, only `_wheel`)
        # are left active instead of stripped, and fail to resolve at env construction.
        self.disable_zero_weight_rewards()


@configclass
class DeeproboticsM20StairsTeacherEnvCfg_PLAY(DeeproboticsM20StairsTeacherEnvCfg):
    """Play/visualization variant (3.3's checks): fewer envs, no pushes, fixed forward command."""

    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.events.randomize_push_robot = None

        if self.scene.terrain.terrain_generator is not None:
            # Keep `curriculum=True` (deterministic column->sub-terrain assignment, see
            # `task_id_per_column`) even though this is just a visualization pass -- with
            # `curriculum=False` a column's sub-terrain type can vary row to row
            # (`TerrainGenerator._generate_random_terrains` samples per-cell, not per-column),
            # which would make the fixed `SUB_TERRAIN_TASK` mapping wrong.
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 8
            # `task_id_per_column`'s output depends on `num_cols` (see its docstring) -- the
            # parent `__post_init__` already baked a `num_cols=20` version into the three
            # `terrain_level_*` curriculum terms above, which is now stale. Recompute and
            # overwrite so the Play variant's own (smaller) column count is reflected.
            columns = task_id_per_column(self.scene.terrain.terrain_generator)
            self.curriculum.terrain_level_flat_rough.params["task_ids_per_column"] = columns
            self.curriculum.terrain_level_stair_ascent.params["task_ids_per_column"] = columns
            self.curriculum.terrain_level_stair_descent.params["task_ids_per_column"] = columns
        self.scene.terrain.max_init_terrain_level = None

        self.commands.base_velocity.ranges.lin_vel_x = (0.5, 0.5)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.rel_standing_envs = 0.0
        self.commands.base_velocity.rel_zero_vel_envs = 0.0
        self.commands.base_velocity.rel_only_lin_x_envs = 0.0
        self.commands.base_velocity.rel_only_lin_y_envs = 0.0
        self.commands.base_velocity.rel_only_ang_z_envs = 0.0


@configclass
class DeeproboticsM20StairsSightedEnvCfg(DeeproboticsM20StairsTeacherEnvCfg):
    """Single-task, single-critic, sighted validation run -- deliberately NOT Milestone A's MoE
    architecture, and not a `command.md` phase of its own.

    Answers one question before any multi-critic/MoE infrastructure gets built: can a plain
    single-MLP actor-critic climb these stairs at all if it can see the terrain? Isolates exactly
    one variable against the blind `DeeproboticsM20StairsTeacherEnvCfg` this subclasses --
    re-enabling height scan for the actor -- one reward function, one critic, one actor, exactly
    like `Rough-Deeprobotics-M20-v0` today (whose blind baseline is the number this is trying to
    beat -- see Phase 1's ~0% blind ascent finding). The 3.4 task-ID plumbing still exists
    underneath for later reuse, but nothing routes on it here.

    2026-09-23 update: the first 2600/3000-iteration run (sighted, unmodified rewards) crossed
    0.10m stairs both ways but timed out (no falls) at 0.20m, with 28 wheel-riser stumbles/trial
    even on the easy 0.10m ascent -- see `command.md` 3.5. Two reward changes added below, per
    Phase 4.4's own recommendations, applied together (not isolated one-at-a-time) since this is
    still a fast validation loop, not the Milestone A ablation study:
    - `feet_stumble` enabled (was 0) -- directly penalizes the wheel-riser hits already observed.
    - `flat_orientation_l2` relaxed -50 -> -15 (global, not stair-task-only -- no per-task reward
      routing exists yet) -- was likely suppressing the pitch needed to climb steeper steps.
    `bad_orientation_2`'s hardcoded 0.7 termination threshold (~44 deg) was deliberately left
    alone: required pitch for a 0.20m/0.30m stair is ~34 deg, comfortably under it, and it isn't
    exposed as a configurable param on the shared mdp function (changing that function would
    affect Rough/Flat too, which command.md's ground rules forbid) -- a candidate follow-up if
    this pass isn't enough, not a first move.
    """

    def __post_init__(self):
        super().__post_init__()

        # Same height-scan sensor/params the critic already reads (`CriticCfg.height_scan` in
        # velocity_env_cfg.py) -- reusing that term as-is rather than designing a new
        # forward-biased pattern yet (that grid redesign is Phase 4.1 proper, once/if this
        # validates the underlying idea is worth building the rest of Milestone A around).
        self.observations.policy.height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            noise=Unoise(n_min=-0.1, n_max=0.1),
            clip=(-1.0, 1.0),
            scale=1.0,
        )

        # ------------------------------ Reward changes (2026-09-23) ------------------------------
        # `feet_stumble` defaults to weight 0, and the parent `__post_init__` (Teacher's) already
        # called `disable_zero_weight_rewards()`, which nulls any zero-weight term to `None` -- so
        # `self.rewards.feet_stumble` is `None` at this point, not a mutable RewTerm. Reconstruct
        # it fresh rather than assigning `.weight` on it (would be an AttributeError on None).
        self.rewards.feet_stumble = RewTerm(
            func=mdp.feet_stumble,
            weight=-1.0,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=self.foot_link_name)},
        )
        # flat_orientation_l2 already exists (weight -50 from the parent, non-zero so it survived
        # disable_zero_weight_rewards()) -- safe to mutate its weight directly.
        self.rewards.flat_orientation_l2.weight = -15.0


@configclass
class DeeproboticsM20StairsSightedEnvCfg_PLAY(DeeproboticsM20StairsSightedEnvCfg):
    """Play/video variant: fewer envs, no pushes, fixed forward command (same treatment as
    `DeeproboticsM20StairsTeacherEnvCfg_PLAY`)."""

    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.events.randomize_push_robot = None

        if self.scene.terrain.terrain_generator is not None:
            # See `DeeproboticsM20StairsTeacherEnvCfg_PLAY`'s docstring: keep `curriculum=True`
            # (deterministic column->sub-terrain assignment) even at a smaller grid.
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 8
            columns = task_id_per_column(self.scene.terrain.terrain_generator)
            self.curriculum.terrain_level_flat_rough.params["task_ids_per_column"] = columns
            self.curriculum.terrain_level_stair_ascent.params["task_ids_per_column"] = columns
            self.curriculum.terrain_level_stair_descent.params["task_ids_per_column"] = columns
        self.scene.terrain.max_init_terrain_level = None

        self.commands.base_velocity.ranges.lin_vel_x = (0.5, 0.5)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.rel_standing_envs = 0.0
        self.commands.base_velocity.rel_zero_vel_envs = 0.0
        self.commands.base_velocity.rel_only_lin_x_envs = 0.0
        self.commands.base_velocity.rel_only_lin_y_envs = 0.0
        self.commands.base_velocity.rel_only_ang_z_envs = 0.0


##
# Sighted spike v2 (2026-09-23): stair-aware rewards, terrain and curriculum.
##

# Pyramid-stairs geometry shared by the v2 terrain and the stair curriculum/progress terms. With an
# 8 m tile and a 1.0 m sub-terrain border, the outermost riser sits at L-inf distance 3.0 m from the
# tile centre (`inverted_pyramid_stairs_terrain` / `pyramid_stairs_terrain` build the steps inward
# from the border).
V2_TILE_SIZE = 8.0
V2_SUB_BORDER_WIDTH = 1.0
V2_PLATFORM_WIDTH = 2.3
V2_STAIRS_OUTER_EDGE = V2_TILE_SIZE / 2 - V2_SUB_BORDER_WIDTH
# Robot root ~0.4 m past the outer riser: whole 0.82 m body is off the stairs.
V2_STAIR_PROMOTE_DISTANCE = V2_STAIRS_OUTER_EDGE + 0.4
# Less than ~a third of the way up/down the flight (inner edge ~1.1 m for 0.32 m treads).
V2_STAIR_DEMOTE_DISTANCE = 1.7

STAIR_TASK_IDS = (TASK_STAIR_ASCENT, TASK_STAIR_DESCENT)


def build_stairs_terrain_generator_v2() -> TerrainGeneratorCfg:
    """Same sub-terrain keys/proportions/order as `build_stairs_terrain_generator` (so
    `task_id_per_column` still applies), with stair geometry changed for the 0.25-0.30 m target:

    - step heights 0.08-0.30 m (low obstacles are covered by `flat_rough`); the narrow-tread
      variant stops at 0.24 m so no flight is steeper than 45 deg.
    - main treads 0.32 m (0.30/0.32 = 43 deg), wide 0.35 m, narrow 0.26 m.
    - `platform_width` 2.3 m -> ~1.8-2.2 m actual centre platform (the generator adds one extra
      step, so the real platform ends up smaller than `platform_width`): less time spent on the
      platform, still > 2x the 0.82 m robot length.
    """
    stair_kwargs = dict(platform_width=V2_PLATFORM_WIDTH, border_width=V2_SUB_BORDER_WIDTH, holes=False)
    sub_terrains = {
        "stair_ascent": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.30, step_height_range=(0.08, 0.30), step_width=0.32, **stair_kwargs
        ),
        "stair_descent": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.25, step_height_range=(0.08, 0.30), step_width=0.32, **stair_kwargs
        ),
        "stair_ascent_narrow_tread": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.075, step_height_range=(0.08, 0.24), step_width=0.26, **stair_kwargs
        ),
        "stair_descent_wide_tread": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.075, step_height_range=(0.08, 0.30), step_width=0.35, **stair_kwargs
        ),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.10, noise_range=(0.02, 0.10), noise_step=0.02, border_width=0.25
        ),
        "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.05, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
        ),
        "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.05, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
        ),
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.10),
    }
    return TerrainGeneratorCfg(
        size=(V2_TILE_SIZE, V2_TILE_SIZE),
        border_width=20.0,
        num_rows=10,
        num_cols=20,
        horizontal_scale=0.1,
        vertical_scale=0.005,
        slope_threshold=0.75,
        use_cache=False,
        curriculum=True,
        sub_terrains=sub_terrains,
    )


def set_task_ids_per_column(env_cfg, columns: list[int]) -> None:
    """Overwrite `task_ids_per_column` in every reward/termination/curriculum term that takes it.

    Needed whenever the terrain's column layout changes after `__post_init__` (Play variants,
    `eval_stairs.py`'s fixed-geometry terrain), since the list is baked in per `num_cols`.
    """
    for group in (env_cfg.rewards, env_cfg.terminations, env_cfg.curriculum):
        for term in vars(group).values():
            params = getattr(term, "params", None)
            if isinstance(params, dict) and "task_ids_per_column" in params:
                params["task_ids_per_column"] = list(columns)


@configclass
class DeeproboticsM20StairsSightedV2EnvCfg(DeeproboticsM20StairsSightedEnvCfg):
    """Sighted spike v2 -- still plain single-MLP actor-critic, one reward function.

    v1 (`DeeproboticsM20StairsSightedEnvCfg`, run 2026-09-23_01-46-05) plateaued at ascent terrain
    level ~5.5 (~0.17 m steps); at 0.20 m it stopped short of the riser (0 stumbles) and teetered
    at the top of descents. Diagnosis: stopping was cheaper than climbing under the reward terms,
    not a lack of stair exposure (stairs are already ~70% of columns). Changes, applied together:

    1. Velocity tracking in the yaw (gravity-aligned) frame with std 0.3 (was base frame, sqrt(0.5)
       -- standing still kept 61-84% of the reward at 0.5/0.3 m/s commands).
    2. Orientation penalty back to -50 but roll-only (+5% pitch) on stair tiles; bad-orientation
       penalty/termination pitch limit 0.9 on stair tiles (0.7 elsewhere, unchanged).
    3. Stair progress reward (episode-best L-inf distance out of the tile + height gained on ascent).
    4. hipy/knee posture penalty relaxed on uneven terrain *ahead* (not "base above env origin").
    5. Wheel clearance shaping over risers in front of airborne wheels; `feet_stumble` -1 -> -3.
    6. Forward-biased height scan (-0.4..+1.6 m, was -0.8..+0.8 m), noise +/-0.025 m (was +/-0.1).
    7. Terrain: steps 0.08-0.30 m, 0.32 m main treads, smaller centre platform, spawn offset
       +/-0.3 m (was +/-1.0), stair-specific promote/demote rule (`mdp.terrain_levels_stairs`).
    """

    def __post_init__(self):
        super().__post_init__()

        # ------------------------------ Terrain / spawn (7) ------------------------------
        self.scene.terrain.terrain_generator = build_stairs_terrain_generator_v2()
        columns = task_id_per_column(self.scene.terrain.terrain_generator)
        self.events.randomize_reset_base.params["pose_range"]["x"] = (-0.3, 0.3)
        self.events.randomize_reset_base.params["pose_range"]["y"] = (-0.3, 0.3)

        # ------------------------------ Height scan (6) ------------------------------
        # `offset.pos` shifts the ray starts in the yaw-aligned sensor frame only; `mdp.height_scan`
        # reads the sensor body height, so the observation's zero point is unchanged.
        self.scene.height_scanner.pattern_cfg.size = [2.0, 1.0]
        self.scene.height_scanner.offset.pos = (0.6, 0.0, 20.0)
        self.observations.policy.height_scan.noise = Unoise(n_min=-0.025, n_max=0.025)

        # ------------------------------ Rewards ------------------------------
        # (1) tracking
        self.rewards.track_lin_vel_xy_exp.func = mdp.track_lin_vel_xy_yaw_frame_exp
        self.rewards.track_lin_vel_xy_exp.params["std"] = 0.3

        # (2) orientation
        self.rewards.flat_orientation_l2.func = mdp.orientation_l2_stair_aware
        self.rewards.flat_orientation_l2.weight = -50.0
        self.rewards.flat_orientation_l2.params.update(
            {"task_ids_per_column": columns, "stair_task_ids": STAIR_TASK_IDS, "stair_pitch_scale": 0.05}
        )
        bad_orientation_params = {
            "task_ids_per_column": columns,
            "stair_task_ids": STAIR_TASK_IDS,
            "stair_pitch_limit": 0.9,
            "pitch_limit": 0.7,
            "roll_limit": 0.7,
        }
        self.rewards.bad_orientation_penalty.func = mdp.bad_orientation_stair_aware_penalty
        self.rewards.bad_orientation_penalty.params.update(dict(bad_orientation_params))
        self.terminations.bad_orientation_2 = DoneTerm(
            func=mdp.bad_orientation_stair_aware, params=dict(bad_orientation_params)
        )

        # (3) progress
        self.rewards.stair_progress = RewTerm(
            func=mdp.stair_progress,
            weight=8.0,
            params={
                "command_name": "base_velocity",
                "task_ids_per_column": columns,
                "ascent_task_ids": (TASK_STAIR_ASCENT,),
                "descent_task_ids": (TASK_STAIR_DESCENT,),
                "max_distance": V2_STAIRS_OUTER_EDGE,
                "height_scale": 1.0,
            },
        )

        # (4) posture penalty gating
        for term in (self.rewards.hipy_joint_pos_penalty, self.rewards.knee_joint_pos_penalty):
            term.func = mdp.joint_pos_penalty_terrain_ahead
            term.params.pop("terrain_height_threshold", None)
            term.params.pop("high_terrain_penalty_scale", None)
            term.params["sensor_cfg"] = SceneEntityCfg("height_scanner")
            term.params["roughness_threshold"] = 0.08
            term.params["rough_penalty_scale"] = 0.1

        # (5) wheel clearance + stumble
        wheel_names = ["fl_wheel", "fr_wheel", "hl_wheel", "hr_wheel"]
        self.rewards.stair_wheel_clearance = RewTerm(
            func=mdp.stair_wheel_clearance,
            weight=1.0,
            params={
                "command_name": "base_velocity",
                "task_ids_per_column": columns,
                "stair_task_ids": STAIR_TASK_IDS,
                "sensor_cfg": SceneEntityCfg("height_scanner"),
                "contact_sensor_cfg": SceneEntityCfg("contact_forces", body_names=wheel_names, preserve_order=True),
                "asset_cfg": SceneEntityCfg("robot", body_names=wheel_names, preserve_order=True),
                "wheel_radius": 0.09,
            },
        )
        self.rewards.feet_stumble.weight = -3.0

        # ------------------------------ Curriculum (7) ------------------------------
        self.curriculum.terrain_levels = CurrTerm(
            func=mdp.terrain_levels_stairs,
            params={
                "task_ids_per_column": columns,
                "stair_task_ids": STAIR_TASK_IDS,
                "stair_promote_distance": V2_STAIR_PROMOTE_DISTANCE,
                "stair_demote_distance": V2_STAIR_DEMOTE_DISTANCE,
            },
        )
        # per-task logging terms from the Teacher cfg were built against the same column layout
        set_task_ids_per_column(self, columns)


@configclass
class DeeproboticsM20StairsSightedV2EnvCfg_PLAY(DeeproboticsM20StairsSightedV2EnvCfg):
    """Play/video variant: fewer envs, no pushes, fixed forward command."""

    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.events.randomize_push_robot = None

        if self.scene.terrain.terrain_generator is not None:
            # keep `curriculum=True`, see `DeeproboticsM20StairsTeacherEnvCfg_PLAY`
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 8
            set_task_ids_per_column(self, task_id_per_column(self.scene.terrain.terrain_generator))
        self.scene.terrain.max_init_terrain_level = None

        self.commands.base_velocity.ranges.lin_vel_x = (0.5, 0.5)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.rel_standing_envs = 0.0
        self.commands.base_velocity.rel_zero_vel_envs = 0.0
        self.commands.base_velocity.rel_only_lin_x_envs = 0.0
        self.commands.base_velocity.rel_only_lin_y_envs = 0.0
        self.commands.base_velocity.rel_only_ang_z_envs = 0.0
