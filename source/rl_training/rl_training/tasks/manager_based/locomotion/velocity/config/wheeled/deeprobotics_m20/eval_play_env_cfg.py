# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause

"""Keyboard-driving the sighted v3b policy on the eval terrains (stairs benchmark and course).

Same geometry as `scripts/tools/eval_stairs.py` / `eval_multi_terrain_course.py` (both build it from
`eval_terrains.py`), one robot, driven with `play.py --keyboard`. The student's front depth camera and
its (clean) `depth` group are attached too, so the depth inputs a student would get on these
terrains can be watched while driving; the v3b policy itself still reads only its height scan.
"""

from __future__ import annotations

from isaaclab.utils import configclass

from .eval_terrains import DEFAULT_COURSE, build_combos, build_course_generator, build_stairs_benchmark_generator
from .skatepark_env_cfg import use_keyboard_play_settings
from .stairs_env_cfg import (
    TASK_STAIR_ASCENT,
    TASK_STAIR_DESCENT,
    DeeproboticsM20StairsSightedV3bEnvCfg,
    add_depth_camera,
    set_task_ids_per_column,
)

# eval_stairs.py's default step heights, at the training tread width, both directions (14 tiles in
# one row, ascent tiles first). N / B in play.py jump to the next / previous tile.
PLAY_STEP_HEIGHTS = (0.08, 0.12, 0.15, 0.18, 0.20, 0.23, 0.25)
PLAY_TREAD_WIDTHS = (0.30,)


def use_fixed_eval_terrain(env_cfg, terrain_generator, task_ids_per_column: list[int]) -> None:
    """Swap in a fixed single-row eval terrain; same curriculum/task-ID handling as the eval scripts."""
    env_cfg.scene.terrain.terrain_generator = terrain_generator
    env_cfg.scene.terrain.max_init_terrain_level = 0
    for name in ("terrain_levels", "gait_level", "command_levels", "terrain_level_flat_rough",
                 "terrain_level_stair_ascent", "terrain_level_stair_descent"):
        if hasattr(env_cfg.curriculum, name):
            setattr(env_cfg.curriculum, name, None)
    set_task_ids_per_column(env_cfg, task_ids_per_column)

    # spawn exactly on the tile/course origin, facing +x (across the stairs / along the course)
    env_cfg.events.randomize_reset_base.params = {
        "pose_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0), "yaw": (0.0, 0.0)},
        "velocity_range": {k: (0.0, 0.0) for k in ("x", "y", "z", "roll", "pitch", "yaw")},
    }


@configclass
class DeeproboticsM20StairsBenchV3bEnvCfg_PLAY(DeeproboticsM20StairsSightedV3bEnvCfg):
    """v3b on the `eval_stairs.py` benchmark tiles. Meant for
    ``play.py --task Stairs-Bench-V3b-Deeprobotics-M20-Play-v0 --keyboard``.
    """

    # tells play.py not to replace this terrain with its usual random 5 x 5 grid
    keep_play_terrain: bool = True

    def __post_init__(self):
        super().__post_init__()

        combos = build_combos(PLAY_STEP_HEIGHTS, PLAY_TREAD_WIDTHS)
        use_fixed_eval_terrain(
            self,
            build_stairs_benchmark_generator(combos),
            [TASK_STAIR_ASCENT if c["direction"] == "ascent" else TASK_STAIR_DESCENT for c in combos],
        )
        use_keyboard_play_settings(self)
        add_depth_camera(self, noisy=False)


@configclass
class DeeproboticsM20CourseV3bEnvCfg_PLAY(DeeproboticsM20StairsSightedV3bEnvCfg):
    """v3b on the `eval_multi_terrain_course.py` course (default course, seed 42, friction 1.0).
    Meant for ``play.py --task Course-V3b-Deeprobotics-M20-Play-v0 --keyboard``.
    """

    keep_play_terrain: bool = True

    def __post_init__(self):
        super().__post_init__()

        # mixed terrain in one column: stair limits everywhere, as in the course eval
        use_fixed_eval_terrain(self, build_course_generator(DEFAULT_COURSE, seed=42), [TASK_STAIR_ASCENT])
        use_keyboard_play_settings(self)
        add_depth_camera(self, noisy=False)

        # the course eval's default fixed friction instead of the DR range
        mat = self.events.randomize_rigid_body_material.params
        mat["static_friction_range"] = [1.0, 1.0]
        mat["dynamic_friction_range"] = [1.0, 1.0]
        mat["restitution_range"] = [0.0, 0.0]
