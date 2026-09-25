# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause

import os

from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

import rl_training.tasks.manager_based.locomotion.velocity.mdp as mdp
from rl_training.assets import ISAACLAB_ASSETS_REPO_DIR

from .rough_env_cfg import DeeproboticsM20RoughEnvCfg
from .stairs_env_cfg import DeeproboticsM20StairsSightedV3bEnvCfg, add_depth_camera

# Produced by `scripts/tools/convert_terrain_mesh.py` from `mesh/mesh.obj` (a ZED SDK scan of a real
# skatepark). That script rotates the Y-up scan into Isaac Sim's Z-up convention and recenters a chosen
# flat/ground point at the world origin, so the robot's normal flat-ground spawn pose (0, 0, 0.58) lands
# right on the pavement.
SKATEPARK_ORIGINAL_USD_PATH = os.path.join(ISAACLAB_ASSETS_REPO_DIR, "mesh", "mesh.usd")

# Second ZED scan, `mesh/mesh-stairs.obj`, converted the same way with
# `--up-axis y --recenter 4.0 -0.194 -2.5` (a flat, clear patch of the lower ground). The upper
# level (~0.94 m) is ~2.4 m to the robot's left (+y) at spawn.
SKATEPARK_STAIRS_USD_PATH = os.path.join(ISAACLAB_ASSETS_REPO_DIR, "mesh", "stairs", "mesh-stairs.usd")

SKATEPARK_USD_PATH = SKATEPARK_STAIRS_USD_PATH


def use_skatepark_terrain(env_cfg) -> None:
    """Swap an M20 env cfg's procedural terrain for the skatepark scan, leaving obs/actions untouched."""
    env_cfg.scene.terrain.terrain_type = "usd"
    env_cfg.scene.terrain.terrain_generator = None
    env_cfg.scene.terrain.usd_path = SKATEPARK_USD_PATH
    # only used to lay out env origins in a grid; irrelevant with a single env for keyboard driving
    env_cfg.scene.terrain.env_spacing = 4.0

    # a single static scan has no difficulty levels to progress through
    env_cfg.curriculum.terrain_levels = None

    # this termination only supports terrain_type in {"plane", "generator"}
    env_cfg.terminations.terrain_out_of_bounds = None

    # spawn upright on the recentered flat spot every reset, instead of randomizing across a
    # (nonexistent) procedural terrain grid -- avoids resets clipping into ramps/bowls nearby
    env_cfg.events.randomize_reset_base.params = {
        "pose_range": {"x": (-0.2, 0.2), "y": (-0.2, 0.2), "z": (0.0, 0.0), "yaw": (-3.14, 3.14)},
        "velocity_range": {
            "x": (0.0, 0.0),
            "y": (0.0, 0.0),
            "z": (0.0, 0.0),
            "roll": (0.0, 0.0),
            "pitch": (0.0, 0.0),
            "yaw": (0.0, 0.0),
        },
    }


def use_keyboard_play_settings(env_cfg) -> None:
    """One robot, no pushes/perturbations, no episode timeout, respawn facing +x."""
    env_cfg.scene.num_envs = 1
    env_cfg.observations.policy.enable_corruption = False

    # no random shoves while driving (play.py only strips an event named "push_robot", which
    # doesn't exist in these cfgs -- the real one is randomize_push_robot)
    env_cfg.events.randomize_push_robot = None
    env_cfg.events.randomize_apply_external_force_torque = None

    # drive indefinitely; only a fall (illegal contact / bad orientation) resets the robot
    env_cfg.terminations.time_out = None

    # respawn facing +x so "up arrow" always starts out driving the same way
    env_cfg.events.randomize_reset_base.params["pose_range"]["yaw"] = (0.0, 0.0)


@configclass
class DeeproboticsM20SkateparkEnvCfg(DeeproboticsM20RoughEnvCfg):
    """M20 driving around a real-world scanned skatepark mesh instead of procedural rough terrain.

    Reuses the exact observation/action/reward wiring of ``DeeproboticsM20RoughEnvCfg`` so the
    checkpoint trained for ``Rough-Deeprobotics-M20-v0`` loads and runs unchanged -- only the terrain
    and a couple of terrain-generator-specific settings are swapped out.
    """

    def __post_init__(self):
        # post init of parent (sets up the procedural rough terrain first)
        super().__post_init__()

        use_skatepark_terrain(self)

        # DeeproboticsM20RoughEnvCfg only prunes zero-weight reward terms for its own exact class
        # (so further subclasses like this one can still tweak weights first) -- redo it here, since
        # otherwise leftover Lite3-shaped params like ".*_foot" (unmatched on M20's wheel body names)
        # fail SceneEntityCfg resolution at env construction.
        if self.__class__.__name__ == "DeeproboticsM20SkateparkEnvCfg":
            self.disable_zero_weight_rewards()


@configclass
class DeeproboticsM20SkateparkEnvCfg_PLAY(DeeproboticsM20SkateparkEnvCfg):
    """Keyboard-driving variant of the rough policy's skatepark env.

    Meant for ``play.py --task Skatepark-Deeprobotics-M20-Play-v0 --keyboard``.
    """

    def __post_init__(self):
        super().__post_init__()

        use_keyboard_play_settings(self)

        # parent only prunes zero-weight rewards for its own exact class name
        self.disable_zero_weight_rewards()


@configclass
class DeeproboticsM20SkateparkV3bEnvCfg_PLAY(DeeproboticsM20StairsSightedV3bEnvCfg):
    """Keyboard-driving the sighted stair-climbing v3b policy around the skatepark scan.

    Keeps v3b's policy observations as trained (incl. its forward-biased height scan, which raycasts
    the scan mesh at /World/ground). Every reward/termination/curriculum term keyed on
    ``task_ids_per_column`` indexes ``terrain.terrain_types``, which only exists for generated
    terrain, so those are dropped -- rewards and curricula don't affect the policy's actions, and the
    stair-aware fall termination falls back to the plain ``bad_orientation_2`` (0.7 pitch/roll limit).

    Meant for ``play.py --task Skatepark-V3b-Deeprobotics-M20-Play-v0 --keyboard``.
    """

    def __post_init__(self):
        super().__post_init__()

        use_skatepark_terrain(self)
        use_keyboard_play_settings(self)
        add_depth_camera(self, noisy=False)

        for group in (self.rewards, self.curriculum):
            for name, term in list(vars(group).items()):
                if isinstance(getattr(term, "params", None), dict) and "task_ids_per_column" in term.params:
                    setattr(group, name, None)
        self.terminations.bad_orientation_2 = DoneTerm(func=mdp.bad_orientation_2)
