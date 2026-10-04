# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause

"""Lite3 sighted stairs teacher: climb (and descend) 0.04-0.18 m stairs from a height scan.

Ports the M20 v2 stairs setup (terrain/task-ID plumbing, stair curriculum, stair-aware orientation
and posture terms) to the Lite3. The only new reward is stair progress: the dense `stair_progress`
term plus its sparse partner `stair_climb_bonus` (finishing a flight at the commanded pace). Every
other change is a weight change or a swap to an existing stair-aware variant of the same term.
"""

from __future__ import annotations

import copy

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
from rl_training.depth import LITE3_D435I_MOUNT
from rl_training.tasks.manager_based.locomotion.velocity.config.wheeled.deeprobotics_m20.stairs_env_cfg import (
    TASK_FLAT_ROUGH,
    TASK_STAIR_ASCENT,
    TASK_STAIR_DESCENT,
    add_d435i_camera,
    set_task_ids_per_column,
    task_id_per_column,
)

from .domain_rand import add_proprio_latency, apply_full_domain_rand
from .rough_env_cfg import DeeproboticsLite3RoughEnvCfg

##
# Terrain
##

TILE_SIZE = 8.0
SUB_BORDER_WIDTH = 1.0
PLATFORM_WIDTH = 1.6
MAX_STEP_HEIGHT = 0.18
# L-inf distance from the tile centre to the outer edge of the flight (top on ascent, bottom on descent).
STAIRS_OUTER_EDGE = TILE_SIZE / 2 - SUB_BORDER_WIDTH
# Inner edge of the flight: the generator adds a step, so the real centre platform is smaller than
# PLATFORM_WIDTH; ~0.7 m covers every tread width used here.
STAIRS_INNER_EDGE = 0.7
# Root ~0.35 m past the outer edge: the whole ~0.6 m Lite3 body is off the stairs.
STAIR_PROMOTE_DISTANCE = STAIRS_OUTER_EDGE + 0.35
# Less than about a third of the way along the ~2.3 m flight.
STAIR_DEMOTE_DISTANCE = 1.5

STAIR_TASK_IDS = (TASK_STAIR_ASCENT, TASK_STAIR_DESCENT)

##
# Height scan: front-biased. Feet sit at x ~ +0.15 (front) and -0.21 (hind) from the base; the grid
# keeps a ~0.1 m strip behind the hind feet (the MLP teacher has no memory of what it walked over)
# and looks ~1.1 m ahead (about 4 treads, ~1.4 s at 0.8 m/s).
##

SCAN_X_RANGE = (-0.30, 1.10)
SCAN_HALF_WIDTH = 0.35
SCAN_RESOLUTION = 0.05


def build_lite3_stairs_terrain_generator() -> TerrainGeneratorCfg:
    """Same sub-terrain keys as the M20 generator, so `task_id_per_column` maps them to tasks.

    Ascent ~45%, descent ~20%, flat/rough/slopes 35%. Main treads 0.27 m (0.18/0.27 = 34 deg), narrow
    0.24 m, wide 0.32 m; rough noise matches the Lite3 rough task.
    """
    stair_kwargs = dict(platform_width=PLATFORM_WIDTH, border_width=SUB_BORDER_WIDTH, holes=False)
    step_height_range = (0.04, MAX_STEP_HEIGHT)
    sub_terrains = {
        "stair_ascent": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.35, step_height_range=step_height_range, step_width=0.27, **stair_kwargs
        ),
        "stair_descent": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.15, step_height_range=step_height_range, step_width=0.27, **stair_kwargs
        ),
        "stair_ascent_narrow_tread": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.10, step_height_range=step_height_range, step_width=0.24, **stair_kwargs
        ),
        "stair_descent_wide_tread": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.05, step_height_range=step_height_range, step_width=0.32, **stair_kwargs
        ),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.15, noise_range=(0.01, 0.06), noise_step=0.01, border_width=0.25
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
        size=(TILE_SIZE, TILE_SIZE),
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


def build_lite3_stairs_terrain_generator_v4() -> TerrainGeneratorCfg:
    """v4 mix: stairs 40% (ascent 20 / descent 10 / narrow-tread ascent 6 / wide-tread descent 4),
    random rough 20%, boxes 10%, slopes 15%, flat 15%. Same stair geometry as v1-v3."""
    gen = build_lite3_stairs_terrain_generator()
    sub = gen.sub_terrains
    for key, proportion in (
        ("stair_ascent", 0.20),
        ("stair_descent", 0.10),
        ("stair_ascent_narrow_tread", 0.06),
        ("stair_descent_wide_tread", 0.04),
        ("random_rough", 0.20),
        ("hf_pyramid_slope", 0.075),
        ("hf_pyramid_slope_inv", 0.075),
        ("flat", 0.15),
    ):
        sub[key].proportion = proportion
    # blocky ground (the course's rough patch the v2/v3 teachers stumbled on): 0.3 m cells, 2-8 cm high
    sub["boxes"] = terrain_gen.MeshRandomGridTerrainCfg(
        proportion=0.10, grid_width=0.3, grid_height_range=(0.02, 0.08), platform_width=1.5
    )
    return gen


def configure_lite3_height_scan(env_cfg) -> None:
    """Front-biased scan grid. `offset.pos` shifts the ray starts in the yaw-aligned sensor frame only;
    `mdp.height_scan` reads the sensor body height, so the observation's zero point is unchanged."""
    scanner = env_cfg.scene.height_scanner
    scanner.pattern_cfg.resolution = SCAN_RESOLUTION
    scanner.pattern_cfg.size = [SCAN_X_RANGE[1] - SCAN_X_RANGE[0], 2 * SCAN_HALF_WIDTH]
    scanner.offset.pos = ((SCAN_X_RANGE[0] + SCAN_X_RANGE[1]) / 2, 0.0, 20.0)


@configclass
class DeeproboticsLite3StairsSightedEnvCfg(DeeproboticsLite3RoughEnvCfg):
    """Sighted (height-scan) Lite3 stairs teacher, plain MLP actor-critic."""

    def __post_init__(self):
        super().__post_init__()

        # ------------------------------ Terrain / spawn ------------------------------
        self.scene.terrain.terrain_generator = build_lite3_stairs_terrain_generator()
        self.scene.terrain.max_init_terrain_level = 3
        columns = task_id_per_column(self.scene.terrain.terrain_generator)
        self.events.randomize_reset_base.params["pose_range"]["x"] = (-0.3, 0.3)
        self.events.randomize_reset_base.params["pose_range"]["y"] = (-0.3, 0.3)

        # ------------------------------ Height scan (actor + critic) ------------------------------
        configure_lite3_height_scan(self)
        self.observations.policy.height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            noise=Unoise(n_min=-0.025, n_max=0.025),
            clip=(-1.0, 1.0),
            scale=1.0,
        )

        # ------------------------------ Commands ------------------------------
        cmd = self.commands.base_velocity
        cmd.ranges.lin_vel_x = (0.2, 0.8)
        cmd.ranges.lin_vel_y = (-0.3, 0.3)
        cmd.ranges.ang_vel_z = (-0.8, 0.8)
        cmd.heading_command = True
        cmd.rel_heading_envs = 1.0
        cmd.rel_standing_envs = 0.0
        cmd.rel_zero_vel_envs = 0.15
        cmd.rel_only_lin_x_envs = 0.25
        cmd.rel_only_lin_y_envs = 0.0
        cmd.rel_only_ang_z_envs = 0.0

        # ------------------------------ Rewards: stair progress (the one new reward) ------------------------------
        self.rewards.stair_progress = RewTerm(
            func=mdp.stair_progress,
            weight=5.0,
            params={
                "command_name": "base_velocity",
                "task_ids_per_column": columns,
                "ascent_task_ids": (TASK_STAIR_ASCENT,),
                "descent_task_ids": (TASK_STAIR_DESCENT,),
                "max_distance": STAIRS_OUTER_EDGE,
                "height_scale": 1.0,
            },
        )
        self.rewards.stair_climb_bonus = RewTerm(
            func=mdp.stair_climb_bonus,
            weight=10.0,
            params={
                "command_name": "base_velocity",
                "task_ids_per_column": columns,
                "task_ids": (TASK_STAIR_ASCENT,),
                "start_distance": STAIRS_INNER_EDGE,
                "finish_distance": STAIRS_OUTER_EDGE,
                "min_efficiency": 0.5,
                "full_efficiency": 0.9,
            },
        )

        # ------------------------------ Rewards: existing terms, retuned ------------------------------
        # climbing needs vertical speed: ~0.28 m/s average up 0.15/0.27 stairs at 0.5 m/s cost ~1.6/s at -20
        self.rewards.lin_vel_z_l2.weight = -2.0
        # pitch only lightly penalised on stair tiles (roll stays fully penalised)
        self.rewards.flat_orientation_l2.func = mdp.orientation_l2_stair_aware
        self.rewards.flat_orientation_l2.params.update(
            {"task_ids_per_column": columns, "stair_task_ids": STAIR_TASK_IDS, "stair_pitch_scale": 0.05}
        )
        # knee posture penalty x0.1 while the terrain ahead is uneven
        knee = self.rewards.knee_joint_pos_penalty
        knee.func = mdp.joint_pos_penalty_terrain_ahead
        knee.params["sensor_cfg"] = SceneEntityCfg("height_scanner")
        knee.params["roughness_threshold"] = 0.08
        knee.params["rough_penalty_scale"] = 0.1
        # the reference swing apex (~5 cm) works against 18 cm risers
        self.rewards.phase_foot_trajectory_exp.weight = 1.0

        # ------------------------------ Terminations ------------------------------
        self.terminations.bad_orientation_2 = DoneTerm(
            func=mdp.bad_orientation_stair_aware,
            params={
                "task_ids_per_column": columns,
                "stair_task_ids": STAIR_TASK_IDS,
                "stair_pitch_limit": 0.9,
                "pitch_limit": 0.7,
                "roll_limit": 0.7,
            },
        )

        # ------------------------------ Curriculum ------------------------------
        self.curriculum.terrain_levels = CurrTerm(
            func=mdp.terrain_levels_stairs,
            params={
                "task_ids_per_column": columns,
                "stair_task_ids": STAIR_TASK_IDS,
                "stair_promote_distance": STAIR_PROMOTE_DISTANCE,
                "stair_demote_distance": STAIR_DEMOTE_DISTANCE,
            },
        )
        for name, task_id in (
            ("terrain_level_flat_rough", TASK_FLAT_ROUGH),
            ("terrain_level_stair_ascent", TASK_STAIR_ASCENT),
            ("terrain_level_stair_descent", TASK_STAIR_DESCENT),
        ):
            setattr(
                self.curriculum,
                name,
                CurrTerm(func=mdp.terrain_level_by_task, params={"task_ids_per_column": columns, "task_id": task_id}),
            )

        # `DeeproboticsLite3RoughEnvCfg` only strips zero-weight terms for its own class name
        self.disable_zero_weight_rewards()


@configclass
class DeeproboticsLite3StairsSightedEnvCfg_PLAY(DeeproboticsLite3StairsSightedEnvCfg):
    """Play/video variant: fewer envs, no pushes, fixed forward command."""

    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False

        if self.scene.terrain.terrain_generator is not None:
            # keep `curriculum=True` so columns keep their sub-terrain types
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 8
            set_task_ids_per_column(self, task_id_per_column(self.scene.terrain.terrain_generator))
        self.scene.terrain.max_init_terrain_level = None

        cmd = self.commands.base_velocity
        cmd.ranges.lin_vel_x = (0.5, 0.5)
        cmd.ranges.lin_vel_y = (0.0, 0.0)
        cmd.ranges.ang_vel_z = (0.0, 0.0)
        cmd.rel_standing_envs = 0.0
        cmd.rel_zero_vel_envs = 0.0
        cmd.rel_only_lin_x_envs = 0.0
        cmd.rel_only_lin_y_envs = 0.0
        cmd.rel_only_ang_z_envs = 0.0


@configclass
class DeeproboticsLite3StairsSightedV2EnvCfg(DeeproboticsLite3StairsSightedEnvCfg):
    """v2: safer, steadier climbing that reaches the top, and no veering. Commands unchanged from v1.

    v1 (run 2026-10-02_03-00-26) sprinted up flights and drifted left/right. Progress paid per metre
    is worth more the sooner it comes (gamma 0.99 ~ 2 s horizon), while overspeed and sideways drift
    were cheap under tracking std sqrt(0.5). Changes (no new reward terms):

    1. `stair_progress` capped at the commanded speed (Extreme Parkour's min(v.d, v_cmd)).
    2. Velocity tracking in the yaw frame (pitch on stairs no longer mixes vz into vx), lin and yaw
       std sqrt(0.5) -> 0.3.
    3. Left-right symmetry augmentation in PPO (runner cfg, `symmetry.compute_symmetric_states_lr`).
    4. The climb bonus only penalises overrunning the command: reaching the top at or below the
       commanded pace pays in full.
    """

    def __post_init__(self):
        super().__post_init__()

        # (1) progress capped at the commanded speed
        self.rewards.stair_progress.params["cap_to_command"] = True
        # (2) tighter tracking, yaw frame
        self.rewards.track_lin_vel_xy_exp.func = mdp.track_lin_vel_xy_yaw_frame_exp
        self.rewards.track_lin_vel_xy_exp.params["std"] = 0.3
        self.rewards.track_ang_vel_z_exp.params["std"] = 0.3
        # (4) reaching the top is what the bonus pays for; only overrunning the command reduces it
        self.rewards.stair_climb_bonus.params["penalize_slow"] = False


@configclass
class DeeproboticsLite3StairsSightedV2EnvCfg_PLAY(DeeproboticsLite3StairsSightedV2EnvCfg):
    """Play/video variant: fewer envs, no pushes, fixed forward command."""

    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False

        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 8
            set_task_ids_per_column(self, task_id_per_column(self.scene.terrain.terrain_generator))
        self.scene.terrain.max_init_terrain_level = None

        cmd = self.commands.base_velocity
        cmd.ranges.lin_vel_x = (0.5, 0.5)
        cmd.ranges.lin_vel_y = (0.0, 0.0)
        cmd.ranges.ang_vel_z = (0.0, 0.0)
        cmd.rel_standing_envs = 0.0
        cmd.rel_zero_vel_envs = 0.0
        cmd.rel_only_lin_x_envs = 0.0
        cmd.rel_only_lin_y_envs = 0.0
        cmd.rel_only_ang_z_envs = 0.0


# Forward-speed allowance on stair tiles (fraction of the command, at least STAIR_MIN_TOLERANCE m/s).
STAIR_SPEED_TOLERANCE = 0.25
STAIR_MIN_TOLERANCE = 0.1


@configclass
class DeeproboticsLite3StairsSightedV3EnvCfg(DeeproboticsLite3StairsSightedV2EnvCfg):
    """v3: on stairs, a bit faster or slower than commanded is free. Commands unchanged from v1.

    v2 (run 2026-10-02_04-46-07) fixed ascent (100% to 0.18 m) and the veer, but fell on fast descents
    (0.8 m/s: 33% success, 12/18 falls at 0.18 m vs v1's 100%): tracking std 0.3 made slowing down on a
    descent expensive. Changes (no new reward terms):

    1. Linear tracking: on stair tiles the error along the command gets a +-25% deadband (min 0.1 m/s);
       sideways error stays at std 0.3 everywhere, flat ground is unchanged.
    2. Progress cap at 1.25 x the commanded speed, and the climb bonus pays in full up to 25% over pace
       (full_efficiency 0.8), so all three terms agree on the allowance.
    """

    def __post_init__(self):
        super().__post_init__()
        columns = task_id_per_column(self.scene.terrain.terrain_generator)

        # (1) forward-speed deadband on stairs
        track = self.rewards.track_lin_vel_xy_exp
        track.func = mdp.track_lin_vel_xy_stair_tolerant_exp
        track.params.update(
            {
                "task_ids_per_column": columns,
                "stair_task_ids": STAIR_TASK_IDS,
                "speed_tolerance": STAIR_SPEED_TOLERANCE,
                "min_tolerance": STAIR_MIN_TOLERANCE,
            }
        )
        # (2) matching allowance in the progress cap and the bonus
        self.rewards.stair_progress.params["cap_factor"] = 1.0 + STAIR_SPEED_TOLERANCE
        self.rewards.stair_climb_bonus.params["full_efficiency"] = 1.0 / (1.0 + STAIR_SPEED_TOLERANCE)


@configclass
class DeeproboticsLite3StairsSightedV3EnvCfg_PLAY(DeeproboticsLite3StairsSightedV3EnvCfg):
    """Play/video variant: fewer envs, no pushes, fixed forward command."""

    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False

        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 8
            set_task_ids_per_column(self, task_id_per_column(self.scene.terrain.terrain_generator))
        self.scene.terrain.max_init_terrain_level = None

        cmd = self.commands.base_velocity
        cmd.ranges.lin_vel_x = (0.5, 0.5)
        cmd.ranges.lin_vel_y = (0.0, 0.0)
        cmd.ranges.ang_vel_z = (0.0, 0.0)
        cmd.rel_standing_envs = 0.0
        cmd.rel_zero_vel_envs = 0.0
        cmd.rel_only_lin_x_envs = 0.0
        cmd.rel_only_lin_y_envs = 0.0
        cmd.rel_only_ang_z_envs = 0.0


##
# v4 teacher and depth students
##


@configclass
class DeeproboticsLite3StairsSightedV4EnvCfg(DeeproboticsLite3StairsSightedV3EnvCfg):
    """v4 teacher: v3's rewards on the v4 terrain mix (40% stairs, more rough/boxes/slopes/flat), with
    the full sim-to-real domain randomization (`domain_rand.apply_full_domain_rand`: payload, pushes,
    motor strength, 0-20 ms actuator delay, 0-20 ms proprioception latency)."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_generator = build_lite3_stairs_terrain_generator_v4()
        set_task_ids_per_column(self, task_id_per_column(self.scene.terrain.terrain_generator))
        apply_full_domain_rand(self)


@configclass
class DeeproboticsLite3StairsSightedV4EnvCfg_PLAY(DeeproboticsLite3StairsSightedV4EnvCfg):
    """Play/video variant: fewer envs, no pushes, fixed forward command."""

    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.events.randomize_push_robot = None

        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 8
            set_task_ids_per_column(self, task_id_per_column(self.scene.terrain.terrain_generator))
        self.scene.terrain.max_init_terrain_level = None

        cmd = self.commands.base_velocity
        cmd.ranges.lin_vel_x = (0.5, 0.5)
        cmd.ranges.lin_vel_y = (0.0, 0.0)
        cmd.ranges.ang_vel_z = (0.0, 0.0)
        cmd.rel_standing_envs = 0.0
        cmd.rel_zero_vel_envs = 0.0
        cmd.rel_only_lin_x_envs = 0.0
        cmd.rel_only_lin_y_envs = 0.0
        cmd.rel_only_ang_z_envs = 0.0


def make_depth_student(env_cfg) -> None:
    """Turn a sighted teacher env into a depth-student env for DAgger distillation.

    - `teacher`: the teacher's own `policy` observations, copied before anything below changes them
      (height scan included, noise off), so the frozen actor loads strictly and sees what it trained on;
    - `policy`: the student's proprioception = the teacher's minus the height scan;
    - `depth`: (N, 1, 36, 64) RealSense D435i image with the full sensor model (stereo occlusion,
      disparity noise, holes, 20-60 ms latency, mount jitter; docs/depth_pipeline.md), from the
      placeholder Lite3 mount `rl_training.depth.LITE3_D435I_MOUNT`.
    """
    add_d435i_camera(env_cfg, mount=LITE3_D435I_MOUNT)
    teacher = copy.deepcopy(env_cfg.observations.policy)
    teacher.enable_corruption = False
    env_cfg.observations.teacher = teacher
    env_cfg.observations.policy.height_scan = None


@configclass
class DeeproboticsLite3StairsStudentEnvCfg(DeeproboticsLite3StairsSightedV3EnvCfg):
    """Stage A depth student, distilled from the frozen v3 teacher (model_19999).

    Terrain is the v4 mix. Only randomization the teacher's actions stay valid under is added: the
    depth sensor model (`make_depth_student`) and 0-20 ms latency on the student's proprioception
    (the teacher's copy has none). Dynamics randomization (payload, pushes, motor strength, actuator
    delay) waits for the v4 teacher: see `DeeproboticsLite3StairsStudentV4EnvCfg`.
    """

    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_generator = build_lite3_stairs_terrain_generator_v4()
        set_task_ids_per_column(self, task_id_per_column(self.scene.terrain.terrain_generator))
        make_depth_student(self)
        add_proprio_latency(self.observations.policy)


@configclass
class DeeproboticsLite3StairsStudentEnvCfg_PLAY(DeeproboticsLite3StairsStudentEnvCfg):
    """Play/video variant: fewer envs, fixed forward command, clean depth and proprioception."""

    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.observations.depth.enable_corruption = False
        self.events.randomize_depth_camera_offset = None

        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 8
            set_task_ids_per_column(self, task_id_per_column(self.scene.terrain.terrain_generator))
        self.scene.terrain.max_init_terrain_level = None

        cmd = self.commands.base_velocity
        cmd.ranges.lin_vel_x = (0.5, 0.5)
        cmd.ranges.lin_vel_y = (0.0, 0.0)
        cmd.ranges.ang_vel_z = (0.0, 0.0)
        cmd.rel_standing_envs = 0.0
        cmd.rel_zero_vel_envs = 0.0
        cmd.rel_only_lin_x_envs = 0.0
        cmd.rel_only_lin_y_envs = 0.0
        cmd.rel_only_ang_z_envs = 0.0


@configclass
class DeeproboticsLite3StairsStudentV4EnvCfg(DeeproboticsLite3StairsSightedV4EnvCfg):
    """Stage B depth student, distilled from a v4 teacher: the v4 env (terrain mix + full domain
    randomization, the teacher trained with the same proprioception latency) plus the depth sensor."""

    def __post_init__(self):
        super().__post_init__()
        make_depth_student(self)
