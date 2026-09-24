# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause

"""Stair-task-aware MDP terms for the M20 stairs envs (command.md Phase 3.5 / 4.4).

Everything here is new rather than a modification of the shared reward/termination/curriculum
functions, so `Rough-`/`Flat-Deeprobotics-M20-v0` behaviour is untouched (command.md ground rules).

Terms that behave differently on stair tiles take `task_ids_per_column` (raw terrain grid column ->
task ID, see `stairs_env_cfg.task_id_per_column`) plus the task IDs they should treat as stairs.
Task IDs are passed as plain ints so this module does not import the env-config module.

Distances on stair tiles are Chebyshev (L-inf) distances from the tile origin: the pyramid stairs
are square, so "further out" in L-inf is exactly "further up" (inverted pyramid, ascent) or
"further down" (pyramid, descent), whatever heading the robot takes out of the centre.
"""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.sensors import ContactSensor, RayCaster
from isaaclab.terrains import TerrainImporter

from .rewards import joint_pos_penalty_except_turn_side_cmd, update_gait_level_from_terrain_mean

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import RewardTermCfg

# `mdp/__init__.py` star-imports this module; export only the new terms so it can't rebind any
# name the shared mdp namespace already has.
__all__ = [
    "env_task_mask",
    "chebyshev_distance_from_origin",
    "terrain_roughness_ahead",
    "orientation_l2_stair_aware",
    "bad_orientation_stair_aware",
    "bad_orientation_stair_aware_penalty",
    "joint_pos_penalty_terrain_ahead",
    "stair_progress",
    "stair_wheel_clearance",
    "terrain_levels_stairs",
    "track_heading_exp",
]


##
# Helpers
##


def env_task_mask(env: ManagerBasedRLEnv, task_ids_per_column: Sequence[int], task_ids: Sequence[int]) -> torch.Tensor:
    """Bool mask of envs whose (fixed) terrain column maps to any of `task_ids`.

    `terrain_types` never changes after terrain generation, so the mask is cached on the env.
    """
    key = (tuple(task_ids_per_column), tuple(task_ids))
    cache = env.__dict__.setdefault("_stairs_task_mask_cache", {})
    if key not in cache:
        terrain: TerrainImporter = env.scene.terrain
        task_of_column = torch.tensor(task_ids_per_column, device=env.device, dtype=torch.long)
        task_of_env = task_of_column[terrain.terrain_types]
        wanted = torch.tensor(list(task_ids), device=env.device, dtype=torch.long)
        cache[key] = torch.isin(task_of_env, wanted)
    return cache[key]


def chebyshev_distance_from_origin(env: ManagerBasedRLEnv, asset: Articulation) -> torch.Tensor:
    delta = asset.data.root_pos_w[:, :2] - env.scene.env_origins[:, :2]
    return torch.max(torch.abs(delta), dim=1).values


def _outward_alignment(env: ManagerBasedRLEnv, asset: Articulation, command_name: str) -> torch.Tensor:
    """cos of the angle between the world-frame velocity command and the tile's outward axis, >= 0.

    The outward axis of a square pyramid is the dominant axis of the robot's offset from the tile
    centre. Used to only pay stair progress when the command actually asks to go that way.
    """
    delta = asset.data.root_pos_w[:, :2] - env.scene.env_origins[:, :2]
    along_x = torch.abs(delta[:, 0]) >= torch.abs(delta[:, 1])
    outward = torch.zeros_like(delta)
    outward[:, 0] = torch.where(along_x, torch.sign(delta[:, 0]), torch.zeros_like(delta[:, 0]))
    outward[:, 1] = torch.where(along_x, torch.zeros_like(delta[:, 1]), torch.sign(delta[:, 1]))

    cmd_b = env.command_manager.get_command(command_name)[:, :2]
    yaw = torch.atan2(
        2.0 * (asset.data.root_quat_w[:, 0] * asset.data.root_quat_w[:, 3]
               + asset.data.root_quat_w[:, 1] * asset.data.root_quat_w[:, 2]),
        1.0 - 2.0 * (asset.data.root_quat_w[:, 2] ** 2 + asset.data.root_quat_w[:, 3] ** 2),
    )
    cos_y, sin_y = torch.cos(yaw), torch.sin(yaw)
    cmd_w = torch.stack((cos_y * cmd_b[:, 0] - sin_y * cmd_b[:, 1], sin_y * cmd_b[:, 0] + cos_y * cmd_b[:, 1]), dim=1)
    cmd_dir = cmd_w / torch.clamp(torch.linalg.norm(cmd_w, dim=1, keepdim=True), min=1e-6)
    return torch.clamp(torch.sum(cmd_dir * outward, dim=1), min=0.0)


def terrain_roughness_ahead(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Max - min terrain height over the (forward-biased) height-scan grid."""
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    hits_z = sensor.data.ray_hits_w[..., 2]
    valid = torch.isfinite(hits_z)
    hi = torch.where(valid, hits_z, torch.full_like(hits_z, -1e6)).max(dim=1).values
    lo = torch.where(valid, hits_z, torch.full_like(hits_z, 1e6)).min(dim=1).values
    return torch.where(valid.any(dim=1), hi - lo, torch.zeros_like(hi))


##
# Orientation: rewards + termination
##


def orientation_l2_stair_aware(
    env: ManagerBasedRLEnv,
    task_ids_per_column: Sequence[int],
    stair_task_ids: Sequence[int],
    stair_pitch_scale: float = 0.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """`flat_orientation_l2`, but pitch is scaled by `stair_pitch_scale` on stair tiles.

    Climbing 0.20/0.30 stairs needs ~34 deg of body pitch; the flat penalty at weight -15 costs
    ~4.7/s there, i.e. nearly the whole velocity-tracking reward. Roll stays fully penalised.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    g = asset.data.projected_gravity_b
    on_stairs = env_task_mask(env, task_ids_per_column, stair_task_ids)
    pitch_scale = torch.where(on_stairs, stair_pitch_scale, 1.0)
    return torch.square(g[:, 1]) + pitch_scale * torch.square(g[:, 0])


def bad_orientation_stair_aware(
    env: ManagerBasedRLEnv,
    task_ids_per_column: Sequence[int],
    stair_task_ids: Sequence[int],
    stair_pitch_limit: float = 0.9,
    pitch_limit: float = 0.7,
    roll_limit: float = 0.7,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """`bad_orientation_2` with a separate (looser) pitch limit on stair tiles.

    `bad_orientation_2` trips at |g_x| or |g_y| > 0.7 (~44 deg). 0.30 m risers on 0.30 m treads
    are 45 deg, so they were infeasible by construction, and nosing over a tall edge on descent
    spikes pitch past it (-1000 + termination), which is what taught the policy to teeter at the top.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    g = asset.data.projected_gravity_b
    on_stairs = env_task_mask(env, task_ids_per_column, stair_task_ids)
    pitch_lim = torch.where(on_stairs, stair_pitch_limit, pitch_limit)
    return (g[:, 2] > 0) | (torch.abs(g[:, 0]) > pitch_lim) | (torch.abs(g[:, 1]) > roll_limit)


def bad_orientation_stair_aware_penalty(
    env: ManagerBasedRLEnv,
    task_ids_per_column: Sequence[int],
    stair_task_ids: Sequence[int],
    stair_pitch_limit: float = 0.9,
    pitch_limit: float = 0.7,
    roll_limit: float = 0.7,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    return bad_orientation_stair_aware(
        env, task_ids_per_column, stair_task_ids, stair_pitch_limit, pitch_limit, roll_limit, asset_cfg
    ).float()


##
# Posture penalty gated on terrain ahead
##


def joint_pos_penalty_terrain_ahead(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    stand_still_scale: float,
    velocity_threshold: float,
    command_threshold: float,
    sensor_cfg: SceneEntityCfg,
    ang_cmd_threshold: float = 0.1,
    y_cmd_threshold: float = 0.1,
    xy_norm_max: float = 0.1,
    xz_norm_max: float = 0.1,
    roughness_threshold: float = 0.08,
    rough_penalty_scale: float = 0.1,
) -> torch.Tensor:
    """`joint_pos_penalty_except_turn_side_cmd`, relaxed when the terrain *ahead* is uneven.

    The original relaxes only once the terrain under the base is 6 cm above the env origin, which
    keeps the full posture penalty on while approaching the first riser (no pre-emptive leg lift)
    and for the whole of a descent (terrain is always below the origin there).
    """
    reward = joint_pos_penalty_except_turn_side_cmd(
        env,
        command_name=command_name,
        asset_cfg=asset_cfg,
        stand_still_scale=stand_still_scale,
        velocity_threshold=velocity_threshold,
        command_threshold=command_threshold,
        ang_cmd_threshold=ang_cmd_threshold,
        y_cmd_threshold=y_cmd_threshold,
        xy_norm_max=xy_norm_max,
        xz_norm_max=xz_norm_max,
        sensor_cfg=None,
    )
    rough = terrain_roughness_ahead(env, sensor_cfg) > roughness_threshold
    return reward * torch.where(rough, rough_penalty_scale, 1.0)


##
# Stair progress + wheel clearance
##


class stair_progress(ManagerTermBase):
    """Reward new progress out of a stair tile's centre: max-so-far, so it can't be farmed.

    Per step: increase of the episode-best Chebyshev distance from the tile origin (capped at
    `max_distance`, the outer edge of the stairs), plus on ascent tiles `height_scale` x the
    increase of the episode-best base height. Returned in m/s (divided by step_dt) so the term
    weight reads as "reward per metre of progress". Only paid on stair tiles, when the command is
    non-trivial, scaled by how well the command points outward (so it never pays for ignoring a
    sideways command).
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.asset: Articulation = env.scene[cfg.params.get("asset_cfg", SceneEntityCfg("robot")).name]
        self.best_dist = torch.zeros(env.num_envs, device=env.device)
        self.best_z = torch.zeros(env.num_envs, device=env.device)
        self.needs_init = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            self.needs_init[:] = True
        else:
            self.needs_init[env_ids] = True

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        command_name: str,
        task_ids_per_column: Sequence[int],
        ascent_task_ids: Sequence[int],
        descent_task_ids: Sequence[int],
        max_distance: float,
        height_scale: float = 1.0,
        command_threshold: float = 0.1,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:
        dist = torch.clamp(chebyshev_distance_from_origin(env, self.asset), max=max_distance)
        z = self.asset.data.root_pos_w[:, 2]

        self.best_dist = torch.where(self.needs_init, dist, self.best_dist)
        self.best_z = torch.where(self.needs_init, z, self.best_z)
        self.needs_init[:] = False

        gain_dist = torch.clamp(dist - self.best_dist, min=0.0)
        gain_z = torch.clamp(z - self.best_z, min=0.0)
        self.best_dist = torch.maximum(self.best_dist, dist)
        self.best_z = torch.maximum(self.best_z, z)

        ascent = env_task_mask(env, task_ids_per_column, ascent_task_ids)
        stairs = ascent | env_task_mask(env, task_ids_per_column, descent_task_ids)
        cmd_active = torch.linalg.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > command_threshold
        align = _outward_alignment(env, self.asset, command_name)

        gain = gain_dist + height_scale * gain_z * ascent.float()
        return gain * stairs.float() * cmd_active.float() * align / env.step_dt


def stair_wheel_clearance(
    env: ManagerBasedRLEnv,
    command_name: str,
    task_ids_per_column: Sequence[int],
    stair_task_ids: Sequence[int],
    sensor_cfg: SceneEntityCfg,
    contact_sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    wheel_radius: float = 0.09,
    target_clearance: float = 0.1,
    lookahead: float = 0.3,
    lateral_window: float = 0.1,
    riser_threshold: float = 0.05,
    min_speed: float = 0.1,
) -> torch.Tensor:
    """Shaping for lifting a wheel over the riser in front of it, before hitting it.

    For each airborne wheel with a riser in the `lookahead` window in front of it (terrain there
    is > `riser_threshold` higher than under the wheel), pays `clamp(1 + clearance/target, 0, 1)`
    where clearance = wheel bottom - highest terrain in that window. Only on stair tiles, only while
    the robot is actually moving along the command (> `min_speed`) so it can't be farmed by
    standing in front of a step with a leg up.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    scanner: RayCaster = env.scene.sensors[sensor_cfg.name]
    contact: ContactSensor = env.scene.sensors[contact_sensor_cfg.name]

    airborne = contact.data.current_contact_time[:, contact_sensor_cfg.body_ids] <= 0.0  # (N, W)
    wheel_pos = asset.data.body_pos_w[:, asset_cfg.body_ids]  # (N, W, 3)
    hits = scanner.data.ray_hits_w  # (N, R, 3)

    quat = asset.data.root_quat_w
    yaw = torch.atan2(
        2.0 * (quat[:, 0] * quat[:, 3] + quat[:, 1] * quat[:, 2]),
        1.0 - 2.0 * (quat[:, 2] ** 2 + quat[:, 3] ** 2),
    )
    cos_y, sin_y = torch.cos(yaw)[:, None, None], torch.sin(yaw)[:, None, None]
    rel = hits[:, None, :, :2] - wheel_pos[:, :, None, :2]  # (N, W, R, 2)
    fwd = rel[..., 0] * cos_y + rel[..., 1] * sin_y
    lat = -rel[..., 0] * sin_y + rel[..., 1] * cos_y
    hits_z = hits[:, None, :, 2].expand_as(fwd)
    valid = torch.isfinite(hits_z) & (torch.abs(lat) <= lateral_window)

    ahead = valid & (fwd > 0.0) & (fwd <= lookahead)
    under = valid & (torch.abs(fwd) <= 0.06)
    neg_inf = torch.full_like(hits_z, -1e6)
    h_ahead = torch.where(ahead, hits_z, neg_inf).max(dim=-1).values
    h_under = torch.where(under, hits_z, neg_inf).max(dim=-1).values
    riser = (h_ahead - h_under > riser_threshold) & ahead.any(-1) & under.any(-1)

    clearance = wheel_pos[..., 2] - wheel_radius - h_ahead
    score = torch.clamp(1.0 + clearance / target_clearance, 0.0, 1.0) * (airborne & riser).float()

    stairs = env_task_mask(env, task_ids_per_column, stair_task_ids)
    cmd = env.command_manager.get_command(command_name)[:, :2]
    cmd_dir = cmd / torch.clamp(torch.linalg.norm(cmd, dim=1, keepdim=True), min=1e-6)
    vel_b = asset.data.root_lin_vel_b[:, :2]
    moving = (torch.sum(vel_b * cmd_dir, dim=1) > min_speed) & (torch.linalg.norm(cmd, dim=1) > 0.1)
    return torch.sum(score, dim=1) * (stairs & moving).float()


##
# Heading
##


class track_heading_exp(ManagerTermBase):
    """Reward holding the commanded heading, not just the commanded yaw rate.

    `track_ang_vel_z_exp` only scores the yaw *rate*; a small constant bias costs almost nothing
    per step but integrates into a large heading error (the course eval saw +31 deg over one
    rough patch with yaw-rate command 0). This scores the heading itself:

    - heading-controlled envs: target = the command term's `heading_target`;
    - all other envs: target = the heading when the command was last (re)sampled, advanced by the
      commanded yaw rate (so a yaw-rate-0 command means "hold the heading you had").

    `exp(-err^2 / std^2)`. If a non-heading env ends up more than `reanchor_error` off (e.g. after
    a push), its reference is re-anchored to the current heading so the term keeps giving a
    usable signal instead of sitting at ~0.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.asset: Articulation = env.scene[cfg.params.get("asset_cfg", SceneEntityCfg("robot")).name]
        self.ref = torch.zeros(env.num_envs, device=env.device)
        self.prev_cmd = torch.zeros(env.num_envs, 3, device=env.device)
        self.needs_init = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            self.needs_init[:] = True
        else:
            self.needs_init[env_ids] = True

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        command_name: str,
        std: float,
        reanchor_error: float = 1.57,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:
        term = env.command_manager.get_term(command_name)
        cmd = term.command
        heading = self.asset.data.heading_w
        heading_env = term.is_heading_env

        # non-heading commands are constant between resamples, so a change means a new command
        resampled = torch.any(cmd != self.prev_cmd, dim=1) & ~heading_env
        self.ref = self.ref + cmd[:, 2] * env.step_dt
        self.ref = torch.where(self.needs_init | resampled, heading, self.ref)
        self.needs_init[:] = False
        self.prev_cmd = cmd.clone()

        target = torch.where(heading_env, term.heading_target, self.ref)
        err = torch.atan2(torch.sin(target - heading), torch.cos(target - heading))
        reward = torch.exp(-torch.square(err) / std**2)

        far = ~heading_env & (torch.abs(err) > reanchor_error)
        self.ref = torch.where(far, heading, self.ref)
        return reward


##
# Curriculum
##


def terrain_levels_stairs(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    task_ids_per_column: Sequence[int],
    stair_task_ids: Sequence[int],
    stair_promote_distance: float,
    stair_demote_distance: float,
    command_threshold: float = 0.2,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """`terrain_levels_vel` with a stair-specific promote/demote rule (+ same global gait_level update).

    Non-stair tiles: unchanged (promote past half a tile, demote below half the commanded distance).
    Stair tiles: promote if the robot got off the stairs (L-inf distance >= `stair_promote_distance`,
    i.e. reached the top on ascent / the bottom on descent); demote only if it made it less than
    `stair_demote_distance` out while being commanded to move; otherwise stay. The shared rule
    demoted every partial climb at commands >= 0.4 m/s (it needs |cmd| x 10 s of travel), so
    partial progress on a tall flight counted the same as not trying.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    terrain: TerrainImporter = env.scene.terrain
    command = env.command_manager.get_command("base_velocity")

    env_ids_t = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
    delta = asset.data.root_pos_w[env_ids_t, :2] - env.scene.env_origins[env_ids_t, :2]
    distance = torch.norm(delta, dim=1)
    cheb = torch.max(torch.abs(delta), dim=1).values
    cmd_norm = torch.norm(command[env_ids_t, :2], dim=1)

    move_up = distance > terrain.cfg.terrain_generator.size[0] / 2
    move_down = (distance < cmd_norm * env.max_episode_length_s * 0.5) & ~move_up

    stairs = env_task_mask(env, task_ids_per_column, stair_task_ids)[env_ids_t]
    stair_up = cheb >= stair_promote_distance
    stair_down = (cheb < stair_demote_distance) & (cmd_norm > command_threshold) & ~stair_up
    move_up = torch.where(stairs, stair_up, move_up)
    move_down = torch.where(stairs, stair_down, move_down)

    terrain.update_env_origins(env_ids_t, move_up, move_down)

    mean_level = torch.mean(terrain.terrain_levels.float())
    update_gait_level_from_terrain_mean(mean_level)
    return mean_level
