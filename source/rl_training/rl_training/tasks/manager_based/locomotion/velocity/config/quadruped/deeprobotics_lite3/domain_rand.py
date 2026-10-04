# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause

"""Lite3 sim-to-real domain randomization on top of what the Lite3 rough task already does.

Already randomized by `Rough-Deeprobotics-Lite3-v0` (and the stairs tasks v1-v3, unchanged):
friction 0.3-1.0 static / 0.3-0.8 dynamic, link masses x0.85-1.15 (inertia recomputed), inertia
x0.85-1.15, torso CoM +-3/3/2 cm, PD gains x0.85-1.15 per reset, actuator delay 0-1 physics step
(0-5 ms), proprioception noise. Off: pushes, external forces, payload.

Added here (constants below are the ranges):
- payload: mass added to the torso (isaaclab's mass event starts from the default mass, so on the
  torso this replaces the x0.85-1.15 link scale rather than adding to it);
- pushes: random base velocity kicks;
- motor strength: per joint, scales Kp, Kd and the torque limit together (replaces the gains event);
- actuator delay: physics steps between the PD target and the applied torque;
- proprioception latency: per-env delay on the base/joint observations of one observation group.

Everything except the depth camera's own noise changes the dynamics or what the actor sees, so a
teacher has to be trained with it before a student distilled from that teacher can use it.
"""

from __future__ import annotations

import copy

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg, SceneEntityCfg

import rl_training.tasks.manager_based.locomotion.velocity.mdp as mdp

PAYLOAD_KG = (-0.5, 2.0)
PUSH_INTERVAL_S = (8.0, 12.0)
PUSH_VELOCITY_MPS = 0.5
MOTOR_STRENGTH = (0.85, 1.15)
ACTUATOR_DELAY_PHYSICS_STEPS = (0, 4)  # x 5 ms -> 0-20 ms
PROPRIO_LATENCY_POLICY_STEPS = (0, 1)  # x 20 ms
# proprioceptive terms that get the latency (commands and the last action are internal: no delay)
PROPRIO_TERMS = ("base_lin_vel", "base_ang_vel", "projected_gravity", "joint_pos", "joint_vel")


def add_payload(env_cfg, mass_kg: tuple[float, float] = PAYLOAD_KG) -> None:
    env_cfg.events.randomize_rigid_body_mass_base = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=[env_cfg.base_link_name]),
            "mass_distribution_params": mass_kg,
            "operation": "add",
            "recompute_inertia": True,
        },
    )


def add_pushes(
    env_cfg, interval_s: tuple[float, float] = PUSH_INTERVAL_S, velocity: float = PUSH_VELOCITY_MPS
) -> None:
    env_cfg.events.randomize_push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=interval_s,
        params={"velocity_range": {"x": (-velocity, velocity), "y": (-velocity, velocity)}},
    )


def add_motor_strength(env_cfg, strength: tuple[float, float] = MOTOR_STRENGTH) -> None:
    gains = env_cfg.events.randomize_actuator_gains.params
    env_cfg.events.randomize_actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains_and_strength,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_scale_range": gains["stiffness_distribution_params"],
            "damping_scale_range": gains["damping_distribution_params"],
            "strength_range": strength,
        },
    )


def set_actuator_delay(env_cfg, physics_steps: tuple[int, int] = ACTUATOR_DELAY_PHYSICS_STEPS) -> None:
    """Copies the robot cfg first, so the shared DEEPROBOTICS_LITE3_CFG constant is not mutated."""
    env_cfg.scene.robot = copy.deepcopy(env_cfg.scene.robot)
    for actuator_cfg in env_cfg.scene.robot.actuators.values():
        actuator_cfg.min_delay, actuator_cfg.max_delay = physics_steps


def add_proprio_latency(group: ObservationGroupCfg, policy_steps: tuple[int, int] = PROPRIO_LATENCY_POLICY_STEPS) -> None:
    for name in PROPRIO_TERMS:
        term = getattr(group, name, None)
        if term is None:
            continue
        term.params = {
            "inner_func": term.func,
            "inner_params": dict(term.params),
            "min_lag": policy_steps[0],
            "max_lag": policy_steps[1],
        }
        term.func = mdp.delayed_observation


def apply_full_domain_rand(env_cfg) -> None:
    """All of the above; the proprioception latency goes on the actor's (`policy`) observations."""
    add_payload(env_cfg)
    add_pushes(env_cfg)
    add_motor_strength(env_cfg)
    set_actuator_delay(env_cfg)
    add_proprio_latency(env_cfg.observations.policy)
