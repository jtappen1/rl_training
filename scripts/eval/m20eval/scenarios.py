# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause

"""Scripted scenarios of the M20 evaluation suite (docs/plans/M20_Evaluation_Suite.md).

Pure Python: no Isaac Sim import, so scenarios and metrics can be checked without a simulator.

Every scenario starts with a SETTLE_S zero-command phase (the robot drops from its spawn height and
settles), followed by its command segments. All times below are in seconds after the settle phase.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

SETTLE_S = 2.0
HOLD_S = 10.0
SS_WINDOW_S = 5.0  # steady state = last 5 s of a 10 s hold
STANDSTILL_S = 20.0
PUSH_AT_S = 3.0  # push time after the start of the hold
TERRAIN_MIN_PROGRESS = 0.5  # terrain success: alive and >= 50 % of the commanded distance covered
STANDSTILL_WHEEL_LIMIT = 0.5  # rad/s
KNEE_TORQUE_LIMIT = 76.4  # N*m, compared with the p99 of per-env knee peaks
ACTION_HF_CUTOFF_HZ = 10.0
RISE_SMOOTHING_S = 0.1  # centred moving average before detecting the 10 % / 90 % crossings

TESTS = ("linear_tracking", "yaw_tracking", "step_response", "standstill", "robustness", "terrain")


@dataclass
class Scenario:
    test: str
    case: str
    kind: str  # "hold" | "step" | "standstill"
    segments: list[tuple[float, tuple[float, float, float]]]  # (duration_s, (vx, vy, wz)) after settle
    terrain: str = "flat"
    spawn_yaw: float = 0.0
    perturbation: dict = field(default_factory=dict)
    # step response only: the signal that is stepped ("vx" or "wz") and the step time after settle
    step_signal: str = ""

    @property
    def duration_s(self) -> float:
        return SETTLE_S + sum(d for d, _ in self.segments)

    @property
    def key(self) -> str:
        return f"{self.test}/{self.case}"


def _yaw_towards_plus_x(vx: float, vy: float) -> float:
    """Spawn yaw that makes a base-frame (vx, vy) command move the robot along world +x."""
    if vx == 0.0 and vy == 0.0:
        return 0.0
    return -math.atan2(vy, vx)


def _fmt(v: float) -> str:
    return f"{v:+.2f}"


def build_scenarios(tests: tuple[str, ...] | list[str] = TESTS) -> list[Scenario]:
    scenarios: list[Scenario] = []

    if "linear_tracking" in tests:
        for vx in (-1.0, -0.5, 0.5, 1.0, 1.5, 2.0):
            scenarios.append(
                Scenario("linear_tracking", f"vx_{_fmt(vx)}", "hold", [(HOLD_S, (vx, 0.0, 0.0))],
                         spawn_yaw=_yaw_towards_plus_x(vx, 0.0))
            )
        for vy in (-0.5, -0.25, 0.25, 0.5):
            scenarios.append(
                Scenario("linear_tracking", f"vy_{_fmt(vy)}", "hold", [(HOLD_S, (0.0, vy, 0.0))],
                         spawn_yaw=_yaw_towards_plus_x(0.0, vy))
            )

    if "yaw_tracking" in tests:
        for wz in (0.5, 1.0, 1.5, 2.0):
            scenarios.append(Scenario("yaw_tracking", f"wz_{_fmt(wz)}", "hold", [(HOLD_S, (0.0, 0.0, wz))]))
        scenarios.append(Scenario("yaw_tracking", "arc_vx_+1.00_wz_+0.50", "hold", [(HOLD_S, (1.0, 0.0, 0.5))]))

    if "step_response" in tests:
        scenarios.append(
            Scenario("step_response", "vx_0_to_1", "step", [(3.0, (0.0, 0.0, 0.0)), (7.0, (1.0, 0.0, 0.0))],
                     step_signal="vx")
        )
        scenarios.append(
            Scenario("step_response", "vx_1_to_0", "step", [(6.0, (1.0, 0.0, 0.0)), (6.0, (0.0, 0.0, 0.0))],
                     step_signal="vx")
        )
        scenarios.append(
            Scenario("step_response", "wz_0_to_1.5", "step", [(3.0, (0.0, 0.0, 0.0)), (7.0, (0.0, 0.0, 1.5))],
                     step_signal="wz")
        )

    if "standstill" in tests:
        scenarios.append(Scenario("standstill", "zero_cmd_20s", "standstill", [(STANDSTILL_S, (0.0, 0.0, 0.0))]))

    if "robustness" in tests:
        fwd = [(HOLD_S, (1.0, 0.0, 0.0))]
        for v in (0.5, 1.0, 1.5):
            scenarios.append(
                Scenario("robustness", f"push_{v:.1f}mps", "hold", list(fwd),
                         perturbation={"type": "push", "speed": v, "at_s": PUSH_AT_S})
            )
        scenarios.append(
            Scenario("robustness", "friction_0.35", "hold", list(fwd), perturbation={"type": "friction", "value": 0.35})
        )
        scenarios.append(
            Scenario("robustness", "latency_15ms_spikes_40ms", "hold", list(fwd),
                     perturbation={"type": "latency", "base_ms": 15.0, "spike_ms": 40.0, "spike_prob": 0.02})
        )

    if "terrain" in tests:
        for tile in ("slope_15deg", "stairs_10cm", "stairs_15cm", "rough_medium"):
            scenarios.append(Scenario("terrain", tile, "hold", [(HOLD_S, (1.0, 0.0, 0.0))], terrain=tile))

    return scenarios


# Reference scenario for "tracking error increase vs nominal" in the robustness test: the same command
# (1 m/s forward) with no perturbation, from the same pass.
ROBUSTNESS_REFERENCE = ("linear_tracking", "vx_+1.00")


def pack_tests(envs_per_case: int, max_envs: int, tests: tuple[str, ...] | list[str] = TESTS) -> list[list[str]]:
    """Group whole tests into chunks of at most ``max_envs`` envs (one simulation process per chunk)."""
    sizes = {t: len(build_scenarios([t])) * envs_per_case for t in tests}
    chunks: list[list[str]] = []
    loads: list[int] = []
    for t in sorted(tests, key=lambda t: -sizes[t]):
        for i, load in enumerate(loads):
            if load + sizes[t] <= max_envs:
                chunks[i].append(t)
                loads[i] += sizes[t]
                break
        else:
            chunks.append([t])
            loads.append(sizes[t])
    # keep the canonical test order inside each chunk
    return [[t for t in tests if t in c] for c in chunks]
