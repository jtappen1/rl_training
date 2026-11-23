# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause

"""Checks of the M20 eval metrics on synthetic signals (no simulator).

Run: python -m pytest -p no:typeguard scripts/eval/tests -q
"""

import copy
import math
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from m20eval import compare as C  # noqa: E402
from m20eval import metrics as M  # noqa: E402
from m20eval import results as R  # noqa: E402
from m20eval import scenarios as S  # noqa: E402

DT = 0.02


def _log(sc, n=4, vel_fn=None, fail_step=None, actions=None):
    T = int(round(sc.duration_s / DT))
    cmd = torch.zeros(T, 3)
    k = int(round(S.SETTLE_S / DT))
    for dur, c in sc.segments:
        m = int(round(dur / DT))
        cmd[k : k + m] = torch.tensor(c)
        k += m
    vel = vel_fn(cmd, T, n) if vel_fn else cmd.unsqueeze(1).repeat(1, n, 1)
    # integrate position in the world frame (heading fixed at spawn yaw, small yaw rates ignored)
    yaw = torch.full((T, n), sc.spawn_yaw)
    vx_w = vel[..., 0] * torch.cos(yaw) - vel[..., 1] * torch.sin(yaw)
    vy_w = vel[..., 0] * torch.sin(yaw) + vel[..., 1] * torch.cos(yaw)
    pos = torch.stack([torch.cumsum(vx_w, 0), torch.cumsum(vy_w, 0)], dim=-1) * DT
    return M.ScenarioLog(
        vel=vel,
        pos=pos,
        heading=yaw,
        actions=actions if actions is not None else torch.zeros(T, n, 16),
        cmd=cmd,
        fail_step=fail_step if fail_step is not None else torch.full((n,), 1 << 30),
        fail_kind=torch.zeros(n, dtype=torch.long),
        wheel_target_max=torch.zeros(n),
        dt=DT,
    )


def _scenario(test, case):
    return next(s for s in S.build_scenarios() if s.test == test and s.case == case)


def test_perfect_tracking_has_zero_error_and_full_survival():
    sc = _scenario("linear_tracking", "vx_+1.00")
    m, _ = M.case_metrics(sc, _log(sc))
    assert m["ss_err_vx"]["mean"] == pytest.approx(0.0, abs=1e-6)
    assert m["survival_pct"] == 100.0


def test_steady_state_error_uses_last_5s_only():
    sc = _scenario("linear_tracking", "vx_+1.00")
    hold_end = int(round((S.SETTLE_S + S.HOLD_S) / DT))
    window_start = hold_end - int(round(S.SS_WINDOW_S / DT))

    def vel_fn(cmd, T, n):
        v = cmd.unsqueeze(1).repeat(1, n, 1)
        v[:window_start, :, 0] = 0.0  # big error before the window is ignored
        v[window_start:hold_end, :, 0] = 0.9  # 0.1 m/s error in the window
        return v

    m, _ = M.case_metrics(sc, _log(sc, vel_fn=vel_fn))
    assert m["ss_err_vx"]["mean"] == pytest.approx(0.1, abs=1e-5)


def test_fallen_envs_are_excluded_and_counted():
    sc = _scenario("linear_tracking", "vx_+1.00")
    m, _ = M.case_metrics(sc, _log(sc, fail_step=torch.tensor([1 << 30, 10, 1 << 30, 1 << 30])))
    assert m["n_fell"] == 1 and m["survival_pct"] == 75.0 and m["ss_err_vx"]["n"] == 3


def test_rise_time_and_overshoot_of_a_known_response():
    # first-order response with time constant tau: 10-90 % rise time = tau * ln(9)
    tau, L = 0.3, 350
    t = torch.arange(L).double() * DT
    y = (1.0 - torch.exp(-t / tau)).unsqueeze(1)
    res = M.rise_and_overshoot(y, 0.0, 1.0, DT)
    assert res["rise_time_s"][0].item() == pytest.approx(tau * math.log(9), abs=DT)
    assert res["overshoot_pct"][0].item() == pytest.approx(0.0, abs=1e-9)
    # 20 % overshoot
    y2 = torch.clamp(t / 0.5, max=1.2).unsqueeze(1)
    res2 = M.rise_and_overshoot(y2, 0.0, 1.0, DT)
    assert res2["overshoot_pct"][0].item() == pytest.approx(20.0, abs=1e-6)
    # downward step 1 -> 0 that undershoots to -0.1: overshoot 10 %
    y3 = (1.0 - torch.clamp(t / 0.5, max=1.1)).unsqueeze(1)
    assert M.rise_and_overshoot(y3, 1.0, 0.0, DT)["overshoot_pct"][0].item() == pytest.approx(10.0, abs=1e-6)


def test_stopping_distance():
    sc = _scenario("step_response", "vx_1_to_0")
    k0 = int(round((S.SETTLE_S + 6.0) / DT))

    def vel_fn(cmd, T, n):
        v = cmd.unsqueeze(1).repeat(1, n, 1)
        # decelerate linearly from 1 to 0 over 1 s after the step: distance 0.5 m
        ramp = torch.clamp(1.0 - torch.arange(T - k0).float() * DT, min=0.0)
        v[k0:, :, 0] = ramp.unsqueeze(1)
        return v

    m, _ = M.case_metrics(sc, _log(sc, vel_fn=vel_fn))
    assert m["stopping_distance_m"]["mean"] == pytest.approx(0.5, abs=0.03)


def test_action_hf_fraction():
    t = torch.arange(500).double() * DT
    low = torch.sin(2 * math.pi * 2.0 * t)  # 2 Hz
    high = torch.sin(2 * math.pi * 15.0 * t)  # 15 Hz, same power
    a = torch.stack([low, high, low + high], dim=-1).unsqueeze(1)  # (T, 1, 3)
    f = M.action_hf_fraction(a, DT)[0]
    assert f[0].item() == pytest.approx(0.0, abs=1e-3)
    assert f[1].item() == pytest.approx(1.0, abs=1e-3)
    assert f[2].item() == pytest.approx(0.5, abs=1e-2)


def test_vy_spawn_yaw_moves_along_plus_x():
    sc = _scenario("linear_tracking", "vy_-0.50")
    log = _log(sc)
    disp = log.pos[-1, 0] - log.pos[0, 0]
    assert disp[0].item() > 4.0 and abs(disp[1].item()) < 1e-3


def test_terrain_success_needs_progress():
    sc = _scenario("terrain", "stairs_15cm")
    hold_start = int(round(S.SETTLE_S / DT))

    def stuck(cmd, T, n):
        v = cmd.unsqueeze(1).repeat(1, n, 1)
        v[hold_start + 100 :, :2, 0] = 0.0  # first two envs stop after 2 m
        return v

    m, _ = M.case_metrics(sc, _log(sc, vel_fn=stuck))
    assert m["success_pct"] == 50.0


def test_pack_tests_respects_budget():
    chunks = S.pack_tests(512, 8192)
    assert sorted(t for c in chunks for t in c) == sorted(S.TESTS)
    for c in chunks:
        assert sum(len(S.build_scenarios([t])) for t in c) * 512 <= 8192


def _fake_result(scale=1.0, label="x"):
    """Minimal finalized result with every test present."""
    cases = {}
    for sc in S.build_scenarios():
        log = _log(sc, vel_fn=lambda cmd, T, n: cmd.unsqueeze(1).repeat(1, n, 1) * scale)
        m, _ = M.case_metrics(sc, log)
        cases.setdefault(sc.test, {})[sc.case] = m
    J = 16
    hw = {
        "peak_computed_Nm": [50.0] * J,
        "peak_applied_Nm": [50.0] * J,
        "sumsq_applied": [100.0] * J,
        "at_limit_samples": [0.0] * J,
        "samples": 100.0,
        "knee_peak_p99_Nm": 45.0,
        "action_hf_sum": [0.1] * J,
        "action_hf_n": 1,
    }
    joints = [f"fl_{n}_joint" for n in ("hipx", "hipy", "knee", "wheel")] * 4
    result = {
        "schema": R.SCHEMA,
        "meta": {"policy_path": f"/tmp/{label}/model_1.pt", "pass": "randomized", "joint_names": joints,
                 "action_names": [f"a{i}" for i in range(J)], "tests_run": list(S.TESTS)},
        "cases": cases,
        "hardware_raw": {t: dict(hw) for t in S.TESTS},
    }
    return R.finalize(result)


def test_finalize_and_verdict():
    vendor = _fake_result(1.0, "vendor")
    worse = _fake_result(0.8, "cand")  # tracks 80 % of every command
    assert vendor["scored"]["linear_tracking.vx_+1.00.ss_err_vx"]["value"] == pytest.approx(0.0, abs=1e-6)
    assert worse["scored"]["linear_tracking.vx_+1.00.ss_err_vx"]["value"] == pytest.approx(0.2, abs=1e-5)
    assert vendor["hardware"]["overall"]["knee_peak_pass"] is True
    v = C.verdict("cand", worse, vendor, ["linear_tracking.mean_ss_err_vx"], C.DEFAULT_REL_TOL)
    assert v.better is False and "linear_tracking.mean_ss_err_vx" in v.target_worse
    v2 = C.verdict("vendor", vendor, worse, ["linear_tracking.mean_ss_err_vx"], C.DEFAULT_REL_TOL)
    assert v2.target_improved == ["linear_tracking.mean_ss_err_vx"]
    table, _ = C.markdown_table([("cand", worse)], ("vendor", vendor), None, ["linear_tracking.*"])
    assert "NOT BETTER" in table


def _subset(full, tests):
    part = copy.deepcopy(full)
    part["cases"] = {t: c for t, c in part["cases"].items() if t in tests}
    part["hardware_raw"] = {t: h for t, h in part["hardware_raw"].items() if t in tests}
    part["meta"]["tests_run"] = list(tests)
    return R.finalize(part)


def test_merge_parts_equals_single_run(tmp_path):
    full = _fake_result(0.9)
    # robustness and its reference case (linear_tracking) in different parts
    a = _subset(full, ["linear_tracking", "yaw_tracking", "standstill"])
    b = _subset(full, ["robustness", "step_response", "terrain"])
    assert b["cases"]["robustness"]["push_0.5mps"]["tracking_err_increase"] is None
    merged = R.merge([a, b])
    assert merged["scored"].keys() == full["scored"].keys()
    for k, e in full["scored"].items():
        assert merged["scored"][k]["value"] == pytest.approx(e["value"])
    R.save(merged, tmp_path / "m.json")
    assert R.load(tmp_path / "m.json")["scored"].keys() == full["scored"].keys()


def test_knee_gate_uses_p99_not_max():
    r = _fake_result()
    for h in r["hardware_raw"].values():
        h["peak_computed_Nm"] = [120.0] * 16  # one-off spike far above the limit
        h["knee_peak_p99_Nm"] = 70.0
    R.finalize(r)
    hw = r["hardware"]["overall"]
    assert hw["knee_peak_Nm"] == 120.0 and hw["knee_peak_p99_Nm"] == 70.0 and hw["knee_peak_pass"] is True
    r["hardware_raw"]["terrain"]["knee_peak_p99_Nm"] = 80.0  # worst test decides
    R.finalize(r)
    assert r["hardware"]["overall"]["knee_peak_pass"] is False
    assert "hardware.knee_peak_p99_Nm" in r["scored"] and "hardware.knee_peak_Nm" not in r["scored"]
