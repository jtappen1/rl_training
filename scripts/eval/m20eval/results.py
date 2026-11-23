# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause

"""Merge and finalize M20 evaluation results. Pure Python.

A result file holds the raw per-scenario metrics written by ``m20_eval.py`` ("cases", "hardware_raw")
and the sections derived from them by :func:`finalize` ("summary", "hardware", "scored"). Finalizing is
deterministic, so results from several simulation chunks can be merged and finalized again.
"""

from __future__ import annotations

import copy
import json
import math
import re
from pathlib import Path

from . import scenarios as S

SCHEMA = "m20_eval/v1"

JOINT_GROUPS = {"hipx": r"hipx", "hipy": r"hipy", "knee": r"knee", "wheel": r"wheel"}

# unit -> absolute tolerance used when comparing to a baseline (differences below it are noise)
ABS_TOL = {
    "m/s": 0.01,
    "rad/s": 0.01,
    "s": 0.02,
    "%": 1.0,
    "m": 0.02,
    "N*m": 1.0,
}


def load(path: str | Path) -> dict:
    with open(path) as f:
        return json.load(f)


def save(result: dict, path: str | Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(_clean(result), f, indent=1, allow_nan=False)


def _clean(obj):
    """Replace NaN/inf with None so the files are strict JSON."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    return obj


def merge(parts: list[dict]) -> dict:
    """Merge raw results of the same policy and pass that cover different tests."""
    if not parts:
        raise ValueError("nothing to merge")
    out = copy.deepcopy(parts[0])
    for p in parts[1:]:
        for key in ("policy_path", "pass"):
            if p["meta"].get(key) != out["meta"].get(key):
                raise ValueError(f"cannot merge results with different {key}: {p['meta'].get(key)} vs {out['meta'].get(key)}")
        for test, cases in p["cases"].items():
            if test in out["cases"]:
                raise ValueError(f"test {test} present in more than one part")
            out["cases"][test] = cases
        out["hardware_raw"].update(p["hardware_raw"])
        out["meta"]["tests_run"] = [t for t in S.TESTS if t in out["meta"]["tests_run"] or t in p["meta"]["tests_run"]]
        out["meta"].setdefault("chunks", []).extend(p["meta"].get("chunks", []))
        out["meta"]["wall_time_s"] = out["meta"].get("wall_time_s", 0.0) + p["meta"].get("wall_time_s", 0.0)
    return finalize(out)


def _mean(values: list[float | None]) -> float | None:
    v = [x for x in values if x is not None and math.isfinite(x)]
    return sum(v) / len(v) if v else None


def _get(d: dict | None, *path, default=None):
    for p in path:
        if not isinstance(d, dict) or p not in d or d[p] is None:
            return default
        d = d[p]
    return d


def _hardware(raws: list[dict], meta: dict) -> dict:
    joints = meta["joint_names"]
    actions = meta["action_names"]
    J, A = len(joints), len(actions)
    peak_c = [max(r["peak_computed_Nm"][j] for r in raws) for j in range(J)]
    peak_a = [max(r["peak_applied_Nm"][j] for r in raws) for j in range(J)]
    samples = sum(r["samples"] for r in raws)
    sumsq = [sum(r["sumsq_applied"][j] for r in raws) for j in range(J)]
    at_lim = [sum(r["at_limit_samples"][j] for r in raws) for j in range(J)]
    hf_n = sum(r["action_hf_n"] for r in raws)
    hf_sum = [sum(r["action_hf_sum"][a] for r in raws) for a in range(A)]

    rms = [math.sqrt(s / samples) if samples else None for s in sumsq]
    pct_lim = [100.0 * a / samples if samples else None for a in at_lim]
    hf_pct = [100.0 * s / hf_n if hf_n else None for s in hf_sum]
    knee_ids = [i for i, n in enumerate(joints) if "knee" in n]
    knee_peak = max(peak_c[i] for i in knee_ids) if knee_ids else None
    p99s = [r["knee_peak_p99_Nm"] for r in raws if r.get("knee_peak_p99_Nm") is not None]
    knee_p99 = max(p99s) if p99s else None

    groups = {}
    for g, pattern in JOINT_GROUPS.items():
        ids = [i for i, n in enumerate(joints) if re.search(pattern, n)]
        if ids:
            groups[g] = {
                "peak_torque_Nm": max(peak_c[i] for i in ids),
                "rms_torque_Nm": max(rms[i] for i in ids if rms[i] is not None) if samples else None,
                "pct_time_at_limit": max(pct_lim[i] for i in ids if pct_lim[i] is not None) if samples else None,
            }
    out = {
        "per_joint": {
            n: {
                "peak_torque_Nm": peak_c[j],
                "peak_applied_torque_Nm": peak_a[j],
                "rms_torque_Nm": rms[j],
                "pct_time_at_limit": pct_lim[j],
            }
            for j, n in enumerate(joints)
        },
        "per_group": groups,
        "knee_peak_Nm": knee_peak,
        # Gate on the 99th percentile of per-env knee peaks (worst test), not the single worst sample:
        # the maximum is dominated by one-off torque spikes from simulated velocity jumps (pushes, impacts).
        "knee_peak_p99_Nm": knee_p99,
        "knee_peak_pass": knee_p99 is not None and knee_p99 < S.KNEE_TORQUE_LIMIT,
        "pct_time_at_limit": max((p for p in pct_lim if p is not None), default=None),
        "action_hf_power_pct": _mean(hf_pct),
        "action_hf_power_pct_per_action": dict(zip(actions, hf_pct)),
        "physics_samples": samples,
    }
    return out


def finalize(result: dict) -> dict:
    """Recompute summary, hardware and scored sections from the raw sections, in place; returns result."""
    cases = result["cases"]
    meta = result["meta"]

    # -- robustness: tracking error increase vs the unperturbed 1 m/s case of the same pass
    ref_test, ref_case = S.ROBUSTNESS_REFERENCE
    ref_err = _get(cases, ref_test, ref_case, "hold_err_xy", "mean")
    for c in cases.get("robustness", {}).values():
        err = _get(c, "hold_err_xy", "mean")
        c["tracking_err_increase"] = err - ref_err if err is not None and ref_err is not None else None
        c["tracking_err_reference"] = f"{ref_test}/{ref_case}"

    # -- per-test summaries
    summary = {}
    for test, tc in cases.items():
        survived = sum(c["n_survived"] for c in tc.values())
        counted = sum(c["n_envs"] - c["n_out_of_bounds"] for c in tc.values())
        s = {"survival_pct": 100.0 * survived / counted if counted else None, "n_cases": len(tc)}
        if test == "linear_tracking":
            s["mean_ss_err_vx"] = _mean([_get(c, "ss_err_vx", "mean") for k, c in tc.items() if k.startswith("vx")])
            s["mean_ss_err_vy"] = _mean([_get(c, "ss_err_vy", "mean") for k, c in tc.items() if k.startswith("vy")])
        elif test == "yaw_tracking":
            s["mean_ss_err_wz"] = _mean([_get(c, "ss_err_wz", "mean") for c in tc.values()])
        elif test == "terrain":
            s["mean_success_pct"] = _mean([c.get("success_pct") for c in tc.values()])
        elif test == "robustness":
            s["mean_tracking_err_increase"] = _mean([c.get("tracking_err_increase") for c in tc.values()])
        summary[test] = s
    result["summary"] = summary

    # -- hardware, per test and over all tests
    hw_raw = result["hardware_raw"]
    result["hardware"] = {
        "overall": _hardware(list(hw_raw.values()), meta) if hw_raw else None,
        "per_test": {t: _hardware([r], meta) for t, r in hw_raw.items()},
    }

    result["scored"] = scored_metrics(result)
    return result


def scored_metrics(result: dict) -> dict:
    """Flat {key: {value, better, unit, abs_tol}} used by comparisons."""
    cases, summary = result["cases"], result.get("summary", {})
    out: dict = {}

    def add(key, value, better, unit):
        if value is None or (isinstance(value, float) and not math.isfinite(value)):
            return
        out[key] = {"value": value, "better": better, "unit": unit, "abs_tol": ABS_TOL[unit]}

    for test, s in summary.items():
        add(f"{test}.survival_pct", s.get("survival_pct"), "higher", "%")

    for case, c in cases.get("linear_tracking", {}).items():
        add(f"linear_tracking.{case}.ss_err_vx", _get(c, "ss_err_vx", "mean"), "lower", "m/s")
        add(f"linear_tracking.{case}.ss_err_vy", _get(c, "ss_err_vy", "mean"), "lower", "m/s")
    if "linear_tracking" in summary:
        add("linear_tracking.mean_ss_err_vx", summary["linear_tracking"].get("mean_ss_err_vx"), "lower", "m/s")
        add("linear_tracking.mean_ss_err_vy", summary["linear_tracking"].get("mean_ss_err_vy"), "lower", "m/s")

    for case, c in cases.get("yaw_tracking", {}).items():
        add(f"yaw_tracking.{case}.ss_err_wz", _get(c, "ss_err_wz", "mean"), "lower", "rad/s")
        if case.startswith("arc"):
            add(f"yaw_tracking.{case}.ss_err_vx", _get(c, "ss_err_vx", "mean"), "lower", "m/s")
    if "yaw_tracking" in summary:
        add("yaw_tracking.mean_ss_err_wz", summary["yaw_tracking"].get("mean_ss_err_wz"), "lower", "rad/s")

    for case, c in cases.get("step_response", {}).items():
        add(f"step_response.{case}.rise_time_s", _get(c, "rise_time_s", "median"), "lower", "s")
        add(f"step_response.{case}.overshoot_pct", _get(c, "overshoot_pct", "mean"), "lower", "%")
        add(f"step_response.{case}.reached_90pct_pct", c.get("reached_90pct_pct"), "higher", "%")
        add(f"step_response.{case}.stopping_distance_m", _get(c, "stopping_distance_m", "mean"), "lower", "m")

    for case, c in cases.get("standstill", {}).items():
        add("standstill.drift_m", _get(c, "drift_m", "mean"), "lower", "m")
        add("standstill.max_wheel_target_radps", _get(c, "max_wheel_target_radps", "max"), "lower", "rad/s")
        add("standstill.wheel_target_pass_pct", c.get("wheel_target_pass_pct"), "higher", "%")

    for case, c in cases.get("robustness", {}).items():
        add(f"robustness.{case}.survival_pct", c.get("survival_pct"), "higher", "%")
        add(f"robustness.{case}.tracking_err_increase", c.get("tracking_err_increase"), "lower", "m/s")

    for case, c in cases.get("terrain", {}).items():
        add(f"terrain.{case}.success_pct", c.get("success_pct"), "higher", "%")
        add(f"terrain.{case}.ss_err_vx", _get(c, "ss_err_vx", "mean"), "lower", "m/s")

    hw = _get(result, "hardware", "overall")
    if hw:
        add("hardware.knee_peak_p99_Nm", hw.get("knee_peak_p99_Nm"), "lower", "N*m")
        add("hardware.pct_time_at_limit", hw.get("pct_time_at_limit"), "lower", "%")
        add("hardware.action_hf_power_pct", hw.get("action_hf_power_pct"), "lower", "%")
        for g, gv in hw.get("per_group", {}).items():
            add(f"hardware.{g}.peak_torque_Nm", gv.get("peak_torque_Nm"), "lower", "N*m")
            add(f"hardware.{g}.rms_torque_Nm", gv.get("rms_torque_Nm"), "lower", "N*m")
    return out
