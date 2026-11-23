# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause

"""Per-scenario metrics of the M20 evaluation suite. Torch only, no Isaac Sim import.

Logged signals are indexed by policy step ``k`` (all scenarios share one clock that starts at the
beginning of the settle phase). ``vel[k]`` is the state after the action taken under ``cmd[k]``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from . import scenarios as S

FAIL_NONE, FAIL_FELL, FAIL_OOB = 0, 1, 2


@dataclass
class ScenarioLog:
    """Signals of one scenario's env block, moved to CPU."""

    vel: torch.Tensor  # (T, n, 3) base-frame vx, vy and yaw rate
    pos: torch.Tensor  # (T, n, 2) world xy
    heading: torch.Tensor  # (T, n) world yaw
    actions: torch.Tensor  # (T, n, A) raw policy actions
    cmd: torch.Tensor  # (T, 3) commanded vx, vy, wz
    fail_step: torch.Tensor  # (n,) first policy step with a termination, or a large number
    fail_kind: torch.Tensor  # (n,) FAIL_NONE / FAIL_FELL / FAIL_OOB
    wheel_target_max: torch.Tensor  # (n,) max |wheel velocity target| after settle (rad/s)
    dt: float


def stats(x: torch.Tensor) -> dict | None:
    """Summary of a 1-D sample, NaNs ignored. None when empty."""
    x = x.flatten().double()
    x = x[~torch.isnan(x)]
    if x.numel() == 0:
        return None
    return {
        "mean": x.mean().item(),
        "std": x.std(unbiased=False).item(),
        "median": x.median().item(),
        "p95": torch.quantile(x, 0.95).item(),
        "min": x.min().item(),
        "max": x.max().item(),
        "n": int(x.numel()),
    }


def _steps(seconds: float, dt: float) -> int:
    return int(round(seconds / dt))


def moving_average(x: torch.Tensor, width: int) -> torch.Tensor:
    """Centred moving average along dim 0 with edge replication. x: (T, n)."""
    if width <= 1:
        return x
    left = (width - 1) // 2
    right = width - 1 - left
    xp = torch.cat([x[:1].expand(left, -1), x, x[-1:].expand(right, -1)], dim=0)
    c = torch.cumsum(torch.cat([torch.zeros_like(xp[:1]), xp], dim=0), dim=0)
    return (c[width:] - c[:-width]) / width


def rise_and_overshoot(signal: torch.Tensor, start: float, target: float, dt: float) -> dict[str, torch.Tensor]:
    """10-90 % rise time and overshoot of a step from ``start`` to ``target``.

    Args:
        signal: (L, n) response starting at the step instant (already smoothed).
    Returns:
        rise_time_s (n,) NaN when 90 % is never reached, overshoot_pct (n,), reached (n,) bool.
    """
    r = (signal - start) / (target - start)
    L = r.shape[0]
    idx = torch.arange(L).unsqueeze(1).expand_as(r)
    big = torch.full_like(idx, L)
    i10 = torch.where(r >= 0.1, idx, big).min(dim=0).values
    # the 90 % crossing is searched at or after the 10 % crossing
    i90 = torch.where((r >= 0.9) & (idx >= i10.unsqueeze(0)), idx, big).min(dim=0).values
    reached = i90 < L
    rise = (i90 - i10).double() * dt
    rise[~reached] = float("nan")
    overshoot = (r.max(dim=0).values - 1.0).clamp(min=0.0).double() * 100.0
    return {"rise_time_s": rise, "overshoot_pct": overshoot, "reached": reached}


def action_hf_fraction(actions: torch.Tensor, dt: float, cutoff_hz: float = S.ACTION_HF_CUTOFF_HZ) -> torch.Tensor:
    """Fraction of (mean-removed) action power above ``cutoff_hz``. actions: (L, n, A) -> (n, A)."""
    x = actions.double()
    x = x - x.mean(dim=0, keepdim=True)
    power = torch.fft.rfft(x, dim=0).abs() ** 2
    freqs = torch.fft.rfftfreq(x.shape[0], d=dt, dtype=torch.float64)
    total = power[freqs > 0].sum(dim=0)
    high = power[freqs > cutoff_hz].sum(dim=0)
    return torch.where(total > 0, high / total.clamp(min=1e-30), torch.zeros_like(total))


def case_metrics(sc: S.Scenario, log: ScenarioLog) -> tuple[dict, dict]:
    """Metrics of one scenario.

    Returns:
        (metrics, hf) where hf holds {"sum": (A,), "n": int} of per-env high-frequency action power
        fractions, for aggregation into the hardware metrics.
    """
    dt = log.dt
    T = _steps(sc.duration_s, dt)
    settle = _steps(S.SETTLE_S, dt)
    n = log.fail_step.numel()

    survived = log.fail_step >= T
    oob = (~survived) & (log.fail_kind == FAIL_OOB)
    fell = (~survived) & ~oob
    counted = ~oob  # out-of-bounds envs are excluded (a layout problem, not the policy's)
    valid = survived  # envs whose full time series is usable

    out: dict = {
        "n_envs": n,
        "n_survived": int(survived.sum()),
        "n_fell": int(fell.sum()),
        "n_out_of_bounds": int(oob.sum()),
        "survival_pct": 100.0 * survived[counted].double().mean().item() if counted.any() else float("nan"),
        "command_segments": [{"duration_s": d, "cmd": list(c)} for d, c in sc.segments],
        "terrain": sc.terrain,
        "perturbation": sc.perturbation or None,
    }

    vel = log.vel[:, valid]
    pos = log.pos[:, valid]
    heading = log.heading[:, valid]
    cmd = log.cmd.unsqueeze(1)  # (T, 1, 3)

    if sc.kind in ("hold", "standstill"):
        hold_s = sc.segments[0][0]
        h0, h1 = settle, settle + _steps(hold_s, dt)
        w0 = h1 - _steps(S.SS_WINDOW_S, dt)
        err = (vel[w0:h1] - cmd[w0:h1]).abs().mean(dim=0)  # (n_valid, 3)
        out["ss_err_vx"] = stats(err[:, 0])
        out["ss_err_vy"] = stats(err[:, 1])
        out["ss_err_wz"] = stats(err[:, 2])
        err_xy = torch.linalg.norm(vel[h0:h1, :, :2] - cmd[h0:h1, :, :2], dim=-1).mean(dim=0)
        out["hold_err_xy"] = stats(err_xy)  # full-hold mean |v_xy - cmd_xy|, used by the robustness test

        disp = pos[h1 - 1] - pos[h0 - 1]
        if sc.kind == "standstill":
            out["drift_m"] = stats(torch.linalg.norm(disp, dim=-1))
            wheel = log.wheel_target_max[valid]
            out["max_wheel_target_radps"] = stats(wheel)
            out["wheel_target_pass_pct"] = (
                100.0 * (wheel < S.STANDSTILL_WHEEL_LIMIT).double().mean().item() if wheel.numel() else float("nan")
            )
            out["wheel_target_pass"] = bool(wheel.numel() and wheel.max().item() < S.STANDSTILL_WHEEL_LIMIT)

        if sc.test == "terrain":
            vx, vy, _ = sc.segments[0][1]
            yaw0 = heading[h0 - 1]
            # commanded direction in the world frame at the start of the hold
            dir_w = torch.stack(
                [vx * torch.cos(yaw0) - vy * torch.sin(yaw0), vx * torch.sin(yaw0) + vy * torch.cos(yaw0)], dim=-1
            )
            speed = math.hypot(vx, vy)
            progress = (disp * dir_w).sum(dim=-1) / max(speed, 1e-9)
            commanded = speed * hold_s
            out["progress_m"] = stats(progress)
            success_valid = progress >= S.TERRAIN_MIN_PROGRESS * commanded
            n_success = int(success_valid.sum())
            n_counted = int(counted.sum())
            out["success_pct"] = 100.0 * n_success / n_counted if n_counted else float("nan")

    elif sc.kind == "step":
        k0 = settle + _steps(sc.segments[0][0], dt)
        k1 = k0 + _steps(sc.segments[1][0], dt)
        axis = {"vx": 0, "vy": 1, "wz": 2}[sc.step_signal]
        a, b = sc.segments[0][1][axis], sc.segments[1][1][axis]
        width = max(1, _steps(S.RISE_SMOOTHING_S, dt))
        smoothed = moving_average(vel[:, :, axis], width)[k0:k1]
        res = rise_and_overshoot(smoothed, a, b, dt)
        out["rise_time_s"] = stats(res["rise_time_s"])
        out["overshoot_pct"] = stats(res["overshoot_pct"])
        out["reached_90pct_pct"] = 100.0 * res["reached"].double().mean().item() if res["reached"].numel() else float("nan")
        if sc.step_signal in ("vx", "vy") and b == 0.0:
            yaw0 = heading[k0 - 1]
            disp = pos[k1 - 1] - pos[k0 - 1]
            along = disp[:, 0] * torch.cos(yaw0) + disp[:, 1] * torch.sin(yaw0)
            out["stopping_distance_m"] = stats(along)

    hf = {"sum": None, "n": 0}
    if valid.any():
        frac = action_hf_fraction(log.actions[settle:T, valid].float(), dt)  # (n_valid, A)
        hf = {"sum": frac.sum(dim=0), "n": int(frac.shape[0])}
    return out, hf


class HardwareAccumulator:
    """Per-env torque statistics accumulated at physics rate (works on the simulation device)."""

    def __init__(self, num_envs: int, num_joints: int, effort_limit: torch.Tensor, device):
        self.effort_limit = effort_limit  # (num_envs, J)
        z = lambda: torch.zeros(num_envs, num_joints, device=device)  # noqa: E731
        self.peak_computed = z()
        self.peak_applied = z()
        self.sumsq_applied = z()
        self.at_limit = z()
        self.samples = torch.zeros(num_envs, device=device)
        self.recording = torch.zeros(num_envs, dtype=torch.bool, device=device)

    def update(self, computed: torch.Tensor, applied: torch.Tensor):
        m = self.recording.unsqueeze(1)
        ca, aa = computed.abs(), applied.abs()
        self.peak_computed = torch.where(m, torch.maximum(self.peak_computed, ca), self.peak_computed)
        self.peak_applied = torch.where(m, torch.maximum(self.peak_applied, aa), self.peak_applied)
        self.sumsq_applied += torch.where(m, aa * aa, torch.zeros_like(aa))
        self.at_limit += (m & (ca >= self.effort_limit)).float()
        self.samples += self.recording.float()

    def raw(
        self, env_ids: torch.Tensor, knee_ids: list[int], hf_sum: torch.Tensor | None, hf_n: int, num_actions: int
    ) -> dict:
        """Mergeable totals for a group of envs (one test)."""
        knee_peak_per_env = self.peak_computed[env_ids][:, knee_ids].max(dim=1).values.double().cpu()
        return {
            "peak_computed_Nm": self.peak_computed[env_ids].max(dim=0).values.tolist(),
            "peak_applied_Nm": self.peak_applied[env_ids].max(dim=0).values.tolist(),
            "sumsq_applied": self.sumsq_applied[env_ids].double().sum(dim=0).tolist(),
            "at_limit_samples": self.at_limit[env_ids].double().sum(dim=0).tolist(),
            "samples": float(self.samples[env_ids].double().sum().item()),
            "knee_peak_p99_Nm": torch.quantile(knee_peak_per_env, 0.99).item() if knee_peak_per_env.numel() else None,
            "action_hf_sum": hf_sum.tolist() if hf_sum is not None else [0.0] * num_actions,
            "action_hf_n": hf_n,
        }
