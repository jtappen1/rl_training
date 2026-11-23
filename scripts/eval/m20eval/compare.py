# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause

"""Compare M20 evaluation results against the two baselines. Pure Python.

Rule (docs/plans/M20_Evaluation_Suite.md): a checkpoint is better only if it improves the target metric
without making any other metric worse than the vendor retrain. A difference counts only when it exceeds
max(rel_tol * |vendor value|, the metric's absolute tolerance); smaller differences are treated as noise.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fnmatch import fnmatch

DEFAULT_REL_TOL = 0.05


@dataclass
class Verdict:
    label: str
    better: bool | None  # None when no target was given
    target_keys: list[str] = field(default_factory=list)
    target_improved: list[str] = field(default_factory=list)
    target_worse: list[str] = field(default_factory=list)
    worse: list[str] = field(default_factory=list)  # non-target metrics worse than the vendor retrain
    improved: list[str] = field(default_factory=list)  # non-target metrics better than the vendor retrain
    missing: list[str] = field(default_factory=list)  # in the vendor result but not in the candidate

    def as_dict(self) -> dict:
        return self.__dict__.copy()


def status(entry: dict, base: dict | None, rel_tol: float) -> tuple[str, float | None]:
    """'better' | 'worse' | 'same' | 'n/a' of ``entry`` relative to ``base``, and the signed delta."""
    if base is None or entry is None:
        return "n/a", None
    delta = entry["value"] - base["value"]
    tol = max(rel_tol * abs(base["value"]), entry["abs_tol"])
    if abs(delta) <= tol:
        return "same", delta
    improved = delta < 0 if entry["better"] == "lower" else delta > 0
    return ("better" if improved else "worse"), delta


def verdict(label: str, cand: dict, vendor: dict, targets: list[str], rel_tol: float) -> Verdict:
    cs, vs = cand["scored"], vendor["scored"]
    target_keys = sorted(k for k in cs if any(fnmatch(k, t) for t in targets))
    v = Verdict(label=label, better=None, target_keys=target_keys)
    v.missing = sorted(k for k in vs if k not in cs)
    for k, e in cs.items():
        st, _ = status(e, vs.get(k), rel_tol)
        if k in target_keys:
            if st == "better":
                v.target_improved.append(k)
            elif st == "worse":
                v.target_worse.append(k)
        elif st == "worse":
            v.worse.append(k)
        elif st == "better":
            v.improved.append(k)
    if targets:
        v.better = bool(v.target_improved) and not v.target_worse and not v.worse and not v.missing
    return v


def _fmt(x: float | None) -> str:
    if x is None:
        return "-"
    a = abs(x)
    if a >= 100:
        return f"{x:.1f}"
    if a >= 10:
        return f"{x:.2f}"
    return f"{x:.3f}"


def markdown_table(
    candidates: list[tuple[str, dict]],
    vendor: tuple[str, dict] | None,
    deployed: tuple[str, dict] | None,
    targets: list[str],
    rel_tol: float = DEFAULT_REL_TOL,
) -> tuple[str, list[Verdict]]:
    """Comparison table of every scored metric plus one verdict per candidate."""
    columns: list[tuple[str, dict]] = []
    if deployed:
        columns.append(deployed)
    if vendor:
        columns.append(vendor)
    columns.extend(candidates)

    keys: list[str] = []
    for _, r in columns:
        for k in r["scored"]:
            if k not in keys:
                keys.append(k)

    def unit_of(k):
        for _, r in columns:
            if k in r["scored"]:
                return r["scored"][k]["unit"], r["scored"][k]["better"]
        return "", ""

    header = ["metric", "unit", "better"] + [f"{lbl}" for lbl, _ in columns]
    if vendor:
        header += [f"Δ {lbl} vs vendor" for lbl, _ in candidates]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    vs = vendor[1]["scored"] if vendor else {}
    for k in keys:
        unit, better = unit_of(k)
        name = f"**{k}**" if any(fnmatch(k, t) for t in targets) else k
        row = [name, unit, better]
        for _, r in columns:
            e = r["scored"].get(k)
            row.append(_fmt(e["value"]) if e else "-")
        if vendor:
            for _, r in candidates:
                st, d = status(r["scored"].get(k), vs.get(k), rel_tol)
                mark = {"better": " ✓", "worse": " ✗ worse", "same": "", "n/a": ""}[st]
                row.append(f"{'+' if d is not None and d >= 0 else ''}{_fmt(d)}{mark}" if d is not None else "-")
        lines.append("| " + " | ".join(row) + " |")

    # pass/fail gates from the plan
    gate_lines = ["", "| gate | " + " | ".join(lbl for lbl, _ in columns) + " |", "|---|" + "---|" * len(columns)]
    gates = [
        ("knee peak p99 < 76.4 N·m", lambda r: ((r.get("hardware") or {}).get("overall") or {}).get("knee_peak_pass")),
        (
            "standstill max wheel target < 0.5 rad/s",
            lambda r: next(iter(r["cases"].get("standstill", {}).values()), {}).get("wheel_target_pass"),
        ),
    ]
    for name, fn in gates:
        vals = []
        for _, r in columns:
            g = fn(r)
            vals.append("-" if g is None else ("pass" if g else "FAIL"))
        gate_lines.append(f"| {name} | " + " | ".join(vals) + " |")
    lines += gate_lines

    verdicts = []
    if vendor:
        lines += ["", f"Verdicts vs vendor retrain `{vendor[0]}` (tolerance: max({rel_tol:.0%} of vendor value, per-unit absolute)):"]
        for lbl, r in candidates:
            v = verdict(lbl, r, vendor[1], targets, rel_tol)
            verdicts.append(v)
            if v.better is None:
                head = "no target metric given"
            else:
                head = "BETTER" if v.better else "NOT BETTER"
            lines.append(f"- **{lbl}: {head}**")
            if targets:
                lines.append(f"  - target {targets}: improved {v.target_improved or 'none'}; worse {v.target_worse or 'none'}")
            lines.append(f"  - other metrics worse than vendor ({len(v.worse)}): {', '.join(v.worse) or 'none'}")
            lines.append(f"  - other metrics better than vendor ({len(v.improved)}): {', '.join(v.improved) or 'none'}")
            if v.missing:
                lines.append(f"  - missing metrics (not evaluated): {', '.join(v.missing)}")
    return "\n".join(lines), verdicts
