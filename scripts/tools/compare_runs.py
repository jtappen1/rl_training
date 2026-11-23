# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause

"""Compare agent.yaml and env.yaml between two training runs.

train.py dumps these files to <log_dir>/params/ at the start of each run.

Usage:
    python scripts/tools/compare_runs.py <path_run1> <path_run2>

Each path should be a run directory (e.g. logs/rsl_rl/deeprobotics_lite3_rough/2025-01-01_12-00-00)
containing params/agent.yaml and params/env.yaml, or the yaml files directly.

If both runs have M20 evaluation results (<run>/eval/*.json, written by scripts/eval/m20_eval.py), the
metric differences are printed after the config differences, per pass (latest result file of each run).
"""

import argparse
import json
import sys
from pathlib import Path

import yaml


# ---------------------------------------------------------------------------
# Custom YAML loader that converts Isaac Lab Python-tagged values to plain
# Python objects so they can be diffed without importing Isaac Lab.
# ---------------------------------------------------------------------------

class _IslabLoader(yaml.SafeLoader):
    pass


def _construct_python_tuple(loader, node):
    return tuple(loader.construct_sequence(node))


def _construct_python_object_apply(loader, tag_suffix, node):
    args = loader.construct_sequence(node)
    return f"{tag_suffix}({', '.join(str(a) for a in args)})"


_IslabLoader.add_constructor(
    "tag:yaml.org,2002:python/tuple", _construct_python_tuple
)
_IslabLoader.add_multi_constructor(
    "tag:yaml.org,2002:python/object/apply:", _construct_python_object_apply
)
_IslabLoader.add_multi_constructor(
    "tag:yaml.org,2002:python/object/new:", _construct_python_object_apply
)


def _load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.load(f, Loader=_IslabLoader) or {}


# ---------------------------------------------------------------------------
# Flatten nested dict/list to dot-notation keys
# ---------------------------------------------------------------------------

def _flatten(d, prefix="") -> dict:
    items = {}
    if isinstance(d, dict):
        for k, v in d.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            items.update(_flatten(v, key))
    elif isinstance(d, (list, tuple)):
        for i, v in enumerate(d):
            items.update(_flatten(v, f"{prefix}[{i}]"))
    else:
        items[prefix] = d
    return items


# ---------------------------------------------------------------------------
# Pretty diff
# ---------------------------------------------------------------------------

def _print_diff(flat1: dict, flat2: dict, name1: str, name2: str) -> None:
    keys1, keys2 = set(flat1), set(flat2)
    only1   = sorted(keys1 - keys2)
    only2   = sorted(keys2 - keys1)
    changed = [(k, flat1[k], flat2[k]) for k in sorted(keys1 & keys2) if flat1[k] != flat2[k]]

    if not (only1 or only2 or changed):
        print("  [identical]")
        return

    col_w = max(len(name1), len(name2), 6)

    if only1:
        print(f"\n  Keys only in {name1}:")
        for k in only1:
            print(f"    {k} = {flat1[k]!r}")

    if only2:
        print(f"\n  Keys only in {name2}:")
        for k in only2:
            print(f"    {k} = {flat2[k]!r}")

    if changed:
        key_w = max(len(k) for k, *_ in changed)
        print(f"\n  Changed values:")
        print(f"    {'key':<{key_w}}  {name1:<{col_w}}  {name2:<{col_w}}")
        print(f"    {'-'*key_w}  {'-'*col_w}  {'-'*col_w}")
        for k, v1, v2 in changed:
            print(f"    {k:<{key_w}}  {str(v1):<{col_w}}  {str(v2):<{col_w}}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _find_yaml(run_dir: Path, filename: str) -> Path:
    """Search for filename directly in run_dir or in run_dir/params/."""
    for candidate in [run_dir / filename, run_dir / "params" / filename]:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Could not find {filename} in {run_dir} or {run_dir / 'params'}"
    )


def _latest_eval(run_dir: Path) -> dict[str, Path]:
    """Newest result file per pass in <run>/eval (comparison tables and chunk parts excluded)."""
    found: dict[str, Path] = {}
    for f in sorted((run_dir / "eval").glob("*.json"), key=lambda f: f.stat().st_mtime):
        if ".part" in f.name:
            continue
        try:
            with open(f) as fh:
                ps = json.load(fh)["meta"]["pass"]
        except (OSError, KeyError, ValueError):
            continue
        found[ps] = f
    return found


def _print_eval_diff(run1: Path, run2: Path, name1: str, name2: str) -> None:
    ev1, ev2 = _latest_eval(run1), _latest_eval(run2)
    passes = [p for p in ("randomized", "nominal") if p in ev1 and p in ev2]
    if not passes:
        return
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))
    from m20eval import compare as C
    from m20eval import results as R

    for ps in passes:
        print(f"\n{'='*70}")
        print(f" eval metrics, {ps} pass: {ev1[ps].name} vs {ev2[ps].name}")
        print(f"{'='*70}")
        s1, s2 = R.load(ev1[ps])["scored"], R.load(ev2[ps])["scored"]
        rows = []
        for k in sorted(set(s1) | set(s2)):
            st, d = C.status(s2.get(k), s1.get(k), C.DEFAULT_REL_TOL)
            if st in ("better", "worse") or k not in s1 or k not in s2:
                v1 = f"{s1[k]['value']:.4g}" if k in s1 else "-"
                v2 = f"{s2[k]['value']:.4g}" if k in s2 else "-"
                rows.append((k, v1, v2, st if k in s1 and k in s2 else "missing"))
        if not rows:
            print("  [no differences beyond tolerance]")
            continue
        key_w = max(len(r[0]) for r in rows)
        col_w = max(len(name1), len(name2), 8)
        print(f"    {'metric':<{key_w}}  {name1:<{col_w}}  {name2:<{col_w}}  {name2} vs {name1}")
        print(f"    {'-'*key_w}  {'-'*col_w}  {'-'*col_w}  {'-'*10}")
        for k, v1, v2, st in rows:
            print(f"    {k:<{key_w}}  {v1:<{col_w}}  {v2:<{col_w}}  {st}")


def main():
    parser = argparse.ArgumentParser(description="Compare two training run configs.")
    parser.add_argument("run1", help="Path to first run directory")
    parser.add_argument("run2", help="Path to second run directory")
    args = parser.parse_args()

    run1, run2 = Path(args.run1), Path(args.run2)
    name1, name2 = run1.name, run2.name

    for yaml_name in ("agent.yaml", "env.yaml"):
        print(f"\n{'='*70}")
        print(f" {yaml_name}")
        print(f"{'='*70}")

        try:
            path1 = _find_yaml(run1, yaml_name)
            path2 = _find_yaml(run2, yaml_name)
        except FileNotFoundError as e:
            print(f"  ERROR: {e}")
            continue

        flat1 = _flatten(_load_yaml(path1))
        flat2 = _flatten(_load_yaml(path2))
        _print_diff(flat1, flat2, name1, name2)

    _print_eval_diff(run1, run2, name1, name2)
    print()


if __name__ == "__main__":
    main()
