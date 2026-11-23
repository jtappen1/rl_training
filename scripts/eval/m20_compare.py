# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause

"""Build the M20 comparison table from existing result files (no simulation).

Usage:
    python scripts/eval/m20_compare.py --candidates <run>/eval/model_10000_randomized.json \
        --vendor <vendor run>/eval/model_10000_randomized.json \
        --deployed <dir>/eval/m20_policy_randomized.json --target "terrain.*.success_pct" [--output table.md]

Several part files of the same policy and pass can be joined with "+": a.part0.json+a.part1.json.
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from m20eval import compare as C  # noqa: E402
from m20eval import results as R  # noqa: E402


def load(spec: str) -> tuple[str, dict]:
    paths = spec.split("+")
    result = R.merge([R.load(p) for p in paths]) if len(paths) > 1 else R.finalize(R.load(paths[0]))
    policy = Path(result["meta"]["policy_path"])
    label = f"{policy.parent.name}/{policy.stem}" if policy.suffix == ".pt" else policy.name
    return label, result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--candidates", nargs="+", required=True)
    parser.add_argument("--vendor", default=None)
    parser.add_argument("--deployed", default=None)
    parser.add_argument("--target", nargs="*", default=[])
    parser.add_argument("--rel_tol", type=float, default=C.DEFAULT_REL_TOL)
    parser.add_argument("--output", default=None, help="Write the table to this Markdown file.")
    args = parser.parse_args()

    cands = [load(c) for c in args.candidates]
    vendor = load(args.vendor) if args.vendor else None
    deployed = load(args.deployed) if args.deployed else None
    passes = {r["meta"]["pass"] for _, r in cands + [x for x in (vendor, deployed) if x]}
    if len(passes) > 1:
        print(f"[WARN] comparing results from different passes: {sorted(passes)}")
    table, _ = C.markdown_table(cands, vendor, deployed, args.target, args.rel_tol)
    print(table)
    if args.output:
        Path(args.output).write_text(table + "\n")


if __name__ == "__main__":
    main()
