# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause

"""Run the M20 evaluation suite on checkpoints and both baselines, then compare them.

For every policy and pass this runs scripts/eval/m20_eval.py (one simulation process per chunk of tests,
so large passes fit in GPU memory), keeps the result JSON next to the policy file
(<policy dir>/eval/<stem>_<pass>.json), and writes one comparison table per pass next to each candidate
(<candidate dir>/eval/comparison_<stem>_<pass>.md). The randomized pass is the headline.

Usage:
    python scripts/eval/m20_eval_suite.py \
        --checkpoints logs/rsl_rl/deeprobotics_m20_dr1/<run> \
        --vendor logs/rsl_rl/deeprobotics_m20_rough/<vendor retrain run> \
        --deployed /path/to/m20_policy.onnx \
        --target "terrain.*.success_pct"

A run directory stands for its latest model_<iter>.pt. Existing result files are reused unless --force.
Arguments after "--" are passed to m20_eval.py (e.g. -- --device cuda:1).
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from m20eval import compare as C  # noqa: E402
from m20eval import results as R  # noqa: E402
from m20eval import scenarios as S  # noqa: E402

EVAL_SCRIPT = Path(__file__).resolve().parent / "m20_eval.py"
DEFAULT_ENVS_PER_CASE = {"nominal": 64, "randomized": 512}


def resolve_policy(path: str) -> Path:
    p = Path(path).resolve()
    if p.is_dir():
        models = sorted(p.glob("model_*.pt"), key=lambda f: int(re.findall(r"\d+", f.stem)[-1]))
        if not models:
            raise FileNotFoundError(f"no model_*.pt in {p}")
        return models[-1]
    if not p.is_file():
        raise FileNotFoundError(p)
    return p


def result_path(policy: Path, eval_pass: str) -> Path:
    return policy.parent / "eval" / f"{policy.stem}_{eval_pass}.json"


def label_of(policy: Path) -> str:
    return f"{policy.parent.name}/{policy.stem}" if policy.suffix == ".pt" else policy.name


def run_eval(policy: Path, eval_pass: str, args, passthrough: list[str]) -> Path:
    out = result_path(policy, eval_pass)
    if out.exists() and not args.force:
        print(f"[suite] reuse {out}")
        return out
    epc = (args.envs_per_case_nominal if eval_pass == "nominal" else args.envs_per_case_randomized) or (
        DEFAULT_ENVS_PER_CASE[eval_pass]
    )
    chunks = S.pack_tests(epc, args.max_envs, args.tests)
    parts = []
    for i, tests in enumerate(chunks):
        part = out if len(chunks) == 1 else out.with_suffix(f".part{i}.json")
        cmd = [
            sys.executable, str(EVAL_SCRIPT),
            "--checkpoint", str(policy),
            "--pass", eval_pass,
            "--envs_per_case", str(epc),
            "--seed", str(args.seed),
            "--output", str(part),
            "--tests", *tests,
        ]
        if args.task:
            cmd += ["--task", args.task]
        cmd += passthrough
        print(f"[suite] {label_of(policy)} {eval_pass} chunk {i + 1}/{len(chunks)}: {' '.join(tests)}", flush=True)
        if part.exists():
            part.unlink()
        subprocess.run(cmd, check=True)
        if not part.exists():
            raise RuntimeError(f"m20_eval.py exited without writing {part}")
        parts.append(part)
    if len(chunks) > 1:
        R.save(R.merge([R.load(p) for p in parts]), out)
        for p in parts:
            p.unlink()
    print(f"[suite] wrote {out}")
    return out


def main():
    argv = sys.argv[1:]
    passthrough = []
    if "--" in argv:
        i = argv.index("--")
        argv, passthrough = argv[:i], argv[i + 1 :]

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoints", nargs="+", required=True, help="Candidate checkpoints (.pt/.onnx) or run dirs.")
    parser.add_argument("--vendor", default=None, help="Fresh retrain of the vendor rough config (.pt or run dir).")
    parser.add_argument("--deployed", default=None, help="The deployed policy, e.g. m20_policy.onnx.")
    parser.add_argument("--passes", nargs="+", choices=("randomized", "nominal"), default=["randomized", "nominal"])
    parser.add_argument("--tests", nargs="+", choices=S.TESTS, default=list(S.TESTS))
    parser.add_argument("--target", nargs="*", default=[], help="Target metric key(s); glob patterns allowed.")
    parser.add_argument("--rel_tol", type=float, default=C.DEFAULT_REL_TOL)
    parser.add_argument("--envs_per_case_nominal", type=int, default=None)
    parser.add_argument("--envs_per_case_randomized", type=int, default=None)
    parser.add_argument("--max_envs", type=int, default=8192, help="Max envs per simulation process.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--task", default=None, help="Override the task the eval config is built from.")
    parser.add_argument("--force", action="store_true", help="Re-run even if result files exist.")
    args = parser.parse_args(argv)

    candidates = [resolve_policy(p) for p in args.checkpoints]
    vendor = resolve_policy(args.vendor) if args.vendor else None
    deployed = resolve_policy(args.deployed) if args.deployed else None
    if not vendor:
        print("[suite] no --vendor given: tables are written without verdicts.")

    policies = list(dict.fromkeys([p for p in (deployed, vendor) if p] + candidates))
    files = {(p, ps): run_eval(p, ps, args, passthrough) for ps in args.passes for p in policies}

    for ps in args.passes:  # randomized first: it is the headline
        # re-finalize so results written by an older version pick up the current metric rules
        load = lambda p: (label_of(p), R.finalize(R.load(files[(p, ps)])))  # noqa: E731
        table, verdicts = C.markdown_table(
            [load(c) for c in candidates],
            load(vendor) if vendor else None,
            load(deployed) if deployed else None,
            args.target,
            args.rel_tol,
        )
        title = f"# M20 evaluation, {ps} pass{' (headline)' if ps == 'randomized' else ' (debugging)'}\n\n"
        sources = "\n".join(f"- `{label_of(p)}`: {files[(p, ps)]}" for p in policies)
        doc = f"{title}Result files:\n{sources}\n\n{table}\n"
        print("\n" + doc)
        for c in candidates:
            path = c.parent / "eval" / f"comparison_{c.stem}_{ps}.md"
            path.write_text(doc)
            print(f"[suite] wrote {path}")


if __name__ == "__main__":
    main()
