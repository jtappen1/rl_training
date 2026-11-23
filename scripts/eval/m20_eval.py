# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause

"""Run the M20 evaluation suite for one policy and one pass (docs/plans/M20_Evaluation_Suite.md).

All scenarios of the selected tests run in parallel in one simulation, each in its own block of envs.
Writes one JSON result (default: <checkpoint dir>/eval/<checkpoint stem>_<pass>.json).

Usage:
    python scripts/eval/m20_eval.py --checkpoint logs/rsl_rl/deeprobotics_m20_rough/<run>/model_10000.pt \
        --pass randomized --headless
    python scripts/eval/m20_eval.py --checkpoint /path/to/m20_policy.onnx --pass nominal --headless

To run several policies/passes and build the comparison table, use scripts/eval/m20_eval_suite.py.
"""

import argparse
import os
import sys
import time

from isaaclab.app import AppLauncher

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from m20eval import scenarios as S  # noqa: E402

VENDOR_TASK = "Rough-Deeprobotics-M20-v0"
DR1_TASK = "Rough-Deeprobotics-M20-DR1-v0"
DEFAULT_ENVS_PER_CASE = {"nominal": 64, "randomized": 512}

parser = argparse.ArgumentParser(description="M20 evaluation suite: one policy, one pass.")
parser.add_argument("--checkpoint", required=True, help="RSL-RL checkpoint (.pt) or exported policy (.onnx).")
parser.add_argument("--pass", dest="eval_pass", choices=("nominal", "randomized"), required=True)
parser.add_argument(
    "--task",
    default=None,
    help=f"Task whose config the eval is built from. Default: {VENDOR_TASK} for the nominal pass, "
    f"{DR1_TASK} for the randomized pass (falls back to {VENDOR_TASK} if DR1 is not registered).",
)
parser.add_argument("--tests", nargs="+", choices=S.TESTS, default=list(S.TESTS))
parser.add_argument("--envs_per_case", type=int, default=None, help="Default: 64 nominal, 512 randomized.")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--output", default=None, help="Result JSON path.")
parser.add_argument(
    "--agent", default="rsl_rl_cfg_entry_point", help="Agent config entry point (used to rebuild .pt policies)."
)
parser.add_argument(
    "--env_cfg_hook",
    default=None,
    help="'path/to/file.py:function' or 'module:function' called as function(env_cfg) on the task config before "
    "the eval overrides, e.g. to switch off one randomization term for an ablation.",
)
parser.add_argument(
    "--snapshot_s",
    type=float,
    nargs="*",
    default=[],
    help="Save PNG snapshots of the whole eval layout at these times (s). Enables rendering, which slows the run.",
)
parser.add_argument("--snapshot_dir", default=None, help="Default: <result dir>/snapshots_<pass>/")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True  # evaluation never needs a viewer
if args_cli.snapshot_s:
    args_cli.enable_cameras = True  # off-screen rendering for snapshots

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import datetime  # noqa: E402
import subprocess  # noqa: E402
from pathlib import Path  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402

import rl_training.tasks  # noqa: E402, F401
from rl_training.tasks.manager_based.locomotion.velocity.config.wheeled.deeprobotics_m20.eval_env_cfg import (  # noqa: E402
    EVAL_START_ROW,
    EVAL_TERRAIN_COLUMNS,
    configure_m20_eval,
)

from m20eval import metrics as M  # noqa: E402
from m20eval import results as R  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "reinforcement_learning", "rsl_rl"))
import cli_args  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
NEVER = 1 << 30


def resolve_task(eval_pass: str, task: str | None) -> tuple[str, str]:
    """Return (task id, description of where the randomization ranges come from)."""
    if task:
        return task, task
    if eval_pass == "nominal":
        return VENDOR_TASK, "none (nominal pass)"
    if DR1_TASK in gym.registry:
        return DR1_TASK, DR1_TASK
    print(f"[WARN] {DR1_TASK} is not registered yet: the randomized pass uses the vendor ranges of {VENDOR_TASK}.")
    return VENDOR_TASK, f"{VENDOR_TASK} (fallback: {DR1_TASK} not registered)"


def load_hook(spec: str):
    """Resolve 'path/to/file.py:function' or 'package.module:function'."""
    import importlib
    import importlib.util

    target, func = spec.rsplit(":", 1)
    if target.endswith(".py"):
        mod_spec = importlib.util.spec_from_file_location("m20_eval_env_cfg_hook", target)
        module = importlib.util.module_from_spec(mod_spec)
        mod_spec.loader.exec_module(module)
    else:
        module = importlib.import_module(target)
    return getattr(module, func)


def camera_views(terrain) -> dict[str, tuple[tuple, tuple]]:
    """(eye, lookat) of the snapshot views, from the tile layout of the eval terrain."""
    o = terrain.terrain_origins
    flat = o[EVAL_START_ROW, EVAL_TERRAIN_COLUMNS["flat"]].tolist()
    tiles = [o[EVAL_START_ROW, c].tolist() for t, c in EVAL_TERRAIN_COLUMNS.items() if t != "flat"]
    ty = sum(p[1] for p in tiles) / len(tiles)
    return {
        "overview": ((flat[0] - 30.0, flat[1] - 26.0, 32.0), (flat[0] + 12.0, ty - 6.0, 0.0)),
        "flat_tests": ((flat[0] - 4.0, flat[1] - 13.0, 7.0), (flat[0] + 7.0, flat[1], 0.0)),
        "terrain_tiles": ((flat[0] - 10.0, ty - 22.0, 12.0), (flat[0] + 6.0, ty, 0.5)),
    }


def save_snapshots(uenv, views: dict, out_dir: Path, stem: str, t: float):
    from PIL import Image

    out_dir.mkdir(parents=True, exist_ok=True)
    for name, (eye, lookat) in views.items():
        uenv.sim.set_camera_view(eye, lookat)
        for _ in range(6):  # let the renderer settle after the camera move
            uenv.sim.render()
        frame = uenv.render()
        if frame is None or frame.size == 0:
            print(f"[WARN] empty frame for view {name}", flush=True)
            continue
        path = out_dir / f"{stem}_t{t:04.1f}s_{name}.png"
        Image.fromarray(frame[..., :3]).save(path)
        print(f"[INFO] snapshot {path}", flush=True)


def git_info() -> dict:
    def run(*cmd):
        try:
            return subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, check=True).stdout.strip()
        except Exception:
            return None

    return {"commit": run("git", "rev-parse", "HEAD"), "dirty": bool(run("git", "status", "--porcelain"))}


def load_policy(path: str, env: RslRlVecEnvWrapper, task: str, device: str):
    """Return (callable obs TensorDict -> actions, policy type)."""
    if path.endswith(".onnx"):
        from m20eval.policy import OnnxPolicy

        obs = env.get_observations()
        onnx_policy = OnnxPolicy(path, obs_dim=obs["policy"].shape[-1], num_actions=env.num_actions)
        return (lambda o: onnx_policy(o["policy"])), "onnx"

    import importlib.metadata as metadata

    from packaging import version
    from rsl_rl.runners import OnPolicyRunner

    agent_cfg = load_cfg_from_registry(task, args_cli.agent)
    agent_cfg.device = device
    train_cfg = agent_cfg.to_dict()
    if version.parse(metadata.version("rsl-rl-lib")) >= version.parse("5.0.0"):
        train_cfg = cli_args.convert_rsl_rl_cfg_dict(train_cfg)
    runner = OnPolicyRunner(env, train_cfg, log_dir=None, device=device)
    runner.load(path)
    # the inference policy returns the action mean (deterministic)
    return runner.get_inference_policy(device=device), "rsl_rl_checkpoint"


def main():
    eval_pass = args_cli.eval_pass
    randomize = eval_pass == "randomized"
    epc = args_cli.envs_per_case or DEFAULT_ENVS_PER_CASE[eval_pass]
    ckpt = os.path.abspath(args_cli.checkpoint)
    if not os.path.isfile(ckpt):
        raise FileNotFoundError(ckpt)
    output = args_cli.output or os.path.join(
        os.path.dirname(ckpt), "eval", f"{Path(ckpt).stem}_{eval_pass}.json"
    )
    device = args_cli.device or "cuda:0"
    task, randomization_source = resolve_task(eval_pass, args_cli.task)
    wall_start = time.time()

    # ---------------- scenarios and env config ----------------
    scenarios = S.build_scenarios(args_cli.tests)
    n_scn = len(scenarios)
    num_envs = n_scn * epc

    env_cfg = load_cfg_from_registry(task, "env_cfg_entry_point")
    if args_cli.env_cfg_hook:
        load_hook(args_cli.env_cfg_hook)(env_cfg)
    orig_delays = {
        name: (int(getattr(a, "min_delay", 0)), int(getattr(a, "max_delay", 0)))
        for name, a in env_cfg.scene.robot.actuators.items()
    }
    total_s = max(sc.duration_s for sc in scenarios)
    configure_m20_eval(env_cfg, randomize=randomize, episode_length_s=total_s + 10.0)
    env_cfg.scene.num_envs = num_envs
    env_cfg.seed = args_cli.seed
    env_cfg.sim.device = device
    if args_cli.snapshot_s:
        env_cfg.viewer.resolution = (1920, 1080)
        env_cfg.viewer.origin_type = "world"

    print(f"[INFO] M20 eval: pass={eval_pass} task={task} scenarios={n_scn} envs/case={epc} envs={num_envs}", flush=True)
    env = gym.make(task, cfg=env_cfg, render_mode="rgb_array" if args_cli.snapshot_s else None)
    env = RslRlVecEnvWrapper(env, clip_actions=load_cfg_from_registry(task, args_cli.agent).clip_actions)
    uenv = env.unwrapped
    robot = uenv.scene["robot"]
    dt = uenv.step_dt
    physics_dt = uenv.physics_dt
    T_total = int(round(total_s / dt))
    settle = int(round(S.SETTLE_S / dt))

    # ---------------- per-env assignment ----------------
    env_group = torch.arange(num_envs, device=device) // epc
    blocks = [torch.arange(i * epc, (i + 1) * epc, device=device) for i in range(n_scn)]
    T_s = torch.tensor([int(round(sc.duration_s / dt)) for sc in scenarios], device=device)
    T_env = T_s[env_group]

    schedule = torch.zeros(n_scn, T_total + 1, 3, device=device)
    for i, sc in enumerate(scenarios):
        k = settle
        for dur, cmd in sc.segments:
            n = int(round(dur / dt))
            schedule[i, k : k + n] = torch.tensor(cmd, device=device)
            k += n

    terrain = uenv.scene.terrain
    spawn_yaw = torch.zeros(num_envs, device=device)
    for i, sc in enumerate(scenarios):
        col = EVAL_TERRAIN_COLUMNS[sc.terrain]
        terrain.env_origins[blocks[i]] = terrain.terrain_origins[EVAL_START_ROW, col]
        terrain.terrain_levels[blocks[i]] = EVAL_START_ROW
        terrain.terrain_types[blocks[i]] = col
        spawn_yaw[blocks[i]] = sc.spawn_yaw
    uenv.eval_spawn_yaw = spawn_yaw
    uenv.command_manager.get_term("base_velocity").set_schedule(env_group, schedule)

    # friction perturbation: robot shapes of those envs get mu = value (ground is 1.0, combine = multiply)
    for i, sc in enumerate(scenarios):
        if sc.perturbation.get("type") == "friction":
            mats = robot.root_physx_view.get_material_properties()
            ids = blocks[i].cpu()
            mats[ids, :, 0] = sc.perturbation["value"]
            mats[ids, :, 1] = sc.perturbation["value"]
            robot.root_physx_view.set_material_properties(mats, ids)

    policy, policy_type = load_policy(ckpt, env, task, device)
    gen = torch.Generator(device=device).manual_seed(args_cli.seed)

    # ---------------- actuator delays ----------------
    delayed = {name: a for name, a in robot.actuators.items() if hasattr(a, "positions_delay_buffer")}
    base_lags = {}
    for name, a in delayed.items():
        lo, hi = orig_delays.get(name, (0, 0))
        if randomize:
            base_lags[name] = torch.randint(lo, hi + 1, (num_envs,), generator=gen, device=device, dtype=torch.long)
        else:
            base_lags[name] = torch.zeros(num_envs, dtype=torch.long, device=device)
    latency_ids, latency_cfg = None, None
    for i, sc in enumerate(scenarios):
        if sc.perturbation.get("type") == "latency":
            latency_ids, latency_cfg = blocks[i], sc.perturbation
            base = int(round(latency_cfg["base_ms"] * 1e-3 / physics_dt))
            for name in delayed:
                base_lags[name][latency_ids] = base
    spike_steps = int(round(latency_cfg["spike_ms"] * 1e-3 / physics_dt)) if latency_cfg else 0

    def set_lags(spikes: torch.Tensor | None = None):
        for name, a in delayed.items():
            lags = base_lags[name]
            if spikes is not None:
                lags = lags.clone()
                lags[latency_ids[spikes]] = spike_steps
            for buf in (a.positions_delay_buffer, a.velocities_delay_buffer, a.efforts_delay_buffer):
                buf.set_time_lag(lags)

    if latency_cfg and not delayed:
        print("[WARN] No delayed actuators found: the latency test runs without added latency.")

    # ---------------- hardware logging at physics rate ----------------
    J = robot.num_joints
    effort_limit = torch.full((num_envs, J), float("inf"), device=device)
    for a in robot.actuators.values():
        effort_limit[:, a.joint_indices] = a.effort_limit
    hw = M.HardwareAccumulator(num_envs, J, effort_limit, device)
    original_write = robot.write_data_to_sim

    def write_and_record():
        original_write()
        hw.update(robot.data.computed_torque, robot.data.applied_torque)

    robot.write_data_to_sim = write_and_record

    # ---------------- reset ----------------
    obs, _ = env.reset()
    set_lags()

    action_names = []
    for term_name in uenv.action_manager.active_terms:
        term = uenv.action_manager.get_term(term_name)
        names = getattr(term, "_joint_names", None) or [f"{term_name}_{j}" for j in range(term.action_dim)]
        action_names += [f"{term_name}:{n}" for n in names]
    wheel_term = uenv.action_manager.get_term("joint_vel") if "joint_vel" in uenv.action_manager.active_terms else None

    # ---------------- logs ----------------
    A = env.num_actions
    vel_log = torch.zeros(T_total, num_envs, 3, device=device)
    pos_log = torch.zeros(T_total, num_envs, 2, device=device)
    head_log = torch.zeros(T_total, num_envs, device=device)
    act_log = torch.zeros(T_total, num_envs, A, device=device, dtype=torch.float16)
    fail_step = torch.full((num_envs,), NEVER, dtype=torch.long, device=device)
    fail_kind = torch.zeros(num_envs, dtype=torch.long, device=device)
    wheel_max = torch.zeros(num_envs, device=device)

    push_events = []  # (step, env ids, speed)
    for i, sc in enumerate(scenarios):
        if sc.perturbation.get("type") == "push":
            push_events.append((settle + int(round(sc.perturbation["at_s"] / dt)), blocks[i], sc.perturbation["speed"]))

    snapshot_steps = {int(round(t / dt)) - 1: t for t in args_cli.snapshot_s if 0 < t <= total_s}
    snapshot_dir = Path(args_cli.snapshot_dir or Path(output).parent / f"snapshots_{eval_pass}")
    snapshot_views = camera_views(terrain) if snapshot_steps else {}

    print(f"[INFO] Simulating {T_total} policy steps ({total_s:.1f} s) ...", flush=True)
    with torch.inference_mode():
        for k in range(T_total):
            alive = fail_step == NEVER
            in_window = (k >= settle) & (k < T_env)
            hw.recording = alive & in_window

            for step_k, ids, speed in push_events:
                if k == step_k:
                    ang = torch.rand(len(ids), generator=gen, device=device) * 2.0 * torch.pi
                    vel_w = robot.data.root_vel_w[ids].clone()
                    vel_w[:, 0] += speed * torch.cos(ang)
                    vel_w[:, 1] += speed * torch.sin(ang)
                    robot.write_root_velocity_to_sim(vel_w, env_ids=ids)

            if latency_cfg and delayed:
                set_lags(torch.rand(len(latency_ids), generator=gen, device=device) < latency_cfg["spike_prob"])

            actions = policy(obs)
            obs, _, _, _ = env.step(actions)
            if k in snapshot_steps:
                save_snapshots(uenv, snapshot_views, snapshot_dir, Path(ckpt).stem, snapshot_steps[k])

            vel_log[k, :, :2] = robot.data.root_lin_vel_b[:, :2]
            vel_log[k, :, 2] = robot.data.root_ang_vel_b[:, 2]
            pos_log[k] = robot.data.root_pos_w[:, :2]
            head_log[k] = robot.data.heading_w
            act_log[k] = actions.to(torch.float16)
            if wheel_term is not None:
                w = wheel_term.processed_actions.abs().max(dim=1).values
                wheel_max = torch.where(alive & in_window, torch.maximum(wheel_max, w), wheel_max)

            tm = uenv.termination_manager
            terminated, time_outs = tm.terminated, tm.time_outs
            new_fail = alive & (terminated | time_outs)
            fail_step[new_fail] = k
            fail_kind[new_fail & terminated] = M.FAIL_FELL
            fail_kind[new_fail & ~terminated] = M.FAIL_OOB
            if k % 250 == 0:
                print(f"  step {k}/{T_total}, envs failed so far: {int((fail_step != NEVER).sum())}", flush=True)

    # ---------------- metrics ----------------
    knee_ids = [i for i, n in enumerate(robot.joint_names) if "knee" in n]
    cases: dict = {}
    hf_by_test: dict = {}
    for i, sc in enumerate(scenarios):
        ids = blocks[i]
        log = M.ScenarioLog(
            vel=vel_log[:, ids].cpu(),
            pos=pos_log[:, ids].cpu(),
            heading=head_log[:, ids].cpu(),
            actions=act_log[:, ids].cpu(),
            cmd=schedule[i, :T_total].cpu(),
            fail_step=fail_step[ids].cpu(),
            fail_kind=fail_kind[ids].cpu(),
            wheel_target_max=wheel_max[ids].cpu(),
            dt=dt,
        )
        case, hf = M.case_metrics(sc, log)
        case["spawn_origin_xyz"] = terrain.env_origins[ids[0]].tolist()
        case["spawn_yaw"] = sc.spawn_yaw
        cases.setdefault(sc.test, {})[sc.case] = case
        acc = hf_by_test.setdefault(sc.test, {"sum": torch.zeros(A, dtype=torch.float64), "n": 0})
        if hf["n"]:
            acc["sum"] += hf["sum"]
            acc["n"] += hf["n"]

    hardware_raw = {}
    for test in dict.fromkeys(sc.test for sc in scenarios):
        ids = torch.cat([blocks[i] for i, sc in enumerate(scenarios) if sc.test == test])
        hardware_raw[test] = hw.raw(ids, knee_ids, hf_by_test[test]["sum"], hf_by_test[test]["n"], A)

    result = {
        "schema": R.SCHEMA,
        "meta": {
            "policy_path": ckpt,
            "policy_type": policy_type,
            "pass": eval_pass,
            "task": task,
            "randomization_source": randomization_source,
            "env_cfg_hook": args_cli.env_cfg_hook,
            "observation_noise": randomize,
            "seed": args_cli.seed,
            "envs_per_case": epc,
            "num_envs": num_envs,
            "tests_run": [t for t in S.TESTS if t in args_cli.tests],
            "chunks": [{"tests": list(args_cli.tests), "num_envs": num_envs}],
            "policy_dt": dt,
            "physics_dt": physics_dt,
            "joint_names": list(robot.joint_names),
            "action_names": action_names,
            "effort_limits_Nm": effort_limit[0].tolist(),
            "actuator_delay_ranges_steps": orig_delays,
            "run_params_dir": str(Path(ckpt).parent / "params") if (Path(ckpt).parent / "params").is_dir() else None,
            "git": git_info(),
            "created": datetime.datetime.now().isoformat(timespec="seconds"),
            "wall_time_s": time.time() - wall_start,
            "definitions": {
                "settle_s": S.SETTLE_S,
                "hold_s": S.HOLD_S,
                "steady_state_window_s": S.SS_WINDOW_S,
                "steady_state_error": "mean |actual - commanded| over the last 5 s of the hold, base frame, surviving envs",
                "rise_time": f"10-90 % of the step, on a {S.RISE_SMOOTHING_S} s centred moving average",
                "overshoot": "max excursion past the commanded value, % of the step",
                "stopping_distance": "displacement along the heading at the step, from the 1->0 step to the end",
                "standstill": "drift = |xy displacement| over the 20 s hold; max wheel target = max |wheel velocity target|",
                "push": f"velocity of the given magnitude added in a random horizontal direction {S.PUSH_AT_S} s into the hold",
                "latency": "actuator delay 15 ms (3 physics steps); each policy step jumps to 40 ms with 2 % probability",
                "friction": "robot friction 0.35 on ground friction 1.0 (multiply)",
                "tracking_err_increase": "full-hold mean |v_xy - cmd_xy| minus the same for linear_tracking/vx_+1.00 of this pass",
                "terrain_success": f"survived and progressed >= {S.TERRAIN_MIN_PROGRESS:.0%} of the commanded distance",
                "survival": "no termination (fall); out-of-bounds envs are excluded",
                "torque": "computed (pre-clip) torque peaks; knee gate = 99th percentile of per-env knee peaks in the worst test; RMS of applied torque; at limit = |computed| >= effort limit; physics rate, until a fall",
                "action_hf_power": f"share of mean-removed raw action power above {S.ACTION_HF_CUTOFF_HZ} Hz, surviving envs",
            },
        },
        "cases": cases,
        "hardware_raw": hardware_raw,
    }
    R.finalize(result)
    R.save(result, output)
    print(f"[INFO] Wrote {output} ({time.time() - wall_start:.0f} s)", flush=True)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
