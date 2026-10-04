"""Unit-test the Lite3 stair progress rewards (`stair_progress` + `stair_climb_bonus`) on scripted paths.

No physics: a fake env moves the robot root along +x out of a stair tile's centre at a chosen speed
under a chosen forward command, and the per-step term outputs are integrated the way the reward
manager does (value x step_dt; weights left out). Env 0 sits on an ascent tile, env 1 on descent.

Usage:
    ./isaaclab.sh -p scripts/tools/test_stair_progress_rewards.py --headless
"""

import argparse
import sys

from isaaclab.app import AppLauncher

sys.stdout.reconfigure(line_buffering=True)

parser = argparse.ArgumentParser(description="Scripted-path tests for the stair progress rewards.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app = AppLauncher(args_cli).app

import math  # noqa: E402
import types  # noqa: E402

import torch  # noqa: E402

from isaaclab.managers import RewardTermCfg  # noqa: E402

import rl_training.tasks.manager_based.locomotion.velocity.mdp as mdp  # noqa: E402
from rl_training.tasks.manager_based.locomotion.velocity.config.quadruped.deeprobotics_lite3.stairs_env_cfg import (  # noqa: E402
    STAIRS_INNER_EDGE,
    STAIRS_OUTER_EDGE,
    TASK_STAIR_ASCENT,
    TASK_STAIR_DESCENT,
)

DT = 0.02
COLUMNS = [TASK_STAIR_ASCENT, TASK_STAIR_DESCENT]


class FakeScene:
    def __init__(self, robot, n):
        self.robot = robot
        self.env_origins = torch.zeros(n, 3)
        self.terrain = types.SimpleNamespace(terrain_types=torch.arange(n))

    def __getitem__(self, name):
        return self.robot


class FakeEnv:
    def __init__(self, n=2):
        self.num_envs, self.device, self.step_dt = n, "cpu", DT
        data = types.SimpleNamespace(
            root_pos_w=torch.zeros(n, 3), root_quat_w=torch.tensor([[1.0, 0, 0, 0]] * n), root_lin_vel_w=torch.zeros(n, 3)
        )
        self.robot = types.SimpleNamespace(data=data)
        self.scene = FakeScene(self.robot, n)
        self.cmd = torch.zeros(n, 3)
        self.command_manager = types.SimpleNamespace(get_command=lambda name: self.cmd)


def make_terms(env, cap_to_command=False, penalize_slow=True, cap_factor=1.0):
    progress_params = {
        "command_name": "base_velocity",
        "task_ids_per_column": COLUMNS,
        "ascent_task_ids": (TASK_STAIR_ASCENT,),
        "descent_task_ids": (TASK_STAIR_DESCENT,),
        "max_distance": STAIRS_OUTER_EDGE,
        "height_scale": 1.0,
        "cap_to_command": cap_to_command,
        "cap_factor": cap_factor,
    }
    bonus_params = {
        "command_name": "base_velocity",
        "task_ids_per_column": COLUMNS,
        "task_ids": (TASK_STAIR_ASCENT,),
        "start_distance": STAIRS_INNER_EDGE,
        "finish_distance": STAIRS_OUTER_EDGE,
        "min_efficiency": 0.5,
        "full_efficiency": 0.9,
        "penalize_slow": penalize_slow,
    }
    progress = mdp.stair_progress(RewardTermCfg(func=mdp.stair_progress, weight=1.0, params=progress_params), env)
    bonus = mdp.stair_climb_bonus(RewardTermCfg(func=mdp.stair_climb_bonus, weight=1.0, params=bonus_params), env)
    return (progress, progress_params), (bonus, bonus_params)


def run(waypoints, speed, cmd_speed, z_per_m=0.0, **term_kwargs):
    """Move both envs' roots through `waypoints` (x, metres) at `speed`; return integrated outputs."""
    env = FakeEnv()
    (progress, pp), (bonus, bp) = make_terms(env, **term_kwargs)
    env.cmd[:, 0] = cmd_speed
    totals = {"progress": torch.zeros(2), "bonus": torch.zeros(2), "bonus_steps": torch.zeros(2)}
    x = waypoints[0]
    for target in waypoints[1:]:
        direction = 1.0 if target > x else -1.0
        while (target - x) * direction > 1e-9:
            x = x + direction * min(speed * DT, abs(target - x))
            env.robot.data.root_pos_w[:, 0] = x
            env.robot.data.root_pos_w[:, 2] = z_per_m * x
            totals["progress"] += progress(env, **pp) * DT
            b = bonus(env, **bp) * DT
            totals["bonus"] += b
            totals["bonus_steps"] += (b > 0).float()
    return totals


failures = []


def check(name, value, expected, tol=0.03):
    ok = abs(value - expected) <= tol
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {value:.3f} (expected {expected:.3f})")
    if not ok:
        failures.append(name)


def main():
    flight = STAIRS_OUTER_EDGE  # progress is paid from the spawn (x=0) out to the outer edge
    past = STAIRS_OUTER_EDGE + 0.5

    t = run([0.0, past], speed=0.5, cmd_speed=0.5)
    check("on-pace climb: dense progress = flight length (m)", t["progress"][0].item(), flight)
    check("on-pace climb: bonus payout", t["bonus"][0].item(), 1.0)
    check("on-pace climb: bonus paid on exactly one step", t["bonus_steps"][0].item(), 1.0, tol=0)
    check("descent tile: dense progress paid", t["progress"][1].item(), flight)
    check("descent tile: no bonus", t["bonus"][1].item(), 0.0, tol=0)

    t = run([0.0, past], speed=0.5, cmd_speed=0.5, z_per_m=0.5)
    # distance is capped at the outer edge, height gain is not (z keeps rising to the end of the path)
    check("ascent with height gain: progress = flight + total height gained",t["progress"][0].item(), flight + 0.5 * past)
    check("descent: height loss not paid", t["progress"][1].item(), flight)

    for speed, expected in ((0.4, 0.75), (0.25, 0.0), (0.75, (2 / 3 - 0.5) / 0.4), (1.0, 0.0)):
        t = run([0.0, past], speed=speed, cmd_speed=0.5)
        check(f"actual {speed} m/s vs command 0.5: bonus payout", t["bonus"][0].item(), expected)

    t = run([0.0, past], speed=0.5, cmd_speed=0.0)
    check("zero command: no dense progress", t["progress"][0].item(), 0.0, tol=0)
    check("zero command: no bonus", t["bonus"][0].item(), 0.0, tol=0)

    t = run([0.0, 2.0, 1.0, 2.0, 1.0, past], speed=0.5, cmd_speed=0.5)
    check("pacing back and forth: dense progress not farmed", t["progress"][0].item(), flight)
    # 6.3 m walked (all under a 0.5 m/s outward command) for 2.3 m of flight: efficiency 0.37
    check("pacing back and forth: no bonus for the slow climb", t["bonus"][0].item(), 0.0, tol=0)

    # v2 options: progress capped at the commanded speed, bonus only penalises overrunning
    t = run([0.0, past], speed=1.0, cmd_speed=0.5, cap_to_command=True)
    check("capped: 2x commanded speed credits half the flight", t["progress"][0].item(), flight / 2)
    t = run([0.0, past], speed=0.4, cmd_speed=0.5, cap_to_command=True)
    check("capped: below commanded speed is unaffected", t["progress"][0].item(), flight)
    t = run([0.0, past], speed=1.0, cmd_speed=0.5, z_per_m=0.5, cap_to_command=True)
    # past the outer edge distance is capped, so the remaining height gain is not scaled
    check("capped: height gain scaled with the distance cap", t["progress"][0].item(),
          (flight + 0.5 * flight) / 2 + 0.5 * (past - flight), tol=0.06)
    t = run([0.0, past], speed=0.2, cmd_speed=0.5, penalize_slow=False)
    check("overspeed-only bonus: slow climb pays in full", t["bonus"][0].item(), 1.0)
    t = run([0.0, past], speed=1.0, cmd_speed=0.5, penalize_slow=False)
    check("overspeed-only bonus: 2x commanded speed pays nothing", t["bonus"][0].item(), 0.0)
    t = run([0.0, past], speed=0.5, cmd_speed=0.0, penalize_slow=False)
    check("overspeed-only bonus: zero command pays nothing", t["bonus"][0].item(), 0.0, tol=0)

    t = run([0.0, past], speed=1.0, cmd_speed=0.5, cap_to_command=True, cap_factor=1.25)
    check("cap factor 1.25: 2x speed credits 62.5% of the flight", t["progress"][0].item(), flight * 0.625)

    # v3 tracking: forward-speed deadband on stairs, sideways never relaxed, unchanged off stairs
    def tracking(vx, vy, cmd_x, stair=True):
        env = FakeEnv()
        env.cmd[:, 0] = cmd_x
        env.robot.data.root_lin_vel_w[:, 0] = vx
        env.robot.data.root_lin_vel_w[:, 1] = vy
        ids = (TASK_STAIR_ASCENT, TASK_STAIR_DESCENT) if stair else ()
        return mdp.track_lin_vel_xy_stair_tolerant_exp(
            env, std=0.3, command_name="base_velocity", task_ids_per_column=COLUMNS, stair_task_ids=ids
        )[0].item()

    check("tracking on stairs: 19% slow is free", tracking(0.65, 0.0, 0.8), 1.0, tol=1e-4)
    check("tracking on stairs: 19% fast is free", tracking(0.95, 0.0, 0.8), 1.0, tol=1e-4)
    check("tracking on stairs: 50% slow pays for the excess 0.2 m/s", tracking(0.4, 0.0, 0.8), math.exp(-0.04 / 0.09), tol=1e-3)
    check("tracking on stairs: 0.2 m/s sideways still costs", tracking(0.8, 0.2, 0.8), math.exp(-0.04 / 0.09), tol=1e-3)
    check("tracking on stairs: min tolerance 0.1 m/s at slow commands", tracking(0.12, 0.0, 0.2), 1.0, tol=1e-4)
    check("tracking off stairs: unchanged yaw-frame reward", tracking(0.65, 0.0, 0.8, stair=False), math.exp(-0.0225 / 0.09), tol=1e-3)

    # reset mid-episode: state clears and the bonus can be earned again
    env = FakeEnv()
    (progress, pp), (bonus, bp) = make_terms(env)
    env.cmd[:, 0] = 0.5
    paid = 0
    for _ in range(2):
        bonus.reset(None)
        for i in range(int(past / (0.5 * DT)) + 1):
            env.robot.data.root_pos_w[:, 0] = i * 0.5 * DT
            paid += int(bonus(env, **bp)[0].item() > 0)
    check("bonus earned again after reset", paid, 2, tol=0)

    print("[RESULT]", "all passed" if not failures else f"{len(failures)} failed: {failures}")


if __name__ == "__main__":
    main()
    app.close()
