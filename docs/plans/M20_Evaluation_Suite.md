# M20 Evaluation Suite

Nov 22, 2026 · John

## Setup

Every checkpoint runs the same scripted tests in two passes and is scored against two baselines.

- **Eval task:** subclass of `DeeproboticsM20RoughEnvCfg`. Curriculum off, random pushes off (except the push test), command resampling off (commands are scripted), fixed seed, deterministic actions (policy mean).
- **Nominal pass:** randomization off, 64 envs per test.
- **Randomized pass:** DR1 ranges on, 512 envs per test.
- **Terrain:** flat ground for tracking, step response and standstill. Fixed tiles for the terrain test: 15° slope, 10 cm stairs, 15 cm stairs, medium rough.
- **Baselines:** the deployed `m20_policy.onnx`, and a fresh retrain of the vendor rough config.
- **Final gate:** MuJoCo closed-loop check (`sim_closed_loop.py`) before anything goes on the robot, not on every checkpoint.

## Tests and metrics

Steady-state error = mean |actual − commanded| over the last 5 s of a 10 s hold, after the robot has settled.

| Test | Commands | Metrics |
| --- | --- | --- |
| Linear tracking | vx −1, −0.5, 0.5, 1, 1.5, 2 m/s; vy ±0.25, ±0.5 m/s; 10 s holds | Steady-state error per axis (m/s) |
| Yaw tracking | In place 0.5, 1, 1.5, 2 rad/s; arc 1 m/s + 0.5 rad/s; 10 s holds | Steady-state yaw error (rad/s) |
| Step response | 0→1 m/s, 1→0 m/s, 0→1.5 rad/s | Rise time 10–90% (s), overshoot (%), stopping distance (m) |
| Standstill | Zero command, 20 s | Base drift (m), max wheel target (rad/s, pass < 0.5) |
| Robustness | Pushes 0.5, 1, 1.5 m/s; friction 0.35; 15 ms latency + 40 ms spikes | Survival (%), tracking error increase vs nominal |
| Terrain | 1 m/s forward across each tile | Success (%), steady-state error (m/s) |
| Hardware | Logged during all tests | Peak and RMS torque per joint (N·m), % time at torque limit, knee peak (pass < 76.4 N·m), action power above 10 Hz (%) |

Gait is not scored, so stepping or rolling can emerge freely.

## Output

Each run writes one JSON per checkpoint per pass, then one comparison table against both baselines.

- A checkpoint is better only if it improves the target metric without making any other metric worse than the vendor retrain.
- Report the randomized pass as the headline; the nominal pass is for debugging.
- Keep every result file with its checkpoint, so `compare_runs.py` config diffs and metric diffs sit side by side.
