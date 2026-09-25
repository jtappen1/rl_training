# rl_training — M20 stairs

[![IsaacSim](https://img.shields.io/badge/IsaacSim-5.1.0-silver.svg)](https://docs.omniverse.nvidia.com/isaacsim/latest/overview.html)
[![Isaac Lab](https://img.shields.io/badge/IsaacLab-2.3.2-silver)](https://isaac-sim.github.io/IsaacLab)
[![RSL-RL](https://img.shields.io/badge/RSL--RL-5.0.1-silver)](https://github.com/leggedrobotics/rsl_rl)
[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://docs.python.org/3/whatsnew/3.11.html)
[![Linux platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/22.04/)
[![License](https://img.shields.io/badge/license-BSD%203--Clause-blue.svg)](https://opensource.org/license/bsd-3-clause)

Stair-climbing and rough-terrain policies for the **DEEP Robotics M20** wheeled quadruped, trained in Isaac Lab. This is a fork of [DeepRoboticsLab/rl_training](https://github.com/DeepRoboticsLab/rl_training). The upstream repo also has Lite3 and DR02 tasks, which are still in the code but not covered here.

<img src="./docs/imgs/deeprobotics_m20.png" alt="Deeprobotics M20" width="300">

What this fork adds on top of the upstream blind `Rough-Deeprobotics-M20-v0` task:

- **Sighted stair tasks** (v1 → v2 → v3 → v3b). The policy reads a forward-biased height scan, and the tasks add stair-aware rewards, terrain and curriculum. v2 was the first to reliably climb and descend stairs; v3/v3b add heading holding. See [docs/stairs_sighted_v2_summary.md](docs/stairs_sighted_v2_summary.md).
- **A depth student env** for distilling a sighted teacher into a depth-camera + proprioception policy. It has a ray-cast depth camera and a depth noise model. See [docs/student_distillation_setup.md](docs/student_distillation_setup.md).
- **Evaluation tools**: a fixed-geometry stairs benchmark and a long mixed-terrain course, both with CSV/markdown reports and video.
- **Keyboard driving** on the stairs benchmark, the course, and real-world ZED scans (a skatepark), with an on-screen speed/depth readout.

The overall plan lives in [command.md](command.md), and the background literature survey in [docs/stairs_research_notes.md](docs/stairs_research_notes.md).

## Installation

- Install **Isaac Sim 5.1.0 and Isaac Lab 2.3.2 with Python 3.11** using the [installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html). These are the tested versions.

- Clone this repository outside the Isaac Lab directory, with submodules (the robot model lives in the `deep_robotics_model` submodule):

  ```bash
  git clone --recurse-submodules <this-repo-url> rl_training
  cd rl_training
  # repairs a clone made without --recurse-submodules
  git submodule update --init --recursive
  ```

- Install the extension into the Python environment that runs Isaac Lab. This also installs **RSL-RL 5.0.1**.

  ```bash
  python -m pip install -e source/rl_training
  ```

  If Isaac Lab isn't installed in a Conda env, use its bundled Python instead: replace `python` with `<path-to-isaaclab>/isaaclab.sh -p` in this and every command below. If a virtualenv is active (for example inherited by a tmux session), `isaaclab.sh` picks up its Python instead of Isaac Sim's; run `unset VIRTUAL_ENV CONDA_PREFIX` first.

- Conda only: configure the environment's C++ runtime before launching Isaac Sim, then reactivate the environment.

  ```bash
  python scripts/tools/setup_conda_runtime.py
  ```

  This checks that Conda's `libstdc++.so.6` provides `CXXABI_1.3.15` and installs activation hooks so Isaac Sim doesn't load an older system runtime. If it reports an outdated runtime, run `conda install -c conda-forge "libstdcxx-ng>=15"` and retry.

- Check the install by listing the registered environments:

  ```bash
  python scripts/tools/list_envs.py
  ```

- **Terrain scans**: the converted skatepark scans (`mesh/mesh.usd`, `mesh/stairs/mesh-stairs.usd`) are committed, so the skatepark tasks work straight after cloning. The source `.obj` scans aren't committed; you only need them to re-convert (see [Real-world scans](#real-world-scans-skatepark)).

## Environments

All tasks are registered in [`config/wheeled/deeprobotics_m20/__init__.py`](source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/wheeled/deeprobotics_m20/__init__.py). Most have a `-Play-v0` twin with fewer envs, no pushes and a fixed forward command.

**Training tasks**

| Task ID | What it is |
|---|---|
| `Flat-Deeprobotics-M20-v0`, `Rough-Deeprobotics-M20-v0` | Upstream blind locomotion. Unchanged. |
| `Stairs-Teacher-Deeprobotics-M20-v0` | Blind, on the stairs terrain mix with the 3-task (flat/rough, stair ascent, stair descent) plumbing. The base for the sighted tasks. |
| `Stairs-Sighted-Deeprobotics-M20-v0` | v1: the height scan is added to the actor. Plateaued at ~0.17 m steps. |
| `Stairs-Sighted-V2-Deeprobotics-M20-v0` | v2: yaw-frame velocity tracking, stair-aware orientation limits, stair progress reward, wheel clearance reward, forward-biased scan, stair curriculum. 100% up to 0.25 m. |
| `Stairs-Sighted-V3-Deeprobotics-M20-v0` | v3: v2 + a heading-hold reward and straight-only commands; wheel clearance reward removed. Fixed heading drift, but ascent regressed. |
| `Stairs-Sighted-V3b-Deeprobotics-M20-v0` | v3b: v3 with the wheel clearance reward restored. The current main policy. |
| `Stairs-Student-Deeprobotics-M20-v0` | Depth student: `teacher` (v2 policy obs, noise-free), `policy` (proprioception), `depth` (64×36 image). The distillation runner is not written yet. |

**Keyboard-drive tasks** (play only, see [Keyboard driving](#keyboard-driving))

| Task ID | Policy | Terrain |
|---|---|---|
| `Stairs-Bench-V3b-Deeprobotics-M20-Play-v0` | v3b | The `eval_stairs.py` tiles: 0.08–0.25 m risers, 0.30 m treads, ascent then descent |
| `Course-V3b-Deeprobotics-M20-Play-v0` | v3b | The `eval_multi_terrain_course.py` course |
| `Skatepark-V3b-Deeprobotics-M20-Play-v0` | v3b | ZED scan |
| `Skatepark-Deeprobotics-M20-Play-v0` | Rough (blind) | ZED scan |

The Stairs-Bench and Course tasks also attach the student's depth camera (clean, no noise), so you can watch what a student would see. The v3b policy itself still reads only its height scan.

## Training

```bash
python scripts/reinforcement_learning/rsl_rl/train.py --task=Stairs-Sighted-V3b-Deeprobotics-M20-v0 --headless
```

- Smoke-test a new task first: `--num_envs 16 --max_iterations 2`.
- Resume: `--resume --load_run <run_folder> --checkpoint model_<N>.pt`.
- Logs and checkpoints go to `logs/rsl_rl/<experiment_name>/<timestamp>/`. The experiment names are `deeprobotics_m20_rough`, `deeprobotics_m20_stairs_sighted`, `..._sighted_v2`, `..._sighted_v3` and `..._sighted_v3b`.
- TensorBoard: `tensorboard --logdir=logs`.
- Diff the configs of two runs (from their saved `params/`): `python scripts/tools/compare_runs.py <run_dir_1> <run_dir_2>`.

## Playing

`play.py` loads the latest checkpoint of the task's experiment unless you pick one. It also exports `policy.pt` / `policy.onnx` to the run's `exported/` folder.

```bash
# 50 robots on a shrunken 5x5 version of the training terrain
python scripts/reinforcement_learning/rsl_rl/play.py --task=Stairs-Sighted-V3b-Deeprobotics-M20-Play-v0

# a specific checkpoint (file, or --load_run <folder> --checkpoint model_<N>.pt)
python scripts/reinforcement_learning/rsl_rl/play.py --task=Stairs-Sighted-V3b-Deeprobotics-M20-Play-v0 \
    --checkpoint logs/rsl_rl/deeprobotics_m20_stairs_sighted_v3b/<run>/model_5999.pt
```

- `--num_envs N`: number of robots (default 50).
- `--video --video_length 200`: record an mp4 into the checkpoint's `videos/play/` folder. Frames are buffered in RAM, so keep clips short.
- `--real-time`: slow the sim down to wall-clock speed.

### Keyboard driving

Add `--keyboard` to drive a single robot. The command from the keyboard replaces the policy's velocity-command input, and the camera follows the robot. Needs a GUI session (no `--headless`). Click in the viewport first so it receives key presses.

```bash
python scripts/reinforcement_learning/rsl_rl/play.py --task=Stairs-Bench-V3b-Deeprobotics-M20-Play-v0 --keyboard
python scripts/reinforcement_learning/rsl_rl/play.py --task=Course-V3b-Deeprobotics-M20-Play-v0 --keyboard
python scripts/reinforcement_learning/rsl_rl/play.py --task=Skatepark-V3b-Deeprobotics-M20-Play-v0 --keyboard
```

| Action | Key |
|---|---|
| Forward / back | Arrow Up / Down (Numpad 8 / 2) |
| Strafe left / right | Arrow Left / Right (Numpad 4 / 6) |
| Turn left / right | Z / X (Numpad 7 / 9) |
| Zero the command | L |
| Faster / slower (linear speed ×0.25–×3) | `=` / `-` (Numpad + / −) |
| Next / previous terrain tile (respawns there) | N / B |

- **Speeds**: at ×1, the top speed is half the task's max forward command, and its full max strafe and turn rates. For v3b that's 0.6 m/s forward, 0.3 m/s sideways and 1.0 rad/s.
- **Overlay**: a small "Keyboard drive" window shows the speed scale, the command, and the measured base velocity. On tasks with a depth camera it also shows the depth image (near = dark, far = bright, invalid = red).
- **Tiles**: N / B only do something on generated terrain. On the stairs benchmark they step through the 14 tiles (7 ascents, then 7 descents).
- **Keyboard-drive tasks**: these run one robot with no pushes and no episode timeout, and respawn facing +x after a fall.

### Real-world scans (skatepark)

The skatepark tasks load a ZED SDK spatial-mapping scan as a static USD terrain. `SKATEPARK_USD_PATH` in [skatepark_env_cfg.py](source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/wheeled/deeprobotics_m20/skatepark_env_cfg.py) picks which scan:

- `mesh/mesh.usd`: the original skatepark scan.
- `mesh/stairs/mesh-stairs.usd`: a second scan with an upper level (~0.94 m) about 2.4 m to the robot's left at spawn. This is the current default.

Both USDs are in the repo. Next to each one, `config.yaml` records the converter settings used to make it (source path, axis rotation, recentering translation).

To convert a new scan, or re-convert one from its `.obj` (not committed), use [`scripts/tools/convert_terrain_mesh.py`](scripts/tools/convert_terrain_mesh.py). It fixes the axis convention (ZED exports are Y-up; Isaac Sim is Z-up) and moves a chosen ground point to the world origin, so the robot spawns on it:

```bash
python scripts/tools/convert_terrain_mesh.py mesh/mesh-stairs.obj mesh/stairs/mesh-stairs.usd \
    --up-axis y --recenter 4.0 -0.194 -2.5 --headless
```

Choose the `--recenter` point (in the original mesh's coordinates) on a flat, open patch of ground. For the v3b task, the stair-aware rewards and curriculum are dropped on a scan (they need generated terrain), and falls use the plain `bad_orientation_2` limit (~44°).

## Evaluation

Both eval scripts take `--checkpoint <model.pt or run dir>`; with a run dir, they use the highest-iteration checkpoint in it. They write to `eval/`, which is git-ignored.

### Stairs benchmark — `eval_stairs.py`

Fixed-geometry pyramid stairs, one tile per (direction, step height, tread width), with many robots in parallel. Each robot gets a fixed forward command (no yaw, no heading control) and must reach the far side of its tile before the time budget runs out.

```bash
python scripts/tools/eval_stairs.py --headless \
    --task Stairs-Sighted-V3b-Deeprobotics-M20-v0 \
    --checkpoint logs/rsl_rl/deeprobotics_m20_stairs_sighted_v3b/<run>
```

| Option | Default |
|---|---|
| `--task` (the obs/robot config to evaluate with; must match the checkpoint) | `Rough-Deeprobotics-M20-v0` |
| `--speeds` | `0.5 1.0` m/s |
| `--step_heights` | `0.08 0.12 0.15 0.18 0.20 0.23 0.25` m |
| `--tread_widths` | `0.25 0.30 0.35` m |
| `--repeats` (trials per tile) | 8 |
| `--time_out_s` | 12 |
| `--video [--video_length N] [--follow_env_id i]` | off; the video has a height-map overlay (`--no_heightmap_viz` to disable) |

Output: `eval/eval_stairs_<timestamp>.csv` (per trial) and `.md`. The `.md` table is per (speed, direction, step height) and reports success rate, time to cross, collisions, wheel-riser stumbles, falls, yaw drift (total and on the stairs only) and lateral offset.

### Multi-terrain course — `eval_multi_terrain_course.py`

One robot on a single long straight course with a fixed forward command. The course is a flat start, then pairs of stairs up/down at 0.10, 0.20, 0.25 and 0.30 m, two rough patches and a 25% ramp up and down. It's defined as `DEFAULT_COURSE` in [eval_terrains.py](source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/wheeled/deeprobotics_m20/eval_terrains.py).

```bash
python scripts/tools/eval_multi_terrain_course.py --headless \
    --task Stairs-Sighted-V3b-Deeprobotics-M20-v0 \
    --checkpoint logs/rsl_rl/deeprobotics_m20_stairs_sighted_v3b/<run> \
    --speed 0.7 --steering straight
```

- `--steering straight` (default): yaw-rate command 0 and no heading control, like a joystick held forward. Any drift comes from the policy.
- `--steering heading_hold`: the command's heading controller holds heading 0, as in training.
- Other options: `--friction` (fixed, default 1.0), `--max_time_s`, `--stuck_time_s`, `--no_video`, `--fps`, `--camera_eye`/`--camera_lookat`.

Output: `eval/multi_terrain_course_<timestamp>_<steering>_v<speed>/` containing:

- `course.mp4`: streamed to disk, so long runs are fine; includes the height-map overlay.
- `trajectory.csv`: includes per-wheel rim speed and contact.
- `course_plot.png`: height, sideways offset and heading along the course.
- `report.md`: per segment, whether it got through, time, speed, heading/sideways drift, stumbles, collisions, peak pitch/roll, and wheel use.

### Depth frames — `dump_depth.py`

A sanity check for the student env. The frozen teacher drives `Stairs-Student-Deeprobotics-M20-v0` for `--steps` policy steps. The script then writes a grid of noisy (training) vs clean depth images, and the observation group shapes, to `debug/depth/depth_grid.png`.

```bash
python scripts/tools/dump_depth.py --headless [--teacher <model.pt>] [--num_envs 16] [--steps 150]
```

## Exporting a policy

`play.py` exports to `exported/` next to the checkpoint. To export straight from the `.pt` without Isaac Sim:

```bash
python scripts/tools/export_onnx_fast.py \
    --checkpoint_path logs/rsl_rl/deeprobotics_m20_rough/<run>/model_<N>.pt \
    --robot m20 --output_path exported/m20_policy.onnx
```

Robot metadata (joint names, gains, default positions, action scales) is embedded in the ONNX file; add `--no_metadata` to skip it. For MuJoCo or real-robot deployment, see the deploy repos in the [Deep Robotics GitHub](https://github.com/DeepRoboticsLab).

## Where things are

```
command.md                                  # project plan (milestones, phases, acceptance criteria)
docs/                                       # research notes, v2 write-up, student setup steps
scripts/reinforcement_learning/rsl_rl/      # train.py, play.py (keyboard drive + overlay)
scripts/tools/
├── eval_stairs.py                          # stairs benchmark
├── eval_multi_terrain_course.py            # mixed-terrain course
├── heightmap_overlay.py                    # height-map video overlay used by the evals
├── dump_depth.py                           # student depth-frame dump
└── convert_terrain_mesh.py                 # scan OBJ -> terrain USD
source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/
├── mdp/stairs.py                           # stair-aware rewards, terminations, curriculum
├── mdp/depth.py                            # depth observation + noise model
└── config/wheeled/deeprobotics_m20/
    ├── rough_env_cfg.py, flat_env_cfg.py   # upstream blind tasks
    ├── stairs_env_cfg.py                   # Teacher, Sighted v1/v2/v3/v3b, Student
    ├── eval_terrains.py                    # stairs benchmark + course terrain builders (shared)
    ├── eval_play_env_cfg.py                # Stairs-Bench / Course keyboard tasks
    └── skatepark_env_cfg.py                # scanned-terrain tasks
```

## Troubleshooting

- **The script exits after a few seconds with no output.** The first Isaac Sim launch in a fresh session sometimes does nothing; run it again. If output is redirected to a file, set `PYTHONUNBUFFERED=1`, or tracebacks can be lost on shutdown.
- **`ModuleNotFoundError: No module named 'isaaclab'`** when using `isaaclab.sh`: a virtualenv is active. Run `unset VIRTUAL_ENV CONDA_PREFIX`.
- **`CXXABI_1.3.15 not found`** (Conda): run `python scripts/tools/setup_conda_runtime.py`, then deactivate and reactivate the environment.
- **Missing robot URDF/USD**: `git submodule update --init --recursive`.
- **Skatepark task fails to load its terrain**: check that `SKATEPARK_USD_PATH` in `skatepark_env_cfg.py` points at a file under `mesh/` that exists.
- **USD cache filling the disk**: `rm -rf /tmp/IsaacLab/usd_*`.
- **Pylance can't find Isaac Lab modules**: add `source/rl_training` and `<isaaclab>/source/isaaclab{,_assets,_rl,_tasks}` to `python.analysis.extraPaths` in `.vscode/settings.json`.

## Acknowledgements

Based on [DeepRoboticsLab/rl_training](https://github.com/DeepRoboticsLab/rl_training), which uses code from [fan-ziqi/robot_lab](https://github.com/fan-ziqi/robot_lab).
