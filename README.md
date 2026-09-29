# rl_training — M20 stairs

[![IsaacSim](https://img.shields.io/badge/IsaacSim-5.1.0-silver.svg)](https://docs.omniverse.nvidia.com/isaacsim/latest/overview.html)
[![Isaac Lab](https://img.shields.io/badge/IsaacLab-2.3.2-silver)](https://isaac-sim.github.io/IsaacLab)
[![RSL-RL](https://img.shields.io/badge/RSL--RL-5.0.1-silver)](https://github.com/leggedrobotics/rsl_rl)
[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://docs.python.org/3/whatsnew/3.11.html)
[![License](https://img.shields.io/badge/license-BSD%203--Clause-blue.svg)](https://opensource.org/license/bsd-3-clause)

Stair-climbing and rough-terrain policies for the **DEEP Robotics M20** wheeled quadruped, trained in Isaac Lab. Fork of [DeepRoboticsLab/rl_training](https://github.com/DeepRoboticsLab/rl_training).

It adds:
- **Sighted policies**: read a height map around the robot, with stair-aware rewards and curriculum.
- **Depth students**: distilled from a sighted policy so they only need proprioception and a depth camera.
- **Evals**: a stairs benchmark and a mixed-terrain course, with reports and video.
- **Keyboard driving** on generated terrain and real-world scans.

## Install

Needs Isaac Sim 5.1.0 and Isaac Lab 2.3.2 (Python 3.11), installed per the [Isaac Lab guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html).

```bash
git clone --recurse-submodules <this-repo-url> rl_training && cd rl_training
python -m pip install -e source/rl_training          # also installs RSL-RL 5.0.1
python scripts/tools/setup_conda_runtime.py          # Conda only: fixes Isaac Sim's C++ runtime
python scripts/tools/list_envs.py                    # check: lists the tasks
```

Without Conda, replace `python` with `<isaaclab>/isaaclab.sh -p` in every command.

## Tasks

| Task ID | What it is |
|---|---|
| `Rough-Deeprobotics-M20-v0` | Blind (proprioception only), upstream task |
| `Stairs-Sighted-V3b-Deeprobotics-M20-v0` | Sighted: proprioception + height map, stair terrain and curriculum |
| `Stairs-Student-Deeprobotics-M20-v0` | Depth student: proprioception + depth camera, distilled from the sighted policy |
| `Stairs-Bench-V3b-Deeprobotics-M20-Play-v0`, `Course-V3b-Deeprobotics-M20-Play-v0` | Keyboard driving on the eval terrains |
| `Skatepark-V3b-Deeprobotics-M20-Play-v0` | Keyboard driving on a real-world scan |

The full list is in [`deeprobotics_m20/__init__.py`](source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/wheeled/deeprobotics_m20/__init__.py). Most tasks have a `-Play-v0` version (fewer robots, no pushes).

## Train and play

```bash
# train (logs/checkpoints: logs/rsl_rl/<experiment>/<run>/)
python scripts/reinforcement_learning/rsl_rl/train.py --task=<task> --headless
#   smoke test: --num_envs 16 --max_iterations 2
#   resume:     --resume --load_run <run> --checkpoint model_<N>.pt

# watch a policy (also exports policy.pt / policy.onnx to <run>/exported/)
python scripts/reinforcement_learning/rsl_rl/play.py --task=<task>-Play-v0 --checkpoint <run>/model_<N>.pt
#   --video --video_length 200   record an mp4 (keep short: frames are held in RAM)
#   --keyboard                   drive one robot (GUI only)

tensorboard --logdir=logs
```

**Keyboard:** arrows = move, Z / X = turn, L = stop, `=` / `-` = faster / slower, N / B = next / previous terrain tile.

## Evaluate

Both scripts take a checkpoint file or a run folder (then they use its latest checkpoint), and write to `eval/`. Pass the same `--task` the policy was trained on.

```bash
# stairs benchmark: many robots, one stair tile each, fixed forward command
python scripts/tools/eval_stairs.py --headless --task <task> --checkpoint <run>
#   --speeds 0.5 1.0   --step_heights ...   --repeats 8   --video

# one robot on a long mixed course (stairs, rough, ramps); always records course.mp4
python scripts/tools/eval_multi_terrain_course.py --headless --task <task> --checkpoint <run> \
    --speed 0.7 --steering heading_hold        # or --steering straight (no heading correction)
```

Reports give success rate, falls, collisions, stumbles and heading drift. Videos show the policy's inputs: the height map and, for depth students, the depth image the network received (invalid pixels in magenta). For students, `--clean_depth` turns the depth noise off.

## Depth students (teacher → student)

A policy that sees terrain is trained in two stages:

1. **Teacher**: a sighted policy trained with RL on a perfect height map. It sets the upper bound, so get its behavior right first.
2. **Student**: learns to copy the frozen teacher's actions using only what the robot has: proprioception (57 dims) and a 64×36 depth image. The image goes through a CNN, then a GRU (memory), then an MLP. The student drives the robot during training, and the teacher labels every step with its own action (MSE loss, no reward).

Placeholders: `<teacher_task>` / `<teacher_experiment>` / `<teacher_run>` / `<N>` are the teacher's task, log folder name, run folder and checkpoint iteration; `<teacher_name>` is any short name for it.

```bash
# 1. train and evaluate the teacher (iterate until it behaves well)
python scripts/reinforcement_learning/rsl_rl/train.py --task=<teacher_task> --headless
python scripts/tools/eval_stairs.py --headless --task <teacher_task> --checkpoint logs/rsl_rl/<teacher_experiment>/<teacher_run>
python scripts/tools/eval_multi_terrain_course.py --headless --task <teacher_task> --checkpoint logs/rsl_rl/<teacher_experiment>/<teacher_run>

# 2. make the teacher checkpoint visible to the student (train.py only looks in the student's log folder)
mkdir -p logs/rsl_rl/deeprobotics_m20_stairs_student
ln -sfn ../<teacher_experiment>/<teacher_run> logs/rsl_rl/deeprobotics_m20_stairs_student/<teacher_name>

# 3. quick checks: network shapes (prints [PASS]), depth frames (debug/depth/), 2-iteration run
python scripts/tools/test_student_cnn.py --headless
python scripts/tools/dump_depth.py --headless --teacher logs/rsl_rl/<teacher_experiment>/<teacher_run>/model_<N>.pt
python scripts/reinforcement_learning/rsl_rl/train.py --task=Stairs-Student-Deeprobotics-M20-v0 --headless \
    --num_envs 16 --max_iterations 2 --load_run <teacher_name> --checkpoint model_<N>.pt

# 4. distil (about an hour; 2048 envs fit on a 12 GB GPU)
python scripts/reinforcement_learning/rsl_rl/train.py --task=Stairs-Student-Deeprobotics-M20-v0 --headless \
    --num_envs 2048 --load_run <teacher_name> --checkpoint model_<N>.pt

# 5. evaluate and watch the student (same evals as the teacher; compare the reports)
python scripts/tools/eval_stairs.py --headless --task Stairs-Student-Deeprobotics-M20-v0 \
    --checkpoint logs/rsl_rl/deeprobotics_m20_stairs_student/<student_run>
python scripts/tools/eval_multi_terrain_course.py --headless --task Stairs-Student-Deeprobotics-M20-v0 \
    --checkpoint logs/rsl_rl/deeprobotics_m20_stairs_student/<student_run>
python scripts/reinforcement_learning/rsl_rl/play.py --task=Stairs-Student-Deeprobotics-M20-Play-v0 \
    --checkpoint logs/rsl_rl/deeprobotics_m20_stairs_student/<student_run>/model_<N>.pt
```

- **Using a different teacher:** change the parent class of `DeeproboticsM20StairsStudentEnvCfg` in `stairs_env_cfg.py` to the teacher's env config. If the teacher's network differs, also update `teacher` in `agents/rsl_rl_distillation_cfg.py` to match it exactly.
- **Smoke test log:** look for `Loading model checkpoint from: .../<teacher_name>/model_<N>.pt` and a `Mean behavior loss` line.
- **Playing a student:** always pass the student's `--checkpoint`; the default points at the teacher.
- **Not supported yet:** exporting the student (its GRU isn't handled by the exporter) and sim-to-sim in MuJoCo. The depth camera is a placeholder until it's matched to the real sensor and mount.

## Export

```bash
python scripts/tools/export_onnx_fast.py --checkpoint_path <run>/model_<N>.pt --robot m20 --output_path m20_policy.onnx
```

Exports an MLP policy (not the depth student) without launching Isaac Sim. For MuJoCo / real-robot deployment, see the [Deep Robotics GitHub](https://github.com/DeepRoboticsLab).

## Code map

```
scripts/reinforcement_learning/rsl_rl/   train.py, play.py, cli_args.py
scripts/tools/                           evals, dump_depth.py, test_student_cnn.py, convert_terrain_mesh.py
source/rl_training/rl_training/
├── models/depth_cnn_gru.py              student network (CNN -> GRU -> MLP)
└── tasks/manager_based/locomotion/velocity/
    ├── mdp/                             rewards, terrain curriculum, depth observation + noise
    └── config/wheeled/deeprobotics_m20/ task configs, eval terrains, agents/ (training configs)
```

## Troubleshooting

- **Exits after a few seconds, no output:** the first Isaac Sim launch sometimes does nothing; run it again. When redirecting output, set `PYTHONUNBUFFERED=1`.
- **`No module named 'isaaclab'`:** a virtualenv is active; `unset VIRTUAL_ENV CONDA_PREFIX`.
- **Missing robot files:** `git submodule update --init --recursive`.
- **Out of GPU memory:** lower `--num_envs`.
- **`CUDNN_STATUS_NOT_INITIALIZED`:** a stray CUDA 13 cuDNN is shadowing torch's. Uninstall `nvidia-cudnn-cu13 nvidia-nccl-cu13 nvidia-cusparselt-cu13`, then `pip install --no-deps --force-reinstall nvidia-cudnn-cu12==9.7.1.26 nvidia-nccl-cu12==2.26.2 nvidia-cusparselt-cu12==0.6.3`.

## Acknowledgements

Based on [DeepRoboticsLab/rl_training](https://github.com/DeepRoboticsLab/rl_training), which uses code from [fan-ziqi/robot_lab](https://github.com/fan-ziqi/robot_lab).
