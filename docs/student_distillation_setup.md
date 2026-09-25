# Student network + distillation runner: setup steps

*2026-09-24. For whoever writes the student model and runner config. The env side is done: see [What already exists](#what-already-exists).*

Plan: sequential, DAgger-style distillation. A **frozen** sighted teacher (v2, `model_5300.pt`) drives a depth + proprioception **student**, and the student is trained to reproduce the teacher's actions. It uses RSL-RL 5.0.1's built-in `DistillationRunner` / `Distillation`. The only custom code is the student network (a CNN + GRU model) and a small config-conversion fix.

## What already exists

| Piece | Where |
|---|---|
| Student env | `Stairs-Student-Deeprobotics-M20-v0` (+ `-Play-v0`), class `DeeproboticsM20StairsStudentEnvCfg` in `config/wheeled/deeprobotics_m20/stairs_env_cfg.py` |
| Depth camera | `scene.depth_camera`: ray-cast, 64×36, ~87°×56° FOV, 10 Hz, clipped 0.1–2.5 m. **Mount pose is a placeholder** (`STUDENT_CAMERA_*` constants) |
| Depth noise model | `mdp/depth.py` (`depth_image`): noise ∝ depth, dropout, holes, edge (flying-pixel) drops, 0–2 step latency, frame drops. Active only while `observations.depth.enable_corruption` is True |
| Debug frames | `scripts/tools/dump_depth.py` → `debug/depth/depth_grid.png` (teacher drives, noisy vs clean frames) |
| Teacher checkpoint | `logs/rsl_rl/deeprobotics_m20_stairs_sighted_v2/2026-09-23_15-04-55/model_5300.pt` |
| Gym registration | already points at `agents/rsl_rl_distillation_cfg.py:DeeproboticsM20StairsStudentRunnerCfg`, which is **the file you create in step 2** |

Observation groups produced by the env (shapes verified with 16 envs):

| Group | Shape | Contents | Used by |
|---|---|---|---|
| `teacher` | (N, 288) | exact noise-free copy of v2's `policy` group (proprio + height scan) | teacher |
| `policy` | (N, 57) | student proprioception (v2 `policy` minus height scan, with noise) | student |
| `depth` | (N, 1, 36, 64) | depth image in [0, 1]; 0 = invalid pixel | student |
| `critic` | (N, 291) | unchanged, unused by distillation | none |

`dump_depth.py` already loads v2's actor **strictly** from the `teacher` group, so the teacher side is verified.

## Step 1: the student network

**Why custom:** RSL-RL 5.0.1 has `CNNModel` (CNN + MLP, no memory) and `RNNModel` (GRU/LSTM on 1D inputs only), but nothing that runs a CNN and then a GRU. You need one small class.

**Where:** `source/rl_training/rl_training/rsl_rl/models/depth_cnn_gru.py`. Create `models/__init__.py` exporting it. `rl_training/rsl_rl/` already holds the repo's custom AMP code. The config will reference it as `"rl_training.rsl_rl.models:DepthCNNGRUModel"` (RSL-RL resolves `module:Class` strings).

**What it must do:**

```python
from rsl_rl.models import CNNModel
from rsl_rl.modules import RNN


class DepthCNNGRUModel(CNNModel):
    """CNN over the depth image, concatenated with (normalized) proprio, then a GRU, then the MLP head."""

    is_recurrent = True

    def __init__(self, obs, obs_groups, obs_set, output_dim, hidden_dims=(512, 256, 128), activation="elu",
                 obs_normalization=False, distribution_cfg=None, cnn_cfg=None, cnns=None,
                 rnn_type="gru", rnn_hidden_dim=256, rnn_num_layers=1):
        # must be set BEFORE super().__init__: MLPModel builds its MLP from self._get_latent_dim()
        self.rnn_hidden_dim = rnn_hidden_dim
        super().__init__(obs, obs_groups, obs_set, output_dim, hidden_dims, activation,
                         obs_normalization, distribution_cfg, cnn_cfg, cnns)
        # GRU input = 1D obs dim (self.obs_dim) + CNN latent (self.cnn_latent_dim), both set by CNNModel
        self.rnn = RNN(self.obs_dim + self.cnn_latent_dim, rnn_hidden_dim, rnn_num_layers, rnn_type)

    def _get_latent_dim(self):
        return self.rnn_hidden_dim  # the MLP head consumes the GRU output

    def get_latent(self, obs, masks=None, hidden_state=None):
        x = super().get_latent(obs)                      # [normalized proprio, CNN(depth)]
        return self.rnn(x, masks, hidden_state).squeeze(0)

    # recurrent bookkeeping, same as rsl_rl.models.RNNModel
    def reset(self, dones=None, hidden_state=None):
        self.rnn.reset(dones, hidden_state)

    def get_hidden_state(self):
        return self.rnn.hidden_state

    def detach_hidden_state(self, dones=None):
        self.rnn.detach_hidden_state(dones)
```

**Things to know:**
- **Input shapes:** `Distillation` feeds **per-timestep** batches (`RolloutStorage.generator()` in distillation mode), so the CNN always sees `(N, 1, 36, 64)`. If you later fine-tune with **PPO**, recurrent mini-batches arrive as padded `(T, N, ...)` sequences. The CNN call then needs a flatten/reshape around it.
- **Export:** `CNNModel.as_onnx()` / `as_jit()` don't know about the GRU. ONNX export needs its own wrapper later (Milestone C); training doesn't need it.
- **Unit check before training:** build the env with 16 envs, instantiate the model from `env.get_observations()`, and run a forward, a `reset(dones)` and a second forward.

## Step 2: the runner config

Create `config/wheeled/deeprobotics_m20/agents/rsl_rl_distillation_cfg.py` with class **`DeeproboticsM20StairsStudentRunnerCfg`**. The name is already registered.

| Field | Value | Why |
|---|---|---|
| base class | `RslRlDistillationRunnerCfg` (`isaaclab_rl.rsl_rl`) | `class_name="DistillationRunner"` |
| `experiment_name` | `"deeprobotics_m20_stairs_student"` | log dir, and where the teacher is looked up (step 4) |
| `num_steps_per_env` | 24 | same as the teacher's PPO |
| `max_iterations` | ~2000–3000 to start | distillation converges much faster than RL |
| `save_interval` | 100 | |
| `obs_groups` | `{"student": ["policy", "depth"], "teacher": ["teacher"]}` | routes env groups to the two models |
| `load_run` / `load_checkpoint` | `"teacher_v2"` / `"model_5300.pt"` | see step 4 |
| `algorithm` | `RslRlDistillationAlgorithmCfg(num_learning_epochs=2, learning_rate=1e-3, gradient_length=15, max_grad_norm=1.0, loss_type="mse")` | `gradient_length` = GRU truncated-backprop steps |

**`teacher`** must reproduce v2's actor **exactly**, or the strict load fails:

```python
RslRlMLPModelCfg(
    hidden_dims=[512, 256, 128], activation="elu", obs_normalization=False,
    distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0, std_type="log"),
)
```

**`student`:** subclass `RslRlCNNModelCfg` in the same file to add the GRU fields, since `configclass` fields become constructor keyword arguments:

```python
@configclass
class DepthCNNGRUModelCfg(RslRlCNNModelCfg):
    class_name: str = "rl_training.rsl_rl.models:DepthCNNGRUModel"
    rnn_type: str = "gru"
    rnn_hidden_dim: int = 256
    rnn_num_layers: int = 1

student = DepthCNNGRUModelCfg(
    hidden_dims=[512, 256, 128], activation="elu", obs_normalization=True,
    distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=0.1, std_type="scalar"),
    cnn_cfg=RslRlCNNModelCfg.CNNCfg(output_channels=[16, 32, 32], kernel_size=[5, 4, 3], stride=[2, 2, 1],
                                    activation="elu", flatten=True),
)
```

- `obs_normalization=True` normalizes the proprio part only; the depth image is already in [0, 1].
- The Gaussian head is required: `Distillation.act()` samples `student(obs, stochastic_output=True)`, which gives a little exploration during rollouts. Loss is computed on the deterministic output.
- The CNN is the [16,32,32] / [5,4,3] / [2,2,1] default that three sources in `docs/stairs_research_notes.md` converge on. On a 36×64 image it gives a 32×5×12 = 1920-dim latent.

## Step 3: strip the deprecated model-config keys

`RslRlMLPModelCfg.to_dict()` also emits `stochastic`, `init_noise_std`, `noise_std_type` and `state_dependent_std` (checked: `{'stochastic': {}, 'init_noise_std': {}, 'noise_std_type': 'scalar', 'state_dependent_std': False, ...}`). RSL-RL passes the model config straight into the constructor as `**kwargs`, so `MLPModel` raises a `TypeError`. This repo's `train.py` uses `cli_args.convert_rsl_rl_cfg_dict` instead of Isaac Lab's `handle_deprecated_rsl_rl_cfg`, so nothing strips them. Add this at the **top** of `convert_rsl_rl_cfg_dict` in `scripts/reinforcement_learning/rsl_rl/cli_args.py`, before its early return:

```python
for name in ("student", "teacher"):
    model_cfg = cfg_dict.get(name)
    if isinstance(model_cfg, dict):
        for key in ("stochastic", "init_noise_std", "noise_std_type", "state_dependent_std"):
            model_cfg.pop(key, None)
```

It's a no-op for the existing PPO configs (their `actor`/`critic` dicts are built by the converter and never contain these keys). The rest of the converter still turns the unused legacy `policy` field into dummy `actor`/`critic` entries, which `Distillation` ignores.

## Step 4: make the teacher checkpoint findable

For distillation, `train.py` always loads a checkpoint from `logs/rsl_rl/<experiment_name>/<load_run>/<load_checkpoint>`, i.e. inside the **student's** experiment folder. Point a stable name at the v2 run:

```bash
cd /workspace/rl_training
mkdir -p logs/rsl_rl/deeprobotics_m20_stairs_student
ln -s ../deeprobotics_m20_stairs_sighted_v2/2026-09-23_15-04-55 logs/rsl_rl/deeprobotics_m20_stairs_student/teacher_v2
```

`Distillation.load()` sees `actor_state_dict` in that file and loads **only** the teacher (iteration reset to 0). Student runs then land beside the symlink as timestamped folders.

Resuming a student run later: set `load_run` to that run's folder. The loader then restores student, teacher and optimizer from the distillation checkpoint.

## Step 5: smoke test, then train

```bash
cd /workspace/rl_training
unset VIRTUAL_ENV CONDA_PREFIX
# smoke test
PYTHONUNBUFFERED=1 /workspace/isaaclab/isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Stairs-Student-Deeprobotics-M20-v0 --num_envs 16 --max_iterations 2 --headless
# full run (inside tmux)
PYTHONUNBUFFERED=1 /workspace/isaaclab/isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Stairs-Student-Deeprobotics-M20-v0 --num_envs 4096 --headless 2>&1 | tee train_student.log
```

- **In the log, look for:** `Loading model checkpoint from: .../teacher_v2/model_5300.pt`, both models printed, and a `behavior` loss that falls steadily.
- **Env count:** 4096 is a guess. The camera adds 2304 rays per env per frame. The 6144-env teacher runs may not fit alongside it, so check GPU memory on the smoke test and scale up.
- **Curriculum:** it stays on. The *student* drives, so terrain levels show whether the student, not the teacher, can climb.

## After training (my side)

- **`eval_stairs.py` and `eval_multi_terrain_course.py`** load the student as-is: `OnPolicyRunner` builds whichever algorithm the config names, and `get_inference_policy()` returns the student. Both scripts only turn off **policy** noise, so the depth noise stays on. That matches Milestone B's "with depth DR enabled" criterion, but I'll add a `--clean_depth` flag for comparison.
- **Milestone B bar:** within 10 points of the teacher on every step height, both directions. Then the latency, dropout and camera-tilt sweeps, and a "depth goes blank mid-flight" test.

## Open items

- [ ] **Real camera mount pose, model, FPS and latency** (command.md Phase 5, **[ASK HUMAN]**). The current mount is a placeholder, and the student must be retrained after it changes.
- [ ] Not modelled yet: self-occlusion by legs/wheels, camera extrinsics jitter, blur (command.md 7.1).
- [ ] Teacher choice: v2 now. If v3b turns out better, change the student env's parent class to `DeeproboticsM20StairsSightedV3bEnvCfg` **and** the symlink together (their teacher obs are identical, 288 dims).
- [ ] Optional after distillation: PPO fine-tuning of the student (command.md 8.4). Needs the padded-sequence handling from step 1.
