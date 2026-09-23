# command.md — Train a depth-aware stair-climbing policy for the DEEP Robotics M20 (wheeled quadruped)

You are a coding agent working in the repo `https://github.com/DeepRoboticsLab/rl_training` (Isaac Lab 2.3.2, Isaac Sim 5.1.0, RSL-RL 5.0.1, Python 3.11). The goal is to take the existing **blind** M20 locomotion task and produce, in stages, a policy that reliably climbs stairs and drives on rough/flat terrain, using a **CTS-MoE-style architecture** (Affonso et al. 2026, arXiv:2606.19633 — see `docs/stairs_research_notes.md` §11, the human's chosen direction as of 2026-09-22): a dense mixture-of-experts actor with a perception-based router, and a multi-critic with one value head per task, trained in a single stage that eventually runs teacher (privileged) and student (deployable) encoders concurrently — no separate freeze-then-distill phase.

This is being built up **sequentially**, not as the full multi-task depth-camera version on day one:

1. **Milestone A — "It climbs stairs" (privileged, 3-task MoE):** an M20 policy using the MoE actor + multi-critic architecture, trained on a *small, fixed task set* — `flat_rough`, `stair_ascent`, `stair_descent` (see Phase 3.4) — using a perfect privileged height scan. No depth camera, no additional tasks yet. This validates the architecture itself (does the router actually differentiate by task? does the multi-critic train stably? does it beat the blind baseline on stairs?) before adding complexity.
2. **Milestone B — "It climbs stairs from depth, concurrently" (deployable, expandable):** add a student encoder (proprioception + simulated depth image) alongside the existing teacher encoder and train them **concurrently** (per-step encoder-alignment loss, no DAgger/β-mixing distillation phase). Only start this after Milestone A's acceptance criteria pass. Expanding the task set beyond the original three (obstacles, gaps, ramps as their own task, etc.) is a deliberate, human-gated decision at this stage, not automatic.
3. **Milestone C — sim-to-sim / deployment readiness:** export the full MoE actor (all experts + router) with gating diagnostics, MuJoCo sim2sim check, and a real-robot sensor pipeline plan.

Milestone A is the priority. Do not start Milestone B (depth, concurrent training, additional tasks) until Milestone A's acceptance criteria are met. Within Milestone A, do not expand the task set beyond the three listed above without asking the human first.

---

## 0. Ground rules for the agent

- Work on a new git branch (`feature/m20-stairs`). **Never edit the existing `Rough-Deeprobotics-M20-v0` / `Flat-Deeprobotics-M20-v0` behaviour.** Add new config classes and new gym IDs that subclass the existing ones.
- Before using any Isaac Lab / RSL-RL API, **read the installed source** (`python -c "import isaaclab, rsl_rl; print(isaaclab.__file__, rsl_rl.__file__)"`) rather than trusting memory or online docs. APIs differ between versions (e.g. quaternion order is `(w,x,y,z)` in Isaac Lab 2.x but `(x,y,z,w)` in 3.x; RSL-RL ≥4 changed the distillation config classes).
- Every new task gets a smoke test first: `--num_envs 16 --max_iterations 2 --headless`. Only then launch a full run.
- Keep an `EXPERIMENTS.md` at repo root. For every run log: date, git commit, task ID, key config diffs (use `scripts/tools/compare_runs.py`), iterations, final metrics, and a one-line conclusion. Never report a result you did not measure.
- Stop and ask the human at every item marked **[ASK HUMAN]**.
- Prefer small, reviewable commits: one logical change per commit.

### Key files you will touch (verified in the repo)

```
source/rl_training/rl_training/
├── assets/deeprobotics.py                         # DEEPROBOTICS_M20_CFG (PD gains, limits, init pose)
└── tasks/manager_based/locomotion/velocity/
    ├── velocity_env_cfg.py                         # base scene, obs, events (DR), terminations, curriculum
    ├── mdp/{rewards,observations,events,curriculums,commands}.py
    └── config/wheeled/deeprobotics_m20/
        ├── __init__.py                             # gym.register(...) — add new IDs here
        ├── rough_env_cfg.py                        # DeeproboticsM20RoughEnvCfg — subclass this
        └── agents/rsl_rl_ppo_cfg.py                # PPO runner cfg — subclass this
scripts/reinforcement_learning/rsl_rl/{train,play}.py
scripts/tools/{export_onnx_fast.py, compare_runs.py, list_envs.py}
```

### Facts about the current M20 task you must account for

- The **policy is blind**: `rough_env_cfg.py` sets `observations.policy.height_scan = None` and `base_lin_vel = None`. The **critic already receives `height_scan`** (asymmetric actor-critic).
- Terrain is Isaac Lab's `ROUGH_TERRAINS_CFG`, which **already includes** `pyramid_stairs` and `pyramid_stairs_inv` (20% each, `step_height_range=(0.05, 0.23)`, `step_width=0.3`, 8 m × 8 m tiles, 10 difficulty rows × 20 columns). So the existing blind policy has seen stairs.
- Robots spawn at the tile centre. On `pyramid_stairs` the centre is the **top** platform (walking out = descending); on `pyramid_stairs_inv` the centre is the **bottom** (walking out = ascending).
- Terrain curriculum is `mdp.terrain_levels_vel`: promote if the robot travels > half a tile, demote if it travels < half the commanded distance. It also drives a global `gait_level` used by the M20 rotation-gait rewards.
- Rewards likely to conflict with stairs:
  - `flat_orientation_l2` weight **−50** (penalises any body pitch; a robot on stairs must pitch).
  - `bad_orientation_penalty` weight **−1000** and termination `bad_orientation_2` both trigger when `|projected_gravity_xy| > 0.7` (~44°). A 25 cm rise / 30 cm tread staircase is ~40°, so this is near the limit.
  - `feet_stumble` (wheel hitting a vertical riser) exists but weight is 0.
- Action space: joint position targets for 12 leg joints (hipx scale 0.125, others 0.25) + joint **velocity** targets for 4 wheels (scale 5.0).
- Real robot spec: max continuous stair height 25 cm, single step up to 80 cm. The stock M20 lists dual 96-line LiDAR and dual wide-angle cameras — **no depth camera is listed**, which matters for Milestone B.

---

## Phase 1 — Setup and baseline (do this first)

1. Install per README; run `python scripts/tools/list_envs.py` and a 1-iteration smoke run of `Rough-Deeprobotics-M20-v0`.
2. **[ASK HUMAN]** Ask for an existing trained blind M20 checkpoint. If none exists, train `Rough-Deeprobotics-M20-v0` to convergence (monitor terrain level and tracking reward in TensorBoard) and record it in `EXPERIMENTS.md`.
3. Build an evaluation script `scripts/tools/eval_stairs.py` that:
   - Creates a stairs-only evaluation terrain with fixed step heights: 0.08, 0.12, 0.15, 0.18, 0.20, 0.23, 0.25 m, and tread widths 0.25 / 0.30 / 0.35 m.
   - Commands a fixed forward velocity (0.5 m/s and 1.0 m/s) with zero lateral/yaw, heading pointed across the stairs.
   - Reports per step height, separately for **ascent** and **descent**: success rate (reached the far platform without termination within T seconds), mean time to cross, number of base/knee collisions, number of wheel–riser stumbles, and falls.
   - Writes a CSV + a short markdown table to `eval/`.
   - Optionally records video (`--video`).
4. Evaluate the blind baseline and put the table in `EXPERIMENTS.md`. This is the number to beat.

**Deliverable:** baseline table. If the blind policy already climbs ≥90% at 0.20 m, tell the human — the focus may shift to higher steps, descent, and depth robustness.

---

## Phase 2 — Research survey (time-boxed: ~1–2 hours of reading)

Read the following and write `docs/stairs_research_notes.md` with, for each source: terrain setup, command/heading scheme, reward terms (with weights where given), observation design, network architecture, domain randomization list, depth noise model, and anything specific to wheeled-legged robots. End the file with a "what we will adopt" section.

Core sources:

- **Extreme Parkour** (Cheng et al., ICRA 2024) — scandot teacher → depth CNN+GRU student via DAgger; student actor initialized from teacher. Code: https://github.com/chengxuxin/extreme-parkour · Paper: https://arxiv.org/abs/2309.14341
- **Isaac Lab port of Extreme Parkour** (teacher + student tasks for Go2 on Isaac Lab; closest structural template for this repo): https://github.com/CAI23sbP/Isaaclab_Parkour
- **Legged Locomotion in Challenging Terrains using Egocentric Vision** (Agarwal et al., 2022) — real depth preprocessing (crop, hole fill, downsample to 58×87), ~100 ms camera period, latency modelling: https://arxiv.org/abs/2211.07638
- **Robot Parkour Learning** (Zhuang et al., 2023): https://arxiv.org/abs/2309.05665
- **Humanoid Parkour Learning** (Zhuang et al., 2024) — concrete depth sim noise (clipping, multi-level Gaussian noise, random artifacts) and matching real-side filters: https://arxiv.org/abs/2406.10759
- **Parkour in the Wild** (Rudin et al., 2025) — multi-expert distillation + RL fine-tuning, depth edge noise and holes: https://arxiv.org/abs/2505.11164
- **RL for Blind Stair Climbing with Legged and Wheeled-Legged Robots** (Chamorro et al., ICRA 2024) — wheeled-legged stair task formulation, privileged terrain info in the critic: https://arxiv.org/abs/2402.06143
- **Learning robust autonomous navigation and locomotion for wheeled-legged robots** (Lee et al., Science Robotics 2024), doi:10.1126/scirobotics.adi9641
- **legged_lab_v3** (Isaac Lab, G1 humanoid, depth history with latency and self-occlusion, stair curriculum with separate up/down promotion): https://github.com/lessonllab/legged_lab_v3
- Upstream of this repo: https://github.com/fan-ziqi/robot_lab
- Isaac Lab docs: `RayCasterCameraCfg`, `TiledCameraCfg`/`CameraCfg`, `MeshPyramidStairsTerrainCfg`, `RslRlDistillationRunnerCfg`.

Also search for newer (2025–2026) work on perceptive **wheeled-legged** stair climbing and note anything relevant; list what you found and how confident you are in each source.

---

## Phase 3 — Stairs task environment (Milestone A, part 1)

Create `config/wheeled/deeprobotics_m20/stairs_env_cfg.py` with `DeeproboticsM20StairsTeacherEnvCfg(DeeproboticsM20RoughEnvCfg)` and register `Stairs-Teacher-Deeprobotics-M20-v0` (+ a `-Play-v0` variant with fewer envs, no pushes, fixed commands).

### 3.1 Terrain

Build a new `TerrainGeneratorCfg` (don't mutate the shared `ROUGH_TERRAINS_CFG` object — deep-copy it or define a new one):

| Sub-terrain | Proportion | Notes |
|---|---|---|
| `pyramid_stairs_inv` (ascent) | 0.30 | step height 0.05 → 0.26 m across difficulty rows |
| `pyramid_stairs` (descent) | 0.25 | same range |
| stairs with different tread | 0.15 | if `step_width` is a scalar in the installed version, add 2 extra entries with 0.25 m and 0.35 m |
| `random_rough` | 0.10 | keep general robustness |
| `hf_pyramid_slope` / `_inv` | 0.10 | ramps |
| flat | 0.10 | prevents forgetting flat driving |

- Use platform widths large enough for the M20 (0.82 m long): ≥ 2.0 m.
- Optional later: a custom straight-staircase terrain (a single flight of 6–12 steps with a landing), which matches real buildings better than pyramids. Implement as a custom `SubTerrainBaseCfg` + mesh function only after the pyramid version works.
- Keep `max_init_terrain_level` low (e.g. 3) so early training isn't dominated by failures.

### 3.2 Commands ("the controller that makes it go forward")

The "controller" during training is the velocity command generator. For stairs:

- Mostly forward commands: `lin_vel_x ∈ [0.3, 1.2]`, reduced `lin_vel_y ∈ [-0.3, 0.3]`, `ang_vel_z ∈ [-1.0, 1.0]`.
- Keep heading control on so the robot faces a random direction and must turn toward/along stairs; also keep a fraction (~10–20%) of zero-velocity envs so it learns to stand on stairs.
- Consider a small fraction of backward commands only if a rear depth sensor will exist (see Phase 6). Otherwise note that descending backwards will be blind.
- Leave room for a future high-level navigation layer; do not bake waypoints into the policy.

### 3.3 Checks

- Visualize the terrain with the play script and a few random-action robots; confirm step heights per row and that spawn points sit on platforms.
- Confirm the terrain curriculum promotes/demotes as expected on stairs (log mean terrain level per sub-terrain type if feasible), **decoupled per task** (3.4) — track ascent-tile and descent-tile difficulty independently rather than one aggregate metric, since Phase 1's baseline already shows a large ascent/descent asymmetry that a coupled curriculum could mask (`docs/stairs_research_notes.md` §9's up/down-decoupled promotion rule).

### 3.4 Task definition (for the multi-critic, Phase 4.2)

Define exactly **three tasks** for Milestone A — do not add more without asking the human first:

| Task ID | Terrain source | Notes |
|---|---|---|
| `flat_rough` | flat + `random_rough` + ramps/slopes | "drive well and stay upright on everything that isn't stairs" |
| `stair_ascent` | `pyramid_stairs_inv` | walking out from centre = ascending |
| `stair_descent` | `pyramid_stairs` | walking out from centre = descending |

Assign each env's task ID from which terrain sub-type its spawn tile belongs to (reuse the per-column terrain-type mapping `eval_stairs.py` already built for fixed-geometry evaluation). Task ID is read by the reward manager (routes to per-task reward weights) and the multi-critic (routes to the right value head) **during training only**. Per CTS-MoE (`docs/stairs_research_notes.md` §11), it must **not** be fed to the actor or the router — the router sees perception only, so gating has to learn to separate behavior from observations alone, the same way it will have to at deployment with no privileged task label available.

### 3.5 Sanity check before Phase 4's architecture work (2026-09-22, human-requested)

Before building the multi-critic/MoE actor, the human asked to validate the much smaller claim
"can a plain single-MLP actor-critic climb these stairs at all, if it can see the terrain" — i.e.
isolate whether sightedness alone (independent of any multi-task architecture) fixes Phase 1's
~0% blind-ascent result. Added `DeeproboticsM20StairsSightedEnvCfg` (`stairs_env_cfg.py`):
identical to 3.1-3.4's terrain/commands/task-ID plumbing, single change is re-enabling the
existing (non-forward-biased) height-scan term for the actor, same plain single-MLP runner cfg
already registered for 3's own smoke test. One reward function, one critic, one actor — no
per-task routing yet even though the task-ID plumbing exists underneath. Registered as
`Stairs-Sighted-Deeprobotics-M20-v0`. This is a spike, not a Milestone A deliverable in its own
right: if it works, it derisks Phase 4's central assumption before adding multi-critic/MoE
complexity on top; if it doesn't, that's a stronger signal that reward-shaping (4.4) or the
forward-biased scan redesign (4.1) is doing more work than sightedness alone, before either gets
built inside the more expensive MoE architecture. **[Result pending — see `EXPERIMENTS.md` /
`docs/stairs_research_notes.md` once the run in progress finishes and gets `eval_stairs.py`'d.]**

---

## Phase 4 — Privileged multi-task teacher training, MoE architecture (Milestone A, part 2)

### 4.1 Observations

- **Actor:** existing proprioception + `height_scan` (re-enable it for the policy) — this also serves as the router's perception input (4.3). Use a forward-biased scan grid: roughly 1.6–2.0 m forward, 0.4–0.6 m behind, ±0.5–0.6 m lateral, 0.1 m resolution; verify the ray-caster offset/pattern so the grid is centred correctly relative to `base_link`. Add small noise (±2–5 cm) and clip.
- Add `base_lin_vel` to the actor only if it will be available on the real robot via a state estimator; otherwise keep it critic-only. **[ASK HUMAN]** if unsure.
- **Critics (one per task, 4.2):** everything in the actor + clean height scan + privileged terms (ground friction, added base mass, CoM offset, motor-strength factor, foot/wheel contact forces). Write observation functions in `mdp/observations.py` that read the randomized values stored by events. Task ID (3.4) is used to route training but is **not** a network input anywhere.
- Optionally follow Extreme Parkour / RMA and add a **proprioceptive history encoder** (last ~10–20 steps) whose latent is regressed from privileged params, feeding both actor and router. Only do this if the simple version plateaus.

### 4.2 Multi-critic (`docs/stairs_research_notes.md` §11)

- One value head per task from 3.4 (`V_flat_rough`, `V_stair_ascent`, `V_stair_descent`), each trained only on trajectories whose env task ID matches. A shared trunk with separate output heads is fine.
- Apply per-task return normalization so a task with naturally larger/smaller returns doesn't dominate or vanish in a shared advantage scale (CTS-MoE uses POPArt). **First check whether the installed RSL-RL 5.0.1 has any multi-head-critic or return-normalization support** (`grep -ri "popart\|multi.critic\|value_head" $(python -c "import rsl_rl, os; print(os.path.dirname(rsl_rl.__file__))")`); if not, a simple per-task running mean/std normalizer is an acceptable first cut.
- Log per-task mean return, episode length, and termination breakdown separately from the start — this is the main signal for whether the multi-critic is preventing interference vs. a single shared critic.
- **Ablation to run once, not optional:** a control run with a single shared critic (everything else identical) to confirm the multi-critic measurably helps before carrying the extra complexity into Milestone B. If it doesn't help at 3 tasks, say so plainly in `EXPERIMENTS.md` and consider dropping it (see the acceptance criteria below).

### 4.3 MoE actor & perception-based router (`docs/stairs_research_notes.md` §11)

- Actor is **E experts**, each an MLP `[512, 256, 128]` (same trunk size as the original single-MLP plan). CTS-MoE uses E=6 for six tasks; **start smaller, E=3–4 for our three tasks**, and only grow E if routing collapses to fewer active experts than tasks.
- Router: MLP `[512, 256]` → softmax over E experts, taking the **same perception input the actor gets** (height scan + proprioception; no task ID, no privileged-only terms) — this rehearses depth-based deployment gating even before depth exists.
- Log per-task expert-usage histograms (mean gate weight per expert, broken out by task ID) throughout training. This is the key Milestone A diagnostic: if gating doesn't differentiate `stair_ascent` from `flat_rough`, the MoE isn't doing anything a single MLP wouldn't.
- **Ablation to run once, not optional (pairs with 4.2's):** a control run with a single plain `[512,256,128]` MLP actor (multi-critic kept) to isolate the MoE actor's contribution specifically. Per Lee et al. (`docs/stairs_research_notes.md` §8), fully emergent mode-switching *without* an explicit MoE gate is a documented possibility at this task scale — don't assume the MoE is pulling weight without checking.

### 4.4 Rewards — per task, changes to evaluate one at a time

Start from the M20 rough rewards, route per-task-specific weights via task ID, then:

1. Reduce `flat_orientation_l2` (e.g. −50 → −5) **on stair tasks only** (or replace with a terrain-slope-relative penalty); keep it at −50 on `flat_rough`. Measure body pitch on 0.25 m stairs before choosing the stair-task value.
2. Relax termination `bad_orientation_2` / `bad_orientation_penalty` threshold from 0.7 to ~0.85 **on stair tasks only**. Check against measured pitch.
3. Enable `feet_stumble` (wheel hitting a riser) at a small negative weight (e.g. −0.5 to −2.0) on stair tasks; tune jointly with wheel-ground friction DR — per `docs/stairs_research_notes.md` §7/§8, the grip-vs-step tradeoff at the wheel-riser contact is the dominant wheeled-specific failure mode (matches Phase 1's "heavy wheel-riser stumbling" finding), more important than the stumble weight alone.
4. Make sure `undesired_contacts` covers base, hips and **knees/thighs** on all tasks — knee-edge collisions are the classic stair failure. Consider terminating on base-to-stair contact.
5. Base height: if enabled, compute it relative to the terrain under the base (the existing `height_scanner_base` sensor), not world z.
6. Consider a foot/wheel clearance reward over edges on stair tasks (reward wheel height above the terrain height sampled under the wheel when not in ground contact), as in Extreme Parkour's edge terms, adapted for wheels.
7. Add waypoint/progress reward terms (`docs/stairs_research_notes.md` §7 Chamorro, §11 CTS-MoE) on stair tasks **layered on top of** the existing velocity-tracking reward, not replacing it — lower risk, stays compatible with `eval_stairs.py`.
8. Keep the rotation-gait rewards on `flat_rough`; check the `gait_level` coupling still makes sense once terrain-level promotion is task-decoupled (3.3).

For each change: short run (~2–3k iterations), evaluate with `eval_stairs.py`, log per-task in `EXPERIMENTS.md`, keep or revert.

### 4.5 Training

- Option 1: train from scratch with the MoE actor + multi-critic.
- Option 2: warm-start each expert from the blind checkpoint's single-MLP weights (replicate the same weights E times) and zero-init the new height-scan input columns; let the router learn differentiation from there. Try this if from-scratch is slow.
- Consider CTS-MoE's staged DR sequencing (`docs/stairs_research_notes.md` §11): converge with the simulator's idealized actuators first, then resume training with the explicit delayed-PD actuator model and fuller DR, rather than applying all DR from step zero.
- Start with the existing PPO hyperparameters (per-expert `[512,256,128]`, lr 1e-3 adaptive, 24 steps/env). Use 4096 envs (revisit downward if MoE/multi-critic overhead forces it). Monitor: terrain level per task (3.3), tracking reward, episode length, termination breakdown, and the router/expert-usage diagnostics from 4.3.

### Milestone A acceptance criteria

On `eval_stairs.py` with 0.5 m/s forward command:
- Ascent success ≥ 90% for steps ≤ 0.20 m and ≥ 75% at 0.25 m.
- Descent success ≥ 90% for steps ≤ 0.20 m.
- No regression on flat ground velocity tracking vs the blind baseline (±10%).
- Clean-looking gait in video (no knee dragging on edges, no hopping).
- **Architecture gates (new — don't skip, this is the point of validating before scaling up):** router shows visibly task-differentiated expert usage across the three tasks (not collapsed to one dominant expert); multi-critic per-task returns are stable; both the multi-critic ablation (4.2) and the MoE-actor ablation (4.3) have been run once and their results reported honestly, even if the answer is "this component didn't measurably help at 3 tasks."

**Report to the human with the eval table, router/expert-usage diagnostics, ablation results, and videos before moving on. Do not expand the task set (obstacles, gaps, extra terrain types) or start Milestone B (depth, concurrent training) without asking the human first.**

---

## Phase 5 — Depth sensor decision **[ASK HUMAN]**

The stock M20 has LiDAR + wide-angle RGB, not a depth camera. Present these options with trade-offs and wait for a decision:

- **Option D1 — add a depth camera** (e.g. Intel RealSense D435i or Orbbec) mounted at the front, angled down (~30–50°). Need: exact mount position/orientation, resolution, FOV, frame rate, whether a rear camera is possible.
- **Option D2 — use the existing LiDAR** to build a robot-centric elevation map on the robot, and feed a height-scan-like observation to the policy. Simpler distillation (the teacher already uses a height scan), but depends on mapping quality and odometry; train the student on a *noisy/delayed/drifting* height map instead of depth.
- **Option D3 — render a depth image from LiDAR points** (spherical or pinhole projection). Rarely needed; mention for completeness.

The rest of this file assumes **D1 (front depth camera)**. If D2 is chosen, keep Phase 7's structure but replace the depth encoder with a height-map noise model (random offsets, drift, missing cells, map latency, occlusion shadows behind steps).

---

## Phase 6 — Simulated depth camera (start of Milestone B — concurrent training, not a separate student env)

Per the CTS-MoE direction (`docs/stairs_research_notes.md` §11), the student is not a separately-trained network in its own gym env — it's a **second encoder added to the same MoE policy validated in Milestone A**, trained concurrently with the existing teacher encoder. Register a new gym ID (e.g. `Stairs-Concurrent-Deeprobotics-M20-v0`) that keeps the Milestone A task/terrain/reward/multi-critic setup and adds the student encoder + depth camera alongside it, rather than inheriting-and-replacing. This phase (6) only builds the simulated sensor; Phase 8 covers how the two encoders train together.

1. **Sensor choice in sim:**
   - Prefer `RayCasterCameraCfg` (with `patterns.PinholeCameraPatternCfg`, `data_types=["distance_to_image_plane"]`) raycasting against `/World/ground`. It is much cheaper than rendering and works with thousands of envs. Limitation: it only sees the listed meshes — **not the robot's own legs/wheels** — so add self-occlusion via masks (Phase 7).
   - Only fall back to `TiledCameraCfg`/`CameraCfg` (RTX rendering) if you need true self-occlusion; expect to drop to ~256–1024 envs.
2. **Mount:** attach to `base_link` using the real mount pose from the human. Double-check the offset convention (`ros`/`opengl`/`world`) and quaternion order for the installed Isaac Lab version by rendering one frame and saving it as PNG.
3. **Resolution and rate:** render at a modest native resolution, then downsample to something like 58×87 or 64×64. Update at the real camera's policy-relevant rate (e.g. 10–15 Hz) while the policy runs at 50 Hz (`decimation=4`, `dt=0.005`); hold the last frame between updates.
4. **Normalization:** clip to [near, far] (e.g. 0.1–2.0 m, match the real sensor), then map to [-0.5, 0.5] or [0,1].
5. Write a debug script that dumps depth frames from a few envs on stairs to `debug/depth/*.png`. Show them to the human.

---

## Phase 7 — Domain randomization

### 7.1 Depth image randomization (apply in the observation function, on GPU, per env)

- Gaussian noise, both per-pixel and low-frequency (multi-scale), scaled with distance.
- Random holes / dropout patches (0–10% of pixels) set to 0 or max range.
- Edge noise / "flying pixels" at depth discontinuities (step edges): detect gradients and perturb or zero nearby pixels.
- Random blur (Gaussian kernel 1–3 px).
- Self-occlusion masks: random leg/wheel-shaped blobs in the lower image region, or real masks captured from the robot later.
- Camera extrinsics jitter per episode: ±1–2 cm position, ±2–3° rotation. Small intrinsics/FOV jitter.
- Latency: 0–100 ms delay (buffer of past frames), plus random frame drops (repeat previous frame).
- Match the **real-side preprocessing** exactly: whatever you do on the robot (crop, hole fill, spatial/temporal filters, downsampling), apply the same in sim after the noise. Document the pipeline in `docs/depth_pipeline.md`.

### 7.2 Physics / dynamics randomization (audit what exists, then add)

Already in `velocity_env_cfg.py` / M20 config: friction (0.35–1.5), restitution, link mass scale, base added mass (−1 to +3 kg), inertia, CoM offset, actuator gain scaling (±15%), reset pose/velocity, external force/torque on reset, interval pushes. Add or verify:

- Payload up to the real limit (M20: 15 kg nominal) — **[ASK HUMAN]** for the payload range they care about.
- Motor strength scaling (0.8–1.2) separately for leg and wheel actuators.
- Action latency (0–20 ms) and observation latency for proprioception.
- Joint friction / armature randomization.
- Separate wheel–ground friction (rubber on concrete vs wood vs metal stair treads, 0.4–1.2).
- Stair geometry randomization: tread width, riser height, nosing overhang if you implement custom stairs, small per-step height noise.

---

## Phase 8 — Concurrent teacher-student training, task expansion (Milestone B)

Per `docs/stairs_research_notes.md` §11, this replaces a sequential "freeze teacher, then DAgger-distill a student" phase with **single-stage concurrent training**: both encoders stay live and aligned throughout, so the student never has to imitate a policy whose input distribution it can't yet reach.

### 8.1 Architecture

- Teacher encoder: whatever Milestone A's actor/router already consume (height scan + proprioception privileged terms), unchanged.
- Student encoder: depth CNN (e.g. 3–4 conv layers → 32–64-d latent, `docs/stairs_research_notes.md` §1/§4/§5's convergent CNN[16,32,32]/kernel[5,4,3]/stride[2,2,1] default) over the latest depth frame (or a short stack), + GRU (hidden ~256–512) over `[proprio, depth_latent]`.
- The **MoE actor and router from Phase 4.3 are shared** between teacher and student passes — router input becomes `z_t` = concatenation of teacher-encoder and student-encoder outputs during training (§11); at deployment only the student-encoder half is live (no privileged inputs exist on the real robot), so the router must already have learned to route sensibly off perception alone — this is exactly why Milestone A required the router to never see task ID or privileged-only terms.
- Optional auxiliary head: predict the teacher's height-scan encoding from the student's recurrent state (MSE loss) — speeds up and stabilizes alignment, same idea as the encoder-alignment loss below.

### 8.2 Training loop

- Every PPO update: after the policy-gradient step, run a supervised alignment step pulling the student encoder toward the teacher encoder's output for the same batch of states: `‖student_encoder(o_t) − teacher_encoder(s_t)‖²` (§11). No β-mixing, no frozen teacher, no separate DAgger rollout phase — the student is trained alongside the teacher from the start of this phase, not after Milestone A's policy is finalized and frozen.
- Implementation route:
  - First check whether the installed RSL-RL 5.0.1 `DistillationRunner`/`RslRlDistillationRunnerCfg` supports this concurrent-encoder pattern with image inputs (CNN + RNN) and a custom alignment loss, vs. only sequential DAgger. It likely does not natively support concurrent dual-encoder training — expect to write a custom runner.
  - If custom: write it in `source/rl_training/rl_training/rsl_rl/` (the repo already has a custom AMP runner there as a style/integration reference). Reference `CAI23sbP/Isaaclab_Parkour`'s student implementation for the CNN+GRU plumbing even though its training loop is sequential DAgger, not concurrent — the network code transfers, the training loop doesn't.
  - **[ASK HUMAN]** if a custom concurrent-training runner turns out to be substantially more implementation risk than falling back to Milestone A's original sequential DAgger plan for just this phase (keeping the MoE actor/router, dropping only the "concurrent" training mechanic) — flag the tradeoff with a time estimate rather than silently picking one.
- Keep the terrain curriculum active during this phase, seeded across all difficulty levels, per-task as in Milestone A (3.3-3.4).

### 8.3 Task-set expansion **[ASK HUMAN]**

Only after Milestone A's three-task architecture is validated and concurrent training (8.1-8.2) is working: decide whether to expand beyond `flat_rough` / `stair_ascent` / `stair_descent` (e.g. add `obstacle_climb`, `gap_cross`, or split `flat_rough` into separate tasks) per CTS-MoE's original six-task setup. Present the tradeoff (more tasks = more of CTS-MoE's actual multi-task benefit demonstrated, but more multi-critic heads, more expert-routing complexity, and terrain/reward work to build) and wait for a decision before adding tasks.

### 8.4 Optional RL fine-tuning

After concurrent training plateaus, optionally fine-tune with PPO at a low learning rate (critic keeps privileged obs, teacher encoder can be frozen at this point), as in "Parkour in the Wild" (`docs/stairs_research_notes.md` §6). Only do this if the student-driven policy is clearly below the teacher-driven one on `eval_stairs.py`.

### Milestone B acceptance criteria

- On `eval_stairs.py`, running the policy in **student-encoder-only mode** (as it will run at deployment) with depth DR **enabled**: within 10 percentage points of the Milestone A teacher-encoder-only results on every step height, ascent and descent.
- Router expert-usage diagnostics (4.3) still show task differentiation when driven by the student encoder, not just the teacher encoder — confirms the alignment loss is actually working, not just that the student encoder produces *some* latent.
- Robust to: +100 ms camera latency, 10% pixel dropout, ±3° camera tilt error (run each as a separate eval sweep and report).
- Blind fallback test: blank the depth image mid-traverse; report behaviour (should degrade gracefully, not flail).

---

## Phase 9 — Export, sim2sim, deployment prep (Milestone C)

1. Extend `scripts/tools/export_onnx_fast.py` to export the full deployable graph: student encoder (CNN + GRU with explicit hidden-state inputs/outputs) → router (perception-gated, student-encoder input only) → all E experts → gate-weighted action output. Embed metadata: joint order, default positions, action scales, PD gains, depth preprocessing params (resolution, clip range, normalization), camera pose, observation ordering, and E/expert indices for debugging gate weights on-device.
2. Write a parity test: same inputs through the PyTorch model and ONNX runtime must match (max abs diff < 1e-4).
3. **[ASK HUMAN]** for the M20 deploy repo from the DEEP Robotics GitHub. Add a MuJoCo stair scene and a depth-camera render in MuJoCo; run the student there (sim2sim). Report success rates at 0.15/0.20/0.25 m.
4. Write `docs/real_robot_checklist.md`: camera driver and filters, time sync, measured latency, extrinsic calibration procedure, safety stop, first-test protocol (tethered, single step, low speed, then short flights).

---

## Reporting cadence

After each phase, post a short summary to the human: what changed, eval table, videos/images, open questions, and the proposed next step. Do not proceed past any **[ASK HUMAN]** item without an answer.

## Summary of open questions to raise early

1. Is there a trained blind M20 checkpoint to use as baseline / warm start?
2. Which exteroceptive sensor for deployment: added depth camera (which model, mount pose, front only or front + rear) or the onboard LiDAR elevation map?
3. Will a velocity estimate (`base_lin_vel`) be available on the real robot?
4. Target stair specs: max riser height, tread depth, open-riser stairs, materials.
5. Payload range to train for.
6. Compute budget: number/type of GPUs (depth rendering and concurrent teacher-student training in Phase 8 are the expensive part).
7. (Resolved 2026-09-22, tracked here for the record) Architecture direction: CTS-MoE-style dense MoE actor + perception-gated router + multi-critic, adopted per the user's decision — see `docs/stairs_research_notes.md` §11 and the intro. Built up sequentially: Milestone A validates the 3-task privileged version (with mandatory MoE-actor and multi-critic ablations, Phase 4.2/4.3) before Milestone B adds depth + concurrent training, and task-set expansion beyond the original three (Phase 8.3) is separately human-gated.
8. Whether RSL-RL 5.0.1 needs a custom multi-critic (4.2) and custom concurrent dual-encoder training loop (8.2), or has partial native support — first thing to check when Phase 4 implementation starts, since it changes the effort estimate materially.
