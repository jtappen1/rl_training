# Stair-climbing perception & RL research survey (Phase 2)

This survey covers the ten sources named in `command.md` plus a scan of newer (2025-2026)
wheeled-legged stair-climbing work, in support of Milestone A (privileged teacher) and
Milestone B (depth student) for the DEEP Robotics M20. Most of the literature is
**fully-legged, point/paw-foot quadrupeds or humanoids** using elevation-map ("scandot")
teachers distilled into depth-CNN+GRU students via DAgger — this gives us a strong template
for network architecture, depth noise modelling, and the teacher→student pipeline, but none
of it deals with wheels. Only two sources (#7 Chamorro et al. and #8 Lee et al.) are actually
wheeled-legged, and they are the ones that matter most for how we handle the M20's 4
velocity-controlled wheels, the walk/drive behavioural trade-off, and the wheel-slip realism
gap. Notes below are extracted from paper abstracts/HTML and repo READMEs via automated
fetches; some numeric details were not retrievable behind paywalls or from compressed PDFs,
and those gaps are marked explicitly rather than guessed.

---

## 1. Extreme Parkour (Cheng et al., ICRA 2024)

Code: https://github.com/chengxuxin/extreme-parkour · Paper: https://arxiv.org/abs/2309.14341

- **Terrain setup:** Curriculum of four obstacle types — tilted ramps, high steps, gaps
  (long-jump), hurdles (high-jump). Exact height/width ranges not quantified in the
  accessible text; scaled relative to the Unitree A1 (thigh height 26 cm, body length 40 cm),
  with jumps up to ~2x body height/length (~50 cm high jump, ~80 cm long jump).
- **Command/heading scheme:** Waypoint-based, not a fixed velocity vector. Tracking reward
  `r_tracking = min(⟨v, d̂_w⟩, v_cmd)` where `d̂_w` is the unit vector to the next waypoint.
  At deployment the student predicts its own heading `θ_pred` and falls back to the
  waypoint direction `d̂_w` only if the prediction error exceeds a threshold (0.6 rad) —
  a "Mixture of Teacher-Student" heading scheme.
- **Reward terms:** Velocity tracking (Eq. 2); foot-clearance penalty near terrain edges,
  `r_clearance = −Σ c_i·M[p_i]` where `M=1` if a stance foot is within 5 cm of a terrain
  edge; a stylized/handstand shaping term `r_stylized = W·[0.5⟨v̂_fwd, ĉ⟩+0.5]²`, toggled by
  a binary "walking flag." No explicit combining coefficients were recoverable from the
  fetched text.
- **Observation design:** Teacher: proprioception, scandots (terrain height grid, resolution
  not specified in the retrieved text), target heading, walking flag, commanded speed.
  Student: depth cropped (dead pixels trimmed from left) and downsampled to **58×87**,
  captured at **10±2 Hz** via an Intel RealSense D435. Depth latency fixed at 0.08 s
  (pipeline pauses if processing < 0.08 s to avoid jitter); proprioception latency 0.016 s.
- **Network architecture:** Convnet-GRU for depth encoding into scandot-like features;
  student initialized from the teacher actor to minimize distribution drift. Exact
  layer/channel sizes and MLP hidden dims not recoverable from the README/abstract fetch —
  would require reading `legged_gym/`/`rsl_rl/` config files directly (not done here; time-boxed).
- **Domain randomization:** Not quantified in the retrieved text (paper references
  "regularized online adaptation," i.e. environment-parameter estimation, but no explicit
  randomization ranges were extracted).
- **Depth noise model:** Only latency is quantified (fixed 0.08 s enforced delay). No
  explicit dropout/blur/edge-noise parameters found in the fetched abstract/HTML.
- **Wheeled content:** None — pure legged quadruped (Unitree A1).
- **Access note:** Abstract page gave little; code README gave architecture description
  without concrete numbers; used as a qualitative reference. Moderate confidence overall.

---

## 2. Isaac Lab port of Extreme Parkour (`CAI23sbP/Isaaclab_Parkour`)

https://github.com/CAI23sbP/Isaaclab_Parkour

- **Structural relevance:** This is the closest available structural template for our repo
  since it already runs Extreme-Parkour-style teacher/student training **inside Isaac Lab**
  with RSL-RL's distillation runner, exactly the stack we're on (Isaac Lab / RSL-RL).
- **Gym IDs:** `Isaac-Extreme-Parkour-Teacher-Unitree-Go2-v0` and
  `Isaac-Extreme-Parkour-Student-Unitree-Go2-v0` — same naming pattern our plan proposes
  (`Stairs-Teacher-...-v0` / `Stairs-Student-...-v0`).
- **Repo layout:** `parkour_isaaclab/` (core env/model definitions), `parkour_tasks/`
  (task configs: terrain, rewards, observation groups split teacher vs. student),
  `parkour_test/`. `scripts/rsl_rl/` for train/eval/deploy, `--headless` for server runs.
- **Terrain, rewards, observations:** README confirms stair-inclusive terrain config exists,
  reward terms with per-term weights exist in `parkour_tasks/`, and observation groups are
  explicitly split into teacher (privileged) and student (depth) — but exact weight values
  and observation dimensions were not retrievable from the README text alone; would need a
  direct read of the config `.py` files for numbers (recommended as a first coding step in
  Phase 3, since this repo's directory shape maps almost 1:1 onto ours).
- **Network architecture:** Confirmed CNN (vision) + GRU (temporal) + MLP (decision) split,
  with different capacity for teacher vs. student, but no concrete sizes recovered.
- **Sensor/noise config:** README confirms a "depth camera simulation with configurable
  noise models" exists as a first-class config object — worth copying the *shape* of this
  config class into our `stairs_env_cfg.py` even before copying exact numbers.
- **Domain randomization:** Present ("systematic parameter randomization") but not itemized
  in the fetched README.
- **Wheeled content:** None — Go2 is a standard point-foot quadruped.
- **Access note:** README-only fetch; recommend a follow-up in Phase 3 where an agent
  actually reads the Python source files (not just README) since this repo is meant to be
  copied from structurally.

---

## 3. Legged Locomotion in Challenging Terrains using Egocentric Vision (Agarwal et al., 2022)

https://arxiv.org/abs/2211.07638 (full text via `arxiv.org/html/2211.07638`)

- **Terrain setup:** Stairs up to **24 cm** height, **28 cm** minimum tread width; curbs up
  to **26 cm** (robot hip height 28 cm); stepping stones and gaps up to **26 cm**
  (100% success from egocentric depth), 94% on "difficult" stepping stones (built from
  tables/stools in the real world). Training terrain: **100 sub-terrains in a 20×10 grid**,
  each **8 m × 8 m**, with fractal roughness up to 10 cm on flat tiles and 4 cm ("medium
  fractal") elsewhere.
- **Command/heading scheme:** Forward velocity command 0.35 m/s while traversing terrain;
  for curve-following, `v_x^cmd ∈ [0.2, 0.75] m/s`. Heading command `±10°` during terrain
  traversal, `±60°` on curves, `±180°` for in-place turning.
- **Reward terms (with weights):** Command tracking weight **7**; absolute work penalty
  **−1e-4**; foot-jerk penalty **−1e-4**; feet-drag penalty **−1e-4**; collision penalty
  **−1**; survival bonus **+1** per step.
- **Observation design:** Two-phase — Phase 1 ("cheap-to-compute" proxy) uses scandots:
  terrain height sampled at (x,y) points in robot frame. Phase 2 replaces this with real
  depth, distilled via supervised learning. Two teacher variants tested: a monolithic
  MLP(scandots)→GRU(proprio+γ+vel) network, and an RMA-style variant with a GRU encoding
  scandots to a latent γ and a separate MLP mapping privileged env parameters to a latent z.
- **Network architecture:** MLP with 2 hidden layers for scandot/latent processing; single-layer
  GRUs; Phase 2 uses a ConvNet to preprocess depth + GRU for temporal fusion of vision and
  proprioception.
- **Domain randomization:** Height-map update frequency 80–120 ms (increased over curriculum);
  height-map latency 10–30 ms; added mass **[−2, +6] kg**; CoM shift **±0.15 m**; random
  pushes every 15 s at 0.3 m/s; friction **[0.3, 1.25]**; fractal terrain height **[2, 4] cm**;
  motor strength **[90%, 110%]**; PD stiffness **[35, 45]**; PD damping **[0.4, 0.6]**.
- **Depth noise model (the most directly reusable part of this source):** Real camera is an
  Intel RealSense D435, capture period **100 ± 20 ms**, native resolution 480×848.
  Preprocessing pipeline: crop **200 pixels from the left** (stereo blind spot), nearest-
  neighbour **hole-filling**, downsample to **58×87**. Depth backbone runs on a Jetson NX,
  sends a compressed latent over UDP to the main compute board with modelled communication
  latency **10 ± 10 ms**; policy runs at **50 Hz**; low-level PD at 400 Hz with
  `Kp=40, Kd=0.5`.
- **Wheeled content:** None — point-foot legged quadruped.
- **Confidence:** High — HTML full-text fetch returned concrete numbers throughout.

---

## 4. Robot Parkour Learning (Zhuang et al., 2023)

https://arxiv.org/abs/2309.05665 (full text)

- **Terrain setup:** Four parkour skills, each with its own obstacle geometry
  (obstacle footprint typically 0.8 m × 0.8 m unless noted):
  - Climb: obstacle height train **[0.2, 0.45] m**, test **[0.25, 0.5] m**.
  - Leap: gap length train **[0.2, 0.8] m**, test **[0.3, 0.9] m**; gap "0.8 m wide × 0.8 m deep."
  - Crawl: clearance train **[0.32, 0.22] m**, test **[0.3, 0.2] m**; obstacle 0.8 m wide, 0.3 m
    in +x.
  - Tilt: path width train **[0.32, 0.28] m**, test **[0.3, 0.26] m**; obstacle 0.6 m along x.
- **Command/heading scheme:** Per-skill target forward speed: running 0.8 m/s, climbing
  1.2 m/s, leaping 1.5 m/s, crawling 0.8 m/s, tilting 0.5 m/s. Policy at 50 Hz; depth camera
  refresh 10 Hz.
- **Reward terms:** `r_skill = r_forward + r_energy + r_alive`, with forward-reward
  sub-weights (climb example): x-vel **α=1.0**, y-vel **α=1.0**, ang-vel **α=0.1**, energy
  penalty **α=2e-6**; a soft-dynamics "penetration" reward for interpenetration depth
  (**α=1e-2**) and volume (**α=1e-2**) during early training. Exact scales vary per skill
  (paper appendix tables).
- **Observation design:** Teacher (per-skill specialist): 29-dim proprioception (roll, pitch,
  angular vel, joint pos/vel), 12-dim last action, privileged visual (distance/height/width
  to the obstacle in front + 4-dim one-hot obstacle category), privileged physics (terrain
  friction, base CoM, motor strength). Vision student (distilled generalist): same
  proprioception + last action + a CNN latent from a **48×64** depth image.
  Skill-specific teacher: GRU (256 hidden) + MLP **[512, 256, 128]**, tanh output, ELU
  activations.
- **Network architecture (parkour/vision policy):** CNN encoder channels **[16, 32, 32]**,
  kernel sizes **[5, 4, 3]**, stride **[2, 2, 1]**, MaxPool, embedding dim **128**; GRU 256
  hidden; MLP **[512, 256, 128]** ELU.
- **Domain randomization:** Added mass **[1.0, 3.0] kg**; CoM x **[−0.05, 0.15] m**, y/z
  **±0.1/±0.05 m**; friction **[0.5, 1.0]**; motor strength **[0.9, 1.1]**; depth latency
  **[0.2, 0.26] s**; camera position jitter x **0.27±0.01 m**, y **0.0075±0.0025 m**, z
  **0.033±0.0005 m**; camera pitch **[0, 5]°**; FOV **[85, 88]°**; proprioception latency
  **[0.0375, 0.0475] s**.
- **Depth noise model:** Simulation side — depth clipping, pixel-level Gaussian noise, random
  artifacts. Real side — depth clipping, hole filling, spatial + temporal smoothing.
- **Wheeled content:** None — Unitree A1/Go2 point-foot quadrupeds.
- **Confidence:** High for terrain/DR numbers (HTML full text fetch); network arch numbers
  cross-checked and consistent with the closely related Humanoid Parkour Learning paper (#5)
  which shares co-authors.

---

## 5. Humanoid Parkour Learning (Zhuang et al., 2024)

https://arxiv.org/abs/2406.10759 (full text)

- **Terrain setup:** 10 obstacle types (jump up, jump down, leap, slope, stairs up/down,
  hurdle, ramp, discrete terrain, wave terrain, tilted ramp) arranged in a **10-row × 40-column**
  grid, difficulty interpolated `(1−i/10)·l_easy + (i/10)·l_hard`. Exact per-obstacle
  height/width ranges are in an appendix table not captured by this fetch.
- **Command/heading scheme:** Velocity commands sampled uniformly: `v_x ∈ [−0.8, 2.0] m/s`,
  `v_y ∈ [−0.8, 0.8] m/s`, `v_yaw ∈ [−1, 1] rad/s`. Autonomous heading term
  `v_yaw^cmd = α_yaw·(θ_goal − θ)`; forward command zeroed if heading error `≥ π/2`.
- **Reward terms:** Linear-vel tracking weight **1.0** (`exp(−‖v−v_cmd‖/0.25)`); angular-vel
  tracking weight **1.5**; energy **−2.5e-7**; DoF-vel penalty **−1e-4**; action-rate
  **−6e-3**; arm/waist/hip-yaw regularizers **−0.3 / −0.1 / −0.1**; penetration penalty
  `r_penetrate = α·Σ_p d(p)·‖v(p)‖`, **α=−5e-3**; footstep reward
  `r_step = α·(−ln‖d_x‖)`, **α=6**.
- **Observation design:** Oracle: scandot grid **11×19** encoded to a 32-dim embedding via
  MLP **[128, 64]**; proprioception 43-dim; estimated base velocity 3-dim; last action
  19-dim. Student: depth image **48×64** (resized from 480×640) + same proprioception/velocity
  estimate.
- **Network architecture:** Actor (both oracle and student): GRU (1 layer, 256 hidden) + MLP
  **[512, 256, 128]**. Separate state-estimator GRU+MLP for velocity. Student depth CNN:
  channels **[16, 32, 32]**, kernels **[5, 4, 3]**, stride **[2, 2, 1]**, MaxPool, 32-dim
  output embedding.
- **Domain randomization:** 4096 parallel envs; added mass **[−1.0, 5.0] kg**; CoM shifts x
  **[−0.1, 0.1]**, y **[−0.15, 0.15]**, z **[−0.2, 0.2] m**; friction **[−0.2, 2.0]** (sic,
  as extracted — likely meant [0.2, 2.0]); motor strength **[0.8, 1.2]**; proprio latency
  **[0.005, 0.045] s**; depth FOV **[86, 90]°**; depth latency **[0.06, 0.12] s**; camera
  position/rotation jitter present but exact ranges not captured.
- **Depth noise model (concrete, matches command.md's framing of this source):**
  Sim-side: "depth clipping, multi-level Gaussian noise, and random artifacts" — the paper
  explicitly names these three mechanisms but the fetch did not surface numeric parameters
  (std devs, artifact rates). Real-side: Intel RealSense D435i-matched pipeline — depth
  clipping, hole filling, spatial smoothing, temporal smoothing.
- **Distillation:** DAgger with L1 loss between student and teacher actions, 4-GPU pipeline
  (3 collectors + 1 trainer). PPO: clip 0.2, GAE λ=0.95, lr 3e-5. Deployment: 10 Hz vision,
  50 Hz policy, on an Intel i7 12-core NUC.
- **Wheeled content:** None — humanoid (presumably a Unitree/custom bipedal platform).
- **Confidence:** High for reward/network/DR numbers; medium for exact depth-noise
  parameters (named but not quantified in the retrieved text — worth reading the PDF
  appendix directly if precise std-devs are needed later).

---

## 6. Parkour in the Wild (Rudin et al., 2025)

https://arxiv.org/abs/2505.11164 (full text)

- **Terrain setup:** 9 "expert" terrains for initial training (Walk, Climb, Climb Down, Jump,
  Tables, Rock pile, Low wall, Beams, Stepping stones); fine-tuning adds **15 terrains** from
  3D scans of real search-and-rescue facilities plus a composite "Parkour line" (boxes, gaps,
  tables, stairs, slopes). Evaluation terrains randomized at 90% of max training difficulty.
- **Command/heading scheme:** Not a velocity command — a **target pose command**: target
  position `r*`, target heading `ψ*`, and a time-to-reach `t*`. Observation includes current
  position `r`, target `r*`, `ψ*`, `t*`.
- **Reward terms (fine-tuning stage):** Track position weight **10**
  (`1 − 0.5‖pos error‖`); track heading weight **5**; joint-vel penalty **−1e-3**; torque
  penalty **−1e-5**; base-acceleration penalty **−1e-3**; feet-acceleration **−2e-3**;
  action-rate **−1e-2**; collision **−1**; termination penalty **−2e3**.
- **Observation design:** Proprioception (base velocities, joint pos/vel, gravity vector,
  previous action) + exteroception = **4 depth images** (two forward-facing, two backward-
  facing), each processed by its own CNN.
- **Network architecture:** Each depth image → 3 conv+maxpool layers → 64-dim feature;
  concatenated with proprioception → **2-layer LSTM**; combined with proprio + task command
  → **3 fully-connected ELU layers**.
- **Multi-expert distillation:** DAgger with per-terrain experts; Gaussian action noise added
  during rollout collection to reduce overfitting; distilled policy learns to implicitly
  classify terrain type and reproduce the matching expert's action.
- **RL fine-tuning after distillation:** Three stabilizers — (1) action noise carried over
  from distillation, (2) conservative PPO hyperparameters, (3) **pre-training the critic with
  frozen policy weights** before joint policy updates resume. This is directly relevant to
  our optional Phase 8.3 ("optional RL fine-tuning" of the distilled student) — this source
  is the one that actually specifies *how* to stabilize that step.
- **Depth noise model (this is the strongest edge-noise/hole reference of all sources):**
  Depth clipped at **2 m**; pixels below **0.15 m** marked empty. Leftmost **1–5 columns**
  removed (stereo blind spot). Downsampled to **48×32**. **Edge noise:** pixels around
  depth-gradient edges are randomly set to empty or shuffled with a neighboring pixel.
  **Holes:** patches set to max-range depth, where patch shape/location comes from
  thresholding a slowly-evolving **Perlin noise** field (gives temporal consistency across
  frames rather than independent per-frame noise). Gaussian blur applied identically to sim
  and real depth after clip/downsample/crop.
- **Emergent behaviour note:** the fine-tuned policy learns "active perception" — pitching
  its body to keep obstacles in the depth camera's FOV — an emergent effect of RL
  fine-tuning directly on raw depth rather than an elevation map. Relevant if we ever
  fine-tune the M20 student directly on depth.
- **Wheeled content:** None — legged quadruped.
- **Confidence:** High — HTML fetch surfaced concrete numbers for nearly every rubric field.

---

## 7. RL for Blind Stair Climbing with Legged and Wheeled-Legged Robots (Chamorro et al., ICRA 2024) — MOST DIRECTLY RELEVANT

https://arxiv.org/abs/2402.06143 (full text via `arxiv.org/html/2402.06143`)

This is the closest published analogue to our Phase 3/4 problem: it trains a **blind**
(no exteroceptive sensing at runtime) stair-climbing controller with an asymmetric
actor-critic, and it explicitly includes a **wheeled-legged biped (Ascento)** alongside
point-foot quadrupeds (Go1, ANYmal-on-wheels quadruped/biped modes) and a biped (Cassie).

- **Terrain setup:** 6 m × 6 m tiles, **12 curriculum rows**. Stair columns split into two
  regimes: 6 rows of a single step of increasing height, then continuous stair flights of
  5 steps. Step heights tested: **14, 18, 22, 26, 30 cm**. Also: random rectangular
  obstacles up to 10 cm high, 1–2 m footprint; smooth slopes 0–8.5°; rough pyramids with up
  to 5 cm noise.
- **Command/heading scheme — the key formulation choice of this paper:** **position-based**,
  not velocity-based. The command is a goal position + heading (sampled at the top of the
  stair tile, heading within ±0.1 rad of straight-up; for other terrains, a goal within a
  1 m radius of the spawn). The paper explicitly argues this position formulation (vs. a
  velocity-tracking controller) is "vital for stair climbing" and that at deployment it
  "behaves similarly to controlling a velocity-based controller" when driven by a remote.
  **This directly contradicts our Phase 3.2 draft plan** (which proposes a
  velocity-command scheme inherited from the existing M20 velocity task) — worth flagging
  as a design decision to revisit, see "what we adopt" below.
- **Reward terms (exact weights):**
  | # | Name | Formula | Weight |
  |---|---|---|---|
  | 1 | Position (terminal) | `1/T_r · 1/(1+‖x−r_goal‖²)`, active only in last `T_r=2 s` of episode | **10.0** |
  | 2 | Position-bias (progress) | `(ẋ·(r_goal−x))/(‖ẋ‖‖r_goal−x‖)` | **1.0** |
  | 3 | Stall penalty | `−1` if `‖ẋ‖<0.1 m/s` and `‖x−r_goal‖>0.5 m` | **1.0** |
  | 4 | Face-goal | `−‖θ−θ_goal‖` if `‖x−r_goal‖>0.5 m` | **0.1** |

  Plus "standard robot-specific shaping rewards" (feet air-time, joint-acceleration penalty)
  reused per-platform from prior locomotion work — legged-specific and not enumerated for
  the wheeled robots in the extracted text.
- **Observation design — actor vs. critic (asymmetric, exactly our current M20 pattern):**
  - **Actor (blind, proprioceptive-only):** angular velocity (coeff 0.25, dim 3, noise ±20%),
    projected gravity (coeff 1.0, dim 3, noise ±5%), direction-to-goal 2D (coeff 1.0, dim 2,
    no noise), heading error (dim 1, no noise), commanded height (dim 1, no noise), **a
    binary "stair-mode" terrain flag** (dim 1, no noise — see below), joint positions (dim 4,
    noise ±1%), joint velocities (dim 6, noise ±150%), last action (dim 6, no noise).
  - **Critic (privileged):** linear velocity (coeff 0.25), base height (dim 3), relative goal
    position (dim 3), **terrain height** (coeff 5.0, dim **187**, sampled on a **1.6 m × 1.0 m**
    grid), contact forces (dim 6, filtered with a 5-step sliding average), friction
    coefficient (dim 1). Networks for actor/critic are fully separate (not shared trunk),
    "allowing them to update independently."
- **The "terrain boolean":** a single bit in the actor's observation that switches a
  stair-climbing behavioural mode on/off. This is the paper's key trick for letting one
  policy retain good flat/rough-terrain gait while specializing a distinct stair gait
  (bigger leg splits, slower cadence) without the two skills interfering — directly
  applicable to us as an alternative/complement to terrain-relative orientation rewards.
- **Network architecture:** Uses RSL-RL PPO (same lineage as our repo); explicitly does not
  report exact MLP layer sizes in the retrieved text. Converges in ~5k iterations on a
  single A100 (16 GB).
- **Domain randomization:** Terrain friction randomized specifically **to avoid the policy
  learning unrealistic wheel-vs-step grip** (see below); random push velocities 0–0.5 m/s
  applied at most every 3 s; **control-loop delay** simulated by reusing the previous
  timestep's state with 50% probability, modelling a ~20 ms delay — flagged by the paper as
  important for real-world transfer.
- **Wheeled/wheeled-legged specifics (the core reason to read this paper):**
  - Action space: leg joints get **position** targets via low-level PD; **wheel joints get
    target angular velocities** — identical action-space pattern to our M20 (12 position +
    N velocity).
  - **Friction is a live design tension, not just a sim2real detail:** if wheel-ground
    friction in sim is too high, the policy learns to climb by using wheel grip against the
    step riser — a strategy that "does not transfer due to unmodeled tire dynamics," where
    real "kick-back and irregularities in the tire profile cause slippage." The paper
    explicitly tunes the *shaping* rewards to find a compromise between "using velocity and
    grip with the step" and "stepping up one leg at a time," rather than relying on friction
    tuning alone.
  - **Real-robot failure mode called out explicitly:** colliding with a step causes a
    momentary loss of contact, which — because collision/friction at the step edge isn't
    modeled perfectly — can make the climb fail even when it worked in sim. This is exactly
    the "wheel-riser stumbling" failure our Phase 1 baseline eval already observed on the
    blind M20, so this paper's mitigation approach (tune shaping reward balance + explicit
    friction randomization, rather than only adding a stumble penalty) is a concrete lead
    for our Phase 4.2 item 3 (`feet_stumble` weight).
  - **Results (Ascento — wheeled-legged biped, "terrain bool" ON):** 14 cm: 87.6%,
    18 cm: 82.3%, 22 cm: 69.4%, 26 cm: 31.4% (exceeds training height), 30 cm: 1.4%
    (exceeds training height); other terrains 97.7%. Real robot: **climbed 15 cm steps**,
    "previously impossible for this robot."
  - **ANYmal-on-wheels, quadruped mode (bool ON):** 14 cm 93.2%, 18 cm 93.8%, 22 cm 90.2%,
    26 cm 61.1%, 30 cm 18.7%, other terrains 95.9%.
  - **ANYmal-on-wheels, biped mode (bool ON):** roughly flat across all tested heights
    (93–94% at 14–26 cm, still 93.4% at 30 cm) — the paper attributes this to the biped
    mode's larger single-leg-lift margin.
  - **Cross-platform comparison at bool ON:** point-foot legged robots (Go1 99.7%@18cm,
    Cassie 100%@18cm) outperform wheeled-legged platforms (Ascento 82.3%@18cm, ANYmal-wheels-
    quad 93.8%@18cm) at matched step height, and the gap widens with step height — the paper
    attributes this gap to unmodeled wheel dynamics and the grip/step tradeoff described
    above, not to an architecture or reward deficiency. **This sets a realistic ceiling
    expectation for our M20 teacher: even a well-tuned wheeled-legged policy should be
    expected to underperform a hypothetical point-foot equivalent, especially near the
    25 cm+ range that matters for M20's real-world stair spec.**
- **Confidence:** High — full HTML text fetch returned nearly all rubric fields with
  concrete numbers.

---

## 8. Learning robust autonomous navigation and locomotion for wheeled-legged robots (Lee et al., Science Robotics 2024)

DOI: 10.1126/scirobotics.adi9641 — **paywalled** (WebFetch returned HTTP 403 Forbidden on
science.org). Retrieved the arXiv preprint instead: https://arxiv.org/pdf/2405.01792 /
https://arxiv.org/abs/2405.01792 (full text fetched successfully from the HTML mirror).
This is the ANYmal-on-wheels (quadruped, wheeled-legged) paper from RSL/ETH Zurich, and
together with #7 it is the most relevant source for our project.

- **Terrain setup:** Parameterized terrain generator (2–3 parameters per type): stairs with
  variable step height, "bumpy" terrain with height deviations comparable to the wheel
  radius, slopes tested to failure, discrete obstacles up to **60 cm** high. Real-world
  validation: Zurich urban area (245 m × 345 m, 8.3 km covered) and Seville, over grass,
  sand, gravel.
- **Command/heading scheme — hierarchical, two update rates:** Low-level controller (LLC)
  at **50 Hz** takes velocity commands `v_x ∈ [−2.5, 2.5] m/s`, `v_y ∈ [−1.2, 1.2] m/s`,
  `ω_z ∈ [−1.5, 1.5] rad/s` (training range); high-level controller (HLC) at **10 Hz**
  (matches the elevation-mapping rate) outputs bounded velocity commands
  `v_x ∈ [−1.0, 2.0]`, `v_y ∈ [−0.75, 0.75]`, `ω_z ∈ [−1.25, 1.25]` as a Beta-distribution
  output, driven by two waypoints sampled 5–20 m apart via pure-pursuit-style interpolation.
- **Reward terms:** HLC dense reward `r_h,dense = 1.0` if within 0.75 m of the next waypoint,
  else `clip(v·ê_wp1, 0, v_thres)/v_thres` with `v_thres=0.5`; an exploration bonus penalizing
  revisiting a position buffer (20 prior positions at 50 cm spacing, up to 10 m back). LLC
  training combines `r_l + r_r`; HLC objective is `r_h + w_l·(r_l + r_r)`. A **cost-of-
  transport** metric `COT_mech = Σ[τθ̇]⁺ / (mg|v_xy^b|)` is used as an evaluation/shaping
  signal, not just a metric — reported COT 0.16 mechanical, vs. 3x the speed and 53% lower
  COT than a point-foot ANYmal comparator, average speed **1.68 m/s**, peak speed **5.0 m/s**
  in sim (hardware capable of 6.3 m/s).
- **Observation design:** LLC — exteroceptive: **height values sampled in a circular pattern
  around each wheel**; proprioceptive: raw IMU (not a filtered state estimator, "to reduce
  use of heuristics"), joint angles/velocities, sequences of both. Privileged (teacher only):
  noiseless joint state, foot/wheel contact state, terrain normal at each contact, contact
  force, true robot velocity, gravity vector. HLC — 4 modalities: local elevation map
  (3 m front, 1.5 m other directions), the LLC's RNN hidden state, a position-history buffer,
  and the two waypoints + short history of 3 previous HLC outputs.
- **Network architecture:** HLC processes the position-history buffer with 1D-CNN + max-pool
  (PointNet-style), the elevation map with a 3-layer 2D CNN + MLP, everything else with plain
  MLPs, output as Beta-distribution parameters. LLC teacher: plain 3-layer MLP. LLC student:
  **GRU**, trained via imitation from the teacher.
- **Domain randomization:** Stair heights/widths randomized via a terrain-tile "Wave Function
  Collapse" generator (ensures navigable connectivity); actuator dynamics modeled by learned
  neural-network models (leg joints: torque model from real data; **wheels: a learned mapping
  from velocity command + velocity-command history to motor current**, i.e. a data-driven
  wheel actuator model rather than an ideal PD/velocity actuator); friction randomized via
  Coulomb + stick-friction constants; dynamic obstacles with randomized count/position/speed
  (0.1–0.5 m/s) during training.
- **Wheeled-legged specifics (core relevance):**
  - **16-dim action space = 12 leg joint position targets + 4 wheel velocity targets** —
    essentially identical to the M20's action space described in `command.md`.
  - **Gait/mode switching is fully emergent, not scripted:** no CPG, no predefined gait
    sequence. Observed behaviours: pure driving on flat ground; an asymmetric "creep + drive"
    gait over large discrete obstacles; full quadrupedal trotting (wheels essentially inert)
    on stairs and steep hills; wheels used as "active suspension" (driven straight over) on
    bumpy terrain. This strongly supports **not hand-designing a stair-specific gait or mode
    switch for the M20** — a well-shaped reward + curriculum can let the policy discover
    when to trot vs. roll on its own, as it does for both Chamorro's "terrain bool" system
    (explicit switch) and this paper's system (fully emergent, no switch at all).
  - **Step-height results (asymmetric ascend/descend):** ascending roughly **30–40 cm** at
    low approach speed; descending roughly **50–60 cm** (descent has a natural advantage,
    consistent with our Phase 1 baseline finding that the blind M20 already descends far
    better than it ascends).
  - **Wheel actuator realism:** the explicit learned current-from-velocity-history wheel
    model (rather than an idealized velocity PD) is presented as important for sim-to-real
    of the wheeled behaviour — a concrete idea for our Phase 7.2 "separate wheel-ground
    friction" and general wheel actuator DR, if the M20 wheel actuator isn't already modeled
    with comparable fidelity.
- **Confidence:** High for the arXiv preprint (full HTML fetch succeeded with concrete
  numbers throughout); the published Science Robotics version was not accessible (403) but
  arXiv preprints of this group's papers are typically near-identical to the camera-ready
  text, so confidence in accuracy is high despite not reading the publisher's version.

---

## 9. legged_lab_v3 (Isaac Lab, G1 humanoid)

https://github.com/lessonllab/legged_lab_v3

- **Terrain/stair curriculum:** README (partly in Chinese) confirms **independent up/down
  promotion and demotion** for the stair curriculum ("上下楼独立晋降级"), i.e. ascent and
  descent difficulty levels are tracked and advanced separately rather than coupled to a
  single terrain-level scalar. Fixed **17-level** staircase curriculum spanning step heights
  **8–30 cm**. This directly matches the concern in our Phase 1 finding (descent already
  works, ascent doesn't) — an up/down-decoupled curriculum promotion rule (rather than the
  existing single `terrain_levels_vel` scalar that promotes/demotes on overall distance
  travelled) would let ascent and descent difficulty scale independently instead of one
  masking the other's progress.
- **Observation design:** An **8-frame depth history buffer** explicitly modelling both
  camera **latency** and the robot's own **self-occlusion** (legs/body appearing in frame).
  Exact buffer indexing/sampling-rate mechanics were not visible in the README text alone —
  flagged as worth reading the actual config source in a later coding phase if we adopt a
  depth-history observation for the M20 student, since self-occlusion masking is exactly the
  gap Isaac Lab's cheap `RayCasterCameraCfg` approach has (per `command.md` Phase 6.1) versus
  full RTX rendering.
- **Reward architecture:** Style-reward coefficient of **0.25**, indicating an AMP
  (adversarial motion prior) component with **30 reference motion clips** — this is a
  humanoid-specific motion-imitation mechanism not applicable to the M20 (no motion capture
  reference data for a wheeled quadruped), but the reward *categories* named — "foothold
  support detection, edge-stepping and collision constraints, low-speed stair-crossing and
  turning training" — map onto exactly the kind of stair-specific reward shaping (edge
  contact, low-speed-on-stairs behaviour) we plan for Phase 4.2.
- **Training framework:** RSL-RL 5.4.1 with PPO/AMP — one version ahead of our pinned
  RSL-RL 5.0.1; worth checking whether any distillation-runner API used here differs from
  what's installed in our repo before copying code directly.
- **Wheeled content:** None — G1 is a bipedal humanoid, no wheels.
- **Confidence:** Medium — README-only fetch, partially non-English, with explicit
  acknowledgement that config-file-level detail (buffer implementation, exact reward
  weights, network sizes) was not visible in the fetched content. Worth a deeper follow-up
  read of the actual Python source before Phase 3/4 implementation if the up/down-decoupled
  curriculum idea is adopted, since that's the most actionable item from this source.

---

## 10. Upstream: `fan-ziqi/robot_lab`

https://github.com/fan-ziqi/robot_lab

- **Relationship to our repo:** confirmed upstream — our repo forked from this project.
- **Robot support:** Explicitly supports three categories, matching what's described in
  `command.md`: legged quadrupeds (Anymal-D, Unitree Go2/A1, Deeprobotics Lite3, others),
  **wheeled robots including the Deeprobotics M20** (also Go2W, B2W, Tita, ZSL1W, Dog-W),
  and humanoids (G1, H1, GR1T1/T2, T1, others). Confirms the M20 wheeled-quadruped config we
  build on originates here.
- **Terrain/rough handling:** README distinguishes `Flat` vs `Rough` env variants by gym ID
  naming convention (e.g. `RobotLab-Isaac-Velocity-Rough-Anymal-D-v0`) but does not expose
  the underlying terrain generator's step-height ranges or sub-terrain proportions in the
  README text — consistent with `command.md`'s own note that the actual terrain config
  (`ROUGH_TERRAINS_CFG` with `pyramid_stairs`/`pyramid_stairs_inv`, 0.05–0.23 m step height)
  lives in the installed Isaac Lab package, not in robot_lab itself.
  robot_lab appears to consume Isaac Lab's stock terrain config rather than defining its own
  stair generator — meaning our planned Phase 3 custom `TerrainGeneratorCfg` is genuinely new
  work, not a divergence from an upstream pattern we should have preserved.
- **Reward terms:** Not enumerated in the README; no wheeled-specific reward terms (e.g.
  wheel velocity tracking, feet_stumble) were visible from this fetch. This matches
  `command.md`'s description of our repo's current reward set as the effective source of
  truth for M20-specific reward terms.
- **Observation design:** README does not describe height_scan usage or actor/critic
  asymmetry explicitly; again consistent with this being config-level detail that lives in
  the Python source rather than documentation.
- **Repo structure:** `source/robot_lab/assets/` (robot definitions), the same
  `tasks/manager_based/locomotion/velocity/` structure our repo already has (`config/
  {robot_name}/`, `velocity_env_cfg.py`), and both RSL-RL and CusRL agent config registration
  — confirming the file layout in `command.md`'s "Key files" section is inherited from here,
  not repo-specific.
- **Wheeled content:** Confirms robot_lab already treats M20-class wheeled robots as a
  first-class category alongside legged and humanoid, but nothing wheel-specific (gait,
  reward, or terrain) was surfaced in the README.
- **Confidence:** Medium-high for repo structure and robot roster (clearly stated in
  README); low for reward/observation/terrain specifics (not in the README, would require
  reading the actual `mdp/rewards.py` etc. in the upstream repo — out of scope for this
  time-boxed survey since our own repo's forked copies of these files are the actual source
  of truth per `command.md`).

---

## 11. CTS-MoE: Implicit Terrain Adaptation via Mixture-of-Experts for Perceptive Locomotion (Affonso et al., 2026) — CHOSEN ARCHITECTURE DIRECTION

Project page: https://cts-moe.github.io/ · Paper: https://arxiv.org/abs/2606.19633 (full HTML
fetched successfully). **The user has indicated this is the architectural direction they want
to pursue for this project**, so this source should be weighted more heavily than "one entry
among many" in downstream planning — see the "What we adopt" section for how it reshapes the
Phase 4/6/8 split.

- **Terrain setup:** Six discrete tasks trained together — flat, stair ascent/descent (step
  height **0.02–0.22 m**), obstacle climb up/down (obstacle height **0.05–0.50 m**), gap
  crossing (gap width **0.05–0.50 m**). Training grid: 7×6 array of 8m×8m patches with
  progressive difficulty. Evaluated (zero-shot) on unseen rough terrain and slopes (0–30°).
- **Command/heading scheme:** Forward-only. Velocity-tracking tasks (flat) sample
  `vx ∈ {0.5, 0.75, 1.0} m/s`, `ωz = 0` at eval; point/waypoint-tracking tasks (stairs,
  obstacles, gaps) restrict to `vx ∈ [0.0, 1.0] m/s` and are rewarded via **waypoint tracking**
  (`r_wp`), not velocity tracking — directly parallel to Chamorro (#7)'s position-based
  argument, from an independent source.
- **Reward terms (Table 7 in paper):** Shared across all six tasks: vertical-velocity penalty
  (−0.25 to −1.5), orientation penalty (−0.05), joint-limit penalty (−2.0). Per-task
  specialization: flat gets velocity/yaw-tracking reward (2.0 / 1.0); stairs/obstacles/gaps get
  waypoint-tracking reward (1.0) plus a *relaxed* orientation term (task-specific, not global)
  — same conceptual move as this survey's own recommended "gate the orientation penalty on a
  stair-mode flag" (see #7's section and "What we adopt" below), but CTS-MoE generalizes it:
  **every task gets its own reward weighting, not just stairs vs. not-stairs.**
- **Observation design:** Proprioceptive (`o_t^p`): velocity commands, base angular velocity,
  projected gravity, joint pos/vel (12 each), previous action (12), short history. Perceptive:
  **48×64 depth image** (Intel RealSense D435i, ~30° downward tilt, 50 Hz). Privileged
  (teacher only): 187-dim heightmap, 1-dim task ID, contact forces, joint torques,
  accelerations, robot mass, joint stiffness/damping — same asymmetric actor/critic split
  already used in our current M20 task and endorsed elsewhere in this survey.
- **Network architecture — the core contribution:**
  - **Dense mixture-of-experts actor**: E=6 experts, each an MLP `[512, 256, 128]` → 12 action
    dims (i.e., each expert is exactly the same size as the single-MLP actor `command.md`
    already plans to use — the MoE wraps multiple copies of the "known-good" trunk rather than
    inventing a new per-expert architecture).
  - **Perception-based gating**: a router MLP `[512, 256]` producing soft mixture weights
    `g_e(z_t)` from a latent `z_t` = concatenation of the teacher and student encoder outputs.
    Critically, **gating depends only on perception at deployment** (no terrain classifier, no
    task ID, no explicit mode flag) — this is a stronger, learned generalization of the
    "stair-mode flag" idea (#7) and the "fully emergent switching" finding (#8): here the
    switching is emergent but *architecturally supported* by having literally separate expert
    weights to route between, rather than asking one monolithic MLP to represent all gaits
    internally.
  - **Multi-critic**: one value head `V_φ^(c_t)` **per task**, trained only on that task's own
    trajectories, with POPArt-style per-task return normalization to prevent value
    interference (a fast task's large returns don't drown out a slow/hard task's small
    returns in a shared critic).
- **Domain randomization:** Two-phase training — first converge using the simulator's built-in
  idealized actuators, then resume with an **explicit delayed PD actuator model** and "extensive"
  DR layered on top (specific ranges not given beyond "follows prior work"). This staged
  DR-after-convergence approach is a useful sequencing idea independent of the MoE
  architecture itself.
- **Training procedure — single-stage concurrent teacher-student (the other core
  contribution, distinct from the MoE actor):** A shared policy conditions on the
  *concatenation* of a teacher encoder (reads privileged state) and a student encoder (reads
  only deployable proprio + depth) latents simultaneously. After every PPO update, a
  supervised step pulls the student encoder toward the teacher encoder's output via MSE:
  `‖ψ_θs(o_t) − ψ_θt(s_t)‖²`. Because both encoders are live and aligned throughout training
  (not teacher-first-then-freeze-and-distill), the student never has to imitate a policy whose
  input distribution it can't yet reach — this is presented as specifically avoiding the
  distribution-mismatch problem that motivates DAgger/β-mixing in the sequential approach
  (#1, #8's LLC distillation step, `command.md`'s own planned Phase 8.2). **Task labels are
  used only to route rewards to the right value head during training** — at deployment there
  is no task ID input at all, only perception-driven gating.
- **Quantitative results:** vs. a monolithic single-task-style perceptive baseline: **+29.3
  percentage points** success rate on gap crossing, **+10.3 points** on obstacle climbing.
  Combined velocity-tracking error on gap terrain roughly matches a "CTS-Single" ablation
  (0.26±0.10 vs 0.25±0.24) — i.e. the MoE gain is concentrated in success rate / hard-task
  competence, not in tracking smoothness. Near-100% success on unseen slopes/rough for all
  methods (ceiling effect, not a discriminating result). Evaluated in sim **and on Unitree Go1
  hardware**.
- **Depth/vision sensor and noise model:** Same RealSense D435i / 48×64 / 50 Hz setup as
  several other sources in this survey (#4, #5 use 48×64 too). No explicit engineered noise
  model beyond DR — the paper notes depth degrades outdoors and at very close range (<0.2 m)
  but doesn't describe a specific synthetic-noise pipeline; for that, this survey's existing
  recommendation (Parkour in the Wild #6's edge/hole algorithm) still stands as the concrete
  implementation reference.
- **Wheeled relevance:** **None explicitly** — Unitree Go1 is a plain point-foot quadruped, no
  wheels, no wheel-specific terms anywhere in the method. The relevance to the M20 is entirely
  architectural (how to share behavior across a small set of qualitatively different gaits/
  terrains without one task's reward dominating or a hierarchical selector losing
  generalization), not mechanical. Everything wheel-specific in this survey still comes from
  #7 and #8 only.
- **Confidence:** High — full HTML text fetched from both the project page and the arXiv HTML
  mirror, with concrete numbers throughout (expert count, MLP sizes, reward table, results).

---

## Newer (2025-2026) wheeled-legged work

Web search terms used: "wheeled-legged robot stair climbing reinforcement learning 2025
2026 perception depth."

- **CTBC — Contact-Triggered Blind Climbing for Wheeled Bipedal Robots with Instruction
  Learning and Reinforcement Learning** (2025), https://arxiv.org/pdf/2509.02986 — combines
  demonstration-based "instruction learning" with RL, using contact/tactile signals as
  triggers for climbing-behaviour transitions on a wheeled bipedal robot. **Low-medium
  confidence**: the PDF fetch only returned title/metadata-level content (compressed stream,
  could not extract body text); could not confirm stair dimensions, reward terms, or success
  rates. Worth a manual PDF read if contact-triggered mode-switching (an alternative to
  Chamorro's explicit "terrain bool" or Lee et al.'s fully emergent switching) becomes a
  design candidate for the M20.
- **Long-Distance Real-World Navigation of the Legged-Wheeled Robot Go2-W Using Deep RL**
  (2026), https://arxiv.org/pdf/2606.21387 — Unitree Go2-W (legged-wheeled hybrid) doing
  long-distance real-world navigation. **Medium confidence, low detail**: PDF fetch returned
  only structural/metadata content; could not confirm specific reward weights, DR ranges, or
  whether stairs/curbs are explicitly handled versus just general rough terrain. Directly
  relevant platform class (4-wheel-leg hybrid, same idea as M20) so worth a manual follow-up
  read.
- **Stand, Walk, Navigate: Recovery-Aware Visual Navigation on a Low-Cost Wheeled Quadruped**
  (2025), https://arxiv.org/html/2510.23902v1 — **High confidence** (full HTML text
  extracted cleanly). Custom low-cost wheeled quadruped, depth camera + IMU (RGB-D +
  inertial fusion improves recovery ~25% over vision-only). Trained/evaluated in Isaac
  Lab/Sim. Terrain includes stairs up to **0.4 m** step height, slopes to 45°, discrete
  terrain with steps up to 0.2 m. Asymmetric actor-critic (terrain height exposed to critic
  only, matching our current M20 pattern). Notable feature: an explicit **fall-recovery**
  objective/reward (+50.0 sparse bonus for successful standing after a randomized drop from
  1–2 m), with recovery success 90.5% (stairs), 93.3% (slopes), 86.1% (discrete terrain),
  93.4% (flat). This is a genuinely new rubric item not covered by any of the ten core
  sources: none of them explicitly reward/evaluate **recovery after a fall on stairs**,
  which could matter for a real M20 given the aggressive pitch angles stairs require.
- **Adaptive multi-mode locomotion for bipedal wheel-legged robots via sparse
  Mixture-of-Experts deep RL** (2026), PubMed 41822818 — a sparse-MoE architecture for
  mode-switching (walk/roll/hybrid) on a bipedal wheel-legged robot. **Low confidence**:
  only the PubMed abstract listing was surfaced by search, not fetched in full; flagged as
  a candidate architecture (MoE over locomotion "experts") if a single monolithic policy
  proves hard to tune across the M20's flat-driving / stair-trotting behavioural range —
  worth comparing against Lee et al.'s finding that a single non-MoE network can still learn
  emergent mode-switching without an explicit gating mechanism.
- **StairMaster: Learning to Conquer Risky Hollow Stairs for Agile Quadrupedal Robots**
  (2026), https://arxiv.org/html/2606.25765v1 — not wheeled-legged (point-foot quadruped),
  so out of primary scope, but flagged because the search surfaced it as directly relevant
  to depth-noise modelling quality: it explicitly claims "a high-fidelity sim-to-real depth
  sensor modeling pipeline that faithfully replicates real-world sensor artifacts" plus a
  cross-attention + "Spatial-aware Recurrent Unit" for robustness to perception blind spots
  on discontinuous (hollow/open-riser) stairs. **Low confidence** — title/abstract-level
  only, not fetched in full; worth a closer read later specifically for its depth-noise
  pipeline design if our Phase 7.1 noise model needs a second concrete reference beyond
  Humanoid Parkour Learning (#5) and Parkour in the Wild (#6).

Overall takeaway from this scan: genuinely wheeled-legged stair-climbing RL work remains
sparse even in 2025-2026 — most of what surfaced is either (a) navigation-focused rather
than stair-specific (Go2-W long-distance nav), (b) recovery-focused rather than climbing-
performance-focused (Stand/Walk/Navigate), or (c) not yet readable in full via automated
fetch (CTBC, Go2-W, StairMaster, MoE bipedal). Sources #7 (Chamorro) and #8 (Lee et al.)
remain by far the most substantive, fully-extracted wheeled-legged stair-climbing references
available, and should be treated as the primary literature anchors for Phase 3/4, with the
2025-2026 items as secondary leads for later refinement (recovery behaviour, MoE
architectures, contact-triggered mode switching) rather than primary design sources.

---

## What we will adopt (Phase 3/4 recommendations)

The M20 is a **wheeled** quadruped (12 position-controlled leg joints + 4 velocity-controlled
wheels), not a point-foot or paw-foot legged robot. Of the ten core sources, only #7
(Chamorro) and #8 (Lee et al.) actually deal with this morphology; everything else (#1–6, #9)
is legged-only and transfers *conceptually* (network shapes, distillation pipeline, depth
noise modelling) but not *mechanically* (their reward/observation terms assume discrete
footfalls, which the M20's rolling wheels don't have in the same way). Recommendations below
are grouped by area, each tagged with which source(s) justify it and an explicit note on
wheeled-vs-legged applicability.

### Terrain mix (Phase 3.1)
- Keep the planned split (ascent/descent pyramid stairs, varied tread, random_rough, ramps,
  flat) — the proportions in `command.md` are already reasonable relative to source #3's
  100-tile 20×10 grid and source #7's 12-row curriculum.
- **Adopt an up/down-decoupled curriculum promotion rule** (source #9, legged_lab_v3): track
  ascent-tile and descent-tile difficulty levels independently rather than promoting on one
  aggregate `terrain_levels_vel` metric. This is directly motivated by our own Phase 1
  finding — descent already succeeds 58-100% while ascent is ~0% — so a coupled curriculum
  risks the aggregate metric being dragged down by ascent failures while descent tiles never
  get harder, or conversely never surfacing enough ascent difficulty signal separately from
  descent. Worth implementing before Phase 4 training starts, not as an afterthought.
- Chamorro (#7) uses 6 rows of single-step-then-flat followed by 5-step continuous flights,
  distinct from Isaac Lab's pyramid style; consider adding a "single step of increasing
  height" sub-terrain type alongside the pyramid stairs, since it isolates the specific
  wheel-vs-riser collision failure mode our Phase 1 baseline already showed (heavy wheel
  stumbling) from the general multi-step endurance problem.

### Command/heading scheme (Phase 3.2) — the one place we should reconsider the draft plan
- `command.md`'s draft assumes a **velocity-command** scheme inherited from the existing M20
  task. Source #7 (Chamorro) explicitly argues and demonstrates that a **position-based**
  goal command (reach a target x,y + heading, rewarded densely only in the final window of
  the episode, non-terminal reward gently pulling the base's velocity toward the goal
  direction) is "vital for stair climbing," specifically because on stairs a fixed forward
  speed is often the wrong objective — sometimes the correct thing is to pause, shift weight,
  and step, not maintain a commanded m/s. Recommendation: **keep the existing velocity-
  tracking scheme as primary (for consistency with the blind baseline and eval_stairs.py's
  velocity-based methodology already run in Phase 1)**, but add Chamorro's position/progress
  reward terms (#1-4 in the table above) *as additional shaping terms layered on top of*
  velocity tracking, rather than replacing the command generator outright — this is lower
  risk than a full command-scheme rewrite and is compatible with our existing eval harness.
  If ascent performance still plateaus after Phase 4.2 reward changes, revisit a full switch
  to Chamorro's position-based formulation as a fallback.
- Adopt Chamorro's **binary "stair-mode" flag** in the actor observation (a near-zero-cost
  addition: one extra bit) as a cheap alternative/complement to terrain-relative orientation
  rewards — it lets the policy learn a genuinely different stair gait (larger leg split,
  slower cadence, more pitch tolerance) without that gait bleeding into flat-ground driving
  behaviour. This directly addresses the reward-conflict problem named in our task brief
  (`flat_orientation_l2` -50 fighting the ~40° pitch stairs require): rather than only
  softening that penalty globally, gate a softer version of it on the stair-mode flag so flat
  driving keeps its current orientation discipline. Contrast with Lee et al. (#8), who show
  mode-switching can also emerge with *no* explicit flag — but given we already have a
  concrete, named reward conflict to resolve and limited compute budget for iteration, the
  explicit flag is the lower-risk, more debuggable choice for Phase 4.2.

### Reward changes (Phase 4.2)
- The `flat_orientation_l2` -50 / `bad_orientation_penalty` -1000 @0.7 conflict identified in
  our own baseline: none of the fully-legged sources hit this problem as sharply because
  paw-foot robots don't carry a rigid wheeled chassis whose pitch directly determines wheel-
  ground contact geometry the way M20's does. Chamorro's stair-mode flag (above) is the most
  directly transferable mitigation. Concretely: keep `flat_orientation_l2` at -50 gated
  off when stair-mode is active (or reduce to roughly -5, as `command.md` already proposes),
  and raise the `bad_orientation_2` termination threshold from 0.7 toward ~0.85 **only** when
  stair-mode is active, consistent with the plan's own Phase 4.2 items 1-2 — this survey adds
  the stair-mode-flag mechanism as the concrete way to gate it rather than a global change.
- Enable `feet_stumble` at a small negative weight, but tune it the way Chamorro (#7) frames
  the problem: **not purely as a stumble penalty**, but jointly with wheel-ground friction
  randomization, because Chamorro found that if sim friction is high enough for wheels to
  "grip" a riser, the policy learns a climbing strategy relying on grip that fails on the
  real robot's actual tire dynamics. Randomize wheel-ground friction explicitly and lower
  than what feels physically "sticky" in sim, per Chamorro's finding — this is a more
  targeted lead for our Phase 1 finding of "heavy wheel-riser stumbling" than a stumble
  penalty alone would be.
- Add a foot/wheel clearance-over-edge reward analogous to Extreme Parkour's (#1) edge-
  clearance term, adapted for wheels: reward wheel height above locally-sampled terrain
  height when a wheel isn't in ground contact, rather than the standard swing-foot clearance
  formula (which assumes discrete footfalls).
- Consider Parkour-in-the-Wild's (#6) critic-pretraining trick (pre-train the critic with
  frozen actor weights before resuming joint PPO updates) if we do the optional Phase 8.3 RL
  fine-tuning of the distilled student, or even when warm-starting the teacher from the blind
  checkpoint (Phase 4.3 option 2) — the new height-scan input columns start uninformative, so
  a brief critic-only warm-up before joint updates may reduce early instability the same way
  it does for their distillation→fine-tuning transition.

### Observation design (Phase 4.1 / Phase 6-8)
- Teacher actor + critic asymmetric split (clean height scan + privileged terrain/physics
  terms to critic only) matches Chamorro (#7, 187-dim height grid to critic, friction/contact
  force also critic-only) and Stand-Walk-Navigate almost exactly — strong convergent evidence
  this is the right default, no change needed to the plan.
  Chamorro's actor-side noise levels are a useful concrete starting point if we don't already
  have our own calibrated numbers: angular velocity noise ±20%, projected gravity ±5%, joint
  position noise ±1%, joint velocity noise ±150% (their number is surprisingly large — worth
  sanity-checking against our own encoder specs rather than copying blindly).
- For the student (Phase 6/8): the CNN-channels-[16,32,32]/kernel-[5,4,3]/stride-[2,2,1] +
  GRU-256 + MLP-[512,256,128] combination recurs almost identically across Extreme Parkour
  (#1), Robot Parkour Learning (#4), and Humanoid Parkour Learning (#5) — treat this as a
  well-validated default starting architecture rather than something to search from scratch.
  Depth resolution: this survey found three different concrete choices (58×87 in #1/#3,
  48×64 in #4/#5, 48×32 in #6) — all are far below native sensor resolution; 58×87 or 64×64
  (as `command.md` already proposes) is a reasonable middle choice.
- **legged_lab_v3's 8-frame depth history with explicit latency + self-occlusion modelling
  (#9)** is the most directly relevant precedent for `command.md`'s Phase 6.1 concern that
  Isaac Lab's cheap `RayCasterCameraCfg` doesn't see the robot's own legs/wheels — worth a
  deeper read of that repo's actual observation code (not just README) before implementing
  Phase 7's self-occlusion mask, since it's solving exactly this problem on a similar
  Isaac Lab stack.
- Given the M20 has **no stock depth camera** and the human hasn't yet chosen between D1
  (add a depth camera) and D2 (LiDAR-derived elevation map) per Phase 5: if D2 is chosen, the
  most directly relevant noise model in this survey is *not* any of the depth-specific
  sources, but rather adapting Chamorro/Lee's **height-scan-based critic observation with
  added noise, latency, and drift** — since D2 essentially trains the student on a noisy
  height map rather than a noisy depth image, sources #7 and #8's actor-side proprioceptive
  noise treatment (not their depth-specific counterparts, since neither uses a depth camera)
  is the better template. If D1 is chosen, sources #3, #5, and #6 (in that order of
  concreteness) are the best templates: #3 for real-camera crop/hole-fill/downsample/latency
  numbers, #5 for the "clipping + multi-level Gaussian noise + random artifacts" framing
  named in `command.md`'s Phase 7.1, and #6 for the most concrete edge-noise/hole
  implementation (Perlin-noise-gated holes, gradient-based edge perturbation) — recommend
  implementing #6's edge/hole mechanism specifically, since it's the only source that gives
  an actual algorithm (not just a named category) for the "flying pixels at step edges"
  failure mode `command.md` Phase 7.1 already anticipates.

### Network architecture (Phase 4.1, 4.3, 8.1) — UPDATED: user has chosen the CTS-MoE (#11) direction
- Superseding the note below (kept for the record of the original single-MLP reasoning): the
  user's stated direction is to adopt **CTS-MoE (#11)** rather than a monolithic teacher +
  sequential-DAgger student. Concretely this means: a **dense mixture-of-experts actor** (each
  expert an MLP the same size as the plan's existing `[512,256,128]` trunk — so the "cross-
  validated MLP size" finding below still applies, just per-expert), a **perception-based
  router** `[512,256]` gated on the concatenated teacher+student latent, a **multi-critic**
  with one value head per task (at minimum: flat, stair-ascent, stair-descent; optionally
  rough/ramp as a fourth), and **single-stage concurrent teacher-student training** with an
  MSE encoder-alignment loss every PPO update, replacing the planned Phase 4-then-Phase 8
  sequential teacher/distillation split. This directly resolves the "second resort, not first
  choice" hedge below — the user has decided to make it the first choice — but note the
  evidence base is entirely legged-only (#11 is Unitree Go1, no wheels); the wheeled-specific
  levers (#7, #8: wheel-ground friction, stair-mode flag as a *task label*, grip-vs-step
  tuning) still apply and should be folded in as task definitions / reward terms *within* this
  architecture, not discarded. See the proposed `command.md` restructuring under discussion
  with the user for how this maps onto phases.
- *(Original single-MLP reasoning, superseded above but kept for context:)* Teacher: no reason
  to deviate from the plan's existing `[512, 256, 128]` MLP — this exact size recurs in #1,
  #4, #5 as the actor/critic MLP trunk after the encoder. Student: CNN[16,32,32]/
  kernel[5,4,3]/stride[2,2,1] → GRU(256) → MLP[512,256,128], actor head initialized from the
  teacher actor. If wheel/leg mode-switching proves hard to tune with a single monolithic
  policy, MoE is a fallback — Lee et al. (#8) showed a single non-MoE network *can* learn
  emergent switching without an explicit gate, so MoE was framed as a second resort. The 2025
  bipedal wheel-legged sparse-MoE PubMed entry (see "newer work" section) was the only prior
  MoE lead before #11 was added.

### Domain randomization (Phase 7.2)
- Wheel-ground friction range: Chamorro (#7) and Lee et al. (#8) both single out wheel
  friction/actuator fidelity as the dominant sim2real risk for this robot class (more so than
  for point-foot robots) — prioritize this over generic friction randomization already listed
  in `command.md`. Concretely: (a) randomize wheel-ground friction independently from
  leg/foot-terrain friction, per `command.md`'s existing plan, but bias the sampled range
  toward the low end to avoid the "grip-climbing" failure mode Chamorro describes; (b)
  consider Lee et al.'s more elaborate approach — a learned/data-driven wheel actuator model
  (velocity-command-history → motor current) rather than an idealized velocity PD — as a
  stretch goal if wheel actuator fidelity turns out to be a bottleneck after Phase 4 training.
- Control-loop / observation latency: Chamorro's simple 50%-probability-of-stale-state trick
  (≈20 ms modeled delay) is a cheap way to get some latency robustness into the teacher even
  before Milestone B's more careful depth-latency modelling.

### Summary of the single most important wheeled-specific takeaway
Every fully-legged source in this survey (#1-6, #9) either has no stumbling/friction-vs-
stepping tension at all, or handles it with footfall-specific rewards that don't map onto a
rolling contact. The two wheeled-legged sources (#7, #8) independently converge on the same
finding: **the grip-vs-step tradeoff at the wheel-riser contact is the dominant, wheel-
specific failure mode**, more so than orientation, gait style, or network capacity — and it
is precisely the failure mode our own Phase 1 baseline already observed ("heavy wheel-riser
stumbling" on ascent). Phase 4.2's reward-tuning work should treat wheel-ground friction
randomization and shaping-reward balance (grip vs. discrete stepping) as the primary lever
for closing the ascent gap, with the orientation-penalty relaxation and stair-mode flag as
important but secondary levers.
