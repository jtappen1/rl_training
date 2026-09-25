# Sighted stairs spike v2: what fixed stair climbing

*2026-09-23. Task `Stairs-Sighted-V2-Deeprobotics-M20-v0`, run `logs/rsl_rl/deeprobotics_m20_stairs_sighted_v2/2026-09-23_15-04-55` (stopped at iteration 5300).*

## TL;DR

The v1 sighted policy (plain MLP, height scan in the actor) could not climb 0.20 m stairs. It stopped in front of the first riser on ascent and teetered at the top edge on descent. The cause was the **reward economics**: stopping earned more than climbing. The robot was not short of stair experience, since stairs were already ~70% of terrain columns. Rebalancing the rewards, loosening the orientation limit on stairs, adding a progress reward and fixing the curriculum rule brought it to **100% ascent and descent up to 0.25 m, 62–92% at 0.30 m, with 0 falls in 864 eval trials.**

Everything below went in together as one run, so there is no per-change attribution yet (see [Open items](#open-items)).

## Results

`eval_stairs.py`, fixed-geometry pyramid stairs, 24 trials per cell (3 tread widths × 8 repeats). Full table: `eval/eval_stairs_20260923_180916.md`.

| Step height | Ascent 0.5 m/s | Ascent 1.0 m/s | Descent 0.5 m/s | Descent 1.0 m/s |
|---|---|---|---|---|
| 0.08–0.20 m | 100% | 100% | 100% | 100% |
| 0.23 m | 100% | 100% | 100% | 100% |
| 0.25 m | 100% | 100% | 100% | 100% |
| 0.28 m | 92% | 71% | 100% | 100% |
| 0.30 m | 71% | 62% | 92% | 96% |

**v1 for comparison** (`eval/eval_stairs_20260923_143505.md`, 1 trial per cell): 0.10 m ascent and descent succeeded; 0.20 m ascent and descent both failed with 0 stumbles and 0 collisions. The robot never touched the stairs.

Training curves: the stair-ascent terrain level ended at ~6.0 (~0.22 m steps with the new 0.08–0.30 m range), against v1's 5.5 (~0.17 m steps with its 0.05–0.26 m range).

Videos (height-map overlay on, 0.30 m tread, 0.5 m/s): `eval/videos_v2_model5300/`.

## Why v1 failed

Each item below was checked against the v1 config and its logged reward terms.

| Problem | Effect |
|---|---|
| Velocity tracking std = √0.5 | A robot standing still under a 0.5 m/s command still earned 61% of the tracking reward (84% at 0.3 m/s). Climbing only gained the remainder, and it carried risk. |
| `flat_orientation_l2` = −15 on all terrain | 0.20/0.30 m stairs need ~34° of body pitch, which costs ~4.7/s. That is almost the whole tracking reward (5/s max). |
| Bad-orientation penalty (−1000) and termination at \|g_xy\| > 0.7 (~44°) | 0.30/0.30 m stairs are 45°, so they were infeasible by construction. Nosing over a tall edge on descent spiked pitch past the limit, and that is what caused the teetering at the top. |
| hipy/knee posture penalty relaxed only when the terrain under the base was 6 cm above the spawn origin | Full penalty applied while approaching the first riser (so no leg lift before the step) and for the whole descent. |
| All air-time/gait rewards gated to turning in place | Nothing rewarded lifting a leg while driving forward. |
| Height scan ±0.8 m around the base, ±0.10 m noise | Only ~0.4 m of lookahead past the front wheels, and the noise was half a 0.2 m riser. |
| Wheel radius 0.09 m | A 0.20 m riser is more than 2× the wheel radius and can't be rolled over. Below ~0.17 m the robot could roll over the first step, which is why v1 plateaued there. |

## What changed in v2

New config: `DeeproboticsM20StairsSightedV2EnvCfg` in `config/wheeled/deeprobotics_m20/stairs_env_cfg.py`. The new reward, termination and curriculum functions live in `mdp/stairs.py`; none of the shared functions were edited, so Rough/Flat behave as before.

"On stairs" below means the env's terrain column belongs to the `stair_ascent` or `stair_descent` task (the Phase 3.4 task IDs).

1. **Tighter velocity tracking.** `track_lin_vel_xy_yaw_frame_exp` (measures horizontal speed, so a pitched body isn't penalised) with std 0.3. Standing still now earns about 6% of the reward instead of 61%.
2. **Orientation that allows climbing.**
   - `orientation_l2_stair_aware`, weight −50: on stairs, roll is fully penalised and pitch at 5%; elsewhere unchanged.
   - `bad_orientation_stair_aware` (termination) and its −1000 penalty: pitch limit 0.9 on stairs, 0.7 everywhere else.
3. **Stair progress reward** (`stair_progress`, weight 8).
   - Pays for increases in the episode-best L∞ distance from the tile centre, plus episode-best base height on ascent.
   - Because it tracks the best so far, oscillating back and forth earns nothing. On a square pyramid, "further out" is "further up/down" in any heading.
   - Only paid on stairs, when the command is non-trivial, and scaled by how well the command points outward. It never rewards ignoring a sideways command.
4. **Posture penalty gated on terrain ahead** (`joint_pos_penalty_terrain_ahead`). The hipy/knee penalty drops to 0.1× when the scan ahead is uneven (max − min > 8 cm), not when the base is above the spawn height.
5. **Wheel clearance shaping** (`stair_wheel_clearance`, weight 1).
   - Pays for an airborne wheel clearing the riser within 0.3 m in front of it.
   - Only while the robot is actually moving along the command, so it can't be earned by standing at a step with a leg up.
   - `feet_stumble` raised from −1 to −3.
6. **Better height scan.** 2.0 × 1.0 m grid shifted forward (0.4 m behind to 1.6 m ahead); noise ±0.1 → ±0.025 m.
7. **Terrain and curriculum.**
   - Step heights 0.08–0.30 m. Main treads 0.32 m, wide 0.35 m, narrow 0.26 m; the narrow variant is capped at 0.24 m so no flight exceeds 45°.
   - Smaller centre platform and spawn offset ±0.3 m (was ±1.0 m), so more of each episode is spent on stairs.
   - New `terrain_levels_stairs` rule for stair columns: promote on getting off the stairs; demote only if the robot got less than ~⅓ of the way while commanded to move; otherwise stay.
   - The old rule demoted every partial climb at commands ≥ 0.4 m/s.
   - Stair share of columns unchanged (~70%).

## Tooling added

- **Height-map video overlay** (`scripts/tools/heightmap_overlay.py`, on by default with `eval_stairs.py --video`; `--no_heightmap_viz` to disable):
  - Colour-coded spheres at the ray hits in the scene.
  - A top-down, forward-up inset of the grid the policy reads.
- **`eval_stairs.py`** now remaps the stair terms' terrain-column → task mapping onto its own eval terrain. Without this, the stair-aware termination would use the wrong limits or index out of bounds.
- **`set_task_ids_per_column(env_cfg, columns)`** in `stairs_env_cfg.py`: updates every term that bakes in the column layout. Use it whenever the terrain grid changes (Play variants, eval).

## Reproduce

```bash
cd /workspace/rl_training
# train (run inside tmux, or with setsid nohup, so it survives a dropped SSH session)
PYTHONUNBUFFERED=1 /workspace/isaaclab/isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Stairs-Sighted-V2-Deeprobotics-M20-v0 --num_envs 6144 --headless

# eval
/workspace/isaaclab/isaaclab.sh -p scripts/tools/eval_stairs.py --task Stairs-Sighted-V2-Deeprobotics-M20-v0 \
  --checkpoint logs/rsl_rl/deeprobotics_m20_stairs_sighted_v2/2026-09-23_15-04-55/model_5300.pt --headless \
  --step_heights 0.08 0.12 0.15 0.18 0.20 0.23 0.25 0.28 0.30

# per-env video (follows one robot; envs are ordered ascent then descent, by height, then by tread)
/workspace/isaaclab/isaaclab.sh -p scripts/tools/eval_stairs.py --task Stairs-Sighted-V2-Deeprobotics-M20-v0 \
  --checkpoint <same> --headless --video --video_length 400 --follow_env_id 0 \
  --repeats 1 --speeds 0.5 --step_heights 0.25 0.30 --tread_widths 0.30 --out_dir eval/vid_env0
```

## Open items

- **No attribution.** All seven changes went in at once. The Phase 4.4 one-at-a-time ablations would show which ones matter; the tracking std, the orientation limits and the progress reward are the strongest suspects.
- **Wheels still hit risers:** ~12–15 wheel-riser stumbles per ascent trial at every height. The clearance reward stayed small (~0.08/s), so the robot mostly bumps and climbs rather than lifting before the step.
- **Body/knee edge collisions at 0.28–0.30 m:** 2–4 per trial.
- **Very steep pitch on ascent.** It stays inside the 0.9 limit, which may be looser than we want on hardware.
- **Candidate next steps:**
  - Raise the clearance weight and the stumble penalty.
  - Penalise knee/thigh contact with stair edges.
  - Tighten the stair pitch limit to ~0.8.
  - Check the gait-quality acceptance item in the videos.
- **Still a plain-MLP spike.** It's not the Milestone A MoE / multi-critic architecture, but it shows that sightedness plus reward shaping is enough at this task scale. That makes a strong baseline for the Phase 4.2/4.3 ablations.
- **Not yet committed.** Nothing is in `EXPERIMENTS.md` yet either.
