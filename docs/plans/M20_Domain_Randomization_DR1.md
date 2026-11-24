# M20 Domain Randomization (DR1)

Nov 22, 2026 · John

## Context

DR1 adds the terms the real robot needs and widens ranges only where a measurement says so. Values come from the [M20 Sim-to-Real README](https://claude.ai/artifact/EA51CNeqgZCAwUr2wSaoge) (bag `s2r_082342`).

- **Task:** `Rough-Deeprobotics-M20-DR1-v0`, subclassing `DeeproboticsM20RoughEnvCfg` so the vendor config stays untouched.
- **Files:** `assets/deeprobotics.py` (actuators) and `deeprobotics_m20/rough_env_cfg.py` (events, noise).

## Changes

| Change | Range | Implementation | Source |
| --- | --- | --- | --- |
| Wheel drive friction (**off for now**: ranges set to 0 until the friction model is confirmed) | 0–3 N·m wheels, 0–0.5 N·m legs (proposed) | Custom actuator subclassing `DelayedPDActuator`: subtract τc · tanh(q̇ / 0.1) after PD; draw τc per env at reset | ≥ 2.3 N·m measured on FL; 0 in training |
| Latency | 1–3 physics steps (5–15 ms) | `min_delay=1`, `max_delay=3` in the same actuator; event repeats the previous action for one policy step with 2% probability | 10–12 ms, spikes to ~40 ms; 0–5 ms in training |
| Observation noise | Halved: ang vel ±0.1, gravity ±0.025, joint vel ±0.75 | `Unoise` ranges in the policy obs group | Real noise 60–100× below training |
| IMU bias | ±1° roll and pitch per env | Reset event draws the offset; custom `projected_gravity` obs rotates by it | +0.20° roll, +0.46° pitch standing |
| Wheel armature | Scale 0.5–1.5 around 0.00243 | `randomize_joint_parameters` on wheels; set 0.00243 in MuJoCo `M20.xml` | Not measured; Isaac and MuJoCo disagree |
| Restitution | 0–0.2 (was 0–0.7) | `randomize_rigid_body_material` | Not measured; 0.7 is too bouncy for rubber |
| Starting joint positions | ±0.1 rad | `reset_joints_by_offset` (replaces `reset_joints_by_scale`, which does nothing) | None today |
| Knee torque limit | Raise only if spec allows | `effort_limit` on knee actuators | 81 N·m seen vs 76.4 N·m limit |

Put friction and delay in one actuator class. The PD model already matches real within 2%, so `RandomPDActuator` isn't needed.

## Unchanged and open

Keep as is: kp/kd ±15%, ground friction 0.35–1.5, base mass and centre-of-mass ranges, pushes, external force.

- [ ] Confirm the knee motor's peak torque spec.
- [ ] Wheel collision is a ~23 cm wide capsule vs a 5.4 cm tyre: fix in DR1 or later?
- [ ] Payload: out of scope for now.

## Rollout

1. Add the DR1 task and register it as its own gym id.
2. Write the new pieces: friction + delay actuator, IMU-bias event and observation, action-repeat event.
3. Train 200 iterations and confirm each new term is active (non-zero friction, delay, bias in logs).
4. Train the full run with the vendor rewards unchanged, so the effect of DR alone is visible.
5. Score it with the [M20 Evaluation Suite](https://claude.ai/artifact/Xu577ZdXpm9wZngauwHD7N), randomized pass, against both baselines.
