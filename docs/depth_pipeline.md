# Depth pipeline: RealSense D435i, sim and robot

The student policy's `depth` input is a **(1, 36, 64) image in [0, 1], where 0 means "no data"**. It
is produced by the same `preprocess()` function in simulation and on the robot. This document is the
contract between the two sides. If you change one side, change the other.

Code:

| File | Runs on | What |
|---|---|---|
| `source/rl_training/rl_training/depth/d435i.py` | both | `D435iSpec` (stream, FOV, range, timing) and `CameraMount` (pose on the M20) |
| `source/rl_training/rl_training/depth/preprocess.py` | both | `preprocess()`: sanitize, zero-ignoring median pool, normalize |
| `source/rl_training/rl_training/depth/stereo_noise.py` | sim | stereo occlusion, disparity noise, random holes |
| `.../velocity/mdp/depth_d435i.py` | sim | Isaac Lab observation term + mount-jitter event |
| `scripts/tools/test_depth_pipeline.py` | CPU | unit tests for the three pure-torch files |

`d435i.py` and `preprocess.py` only import `math` / `torch`, so the robot can copy the `depth/` folder
without Isaac Lab.

## Robot side

1. **Stream:** depth `z16` at **848 x 480, 60 fps** (`D435iSpec.native_*`, `fps`). Emitter on.
2. **RealSense post-processing filters: OFF** (no decimation / spatial / temporal / hole-filling).
   The simulator does not model them. If one turns out to be needed, it has to be added to the
   simulator too.
3. Convert to metres: `depth_m = frame.astype(float32) * depth_scale` (depth_scale from the device,
   normally 0.001). Leave 0 as 0.
4. `img = preprocess(torch.from_numpy(depth_m), 36, 64, near=spec.min_z_m, far=spec.far_m,
   min_valid_frac=spec.min_valid_frac)` gives the (36, 64) image. Add batch and channel dims to get
   (1, 1, 36, 64).
5. Feed the newest image to the policy at every 50 Hz step. If no new frame has arrived, reuse the
   last one (the sim models this as latency + frame drops).

What `preprocess()` does:

- **sanitize:** non-finite, <= 0, or closer than `min_z_m` (0.2 m) becomes 0 (invalid). Anything
  beyond `far_m` (3.0 m) is clamped to 3.0.
- **pool:** each output pixel covers a 13 x 13 block of the 848 x 480 frame. It takes the median of
  the block's *valid* pixels, or 0 if fewer than `min_valid_frac` (50%) of them are valid. A median
  keeps step edges sharp (it returns one surface's depth, never an average of two), which is what
  the simulator's one ray per pixel produces.
- **normalize:** 0.2..3.0 m maps to 0.001..1. Invalid stays exactly 0.

## Simulator side (`mdp.d435i_depth_image`)

The ray caster casts one ray per output pixel (64 x 36) against the terrain mesh and updates at
60 Hz. Per policy step:

1. **sanitize** (same as robot).
2. **Stereo occlusion** (geometry, always on). Depth is referenced to the left imager, and the right
   imager is 50 mm to its right. Pixels the right imager can't see become invalid: a band on the
   left of near objects, and a strip on the left image border. The baseline is horizontal, so this
   happens beside vertical edges (walls, the ends of steps, stairs approached at an angle), not
   along head-on stair nosings.
3. **Disparity noise** (only with corruption on): Gaussian noise in disparity space (so depth error
   grows with z^2: about 4 / 14 / 32 mm at 1 / 2 / 3 m for 0.08 px), a smooth low-frequency field,
   1/32 px quantization, and a per-env depth scale error of +/-2%.
4. **Random holes** (only with corruption on): per-env pixel dropout, rectangular holes, and dropped
   pixels at depth edges.
5. **preprocess()** (same as robot; pooling is the identity at 64 x 36).
6. **Latency 20-60 ms** (1-3 policy steps, per env), plus 2% frame repeats.

Per-env mount jitter at reset (`mdp.randomize_camera_offset`): +/-1 cm, +/-2 deg roll/yaw, +/-3 deg
pitch.

`Stairs-Student-Deeprobotics-M20-Play-v0` turns off steps 3, 4 and 6 and the jitter. Stereo occlusion
stays on because it is geometry, not noise.

## Open items / VERIFY on the real unit

- [ ] **Mount pose.** `M20_D435I_MOUNT` is still the placeholder (0.38 m forward, 0.05 m up, 40 deg
      down). Measure the real bracket.
- [ ] **Self-occlusion.** The ray caster only sees `/World/ground`. If the M20's front wheels or legs
      are in the image at the real mount, model it: a fixed mask, or Isaac Lab's
      `MultiMeshRayCasterCamera`, which can also hit the robot's own meshes.
- [ ] **Min-Z** at 848 x 480 (assumed 0.2 m) and the **latency** (assumed 20-60 ms). Measure both.
- [ ] **Noise levels.** Record the D435i on real stairs at the mount height and angle, run the frames
      through `preprocess()`, and compare with `scripts/tools/dump_depth.py` output: invalid fraction,
      depth noise vs distance, and how edges and the left-side occlusion bands look. Adjust the
      `sigma_px_range` / `lowfreq_px_range` / hole params in `mdp.d435i_depth_image` to match.
- [ ] If real edges look clearly different from sim after pooling, cast rays at 2x resolution in sim
      and pool with the same `preprocess()` ("option B"). Costs 4x rays.
