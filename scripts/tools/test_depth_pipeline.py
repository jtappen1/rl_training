"""CPU unit tests for the D435i depth pipeline (rl_training/depth). No Isaac Sim needed.

Usage:
    /workspace/isaaclab/_isaac_sim/python.sh scripts/tools/test_depth_pipeline.py
"""

import os
import sys

# import the pure-torch modules directly: `import rl_training` would pull in Isaac Lab
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../source/rl_training/rl_training/depth"))

import torch  # noqa: E402

from d435i import D435I  # noqa: E402
from preprocess import pool_depth, preprocess, sanitize  # noqa: E402
from stereo_noise import disparity_noise, stereo_invalid_mask  # noqa: E402

torch.manual_seed(0)
S = D435I
NEAR, FAR = S.min_z_m, S.far_m


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")
    if not cond:
        sys.exit(1)


def test_spec():
    check("native focal ~447 px", abs(S.native_focal_px - 447) < 2, f"{S.native_focal_px:.1f}")
    check("policy vfov ~56-58 deg", 55 < S.vfov_deg < 59, f"{S.vfov_deg:.1f}")
    check("rms @2m ~14 mm", 0.012 < S.depth_rms_m(2.0) < 0.016, f"{S.depth_rms_m(2.0) * 1000:.1f} mm")


def test_sanitize():
    d = torch.tensor([[float("nan"), 0.0, 0.1, 1.0, 5.0, float("inf")]])
    s = sanitize(d, NEAR, FAR)
    check("sanitize", torch.equal(s, torch.tensor([[0.0, 0.0, 0.0, 1.0, FAR, FAR]])), str(s.tolist()))


def test_pool():
    same = torch.rand(2, S.height, S.width) + 0.5
    check("pool identity at policy res", torch.equal(pool_depth(same, S.height, S.width), same))

    plane = torch.full((S.native_height, S.native_width), 1.3)
    out = pool_depth(plane, S.height, S.width)
    check("pool constant plane", out.shape == (36, 64) and torch.allclose(out, torch.tensor(1.3)))

    # step edge inside every block: top half 0.8 m, bottom half 1.0 m -> median is one side, never ~0.9
    blocks = torch.ones(S.height * 13, S.width * 13)
    for r in range(S.height):
        blocks[r * 13 : r * 13 + 6] = 0.8
    out = pool_depth(blocks, S.height, S.width)
    ghost = ((out > 0.82) & (out < 0.98)).sum().item()
    check("median never returns a ghost depth", ghost == 0, f"values={sorted(set(out.flatten().tolist()))}")

    # invalid fraction rule
    b = torch.full((13, 13), 1.0)
    b.view(-1)[:101] = 0.0  # 60% invalid
    check("block 60% invalid -> invalid", pool_depth(b, 1, 1).item() == 0.0)
    b = torch.full((13, 13), 1.0)
    b.view(-1)[:67] = 0.0  # 40% invalid
    check("block 40% invalid -> valid", pool_depth(b, 1, 1).item() == 1.0)


def test_preprocess():
    raw = torch.full((S.native_height, S.native_width), 2.0)
    raw[:, :100] = 0.0  # camera "no data"
    img = preprocess(raw, S.height, S.width, NEAR, FAR, S.min_valid_frac)
    check("preprocess shape", img.shape == (36, 64))
    check("preprocess invalid -> 0", (img[:, :7] == 0).all().item())
    exp = (2.0 - NEAR) / (FAR - NEAR) * (1 - 1e-3) + 1e-3
    check("preprocess valid value", torch.allclose(img[:, 10:], torch.tensor(exp)), f"{img[0, 10].item():.4f}")
    edge = preprocess(torch.tensor([[NEAR, FAR]]), 1, 2, NEAR, FAR)
    check("near -> 1e-3, far -> 1", torch.allclose(edge, torch.tensor([[1e-3, 1.0]])), str(edge.tolist()))


def test_disparity_noise():
    n = 4096
    for z0 in (1.0, 2.0, 3.0):
        z = torch.full((n, 4, 4), z0)
        sig = torch.full((n,), 0.08)
        zn = disparity_noise(z, S.native_focal_px, S.baseline_m, sig, torch.zeros(n), torch.ones(n))
        std = zn.std().item()
        exp = S.depth_rms_m(z0, 0.08)
        check(f"disparity noise std @ {z0} m", abs(std - exp) / exp < 0.15, f"{std * 1000:.1f} mm vs {exp * 1000:.1f} mm")
    z = torch.zeros(2, 3, 3)
    zn = disparity_noise(z, S.native_focal_px, S.baseline_m, torch.ones(2), torch.ones(2), torch.ones(2))
    check("invalid stays invalid under noise", (zn == 0).all().item())


def test_stereo_mask():
    f = S.focal_px(S.width)
    z = torch.full((1, 4, S.width), 2.0)
    z[:, :, 30:40] = 0.5  # near box in front of a 2 m wall
    m = stereo_invalid_mask(z, f, S.baseline_m)[0, 0]
    band = f * S.baseline_m * (1 / 0.5 - 1 / 2.0)  # expected shadow width (px)
    shadow = (m[20:30].nonzero().flatten() + 20).tolist()
    check(
        "occlusion band LEFT of near box",
        len(shadow) in (int(band), int(band) + 1) and shadow[-1] == 29 and not m[40:].any(),
        f"cols {shadow}, expected ~{band:.1f} px",
    )
    check("box itself valid", not m[30:40].any())
    border = f * S.baseline_m / 2.0
    check("left border band", m[0].item() and not m[2].item(), f"expected ~{border:.2f} px at 2 m")
    flat = torch.full((1, 4, S.width), 1.0)
    check("flat plane: only left border", stereo_invalid_mask(flat, f, S.baseline_m)[0, :, 2:].sum().item() == 0)


if __name__ == "__main__":
    test_spec()
    test_sanitize()
    test_pool()
    test_preprocess()
    test_disparity_noise()
    test_stereo_mask()
    print("[ALL PASS]")
