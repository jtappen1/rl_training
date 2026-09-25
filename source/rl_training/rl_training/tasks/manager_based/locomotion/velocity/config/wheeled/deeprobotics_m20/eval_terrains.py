# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause

"""Fixed evaluation terrains for the M20: the stairs benchmark and the multi-terrain course.

Shared by the headless evals (`scripts/tools/eval_stairs.py`, `scripts/tools/eval_multi_terrain_course.py`)
and the keyboard play envs in `eval_play_env_cfg.py`, so driving by hand happens on exactly the
geometry the evals score.
"""

from __future__ import annotations

import itertools
from collections import OrderedDict
from dataclasses import MISSING

import numpy as np
import trimesh

import isaaclab.terrains as terrain_gen
from isaaclab.terrains import SubTerrainBaseCfg, TerrainGeneratorCfg
from isaaclab.terrains.height_field.utils import convert_height_field_to_mesh
from isaaclab.utils import configclass

# ---------------------------------------------------------------------------
# Stairs benchmark: one pyramid-stairs tile per (direction, step height, tread width).
#
# These mirror the values ROUGH_TERRAINS_CFG already uses for its pyramid_stairs /
# pyramid_stairs_inv sub-terrains (isaaclab/terrains/config/rough.py), so the eval tiles are the
# same "shape" of stairs the blind policy has already seen during training -- only the step
# height/tread width are pinned to exact values instead of being difficulty-sampled.
# ---------------------------------------------------------------------------
TILE_SIZE = 8.0
PLATFORM_WIDTH = 2.5
SUB_BORDER_WIDTH = 1.0
GEN_BORDER_WIDTH = 4.0

DIRECTIONS = ("ascent", "descent")


def num_stairs_steps(step_width: float) -> int:
    """Mirror the step count formula in isaaclab's pyramid_stairs_terrain / inverted_pyramid_stairs_terrain."""
    usable = TILE_SIZE - 2 * SUB_BORDER_WIDTH - PLATFORM_WIDTH
    return int(usable // (2 * step_width) + 1)


def build_combos(step_heights, tread_widths):
    combos = []
    for direction, height, width in itertools.product(DIRECTIONS, step_heights, tread_widths):
        combos.append({"direction": direction, "step_height": height, "tread_width": width})
    return combos


def build_stairs_benchmark_generator(combos: list[dict]) -> TerrainGeneratorCfg:
    """One sub-terrain per combo, equal proportion, one difficulty row.

    With `curriculum=True` and equal proportions, `TerrainGenerator._generate_curriculum_terrains`
    assigns column `i` to `list(sub_terrains.values())[i]` (see terrain_generator.py) -- so combo
    order here is exactly the column order the terrain importer later hands back as
    `terrain.terrain_types`. `step_height_range=(h, h)` makes the per-row difficulty jitter a
    no-op for pyramid_stairs (only step height depends on difficulty; see mesh_terrains.py), so a
    single row (no promotion to chase) is safe.
    """
    sub_terrains = OrderedDict()
    for i, combo in enumerate(combos):
        cls = (
            terrain_gen.MeshInvertedPyramidStairsTerrainCfg
            if combo["direction"] == "ascent"
            else terrain_gen.MeshPyramidStairsTerrainCfg
        )
        key = f"{i:03d}_{combo['direction']}_h{combo['step_height']:.2f}_w{combo['tread_width']:.2f}"
        sub_terrains[key] = cls(
            proportion=1.0,
            step_height_range=(combo["step_height"], combo["step_height"]),
            step_width=combo["tread_width"],
            platform_width=PLATFORM_WIDTH,
            border_width=SUB_BORDER_WIDTH,
            holes=False,
        )
    return TerrainGeneratorCfg(
        size=(TILE_SIZE, TILE_SIZE),
        border_width=GEN_BORDER_WIDTH,
        num_rows=1,
        num_cols=len(combos),
        horizontal_scale=0.1,
        vertical_scale=0.005,
        slope_threshold=0.75,
        use_cache=False,
        curriculum=True,
        sub_terrains=sub_terrains,
    )


# ---------------------------------------------------------------------------
# Course definition. Segments are laid end to end along +x, spanning the full course width.
#   flat / landing: length            -- level ground (landing = top of a flight)
#   ascent / descent: step_height, steps, tread
#   slope: length, grade (+ up / - down)
#   rough: length, max_height (random 0.1 m cells, like the training `random_rough`)
# Every ascent is paired with a descent and every slope up with one down, so the course ends at
# its starting height (level with the flat border around it).
# ---------------------------------------------------------------------------
DEFAULT_COURSE = [
    dict(kind="flat", length=5.0),
    dict(kind="ascent", step_height=0.10, steps=4, tread=0.30),
    dict(kind="landing", length=2.0),
    dict(kind="descent", step_height=0.10, steps=4, tread=0.30),
    dict(kind="flat", length=2.5),
    dict(kind="rough", length=4.0, max_height=0.05),
    dict(kind="flat", length=2.0),
    dict(kind="ascent", step_height=0.20, steps=5, tread=0.30),
    dict(kind="landing", length=2.5),
    dict(kind="descent", step_height=0.20, steps=5, tread=0.30),
    dict(kind="flat", length=2.5),
    dict(kind="slope", length=3.0, grade=0.25),
    dict(kind="landing", length=2.0),
    dict(kind="slope", length=3.0, grade=-0.25),
    dict(kind="flat", length=2.0),
    dict(kind="rough", length=4.0, max_height=0.10),
    dict(kind="flat", length=2.0),
    dict(kind="ascent", step_height=0.25, steps=5, tread=0.32),
    dict(kind="landing", length=2.5),
    dict(kind="descent", step_height=0.25, steps=5, tread=0.32),
    dict(kind="flat", length=2.5),
    dict(kind="ascent", step_height=0.30, steps=4, tread=0.32),
    dict(kind="landing", length=2.5),
    dict(kind="descent", step_height=0.30, steps=4, tread=0.35),
    dict(kind="flat", length=5.0),
]
COURSE_WIDTH = 8.0
COURSE_BORDER_WIDTH = 6.0
START_X = 2.0  # robot root spawn, metres into the first flat segment
HORIZONTAL_SCALE = 0.05
VERTICAL_SCALE = 0.005


def segment_label(seg: dict) -> str:
    k = seg["kind"]
    if k in ("ascent", "descent"):
        return f"{k} {seg['step_height']:.2f} m x{seg['steps']}"
    if k == "slope":
        return f"slope {'up' if seg['grade'] > 0 else 'down'} {abs(seg['grade']):.2f}"
    if k == "rough":
        return f"rough {seg['max_height']:.2f} m"
    return k


def segment_length(seg: dict) -> float:
    if seg["kind"] in ("ascent", "descent"):
        return seg["steps"] * seg["tread"]
    return seg["length"]


def layout(course: list[dict]) -> list[dict]:
    """Absolute x-extent and start/end heights of every segment."""
    out, x, z = [], 0.0, 0.0
    for seg in course:
        length = segment_length(seg)
        dz = 0.0
        if seg["kind"] == "ascent":
            dz = seg["steps"] * seg["step_height"]
        elif seg["kind"] == "descent":
            dz = -seg["steps"] * seg["step_height"]
        elif seg["kind"] == "slope":
            dz = seg["grade"] * length
        out.append({**seg, "label": segment_label(seg), "x0": x, "x1": x + length, "z0": z, "z1": z + dz})
        x, z = x + length, z + dz
    return out


def course_terrain(difficulty: float, cfg: CourseTerrainCfg):
    """Sub-terrain function: the whole course as one height field -> trimesh."""
    segs = layout(cfg.course)
    length, width = cfg.size
    hs, vs = HORIZONTAL_SCALE, VERTICAL_SCALE
    nx, ny = int(length / hs) + 1, int(width / hs) + 1
    xs = np.arange(nx) * hs
    heights = np.zeros((nx, ny))
    rng = np.random.default_rng(cfg.seed)
    for seg in segs:
        m = (xs >= seg["x0"]) & (xs < seg["x1"])
        if not m.any():
            continue
        local = xs[m] - seg["x0"]
        k = seg["kind"]
        if k in ("flat", "landing"):
            prof = np.full(local.shape, seg["z0"])
        elif k in ("ascent", "descent"):
            sign = 1.0 if k == "ascent" else -1.0
            step_idx = np.floor(local / seg["tread"] + 1e-9) + 1  # first tread is already one step up/down
            prof = seg["z0"] + sign * seg["step_height"] * step_idx
        elif k == "slope":
            prof = seg["z0"] + seg["grade"] * local
        elif k == "rough":
            prof = np.full(local.shape, seg["z0"])
        else:
            raise ValueError(f"unknown segment kind {k}")
        heights[m, :] = prof[:, None]
        if k == "rough":
            cell = int(round(0.1 / hs))
            cx, cy = int(np.ceil(m.sum() / cell)), int(np.ceil(ny / cell))
            levels = np.arange(0.0, seg["max_height"] + 1e-9, 0.02)
            noise = rng.choice(levels, size=(cx, cy))
            noise = np.kron(noise, np.ones((cell, cell)))[: m.sum(), :ny]
            heights[m, :] += noise
    hf = np.round(heights / vs).astype(np.int16)
    vertices, triangles = convert_height_field_to_mesh(hf, hs, vs, cfg.slope_threshold)
    mesh = trimesh.Trimesh(vertices=vertices, faces=triangles)
    origin = np.array([START_X, width / 2.0, 0.0])
    return [mesh], origin


@configclass
class CourseTerrainCfg(SubTerrainBaseCfg):
    function = course_terrain
    course: list = MISSING
    slope_threshold: float = 0.75
    seed: int = 0


def course_length(course: list[dict]) -> float:
    return layout(course)[-1]["x1"]


def build_course_generator(course: list[dict], seed: int) -> TerrainGeneratorCfg:
    """The whole course as a single 1 x 1 tile; the robot spawns at `START_X` on its centreline."""
    return TerrainGeneratorCfg(
        size=(course_length(course), COURSE_WIDTH),
        border_width=COURSE_BORDER_WIDTH,
        num_rows=1,
        num_cols=1,
        horizontal_scale=HORIZONTAL_SCALE,
        vertical_scale=VERTICAL_SCALE,
        slope_threshold=0.75,
        use_cache=False,
        curriculum=True,
        sub_terrains={"course": CourseTerrainCfg(proportion=1.0, course=course, seed=seed)},
    )
