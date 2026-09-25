# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause

"""Utility to convert a scanned OBJ/STL/FBX mesh (e.g. a ZED SDK spatial-mapping export) into a
static-terrain USD asset that can be loaded via ``TerrainImporterCfg(terrain_type="usd", ...)``.

Unlike ``isaaclab``'s stock ``scripts/tools/convert_mesh.py``, this script exposes the
translation/rotation used to correct the source mesh's axis convention and to recenter it, since
scan exports are frequently Y-up (common for ZED/OpenGL-style tools) while Isaac Sim/PhysX are Z-up.

Example (Y-up scan, recenter a chosen ground point to the world origin)::

    ./isaaclab.sh -p /workspace/rl_training/scripts/tools/convert_terrain_mesh.py \\
        /workspace/rl_training/mesh/mesh.obj \\
        /workspace/rl_training/mesh/mesh.usd \\
        --up-axis y \\
        --recenter -1.869 0.016 -7.013 \\
        --headless
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import math

from isaaclab.app import AppLauncher

_valid_collision_approx = [
    "convexDecomposition",
    "convexHull",
    "triangleMesh",
    "meshSimplification",
    "sdf",
    "boundingCube",
    "boundingSphere",
    "none",
]

parser = argparse.ArgumentParser(description="Convert a scanned mesh into a static-terrain USD asset.")
parser.add_argument("input", type=str, help="The path to the input mesh (.obj/.stl/.fbx) file.")
parser.add_argument("output", type=str, help="The path to store the output USD file.")
parser.add_argument(
    "--up-axis",
    type=str,
    default="y",
    choices=["y", "z"],
    help=(
        "The up axis of the source mesh. 'y' rotates the mesh +90 deg about X so that mesh-Y becomes"
        " world-Z (Isaac Sim/PhysX convention). 'z' applies no rotation. Defaults to 'y'."
    ),
)
parser.add_argument(
    "--recenter",
    type=float,
    nargs=3,
    default=None,
    metavar=("X", "Y", "Z"),
    help=(
        "A point in the SOURCE mesh's own (pre-rotation) coordinates that should end up at the world"
        " origin after conversion. Use this to place a known flat/ground point at (0, 0, 0) so the robot"
        " can spawn at the same pose used for flat-ground envs. If omitted, no recentering is applied."
    ),
)
parser.add_argument(
    "--collision-approximation",
    type=str,
    default="triangleMesh",
    choices=_valid_collision_approx,
    help=(
        "Collision mesh approximation. Defaults to 'triangleMesh' (exact, concave-capable, and only"
        " valid for static colliders -- which is what a terrain is). This matches what"
        " isaaclab.terrains.TerrainImporter uses for procedurally generated terrain meshes."
    ),
)
parser.add_argument(
    "--static-friction",
    type=float,
    default=1.0,
    help=(
        "Static friction to bind onto the terrain mesh. Defaults to 1.0, matching the"
        " RigidBodyMaterialCfg used by MySceneCfg.terrain for procedurally generated training terrain."
        " NOTE: unlike terrain_type='generator', TerrainImporter.import_usd() does NOT apply"
        " cfg.scene.terrain.physics_material -- for terrain_type='usd' the material must be baked into"
        " the USD file itself (which is what this flag does), otherwise PhysX falls back to its default"
        " material (~0.5 friction, 'average' combine) and wheeled/legged policies trained against a"
        " higher-friction terrain can visibly creep or slide even under a zero velocity command."
    ),
)
parser.add_argument("--dynamic-friction", type=float, default=1.0, help="Dynamic friction. Defaults to 1.0.")
parser.add_argument("--restitution", type=float, default=1.0, help="Restitution. Defaults to 1.0.")
parser.add_argument(
    "--friction-combine-mode",
    type=str,
    default="multiply",
    choices=["average", "min", "multiply", "max"],
    help="Defaults to 'multiply', matching MySceneCfg.terrain's training material.",
)
parser.add_argument(
    "--restitution-combine-mode",
    type=str,
    default="multiply",
    choices=["average", "min", "multiply", "max"],
    help="Defaults to 'multiply', matching MySceneCfg.terrain's training material.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os

import isaaclab.sim as sim_utils
from isaaclab.sim.converters import MeshConverter, MeshConverterCfg
from isaaclab.sim.schemas import schemas_cfg
from isaaclab.sim.utils import bind_physics_material, get_current_stage, open_stage
from isaaclab.utils.assets import check_file_path
from isaaclab.utils.dict import print_dict

collision_approximation_map = {
    "convexDecomposition": schemas_cfg.ConvexDecompositionPropertiesCfg,
    "convexHull": schemas_cfg.ConvexHullPropertiesCfg,
    "triangleMesh": schemas_cfg.TriangleMeshPropertiesCfg,
    "meshSimplification": schemas_cfg.TriangleMeshSimplificationPropertiesCfg,
    "sdf": schemas_cfg.SDFMeshPropertiesCfg,
    "boundingCube": schemas_cfg.BoundingCubePropertiesCfg,
    "boundingSphere": schemas_cfg.BoundingSpherePropertiesCfg,
    "none": None,
}


def main():
    mesh_path = os.path.abspath(args_cli.input)
    if not check_file_path(mesh_path):
        raise ValueError(f"Invalid mesh file path: {mesh_path}")
    dest_path = os.path.abspath(args_cli.output)

    # rotation: identity for z-up sources; +90 deg about X maps mesh-Y -> world-Z for y-up sources
    if args_cli.up_axis == "y":
        half = math.sqrt(0.5)
        rotation = (half, half, 0.0, 0.0)  # (w, x, y, z), 90 deg about X
    else:
        rotation = (1.0, 0.0, 0.0, 0.0)

    # translation: applied AFTER rotation/scale, so it must be expressed by rotating the requested
    # recenter point first, then negating it.
    if args_cli.recenter is not None:
        rx, ry, rz = args_cli.recenter
        if args_cli.up_axis == "y":
            rotated = (rx, -rz, ry)
        else:
            rotated = (rx, ry, rz)
        translation = (-rotated[0], -rotated[1], -rotated[2])
    else:
        translation = (0.0, 0.0, 0.0)

    collision_props = schemas_cfg.CollisionPropertiesCfg(collision_enabled=args_cli.collision_approximation != "none")
    cfg_class = collision_approximation_map[args_cli.collision_approximation]
    collision_cfg = cfg_class() if cfg_class is not None else None

    mesh_converter_cfg = MeshConverterCfg(
        asset_path=mesh_path,
        force_usd_conversion=True,
        usd_dir=os.path.dirname(dest_path),
        usd_file_name=os.path.basename(dest_path),
        make_instanceable=False,
        collision_props=collision_props,
        mesh_collision_props=collision_cfg,
        translation=translation,
        rotation=rotation,
    )

    print("-" * 80)
    print(f"Input mesh file: {mesh_path}")
    print(f"Up axis: {args_cli.up_axis}  ->  rotation (w,x,y,z): {rotation}")
    print(f"Recenter point (source coords): {args_cli.recenter}  ->  translation: {translation}")
    print("Mesh importer config:")
    print_dict(mesh_converter_cfg.to_dict(), nesting=0)
    print("-" * 80)

    mesh_converter = MeshConverter(mesh_converter_cfg)
    print(f"Generated USD file: {mesh_converter.usd_path}")

    # Bind a physics material matching the procedural training terrain's friction/restitution.
    # TerrainImporter.import_usd() does not apply scene.terrain.physics_material for terrain_type="usd",
    # so this has to be authored directly into the USD asset. Note: spawn_rigid_body_material() and
    # bind_physics_material() both operate on the omniverse USD *context's* current stage, not on a
    # bare pxr.Usd.Stage.Open() handle -- so the file must be opened via open_stage() first.
    open_stage(mesh_converter.usd_path)
    stage = get_current_stage()
    default_prim = stage.GetDefaultPrim()
    geometry_prim_path = f"{default_prim.GetPath()}/geometry"
    material_path = f"{geometry_prim_path}/PhysicsMaterial"
    material_cfg = sim_utils.RigidBodyMaterialCfg(
        static_friction=args_cli.static_friction,
        dynamic_friction=args_cli.dynamic_friction,
        restitution=args_cli.restitution,
        friction_combine_mode=args_cli.friction_combine_mode,
        restitution_combine_mode=args_cli.restitution_combine_mode,
    )
    material_cfg.func(material_path, material_cfg)
    bind_physics_material(geometry_prim_path, material_path, stage=stage)
    stage.Save()
    print(
        f"Bound physics material at {material_path} (static={args_cli.static_friction},"
        f" dynamic={args_cli.dynamic_friction}, restitution={args_cli.restitution},"
        f" combine={args_cli.friction_combine_mode}/{args_cli.restitution_combine_mode}) to {geometry_prim_path}"
    )
    print("-" * 80)


if __name__ == "__main__":
    main()
    simulation_app.close()
