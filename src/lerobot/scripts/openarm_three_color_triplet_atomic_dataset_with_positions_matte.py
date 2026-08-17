#!/usr/bin/env python3
"""Atomic three-color triplet collector with saturated matte cube materials.

The collection, IK, atomic rejection, staging/merge, camera recording, and
position-log behavior are exactly the behavior of
openarm_three_color_triplet_atomic_dataset_with_positions.py.  This launcher
changes only the three cube visual materials, locally in this file.  It does
not modify any of the existing Python files.
"""

from __future__ import annotations

import openarm_three_color_triplet_atomic_dataset_with_positions as atomic


generator = atomic.generator
base = atomic.base

# Non-emissive, highly saturated colors matching the standalone scene viewer.
MATTE_CUBE_RGB = {
    "red": (1.00, 0.01, 0.01),
    "blue": (0.01, 0.08, 1.00),
    "yellow": (1.00, 0.95, 0.01),
}


def matte_cube_cfg(color: str):
    """Recreate the original cube physics, changing only visual material."""
    return base.RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{color.title()}Cube",
        spawn=base.sim_utils.CuboidCfg(
            size=(base.CUBE_SIZE,) * 3,
            rigid_props=base.sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                linear_damping=0.5,
                angular_damping=0.5,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=4,
                max_depenetration_velocity=0.2,
            ),
            mass_props=base.sim_utils.MassPropertiesCfg(mass=0.05),
            collision_props=base.sim_utils.CollisionPropertiesCfg(
                contact_offset=0.002,
                rest_offset=0.0,
            ),
            physics_material=base.sim_utils.RigidBodyMaterialCfg(
                static_friction=1.0,
                dynamic_friction=0.8,
                restitution=0.0,
            ),
            visual_material=base.sim_utils.PreviewSurfaceCfg(
                diffuse_color=MATTE_CUBE_RGB[color],
                emissive_color=(0.0, 0.0, 0.0),
                roughness=1.0,
                metallic=0.0,
                opacity=1.0,
            ),
        ),
        init_state=base.RigidObjectCfg.InitialStateCfg(pos=base.CUBE_POS),
    )


@base.configclass
class MatteThreeColorPickPlaceSceneCfg(base.DualRealSensePickPlaceSceneCfg):
    # Keep the original asset keys expected by the controller and logger.
    cube = matte_cube_cfg("red")
    blue_cube = matte_cube_cfg("blue")
    yellow_cube = matte_cube_cfg("yellow")


# atomic.main() looks this class up through the generator module at runtime.
# Replacing the reference changes only this process; no source file is edited.
generator.ThreeColorPickPlaceSceneCfg = MatteThreeColorPickPlaceSceneCfg
generator.CUBE_RGB = MATTE_CUBE_RGB


if __name__ == "__main__":
    print("[MATERIAL] Saturated matte cubes enabled (non-emissive, roughness=1, metallic=0).")
    try:
        atomic.main()
    finally:
        base.simulation_app.close()
