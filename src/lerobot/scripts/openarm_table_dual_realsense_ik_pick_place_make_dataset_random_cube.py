"""Generate a dual-RealSense OpenArm IK pick-and-place LeRobot dataset.

Sequence:
    wait for cube -> open -> safe-height transit -> 5 cm pre-grasp
    -> grasp -> lift -> storage-box transit -> place -> open -> retreat

Each episode runs the complete scripted expert trajectory (settle, grasp, lift,
place, release, retreat), records joint state/action and optional RGB, then
resets the same environment for the next episode. The recorded state/action
contains seven arm joints plus one gripper value in radians. Both the fixed top
RealSense and the left-TCP-mounted wrist RealSense are recorded at 640x480,
30 FPS under observation.images.top and observation.images.wrist.

Usage:
    /home/zxro/IsaacLab/isaaclab.sh -p \
        /home/zxro/smolVLA-isaacLab/scripts/openarm_table_dual_realsense_ik_pick_place_make_dataset.py \
        --num_episodes 10
"""

from __future__ import annotations

import argparse
import math
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np

from isaaclab.app import AppLauncher


DEFAULT_DATASET_ROOT = (
    f"outputs/lerobot_datasets/openarm_pick_place_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
)
DEFAULT_CAMERA_WIDTH = 640
DEFAULT_CAMERA_HEIGHT = 480
DATASET_FPS = 30

parser = argparse.ArgumentParser(description="OpenArm IK pick-and-place LeRobot dataset generator.")
parser.add_argument("--num_episodes", type=int, default=1, help="Number of episodes to record.")
parser.add_argument("--dataset_repo_id", type=str, default="a126-kitech/openarm_pick_place")
parser.add_argument("--dataset_root", type=str, default=DEFAULT_DATASET_ROOT)
parser.add_argument("--task", type=str, default="Pick up the red cube and place it in the storage box.")
parser.add_argument("--overwrite_dataset", action="store_true")
parser.add_argument("--push_to_hub", action="store_true")
parser.add_argument("--private", action="store_true")
parser.add_argument("--use_videos", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--record_camera", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--image_writer_threads", type=int, default=4)
parser.add_argument("--image_writer_processes", type=int, default=0)
parser.add_argument("--warmup_time_s", type=float, default=1.0)
parser.add_argument("--cube_x", type=float, default=-0.52772)
parser.add_argument("--cube_y", type=float, default=0.0974)
parser.add_argument("--cube_z", type=float, default=0.1, help="Initial cube drop height.")
parser.add_argument(
    "--cube_x_range",
    type=float,
    nargs=2,
    metavar=("MIN", "MAX"),
    default=None,
    help="Randomize the cube x position uniformly in [MIN, MAX] for each episode.",
)
parser.add_argument(
    "--cube_y_range",
    type=float,
    nargs=2,
    metavar=("MIN", "MAX"),
    default=None,
    help="Randomize the cube y position uniformly in [MIN, MAX] for each episode.",
)
parser.add_argument(
    "--cube_random_seed",
    type=int,
    default=None,
    help="Seed for reproducible cube-position randomization.",
)
parser.add_argument("--tilt_deg", type=float, default=45.0, help="TCP tilt about the robot-base y axis.")
parser.add_argument(
    "--tilt_at_safe_height",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Apply TCP tilt while transiting at safe height toward the cube.",
)
parser.add_argument("--settle_speed", type=float, default=0.015)
parser.add_argument("--settle_time_s", type=float, default=0.5)
parser.add_argument(
    "--settle_failure_timeout_s",
    type=float,
    default=3.0,
    help="Fail and retry an episode if its target cube does not settle within this time.",
)
parser.add_argument("--open_time_s", type=float, default=0.7)
parser.add_argument("--close_time_s", type=float, default=1.0)
parser.add_argument("--move_time_s", type=float, default=2.5)
parser.add_argument("--short_move_time_s", type=float, default=1.4)
parser.add_argument("--pregrasp_clearance", type=float, default=0.05)
parser.add_argument(
    "--grasp_offset",
    type=float,
    default=0.0,
    help="TCP z offset from the settled cube center; zero grasps the exact center.",
)
parser.add_argument("--safe_z", type=float, default=0.28)
parser.add_argument("--place_clearance", type=float, default=0.01)
parser.add_argument("--max_cartesian_step", type=float, default=0.008)
parser.add_argument("--max_rotation_step", type=float, default=0.035)
parser.add_argument("--grasp_position_tolerance", type=float, default=0.005)
parser.add_argument("--grasp_rotation_tolerance", type=float, default=0.05, help="Radians.")
parser.add_argument("--grasp_reached_hold_s", type=float, default=0.25)
parser.add_argument(
    "--ik_failure_timeout_s",
    type=float,
    default=3.0,
    help="Discard and retry when the grasp IK target remains unreached this long after descent.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.num_episodes < 1:
    parser.error("--num_episodes must be at least 1")
if args_cli.settle_failure_timeout_s <= 0.0:
    parser.error("--settle_failure_timeout_s must be greater than zero")
if args_cli.ik_failure_timeout_s <= 0.0:
    parser.error("--ik_failure_timeout_s must be greater than zero")
for range_name in ("cube_x_range", "cube_y_range"):
    limits = getattr(args_cli, range_name)
    if limits is not None and limits[0] > limits[1]:
        parser.error(f"--{range_name} requires MIN <= MAX")
if args_cli.record_camera and hasattr(args_cli, "enable_cameras"):
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch

import isaaclab.sim as sim_utils
if args_cli.record_camera:
    import omni.ui as ui
    from omni.kit.viewport.utility import create_viewport_window
else:
    ui = None
    create_viewport_window = None
from isaacsim.core.utils.stage import get_current_stage
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.utils.math import (
    compute_pose_error,
    matrix_from_quat,
    quat_from_angle_axis,
    quat_inv,
    quat_mul,
    quat_slerp,
    subtract_frame_transforms,
)
from pxr import Gf, Sdf, Usd, UsdGeom


ASSET_DIR = Path(__file__).resolve().parents[1] / "assets" / "openarm_use"
TABLE_USD = str(ASSET_DIR / "table.usd")
ROBOT_USD = str(ASSET_DIR / "openarm_half_tesollo_tactile.usd")
REALSENSE_USD = str(ASSET_DIR / "realsense.usd")
STORAGE_BOX_USD = str(ASSET_DIR / "storage_box.usd")

ROBOT_POS = (-0.17858, 0.24336, -0.00468)
ROBOT_ROT = (0.0, 0.0, 0.0, 1.0)
REALSENSE_POS = (-0.38725, 0.09957, 0.78544)
REALSENSE_ROT = (math.cos(math.pi / 4.0), 0.0, math.sin(math.pi / 4.0), 0.0)
WRIST_REALSENSE_POS = (0.06336, -0.01314, -0.05143)
WRIST_REALSENSE_ROT = (0.0, 0.5, 0.0, 0.86603)  # wxyz, local to left-hand TCP
WRIST_PARENT_PRIM_NAME = "openarm_left_hand_tcp"
FIXED_BOX_SIZE = (0.45, 1.0, 0.05)
FIXED_BOX_POS = (-0.57337, 0.0, 0.02164)
STORAGE_BOX_SIZE = (0.55, 0.75, 0.15)
STORAGE_BOX_POS = (-0.53584, -0.15, 0.04664)
FLOOR_Z = -0.754904
CUBE_SIZE = 0.05
CUBE_POS = (args_cli.cube_x, args_cli.cube_y, args_cli.cube_z)
CUBE_POSITION_RNG = np.random.default_rng(args_cli.cube_random_seed)

INITIAL_JOINT_POS_DEG = {
    "openarm_left_joint1": 20.0,
    "openarm_left_joint2": 0.0,
    "openarm_left_joint3": 0.0,
    "openarm_left_joint4": 60.0,
    "openarm_left_joint5": 0.0,
    "openarm_left_joint6": 0.0,
    "openarm_left_joint7": -44.0,
    "openarm_right_joint1": -20.0,
    "openarm_right_joint2": 40.0,
    "openarm_right_joint3": -10.0,
    "openarm_right_joint4": 60.0,
    "openarm_right_joint5": 30.0,
    "openarm_right_joint6": 15.0,
    "openarm_right_joint7": 44.0,
}


def joint_positions_rad() -> dict[str, float]:
    return {name: math.radians(value) for name, value in INITIAL_JOINT_POS_DEG.items()}


@configclass
class DualRealSensePickPlaceSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.35, dynamic_friction=0.30, restitution=0.0
            )
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, FLOOR_Z)),
    )
    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
    )
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.UsdFileCfg(
            usd_path=TABLE_USD,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(),
    )
    table_top_proxy = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/TableTopProxyCollider",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.0211)),
        spawn=sim_utils.CuboidCfg(
            size=(1.6, 1.0, 0.04),
            visible=False,
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True, contact_offset=0.01, rest_offset=0.0
            ),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.0, dynamic_friction=0.8, restitution=0.0
            ),
        ),
    )
    fixed_box = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/FixedBox",
        init_state=AssetBaseCfg.InitialStateCfg(pos=FIXED_BOX_POS),
        spawn=sim_utils.CuboidCfg(
            size=FIXED_BOX_SIZE,
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True, contact_offset=0.005, rest_offset=0.0
            ),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.8, dynamic_friction=0.6, restitution=0.0
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.5, 0.5)),
        ),
    )
    storage_box = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/StorageBox",
        spawn=sim_utils.UsdFileCfg(
            usd_path=STORAGE_BOX_USD,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True, contact_offset=0.005, rest_offset=0.0
            ),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=STORAGE_BOX_POS),
    )
    realsense = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Realsense",
        spawn=sim_utils.UsdFileCfg(
            usd_path=REALSENSE_USD,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=REALSENSE_POS, rot=REALSENSE_ROT),
    )
    top_color = CameraCfg(
        prim_path="{ENV_REGEX_NS}/TopColorCamera",
        update_period=1.0 / DATASET_FPS,
        height=DEFAULT_CAMERA_HEIGHT,
        width=DEFAULT_CAMERA_WIDTH,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.01, 10.0),
        ),
        offset=CameraCfg.OffsetCfg(pos=REALSENSE_POS, rot=REALSENSE_ROT, convention="world"),
    )
    robot = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=ROBOT_USD,
            activate_contact_sensors=True,
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False, fix_root_link=True
            ),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True, max_depenetration_velocity=5.0
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=ROBOT_POS, rot=ROBOT_ROT, joint_pos=joint_positions_rad()
        ),
        actuators={
            "arm_pd": ImplicitActuatorCfg(
                joint_names_expr=["openarm_.*_joint[1-7]"],
                stiffness=800.0,
                damping=80.0,
                effort_limit_sim=150.0,
                velocity_limit_sim=10.0,
            ),
            "gripper_pd": ImplicitActuatorCfg(
                joint_names_expr=["openarm_left_finger_joint[1-2]", "rj_dg_.*"],
                stiffness=800.0,
                damping=50.0,
                effort_limit_sim=15.0,
                velocity_limit_sim=2.0,
            ),
        },
    )
    wrist_color = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/openarm_left_hand_tcp/WristColorCamera",
        update_period=1.0 / DATASET_FPS,
        height=DEFAULT_CAMERA_HEIGHT,
        width=DEFAULT_CAMERA_WIDTH,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.01, 10.0),
        ),
        # This temporary local pose is overwritten from the wrist RealSense
        # USD's actual color-camera prim before recording starts.
        offset=CameraCfg.OffsetCfg(
            pos=WRIST_REALSENSE_POS,
            rot=WRIST_REALSENSE_ROT,
            convention="ros",
        ),
    )
    cube = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube",
        spawn=sim_utils.CuboidCfg(
            size=(CUBE_SIZE, CUBE_SIZE, CUBE_SIZE),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                linear_damping=0.5,
                angular_damping=0.5,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=4,
                max_depenetration_velocity=0.2,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.002, rest_offset=0.0),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.0, dynamic_friction=0.8, restitution=0.0
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.85, 0.05, 0.03)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=CUBE_POS),
    )


def sample_cube_position() -> tuple[float, float, float]:
    """Sample an episode-local cube position, or use the fixed CLI position."""
    x = (
        float(CUBE_POSITION_RNG.uniform(*args_cli.cube_x_range))
        if args_cli.cube_x_range is not None
        else args_cli.cube_x
    )
    y = (
        float(CUBE_POSITION_RNG.uniform(*args_cli.cube_y_range))
        if args_cli.cube_y_range is not None
        else args_cli.cube_y
    )
    return x, y, args_cli.cube_z


def reset_scene(scene: InteractiveScene) -> tuple[float, float, float]:
    robot = scene["robot"]
    cube = scene["cube"]
    robot_state = robot.data.default_root_state.clone()
    robot_state[:, :3] += scene.env_origins
    robot.write_root_pose_to_sim(robot_state[:, :7])
    robot.write_root_velocity_to_sim(robot_state[:, 7:])
    joint_pos = robot.data.default_joint_pos.clone()
    robot.write_joint_state_to_sim(joint_pos, torch.zeros_like(robot.data.default_joint_vel))
    robot.set_joint_position_target(joint_pos)
    cube_state = cube.data.default_root_state.clone()
    cube_position = sample_cube_position()
    cube_state[:, :3] = torch.tensor(
        cube_position, device=cube_state.device, dtype=cube_state.dtype
    )
    cube_state[:, :3] += scene.env_origins
    cube.write_root_pose_to_sim(cube_state[:, :7])
    cube.write_root_velocity_to_sim(torch.zeros_like(cube_state[:, 7:]))
    scene.reset()
    return cube_position


def find_descendant_prim_path(root_path: str, prim_name: str) -> Sdf.Path:
    """Find one named prim below a USD subtree."""
    stage = get_current_stage()
    root_prim = stage.GetPrimAtPath(root_path)
    matches = [
        prim.GetPath()
        for prim in Usd.PrimRange(root_prim)
        if prim.GetName() == prim_name
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one '{prim_name}' below {root_path}, found {len(matches)}: {matches}"
        )
    return matches[0]


def attach_wrist_realsense(scene: InteractiveScene) -> Sdf.Path:
    """Attach the existing RealSense USD below openarm_left_hand_tcp."""
    stage = get_current_stage()
    robot_root = f"{scene.env_prim_paths[0]}/Robot"
    tcp_path = find_descendant_prim_path(robot_root, WRIST_PARENT_PRIM_NAME)
    tcp_prim = stage.GetPrimAtPath(tcp_path)
    if tcp_prim.IsInstanceProxy():
        raise RuntimeError(f"Cannot author wrist RealSense below instance proxy: {tcp_path}")

    wrist_root_path = tcp_path.AppendChild("WristRealsense")
    wrist_root = UsdGeom.Xform.Define(stage, wrist_root_path)
    wrist_root.GetPrim().GetReferences().AddReference(REALSENSE_USD)
    xformable = UsdGeom.Xformable(wrist_root.GetPrim())
    xformable.ClearXformOpOrder()
    xformable.AddTranslateOp(opSuffix="wristMount").Set(Gf.Vec3d(*WRIST_REALSENSE_POS))
    quat = WRIST_REALSENSE_ROT
    xformable.AddOrientOp(opSuffix="wristMount").Set(
        Gf.Quatf(quat[0], Gf.Vec3f(*quat[1:]))
    )
    print(f"[INFO] Wrist RealSense attached below: {tcp_path}")
    return wrist_root_path


def find_color_camera_path(realsense_root: Sdf.Path | str) -> Sdf.Path:
    """Find the RGB/color camera contained in a RealSense USD reference."""
    stage = get_current_stage()
    root_prim = stage.GetPrimAtPath(realsense_root)
    source_paths = [
        prim.GetPath()
        for prim in Usd.PrimRange(root_prim)
        if prim.IsA(UsdGeom.Camera)
    ]
    if not source_paths:
        raise RuntimeError(f"No camera found below {realsense_root}")
    return next(
        (
            path
            for path in source_paths
            if "color" in path.pathString.lower() or "rgb" in path.pathString.lower()
        ),
        source_paths[0],
    )


def match_record_camera_to_source(source_path: Sdf.Path, record_path: Sdf.Path) -> None:
    """Copy the source RealSense color camera pose and optics to CameraCfg."""
    stage = get_current_stage()
    source_prim = stage.GetPrimAtPath(source_path)
    record_prim = stage.GetPrimAtPath(record_path)
    if not record_prim.IsValid() or not record_prim.IsA(UsdGeom.Camera):
        raise RuntimeError(f"Dataset RGB camera was not found at {record_path}")

    xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    source_world = xform_cache.GetLocalToWorldTransform(source_prim)
    record_parent_world = xform_cache.GetLocalToWorldTransform(record_prim.GetParent())
    record_local = source_world * record_parent_world.GetInverse()
    record_xform = UsdGeom.Xformable(record_prim)
    record_xform.ClearXformOpOrder()
    record_xform.AddTransformOp().Set(record_local)

    for attribute_name in (
        "projection",
        "focalLength",
        "focusDistance",
        "fStop",
        "horizontalAperture",
        "verticalAperture",
        "horizontalApertureOffset",
        "verticalApertureOffset",
        "clippingRange",
    ):
        source_value = source_prim.GetAttribute(attribute_name).Get()
        if source_value is not None:
            record_prim.GetAttribute(attribute_name).Set(source_value)

    print(f"[INFO] Dataset camera {record_path} matched to {source_path}")


def create_dual_realsense_views(
    scene: InteractiveScene, wrist_root_path: Sdf.Path
) -> tuple[object, object]:
    """Match both recording sensors and show both source color cameras."""
    env_root = scene.env_prim_paths[0]
    top_source_path = find_color_camera_path(Sdf.Path(f"{env_root}/Realsense"))
    wrist_source_path = find_color_camera_path(wrist_root_path)
    match_record_camera_to_source(
        top_source_path,
        Sdf.Path(f"{env_root}/TopColorCamera"),
    )
    match_record_camera_to_source(
        wrist_source_path,
        Sdf.Path(f"{env_root}/Robot/openarm_left_hand_tcp/WristColorCamera"),
    )

    top_view = create_viewport_window(
        name="Top RealSense Color",
        width=DEFAULT_CAMERA_WIDTH,
        height=DEFAULT_CAMERA_HEIGHT,
        camera_path=top_source_path,
    )
    wrist_view = create_viewport_window(
        name="Wrist RealSense Color",
        width=DEFAULT_CAMERA_WIDTH,
        height=DEFAULT_CAMERA_HEIGHT,
        camera_path=wrist_source_path,
    )
    scene_view = ui.Workspace.get_window("Viewport")
    if top_view is not None and scene_view is not None:
        top_view.dock_in(scene_view, ui.DockPosition.RIGHT, 0.5)
    top_handle = ui.Workspace.get_window("Top RealSense Color")
    if wrist_view is not None and top_handle is not None:
        wrist_view.dock_in(top_handle, ui.DockPosition.BOTTOM, 0.5)
    elif wrist_view is not None and scene_view is not None:
        wrist_view.dock_in(scene_view, ui.DockPosition.RIGHT, 0.5)
    return top_view, wrist_view


class PickPlaceController:
    """Waypoint state machine with 6-D differential IK for the left TCP."""

    def __init__(self, robot: Articulation, scene: InteractiveScene):
        self.robot = robot
        self.cube = scene["cube"]
        self.scene = scene
        self.state = "wait_settle"
        self.state_time = 0.0
        self.quiet_time = 0.0
        self.goal_reached_time = 0.0
        self.failure_reason: str | None = None
        self.settled_cube_pos_b: torch.Tensor | None = None
        self.start_pos_b: torch.Tensor | None = None
        self.start_quat_b: torch.Tensor | None = None
        self.goal_pos_b: torch.Tensor | None = None

        self.arm_names = [f"openarm_left_joint{i}" for i in range(1, 8)]
        self.arm_ids, names = robot.find_joints(self.arm_names, preserve_order=True)
        if len(self.arm_ids) != 7:
            raise RuntimeError(f"Expected seven left-arm joints, found {names}")
        self.gripper_names = ["openarm_left_finger_joint1", "openarm_left_finger_joint2"]
        self.gripper_ids, names = robot.find_joints(self.gripper_names, preserve_order=True)
        if len(self.gripper_ids) != 2:
            raise RuntimeError(f"Expected two left-gripper joints, found {names}")
        # Both opposing finger joints are needed in simulation.  The real
        # robot exposes one gripper command, represented by joint1 here;
        # joint2 therefore remains simulation-only.
        self.record_gripper_id = self.gripper_ids[0]

        self.entity = SceneEntityCfg(
            "robot", joint_names=self.arm_names, body_names=["openarm_left_hand_tcp"], preserve_order=True
        )
        self.entity.resolve(scene)
        self.jacobian_body_id = self.entity.body_ids[0] - 1 if robot.is_fixed_base else self.entity.body_ids[0]
        cfg = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls")
        self.ik = DifferentialIKController(cfg, num_envs=scene.num_envs, device=robot.device)
        self.command = torch.zeros(scene.num_envs, self.ik.action_dim, device=robot.device)
        self.lower = robot.data.soft_joint_pos_limits[0, :, 0]
        self.upper = robot.data.soft_joint_pos_limits[0, :, 1]
        # Last commanded joint targets are recorded as the policy action.
        self.commanded_joint_pos = robot.data.joint_pos.detach().clone()

        _, home_quat_b = self.ee_pose_b()
        self.home_quat_b = home_quat_b.detach().clone()
        axis_y = torch.tensor([0.0, 1.0, 0.0], device=robot.device).repeat(scene.num_envs, 1)
        angle = torch.full((scene.num_envs,), math.radians(args_cli.tilt_deg), device=robot.device)
        # Pre-multiplication makes this a robot-base-y tilt, independent of cube orientation.
        self.grasp_quat_b = quat_mul(quat_from_angle_axis(angle, axis_y), home_quat_b).detach()
        print(f"[AUTO] TCP target tilt: base y-axis {args_cli.tilt_deg:.1f} deg")

    @staticmethod
    def smoothstep(alpha: float) -> float:
        alpha = min(max(alpha, 0.0), 1.0)
        return alpha * alpha * (3.0 - 2.0 * alpha)

    def ee_pose_b(self) -> tuple[torch.Tensor, torch.Tensor]:
        ee_w = self.robot.data.body_pose_w[:, self.entity.body_ids[0]]
        root_w = self.robot.data.root_pose_w
        return subtract_frame_transforms(root_w[:, :3], root_w[:, 3:7], ee_w[:, :3], ee_w[:, 3:7])

    def world_point_to_base(self, point_w: tuple[float, float, float] | torch.Tensor) -> torch.Tensor:
        point = torch.as_tensor(point_w, dtype=torch.float32, device=self.robot.device).reshape(1, 3)
        root_w = self.robot.data.root_pose_w
        identity = torch.zeros((1, 4), device=self.robot.device)
        identity[:, 0] = 1.0
        position, _ = subtract_frame_transforms(root_w[:, :3], root_w[:, 3:7], point, identity)
        return position[0]

    def cube_pos_b(self) -> torch.Tensor:
        return self.world_point_to_base(self.cube.data.root_pos_w[0])

    def settled_cube_position(self) -> torch.Tensor:
        """Return the cube position captured once after landing and settling."""
        if self.settled_cube_pos_b is None:
            raise RuntimeError("Cube landing position has not been captured yet.")
        return self.settled_cube_pos_b.clone()

    def set_gripper(self, opened: bool) -> None:
        targets = []
        for joint_id, name in zip(self.gripper_ids, self.gripper_names, strict=True):
            if name.endswith("joint1"):
                target = self.upper[joint_id] if opened else self.lower[joint_id]
            else:
                target = self.lower[joint_id] if opened else self.upper[joint_id]
            targets.append(target)
        values = torch.stack(targets).unsqueeze(0).repeat(self.scene.num_envs, 1)
        self.commanded_joint_pos[:, self.gripper_ids] = values
        self.robot.set_joint_position_target(values, joint_ids=self.gripper_ids)

    def move_tcp(self, target_pos_b: torch.Tensor, target_quat_b: torch.Tensor | None = None) -> None:
        ee_pos_b, ee_quat_b = self.ee_pose_b()
        if target_quat_b is None:
            target_quat_b = self.home_quat_b
        pos_error, rot_error = compute_pose_error(
            ee_pos_b, ee_quat_b, target_pos_b.unsqueeze(0), target_quat_b, rot_error_type="axis_angle"
        )
        pos_norm = torch.linalg.norm(pos_error, dim=1, keepdim=True).clamp_min(1e-9)
        rot_norm = torch.linalg.norm(rot_error, dim=1, keepdim=True).clamp_min(1e-9)
        self.command[:, :3] = pos_error * torch.clamp(args_cli.max_cartesian_step / pos_norm, max=1.0)
        self.command[:, 3:] = rot_error * torch.clamp(args_cli.max_rotation_step / rot_norm, max=1.0)

        jacobian_w = self.robot.root_physx_view.get_jacobians()[
            :, self.jacobian_body_id, :, self.entity.joint_ids
        ]
        world_to_base = matrix_from_quat(quat_inv(self.robot.data.root_quat_w))
        jacobian = jacobian_w.clone()
        jacobian[:, :3, :] = torch.bmm(world_to_base, jacobian_w[:, :3, :])
        jacobian[:, 3:, :] = torch.bmm(world_to_base, jacobian_w[:, 3:, :])
        joint_pos = self.robot.data.joint_pos[:, self.entity.joint_ids]
        self.ik.set_command(self.command, ee_pos=ee_pos_b, ee_quat=ee_quat_b)
        desired = self.ik.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos)
        desired = torch.clamp(desired, self.lower[self.entity.joint_ids], self.upper[self.entity.joint_ids])
        self.commanded_joint_pos[:, self.entity.joint_ids] = desired
        self.robot.set_joint_position_target(desired, joint_ids=self.entity.joint_ids)

    @property
    def record_joint_names(self) -> list[str]:
        return [f"joint_{i}" for i in range(1, 8)] + ["gripper"]

    def get_record_state_action(self) -> tuple[np.ndarray, np.ndarray]:
        # Isaac Lab stores revolute joints in radians. The fixed dataset layout
        # is [joint_1, ..., joint_7, gripper], always in radians.
        state_rad = torch.cat(
            (
                self.robot.data.joint_pos[0, self.arm_ids],
                self.robot.data.joint_pos[0, self.record_gripper_id].reshape(1),
            )
        )
        action_rad = torch.cat(
            (
                self.commanded_joint_pos[0, self.arm_ids],
                self.commanded_joint_pos[0, self.record_gripper_id].reshape(1),
            )
        )
        state = state_rad.detach().cpu().numpy().astype(np.float32)
        action = action_rad.detach().cpu().numpy().astype(np.float32)
        return state, action

    def enter(self, state: str, goal: torch.Tensor | None = None) -> None:
        self.state = state
        self.state_time = 0.0
        self.goal_reached_time = 0.0
        start_pos_b, start_quat_b = self.ee_pose_b()
        self.start_pos_b = start_pos_b[0].detach().clone()
        self.start_quat_b = start_quat_b.detach().clone()
        self.goal_pos_b = goal.detach().clone() if goal is not None else None
        print(f"[AUTO] state -> {state}")

    def fail(self, reason: str) -> None:
        self.failure_reason = reason
        self.state = "failed"
        print(f"[AUTO][FAIL] {reason}")

    def interpolated_move(self, duration: float, target_quat_b: torch.Tensor | None = None) -> bool:
        assert self.start_pos_b is not None and self.start_quat_b is not None and self.goal_pos_b is not None
        alpha = self.smoothstep(self.state_time / max(duration, 1e-6))
        if target_quat_b is None:
            target_quat_b = self.home_quat_b
        # Isaac Lab's quat_slerp accepts individual (4,) quaternions, while
        # controller poses are batched as (num_envs, 4).
        interpolated_quat = torch.stack(
            [
                quat_slerp(self.start_quat_b[env_id], target_quat_b[env_id], alpha)
                for env_id in range(self.scene.num_envs)
            ],
            dim=0,
        )
        self.move_tcp((1.0 - alpha) * self.start_pos_b + alpha * self.goal_pos_b, interpolated_quat)
        return self.state_time >= duration

    def grasp_pose_reached(self) -> bool:
        """Check the measured TCP pose, rather than just the trajectory timer."""
        assert self.goal_pos_b is not None
        ee_pos_b, ee_quat_b = self.ee_pose_b()
        pos_error, rot_error = compute_pose_error(
            ee_pos_b,
            ee_quat_b,
            self.goal_pos_b.unsqueeze(0),
            self.grasp_quat_b,
            rot_error_type="axis_angle",
        )
        return (
            torch.linalg.norm(pos_error[0]).item() <= args_cli.grasp_position_tolerance
            and torch.linalg.norm(rot_error[0]).item() <= args_cli.grasp_rotation_tolerance
        )

    def storage_waypoint(self, z: float) -> torch.Tensor:
        return self.world_point_to_base((STORAGE_BOX_POS[0], STORAGE_BOX_POS[1], z))

    def advance(self, dt: float) -> None:
        self.state_time += dt
        if self.state == "wait_settle":
            velocity = self.cube.data.root_vel_w[0]
            quiet = (
                torch.linalg.norm(velocity[:3]).item() < args_cli.settle_speed
                and torch.linalg.norm(velocity[3:]).item() < args_cli.settle_speed
                and self.cube.data.root_pos_w[0, 2].item() < args_cli.cube_z - 0.01
            )
            self.quiet_time = self.quiet_time + dt if quiet else 0.0
            if self.quiet_time >= args_cli.settle_time_s:
                # Capture the post-contact pose exactly once.  All pick waypoints
                # use this fixed position instead of chasing later cube motion.
                self.settled_cube_pos_b = self.cube_pos_b().detach().clone()
                settled_world = self.cube.data.root_pos_w[0].detach().cpu().tolist()
                print(f"[AUTO] cube landed and settled at world xyz={settled_world}")
                self.set_gripper(True)
                self.enter("open_gripper")
            elif self.state_time >= args_cli.settle_failure_timeout_s:
                position = self.cube.data.root_pos_w[0].detach().cpu().tolist()
                velocity = self.cube.data.root_vel_w[0].detach().cpu().tolist()
                self.fail(
                    "target cube did not settle within "
                    f"{args_cli.settle_failure_timeout_s:.2f}s; "
                    f"xyz={position}, velocity={velocity}"
                )

        elif self.state == "open_gripper":
            self.set_gripper(True)
            if self.state_time >= args_cli.open_time_s:
                current = self.ee_pose_b()[0][0].detach().clone()
                current[2] = self.world_point_to_base((0.0, 0.0, args_cli.safe_z))[2]
                self.enter("raise_to_safe", current)

        elif self.state == "raise_to_safe":
            self.set_gripper(True)
            if self.interpolated_move(args_cli.short_move_time_s):
                cube = self.settled_cube_position()
                above = cube.clone()
                above[2] = self.world_point_to_base((0.0, 0.0, args_cli.safe_z))[2]
                self.enter("transit_to_cube", above)

        elif self.state == "transit_to_cube":
            self.set_gripper(True)
            transit_quat = self.grasp_quat_b if args_cli.tilt_at_safe_height else self.home_quat_b
            if self.interpolated_move(args_cli.move_time_s, transit_quat):
                pregrasp = self.settled_cube_position()
                pregrasp[2] += CUBE_SIZE / 2.0 + args_cli.pregrasp_clearance
                self.enter("pregrasp", pregrasp)

        elif self.state == "pregrasp":
            self.set_gripper(True)
            pregrasp_quat = self.grasp_quat_b if args_cli.tilt_at_safe_height else self.home_quat_b
            if self.interpolated_move(args_cli.short_move_time_s, pregrasp_quat):
                # The grasp TCP target is the captured cube center.  The optional
                # offset defaults to zero and is only exposed for TCP calibration.
                grasp = self.settled_cube_position()
                grasp[2] += args_cli.grasp_offset
                self.enter("descend_to_grasp", grasp)

        elif self.state == "descend_to_grasp":
            self.set_gripper(True)
            # With safe-height tilt enabled this is a straight, fixed-orientation descent.
            trajectory_finished = self.interpolated_move(
                args_cli.short_move_time_s, target_quat_b=self.grasp_quat_b
            )
            if trajectory_finished and self.grasp_pose_reached():
                self.goal_reached_time += dt
            else:
                self.goal_reached_time = 0.0
            if self.goal_reached_time >= args_cli.grasp_reached_hold_s:
                self.enter("close_gripper", self.goal_pos_b)
            elif self.state_time >= args_cli.short_move_time_s + args_cli.ik_failure_timeout_s:
                self.fail(
                    "grasp IK target was not reached within "
                    f"{args_cli.ik_failure_timeout_s:.2f}s after trajectory completion"
                )

        elif self.state == "close_gripper":
            assert self.goal_pos_b is not None
            self.move_tcp(self.goal_pos_b, self.grasp_quat_b)
            self.set_gripper(False)
            if self.state_time >= args_cli.close_time_s:
                lift = self.goal_pos_b.clone()
                lift[2] = self.world_point_to_base((0.0, 0.0, args_cli.safe_z))[2]
                self.enter("lift", lift)

        elif self.state == "lift":
            self.set_gripper(False)
            if self.interpolated_move(args_cli.short_move_time_s, self.grasp_quat_b):
                self.enter("transit_to_storage", self.storage_waypoint(args_cli.safe_z))

        elif self.state == "transit_to_storage":
            self.set_gripper(False)
            if self.interpolated_move(args_cli.move_time_s, self.grasp_quat_b):
                box_top = STORAGE_BOX_POS[2] + STORAGE_BOX_SIZE[2] / 2.0
                place_z = box_top + CUBE_SIZE / 2.0 + args_cli.place_clearance
                self.enter("lower_into_storage", self.storage_waypoint(place_z + args_cli.grasp_offset))

        elif self.state == "lower_into_storage":
            self.set_gripper(False)
            if self.interpolated_move(args_cli.short_move_time_s, self.grasp_quat_b):
                self.enter("release", self.goal_pos_b)

        elif self.state == "release":
            assert self.goal_pos_b is not None
            self.move_tcp(self.goal_pos_b, self.grasp_quat_b)
            self.set_gripper(True)
            if self.state_time >= args_cli.open_time_s:
                self.enter("retreat", self.storage_waypoint(args_cli.safe_z))

        elif self.state == "retreat":
            self.set_gripper(True)
            if self.interpolated_move(args_cli.short_move_time_s):
                self.enter("done", self.goal_pos_b)

        elif self.state == "done":
            assert self.goal_pos_b is not None
            self.move_tcp(self.goal_pos_b)
            self.set_gripper(True)

        elif self.state == "failed":
            pass


class IsaacLabRGBSensorGrabber:
    """Convert the Isaac Lab camera output to LeRobot's HWC uint8 format."""

    def __init__(self, camera_sensor, width: int, height: int):
        self.camera_sensor = camera_sensor
        self.latest_rgb = np.zeros((height, width, 3), dtype=np.uint8)
        self._warned = False

    def read(self) -> np.ndarray:
        try:
            rgb = self.camera_sensor.data.output.get("rgb")
            if rgb is None:
                raise RuntimeError("camera_sensor.data.output['rgb'] is unavailable")
            if isinstance(rgb, torch.Tensor):
                arr = rgb[0].detach().cpu().numpy() if rgb.ndim == 4 else rgb.detach().cpu().numpy()
            else:
                arr = np.asarray(rgb)
                if arr.ndim == 4:
                    arr = arr[0]
            if arr.ndim != 3 or arr.shape[-1] != 3:
                raise RuntimeError(f"unexpected RGB shape: {arr.shape}")
            if arr.dtype != np.uint8:
                if arr.size and float(np.nanmax(arr)) <= 1.0:
                    arr = arr * 255.0
                arr = np.clip(arr, 0, 255).astype(np.uint8)
            self.latest_rgb = np.ascontiguousarray(arr)
        except Exception as exc:
            if not self._warned:
                print(f"[WARN] RGB capture failed; using the last valid frame: {exc}")
                self._warned = True
        return self.latest_rgb.copy()


class MultiEpisodeLeRobotRecorder:
    """One LeRobotDataset instance that accumulates multiple saved episodes."""

    def __init__(
        self,
        controller: PickPlaceController,
        top_rgb_sensor,
        wrist_rgb_sensor,
    ) -> None:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        root = Path(args_cli.dataset_root).expanduser() if args_cli.dataset_root else None
        if root is not None and root.exists():
            if args_cli.overwrite_dataset:
                shutil.rmtree(root)
            else:
                raise RuntimeError(
                    f"Dataset root already exists: {root}. Use --overwrite_dataset or another --dataset_root."
                )

        names = controller.record_joint_names
        features = {
            "observation.state": {
                "dtype": "float32", "shape": (len(names),), "names": [names]
            },
            "action": {
                "dtype": "float32", "shape": (len(names),), "names": [names]
            },
        }
        self.top_camera = None
        self.wrist_camera = None
        if args_cli.record_camera:
            for camera_key in ("observation.images.top", "observation.images.wrist"):
                features[camera_key] = {
                    "dtype": "video" if args_cli.use_videos else "image",
                    "shape": (DEFAULT_CAMERA_HEIGHT, DEFAULT_CAMERA_WIDTH, 3),
                    "names": ["height", "width", "channels"],
                }
            self.top_camera = IsaacLabRGBSensorGrabber(
                top_rgb_sensor, DEFAULT_CAMERA_WIDTH, DEFAULT_CAMERA_HEIGHT
            )
            self.wrist_camera = IsaacLabRGBSensorGrabber(
                wrist_rgb_sensor, DEFAULT_CAMERA_WIDTH, DEFAULT_CAMERA_HEIGHT
            )
        self.dataset = LeRobotDataset.create(
            repo_id=args_cli.dataset_repo_id,
            root=root,
            fps=DATASET_FPS,
            robot_type="openarm_isaaclab",
            features=features,
            use_videos=args_cli.use_videos,
            image_writer_threads=args_cli.image_writer_threads,
            image_writer_processes=args_cli.image_writer_processes,
        )
        self.controller = controller
        self.elapsed_s = 0.0
        self.next_frame_t = 0.0
        self.episode_frames = 0
        self.saved_episodes = 0
        self.finalized = False
        print(f"[RECORD] Dataset root: {self.dataset.root}")
        print(f"[RECORD] Recording {args_cli.num_episodes} episode(s) at {DATASET_FPS} FPS")
        print("[RECORD] State/action schema: [joint_1, ..., joint_7, gripper], radians")
        print(
            f"[RECORD] Cameras: top + wrist, "
            f"{DEFAULT_CAMERA_WIDTH}x{DEFAULT_CAMERA_HEIGHT} RGB"
        )

    def begin_episode(self, controller: PickPlaceController) -> None:
        self.controller = controller
        self.elapsed_s = 0.0
        self.next_frame_t = 0.0
        self.episode_frames = 0

    def record_if_needed(self, dt: float, force: bool = False) -> None:
        self.elapsed_s += dt
        period = 1.0 / float(DATASET_FPS)
        if force or self.elapsed_s + 1e-9 >= self.next_frame_t:
            state, action = self.controller.get_record_state_action()
            frame = {"observation.state": state, "action": action, "task": args_cli.task}
            if self.top_camera is not None and self.wrist_camera is not None:
                frame["observation.images.top"] = self.top_camera.read()
                frame["observation.images.wrist"] = self.wrist_camera.read()
            if hasattr(self.dataset, "features") and "next.done" in self.dataset.features:
                frame["next.done"] = np.array([False], dtype=bool)
            # LeRobotDataset.add_frame() automatically assigns:
            # timestamp = frame_index / self.dataset.fps = frame_index / 30.
            self.dataset.add_frame(frame)
            self.episode_frames += 1
            if not force:
                self.next_frame_t += period

    def save_episode(self) -> None:
        if self.episode_frames == 0:
            raise RuntimeError("Cannot save an episode with zero frames")
        self.dataset.save_episode()
        self.saved_episodes += 1
        print(f"[RECORD] Saved episode {self.saved_episodes}/{args_cli.num_episodes} "
              f"({self.episode_frames} frames)")

    def discard_episode(self, reason: str) -> None:
        discarded_frames = self.episode_frames
        self.dataset.clear_episode_buffer()
        self.elapsed_s = 0.0
        self.next_frame_t = 0.0
        self.episode_frames = 0
        print(f"[RECORD] Discarded {discarded_frames} frame(s): {reason}")

    def finalize(self) -> None:
        if self.finalized:
            return
        if hasattr(self.dataset, "finalize"):
            self.dataset.finalize()
        if args_cli.push_to_hub:
            self.dataset.push_to_hub(private=args_cli.private)
            print(f"[RECORD] Pushed to Hub: {args_cli.dataset_repo_id}")
        self.finalized = True
        print(f"[RECORD] Finalized {self.saved_episodes} episode(s) at: {self.dataset.root}")


def run_simulator(
    sim: SimulationContext,
    scene: InteractiveScene,
    top_rgb_sensor,
    wrist_rgb_sensor,
) -> None:
    dt = sim.get_physics_dt()
    recorder = None
    attempt_index = 0
    try:
        while recorder is None or recorder.saved_episodes < args_cli.num_episodes:
            if not simulation_app.is_running():
                break
            attempt_index += 1
            episode_number = 1 if recorder is None else recorder.saved_episodes + 1
            print(
                f"\n[EPISODE] Starting dataset episode "
                f"{episode_number}/{args_cli.num_episodes} (attempt {attempt_index})"
            )
            cube_position = reset_scene(scene)
            print(f"[EPISODE] Sampled cube xyz={cube_position}")
            scene.write_data_to_sim()
            sim.step()
            scene.update(dt)

            warmup_steps = max(1, int(args_cli.warmup_time_s / dt))
            for _ in range(warmup_steps):
                scene.write_data_to_sim()
                sim.step()
                scene.update(dt)

            controller = PickPlaceController(scene["robot"], scene)
            if recorder is None:
                recorder = MultiEpisodeLeRobotRecorder(
                    controller,
                    top_rgb_sensor,
                    wrist_rgb_sensor,
                )
            else:
                recorder.begin_episode(controller)

            while simulation_app.is_running() and controller.state not in ("done", "failed"):
                controller.advance(dt)
                scene.write_data_to_sim()
                sim.step()
                scene.update(dt)
                recorder.record_if_needed(dt)

            if controller.state == "failed":
                recorder.discard_episode(controller.failure_reason or "IK failure")
                print(f"[EPISODE] Retrying dataset episode {episode_number}")
                continue
            if controller.state != "done":
                break
            recorder.record_if_needed(0.0, force=True)
            recorder.save_episode()
    finally:
        if recorder is not None:
            recorder.finalize()


def main() -> None:
    sim = SimulationContext(sim_utils.SimulationCfg(device=args_cli.device, dt=1.0 / 120.0))
    sim.set_camera_view(eye=[1.0, -1.35, 0.85], target=[-0.35, -0.2, 0.15])
    scene = InteractiveScene(DualRealSensePickPlaceSceneCfg(num_envs=1, env_spacing=1.0))
    wrist_root_path = attach_wrist_realsense(scene)
    sim.reset()
    camera_views = create_dual_realsense_views(scene, wrist_root_path)
    top_rgb_sensor = scene["top_color"] if args_cli.record_camera else None
    wrist_rgb_sensor = scene["wrist_color"] if args_cli.record_camera else None
    print("[INFO] OpenArm dual-RealSense tilted IK dataset scene ready.")
    if args_cli.cube_x_range is not None or args_cli.cube_y_range is not None:
        print(
            f"[INFO] Cube randomization x={args_cli.cube_x_range or args_cli.cube_x}, "
            f"y={args_cli.cube_y_range or args_cli.cube_y}, z={args_cli.cube_z}, "
            f"seed={args_cli.cube_random_seed}"
        )
    else:
        print(f"[INFO] Cube drop xyz={CUBE_POS}")
    print(f"[INFO] Storage center xyz={STORAGE_BOX_POS}")
    run_simulator(sim, scene, top_rgb_sensor, wrist_rgb_sensor)


if __name__ == "__main__":
    main()
    simulation_app.close()
