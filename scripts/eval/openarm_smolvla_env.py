"""
Three-cube OpenArm Isaac Lab eval environment for SmolVLA.

The policy contract matches the dataset contract:
- observation.state: 8D degrees
- action: 8D degrees
- observation.images.top / wrist: 480x640 RGB
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium import spaces


def resolve_asset_dir() -> Path:
    legacy_asset_dir = Path(__file__).resolve().parents[2] / "assets" / "openarm_use"
    if legacy_asset_dir.exists():
        return legacy_asset_dir

    src_asset_dir = Path(__file__).resolve().parents[2] / "src" / "lerobot" / "assets" / "openarm_use"
    if src_asset_dir.exists():
        return src_asset_dir

    import lerobot

    package_asset_dir = Path(lerobot.__file__).resolve().parent / "assets" / "openarm_use"
    if package_asset_dir.exists():
        return package_asset_dir

    raise FileNotFoundError("Could not locate openarm assets.")


TASK = "Pick up the red cube and place it in the storage box."
TASKS_BY_COLOR = {
    "red": "Pick up the red cube and place it in the storage box.",
    "blue": "Pick up the blue cube and place it in the storage box.",
    "yellow": "Pick up the yellow cube and place it in the storage box.",
}
ROBOT_TYPE = "openarm_isaaclab"
OBSERVATION_KEYS = (
    "observation.state",
    "observation.images.top",
    "observation.images.wrist",
)
ACTION_DIM = 8
COLOR_NAMES = ("red", "blue", "yellow")
OBJECT_COLORS: dict[str, tuple[float, float, float]] = {
    "red": (0.85, 0.05, 0.03),
    "blue": (0.03, 0.15, 0.90),
    "yellow": (0.95, 0.80, 0.03),
}


def infer_instruction_color(task_text: str) -> str | None:
    task_text = task_text.lower()
    for color in COLOR_NAMES:
        if f"{color} cube" in task_text:
            return color
    return None

ASSET_DIR = resolve_asset_dir()
TABLE_USD = str(ASSET_DIR / "table.usd")
ROBOT_USD = str(ASSET_DIR / "openarm_half_tesollo_tactile.usd")
REALSENSE_USD = str(ASSET_DIR / "realsense.usd")
STORAGE_BOX_USD = str(ASSET_DIR / "storage_box.usd")

ROBOT_POS = (-0.17858, 0.24336, -0.00468)
ROBOT_ROT = (0.0, 0.0, 0.0, 1.0)
REALSENSE_POS = (-0.38725, 0.09957, 0.78544)
REALSENSE_ROT = (math.cos(math.pi / 4.0), 0.0, math.sin(math.pi / 4.0), 0.0)
FIXED_BOX_SIZE = (0.45, 1.0, 0.05)
FIXED_BOX_POS = (-0.57337, 0.0, 0.02164)
STORAGE_BOX_SIZE = (0.55, 0.75, 0.15)
STORAGE_BOX_POS = (-0.53584, -0.15, 0.04664)
FLOOR_Z = -0.754904
CUBE_SIZE = 0.05
CUBE_Z = 0.1
CUBE_MIN_SEPARATION = 0.09
CUBE_SAMPLING_MAX_ATTEMPTS = 1000
CONTROL_HZ = 30
SUCCESS_INNER_MARGIN_X = 0.155
SUCCESS_INNER_MARGIN_Y = 0.275
SUCCESS_Z_MARGIN_BOTTOM = 0.005
SUCCESS_Z_MARGIN_TOP = 0.015
SUCCESS_MAX_LINEAR_SPEED = 0.05
WORKSPACE_X_RANGE = (-0.62, -0.45)
WORKSPACE_Y_RANGE = (0.00, 0.15)
DATASET_GENERATION_TCP_TILT_DEG = 50.0
WRIST_REALSENSE_POS = (0.06336, -0.01314, -0.05143)
WRIST_REALSENSE_ROT = (0.0, 0.5, 0.0, 0.86603)
WRIST_PARENT_PRIM_NAME = "openarm_left_hand_tcp"

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

# Isaac app must initialize before other Isaac imports.
from isaaclab.app import AppLauncher

original_argv = sys.argv.copy()
sys.argv = [sys.argv[0]]
HEADLESS = os.environ.get("OPENARM_SMOLVLA_HEADLESS", "0") == "1"
app_launcher = AppLauncher({"headless": HEADLESS, "enable_cameras": True})
simulation_app = app_launcher.app
sys.argv = original_argv

import isaaclab.sim as sim_utils
import torch
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg, RigidObject, RigidObjectCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import Camera, CameraCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaacsim.core.utils.stage import get_current_stage
from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade


def joint_positions_rad() -> dict[str, float]:
    return {name: math.radians(value) for name, value in INITIAL_JOINT_POS_DEG.items()}


def cube_cfg(color: str) -> RigidObjectCfg:
    return RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{color.title()}Cube",
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
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=OBJECT_COLORS[color]),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.52772, 0.0974, CUBE_Z)),
    )


def apply_cube_color_override(cube_root_path: Sdf.Path | str, color_rgb: tuple[float, float, float]) -> None:
    stage = get_current_stage()
    cube_prim = stage.GetPrimAtPath(cube_root_path)
    if not cube_prim.IsValid():
        raise RuntimeError(f"Cube prim not found at {cube_root_path}")
    diffuse_color = Gf.Vec3f(*color_rgb)
    for prim in Usd.PrimRange(cube_prim):
        if prim.IsA(UsdShade.Shader):
            shader = UsdShade.Shader(prim)
            if shader.GetIdAttr().Get() == "UsdPreviewSurface":
                shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(diffuse_color)
        if prim.IsA(UsdGeom.Gprim):
            UsdGeom.Gprim(prim).CreateDisplayColorAttr([diffuse_color])


def attach_wrist_realsense(scene: InteractiveScene) -> Sdf.Path:
    stage = get_current_stage()
    robot_root = f"{scene.env_prim_paths[0]}/Robot"
    root_prim = stage.GetPrimAtPath(robot_root)
    matches = [p.GetPath() for p in Usd.PrimRange(root_prim) if p.GetName() == WRIST_PARENT_PRIM_NAME]
    if not matches:
        raise RuntimeError(f"Could not find TCP prim '{WRIST_PARENT_PRIM_NAME}' under {robot_root}")
    tcp_path = matches[0]
    wrist_root_path = tcp_path.AppendChild("WristRealsense")
    wrist_root = UsdGeom.Xform.Define(stage, wrist_root_path)
    wrist_root.GetPrim().GetReferences().AddReference(REALSENSE_USD)
    xformable = UsdGeom.Xformable(wrist_root.GetPrim())
    xformable.ClearXformOpOrder()
    xformable.AddTranslateOp(opSuffix="wristMount").Set(Gf.Vec3d(*WRIST_REALSENSE_POS))
    quat = WRIST_REALSENSE_ROT
    xformable.AddOrientOp(opSuffix="wristMount").Set(Gf.Quatf(quat[0], Gf.Vec3f(*quat[1:])))
    return wrist_root_path


def find_color_camera_path(realsense_root: Sdf.Path | str) -> Sdf.Path:
    stage = get_current_stage()
    root_prim = stage.GetPrimAtPath(realsense_root)
    source_paths = [prim.GetPath() for prim in Usd.PrimRange(root_prim) if prim.IsA(UsdGeom.Camera)]
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
    stage = get_current_stage()
    source_prim = stage.GetPrimAtPath(source_path)
    record_prim = stage.GetPrimAtPath(record_path)
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


def sample_three_cube_positions(rng: np.random.Generator) -> dict[str, tuple[float, float, float]]:
    sampled_positions: list[tuple[float, float, float]] = []
    while True:
        sampled_positions.clear()
        layout_is_valid = True
        for _ in range(len(COLOR_NAMES)):
            for _ in range(CUBE_SAMPLING_MAX_ATTEMPTS):
                candidate = (
                    float(rng.uniform(*WORKSPACE_X_RANGE)),
                    float(rng.uniform(*WORKSPACE_Y_RANGE)),
                    float(CUBE_Z),
                )
                if all(
                    math.hypot(candidate[0] - other[0], candidate[1] - other[1]) >= CUBE_MIN_SEPARATION
                    for other in sampled_positions
                ):
                    sampled_positions.append(candidate)
                    break
            else:
                layout_is_valid = False
                break
        if layout_is_valid:
            shuffled_colors = list(COLOR_NAMES)
            rng.shuffle(shuffled_colors)
            return dict(zip(shuffled_colors, sampled_positions, strict=True))


def cube_success(cube: RigidObject) -> tuple[bool, dict[str, object]]:
    position = cube.data.root_pos_w[0].detach().cpu().numpy()
    velocity = cube.data.root_vel_w[0].detach().cpu().numpy()
    (x_min, x_max), (y_min, y_max), (z_min, z_max) = get_success_bounds()
    x_inside = x_min < float(position[0]) < x_max
    y_inside = y_min < float(position[1]) < y_max
    z_inside = z_min < float(position[2]) < z_max
    in_box = x_inside and y_inside and z_inside
    speed = float(np.linalg.norm(velocity[:3]))
    stationary = speed < SUCCESS_MAX_LINEAR_SPEED
    return bool(in_box and stationary), {
        "position": position.astype(np.float32).tolist(),
        "linear_velocity": velocity[:3].astype(np.float32).tolist(),
        "in_box": in_box,
        "x_inside": x_inside,
        "y_inside": y_inside,
        "z_inside": z_inside,
        "stationary": stationary,
        "speed": speed,
    }


def get_success_bounds() -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    """Conservative box-interior volume for per-step eval termination.

    Dataset generation used a broader post-controller check. For eval we need a
    tighter interior region because the same criterion is queried every step,
    including immediately after reset.
    """
    outer_half_x = STORAGE_BOX_SIZE[0] / 2.0
    outer_half_y = STORAGE_BOX_SIZE[1] / 2.0
    inner_half_x = max(outer_half_x - SUCCESS_INNER_MARGIN_X, CUBE_SIZE / 2.0 + 0.02)
    inner_half_y = max(outer_half_y - SUCCESS_INNER_MARGIN_Y, CUBE_SIZE / 2.0 + 0.02)
    x_bounds = (STORAGE_BOX_POS[0] - inner_half_x, STORAGE_BOX_POS[0] + inner_half_x)
    y_bounds = (STORAGE_BOX_POS[1] - inner_half_y, STORAGE_BOX_POS[1] + inner_half_y)
    z_bounds = (
        STORAGE_BOX_POS[2] + SUCCESS_Z_MARGIN_BOTTOM,
        STORAGE_BOX_POS[2] + STORAGE_BOX_SIZE[2] / 2.0 - SUCCESS_Z_MARGIN_TOP,
    )
    return x_bounds, y_bounds, z_bounds


@configclass
class PickPlaceSceneCfg(InteractiveSceneCfg):
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
    )
    table_top_proxy = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/TableTopProxyCollider",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.0211)),
        spawn=sim_utils.CuboidCfg(
            size=(1.6, 1.0, 0.04),
            visible=False,
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True, contact_offset=0.01, rest_offset=0.0),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=0.8, restitution=0.0),
        ),
    )
    fixed_box = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/FixedBox",
        init_state=AssetBaseCfg.InitialStateCfg(pos=FIXED_BOX_POS),
        spawn=sim_utils.CuboidCfg(
            size=FIXED_BOX_SIZE,
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True, contact_offset=0.005, rest_offset=0.0),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.8, dynamic_friction=0.6, restitution=0.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.5, 0.5)),
        ),
    )
    storage_box = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/StorageBox",
        spawn=sim_utils.UsdFileCfg(
            usd_path=STORAGE_BOX_USD,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True, contact_offset=0.005, rest_offset=0.0),
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
        update_period=0.0,
        height=480,
        width=640,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955),
        offset=CameraCfg.OffsetCfg(pos=REALSENSE_POS, rot=REALSENSE_ROT, convention="world"),
    )
    robot = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=ROBOT_USD,
            activate_contact_sensors=True,
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(enabled_self_collisions=False, fix_root_link=True),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True, max_depenetration_velocity=5.0),
        ),
        init_state=ArticulationCfg.InitialStateCfg(pos=ROBOT_POS, rot=ROBOT_ROT, joint_pos=joint_positions_rad()),
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
    red_cube = cube_cfg("red")
    blue_cube = cube_cfg("blue")
    yellow_cube = cube_cfg("yellow")


DEFAULT_MIN_STEPS_BEFORE_SUCCESS = 50


class OpenArmEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}
    task = TASK
    task_description = TASK
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._is_initialized = False
        return cls._instance

    def __init__(
        self,
        obs_type="pixels_agent_pos",
        render_mode="rgb_array",
        max_episode_steps=1000,
        seed: int | None = None,
        task: str | None = None,
        min_steps_before_success: int = DEFAULT_MIN_STEPS_BEFORE_SUCCESS,
        debug_success: bool = False,
        diagnostic_instrumentation: bool = False,
    ):
        if getattr(self, "_is_initialized", False):
            return
        super().__init__()
        self.obs_type = obs_type
        self.render_mode = render_mode
        self.max_episode_steps = max_episode_steps
        self.current_step = 0
        self.reset_counter = 0
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.task_str = task if task is not None else TASK
        self.task = self.task_str
        self.task_description = self.task_str
        self.layout_rng = np.random.default_rng(seed)
        self.min_steps_before_success = min_steps_before_success
        self.debug_success = debug_success
        self.diagnostic_instrumentation = diagnostic_instrumentation
        self.action_space = spaces.Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32)
        self.observation_space = spaces.Dict(
            {
                "observation.state": spaces.Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32),
                "observation.images.top": spaces.Box(low=0, high=255, shape=(480, 640, 3), dtype=np.uint8),
                "observation.images.wrist": spaces.Box(low=0, high=255, shape=(480, 640, 3), dtype=np.uint8),
            }
        )
        self.sim = SimulationContext(sim_utils.SimulationCfg(device=self.device, dt=1.0 / 120.0))
        self.scene = InteractiveScene(PickPlaceSceneCfg(num_envs=1, env_spacing=1.0))
        wrist_root_path = attach_wrist_realsense(self.scene)
        self.sim.reset()
        env_root = self.scene.env_prim_paths[0]
        top_source_path = find_color_camera_path(Sdf.Path(f"{env_root}/Realsense"))
        wrist_source_path = find_color_camera_path(wrist_root_path)
        match_record_camera_to_source(top_source_path, Sdf.Path(f"{env_root}/TopColorCamera"))
        self.robot: Articulation = self.scene["robot"]
        self.cubes: dict[str, RigidObject] = {
            "red": self.scene["red_cube"],
            "blue": self.scene["blue_cube"],
            "yellow": self.scene["yellow_cube"],
        }
        self.camera_top = self.scene["top_color"]
        for color in COLOR_NAMES:
            apply_cube_color_override(f"{env_root}/{color.title()}Cube", OBJECT_COLORS[color])
        self._cube_default_quat = {
            color: self.cubes[color].data.default_root_state[0, 3:7].detach().cpu().numpy().astype(np.float32)
            for color in COLOR_NAMES
        }
        wrist_cam_cfg = CameraCfg(
            prim_path=str(wrist_source_path),
            update_period=0.0,
            height=480,
            width=640,
            data_types=["rgb"],
            spawn=None,
        )
        self.camera_wrist = Camera(wrist_cam_cfg)
        self.camera_wrist._initialize_impl()
        arm_names = [f"openarm_left_joint{i}" for i in range(1, 8)]
        self.arm_ids, _ = self.robot.find_joints(arm_names, preserve_order=True)
        gripper_names = ["openarm_left_finger_joint1", "openarm_left_finger_joint2"]
        self.gripper_ids, _ = self.robot.find_joints(gripper_names, preserve_order=True)
        self.record_gripper_id = self.gripper_ids[0]
        self.ee_entity = SceneEntityCfg(
            "robot",
            joint_names=arm_names,
            body_names=["openarm_left_hand_tcp"],
            preserve_order=True,
        )
        self.ee_entity.resolve(self.scene)
        self.ee_body_id = self.ee_entity.body_ids[0]
        self.initial_positions: dict[str, list[float]] = {}
        self._is_initialized = True

    def describe_configuration(self) -> dict[str, object]:
        return {
            "task": self.task_str,
            "tasks_by_color": TASKS_BY_COLOR,
            "cube_workspace_x": list(WORKSPACE_X_RANGE),
            "cube_workspace_y": list(WORKSPACE_Y_RANGE),
            "cube_min_separation": CUBE_MIN_SEPARATION,
            "cube_count": 3,
            "gripper_angle_deg": DATASET_GENERATION_TCP_TILT_DEG,
            "storage_box_usd": STORAGE_BOX_USD,
            "storage_box_position": list(STORAGE_BOX_POS),
            "success_bounds": {
                "x": list(get_success_bounds()[0]),
                "y": list(get_success_bounds()[1]),
                "z": list(get_success_bounds()[2]),
                "velocity": SUCCESS_MAX_LINEAR_SPEED,
            },
            "min_steps_before_success": self.min_steps_before_success,
            "top_camera_pose": {"pos": list(REALSENSE_POS), "rot_wxyz": list(REALSENSE_ROT)},
            "observation_keys": list(OBSERVATION_KEYS),
            "state_dimension": 8,
            "action_dimension": ACTION_DIM,
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        if seed is not None:
            self.layout_rng = np.random.default_rng(seed)
        robot_state = self.robot.data.default_root_state.clone()
        robot_state[:, :3] += self.scene.env_origins
        self.robot.write_root_pose_to_sim(robot_state[:, :7])
        self.robot.write_root_velocity_to_sim(robot_state[:, 7:])
        joint_pos = self.robot.data.default_joint_pos.clone()
        self.robot.write_joint_state_to_sim(joint_pos, torch.zeros_like(self.robot.data.default_joint_vel))
        self.robot.set_joint_position_target(joint_pos)
        positions = sample_three_cube_positions(self.layout_rng)
        self.initial_positions = {}
        for color, cube in self.cubes.items():
            cube_state = cube.data.default_root_state.clone()
            requested_position = np.asarray(positions[color], dtype=np.float32)
            cube_state[:, :3] = torch.tensor(requested_position, dtype=cube_state.dtype, device=cube_state.device)
            cube_state[:, 3:7] = torch.tensor(self._cube_default_quat[color], dtype=cube_state.dtype, device=cube_state.device)
            cube_state[:, :3] += self.scene.env_origins
            cube.write_root_pose_to_sim(cube_state[:, :7])
            cube.write_root_velocity_to_sim(torch.zeros_like(cube_state[:, 7:]))
        self.scene.reset()
        for _ in range(120):
            self.scene.write_data_to_sim()
            self.sim.step()
            self.scene.update(self.sim.get_physics_dt())
        info: dict[str, object] = {"is_success": False, "task": self.task_str, "task_description": self.task_str}
        reset_debug: dict[str, dict[str, object]] = {}
        for color, cube in self.cubes.items():
            settled_position = cube.data.root_pos_w[0].detach().cpu().numpy().astype(np.float32)
            self.initial_positions[color] = settled_position.tolist()
            info[f"{color}_initial_x"] = float(settled_position[0])
            info[f"{color}_initial_y"] = float(settled_position[1])
            info[f"{color}_initial_z"] = float(settled_position[2])
            success, debug = cube_success(cube)
            reset_debug[color] = {
                "position": debug["position"],
                "linear_velocity": debug["linear_velocity"],
                "success": success,
            }
        if self.debug_success:
            x_bounds, y_bounds, z_bounds = get_success_bounds()
            print("[RESET DEBUG]", flush=True)
            print(f"success_bounds x={x_bounds} y={y_bounds} z={z_bounds} velocity<{SUCCESS_MAX_LINEAR_SPEED}", flush=True)
            for color in COLOR_NAMES:
                debug = reset_debug[color]
                print(
                    f"{color} pos={debug['position']} vel={debug['linear_velocity']} success={debug['success']}",
                    flush=True,
                )
        self.reset_counter += 1
        return self._get_obs(), info

    def _get_end_effector_pose_world(self) -> tuple[list[float], list[float]]:
        ee_pose = self.robot.data.body_pose_w[0, self.ee_body_id].detach().cpu().numpy().astype(np.float32)
        return ee_pose[:3].tolist(), ee_pose[3:7].tolist()

    def _get_cube_position(self, color: str) -> list[float]:
        return self.cubes[color].data.root_pos_w[0].detach().cpu().numpy().astype(np.float32).tolist()

    def _build_diagnostic_state(self, action_deg: np.ndarray) -> dict[str, object]:
        ee_position, ee_orientation = self._get_end_effector_pose_world()
        cube_positions = {color: self._get_cube_position(color) for color in COLOR_NAMES}
        gripper_sim_joint_deg = float(np.rad2deg(self.robot.data.joint_pos[0, self.record_gripper_id].item()))
        return {
            "frame": "world",
            "current_step": int(self.current_step),
            "task": self.task_str,
            "target_color": infer_instruction_color(self.task_str),
            "ee_position": ee_position,
            "ee_orientation_wxyz": ee_orientation,
            "cube_positions": cube_positions,
            "storage_box_position": list(STORAGE_BOX_POS),
            "policy_action_deg": np.asarray(action_deg, dtype=np.float32).tolist(),
            "gripper_action_deg": float(action_deg[7]),
            "gripper_state_deg": gripper_sim_joint_deg,
            "gripper_sim_joint_deg": gripper_sim_joint_deg,
            "initial_positions": self.initial_positions,
            "cube_size_m": float(CUBE_SIZE),
        }

    def step(self, action: np.ndarray):
        action_deg = np.asarray(action, dtype=np.float32)
        if action_deg.shape != (8,):
            raise ValueError(f"Expected action shape (8,), got {action_deg.shape}")
        action = np.deg2rad(action_deg).astype(np.float32)
        self.current_step += 1
        action_tensor = torch.tensor(action, dtype=torch.float32, device=self.device).unsqueeze(0)
        real_gripper_closed = math.radians(-15.0)
        real_gripper_open = math.radians(-60.0)
        sim_gripper_closed = 0.0
        sim_gripper_open = 0.044
        raw_gripper_action = action_tensor[:, 7]
        alpha = (raw_gripper_action - real_gripper_closed) / (real_gripper_open - real_gripper_closed)
        alpha = torch.clamp(alpha, min=0.0, max=1.0)
        sim_gripper_action = sim_gripper_closed + alpha * (sim_gripper_open - sim_gripper_closed)
        arm_target = action_tensor[:, :7]
        joint1_target = sim_gripper_action
        joint2_target = sim_gripper_open - sim_gripper_action
        gripper_target = torch.stack([joint1_target, joint2_target], dim=-1)
        self.robot.set_joint_position_target(arm_target, joint_ids=self.arm_ids)
        self.robot.set_joint_position_target(gripper_target, joint_ids=self.gripper_ids)
        for _ in range(4):
            self.scene.write_data_to_sim()
            self.sim.step()
            self.scene.update(self.sim.get_physics_dt())
        obs = self._get_obs()
        success_by_color: dict[str, bool] = {}
        cube_debug: dict[str, dict[str, object]] = {}
        picked_candidates: list[str] = []
        for color, cube in self.cubes.items():
            success, debug = cube_success(cube)
            if self.current_step < self.min_steps_before_success:
                success = False
            success_by_color[color] = success
            cube_debug[color] = debug
            if success:
                picked_candidates.append(color)
        if len(picked_candidates) == 1:
            picked_color = picked_candidates[0]
            termination_reason = "success"
        elif len(picked_candidates) > 1:
            picked_color = "ambiguous"
            termination_reason = "ambiguous_multiple_cubes_in_box"
        else:
            picked_color = "failure"
            termination_reason = "max_steps" if self.current_step >= self.max_episode_steps else "running"
        is_success = len(picked_candidates) >= 1
        terminated = is_success
        truncated = self.current_step >= self.max_episode_steps
        info: dict[str, object] = {
            "is_success": is_success,
            "picked_color": picked_color,
            "task": self.task_str,
            "task_description": self.task_str,
            "termination_reason": termination_reason,
            "success_by_color": success_by_color,
        }
        for color in COLOR_NAMES:
            debug = cube_debug[color]
            pos = debug["position"]
            vel = debug["linear_velocity"]
            info[f"{color}_success"] = success_by_color[color]
            info[f"{color}_final_x"] = float(pos[0])
            info[f"{color}_final_y"] = float(pos[1])
            info[f"{color}_final_z"] = float(pos[2])
            info[f"{color}_linear_velocity"] = vel
        if self.diagnostic_instrumentation:
            info["diagnostic_state"] = self._build_diagnostic_state(action_deg)
        if self.debug_success and is_success:
            print("[SUCCESS DEBUG]", flush=True)
            print(
                f"step={self.current_step} target={self.task_str} red_success={success_by_color['red']} "
                f"blue_success={success_by_color['blue']} yellow_success={success_by_color['yellow']} "
                f"picked_color={picked_color}",
                flush=True,
            )
        return obs, float(is_success), terminated, truncated, info

    def _get_obs(self):
        real_gripper_closed = math.radians(-15.0)
        real_gripper_open = math.radians(-60.0)
        sim_gripper_closed = 0.0
        sim_gripper_open = 0.044
        sim_gripper_pos = self.robot.data.joint_pos[0, self.record_gripper_id].item()
        alpha = (sim_gripper_pos - sim_gripper_closed) / (sim_gripper_open - sim_gripper_closed)
        alpha = min(max(alpha, 0.0), 1.0)
        real_gripper_pos = real_gripper_closed + alpha * (real_gripper_open - real_gripper_closed)
        mapped_gripper_tensor = torch.tensor([real_gripper_pos], dtype=torch.float32, device=self.device)
        state_rad = torch.cat((self.robot.data.joint_pos[0, self.arm_ids], mapped_gripper_tensor))
        agent_pos = np.rad2deg(state_rad.detach().cpu().numpy()).astype(np.float32)

        def get_rgb(sensor):
            rgb = sensor.data.output.get("rgb")
            if rgb is not None:
                arr = (
                    rgb[0].detach().cpu().numpy()
                    if isinstance(rgb, torch.Tensor) and rgb.ndim == 4
                    else (rgb.detach().cpu().numpy() if isinstance(rgb, torch.Tensor) else np.asarray(rgb))
                )
                if arr.ndim == 4:
                    arr = arr[0]
                if arr.dtype != np.uint8:
                    if arr.size and float(np.nanmax(arr)) <= 1.0:
                        arr = arr * 255.0
                    arr = np.clip(arr, 0, 255).astype(np.uint8)
                return np.ascontiguousarray(arr)
            return np.zeros((480, 640, 3), dtype=np.uint8)

        self.camera_wrist.update(self.sim.get_physics_dt())
        return {
            "observation.state": agent_pos,
            "observation.images.top": get_rgb(self.camera_top),
            "observation.images.wrist": get_rgb(self.camera_wrist),
        }

    def render(self):
        if self.render_mode == "rgb_array":
            return self._get_obs()["observation.images.top"]

    def close(self):
        pass
