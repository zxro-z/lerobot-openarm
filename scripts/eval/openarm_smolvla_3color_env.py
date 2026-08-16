#!/usr/bin/env python3
"""Three-cube OpenArm eval environment for real 3-color SmolVLA checkpoints."""

from __future__ import annotations

import math
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces

from openarm_smolvla_env import (
    CUBE_SIZE,
    FIXED_BOX_POS,
    FIXED_BOX_SIZE,
    FLOOR_Z,
    REALSENSE_POS as LEGACY_REALSENSE_POS,
    REALSENSE_ROT as LEGACY_REALSENSE_ROT,
    REALSENSE_USD,
    ROBOT_POS,
    ROBOT_ROT,
    ROBOT_USD,
    STORAGE_BOX_SIZE,
    STORAGE_BOX_USD,
    TABLE_USD,
    WRIST_REALSENSE_POS,
    WRIST_REALSENSE_ROT,
    AppLauncher,
    Articulation,
    ArticulationCfg,
    AssetBaseCfg,
    Camera,
    CameraCfg,
    Gf,
    ImplicitActuatorCfg,
    InteractiveScene,
    InteractiveSceneCfg,
    OBJECT_COLORS,
    Path as _UnusedPathAlias,
    RigidObjectCfg,
    Sdf,
    SimulationContext,
    UsdGeom,
    UsdShade,
    app_launcher,
    apply_cube_color_override,
    attach_wrist_realsense,
    configclass,
    find_color_camera_path,
    joint_positions_rad,
    match_record_camera_to_source,
    sim_utils,
    simulation_app,
)


COLOR_NAMES = ("red", "blue", "yellow")
TASK_BY_COLOR = {
    "red": "Pick up the red cube and place it in the storage box.",
    "blue": "Pick up the blue cube and place it in the storage box.",
    "yellow": "Pick up the yellow cube and place it in the storage box.",
}
DEFAULT_TARGET_COLOR = "red"
ROBOT_TYPE = "openarm_follower"
OBSERVATION_KEYS = (
    "observation.state",
    "observation.images.front",
    "observation.images.wrist",
)
STATE_DIM = 24
ACTION_DIM = 24
MOTOR_NAMES = (
    "joint_1",
    "joint_2",
    "joint_3",
    "joint_4",
    "joint_5",
    "joint_6",
    "joint_7",
    "gripper",
)
SLOT_NAMES = ("left", "center", "right")
REAL_3COLOR_LAYOUT = {
    "source": "approximated_from_real_dataset_front_view_and_existing_openarm_assets",
    "front_view_structure": "storage_box_above_cube_row_robot_below",
    "layout_name": "real_3color_layout",
    "robot_world_pos": ROBOT_POS,
    "robot_world_rot": ROBOT_ROT,
    "front_camera_world_pos": LEGACY_REALSENSE_POS,
    "front_camera_world_rot": LEGACY_REALSENSE_ROT,
    # New storage pose separated from the legacy single-cube scene.
    # This is an approximation intended to move the box above the 3-cube row in the front view.
    "storage_box_world_pos": (-0.53584, 0.235, 0.04664),
    "cube_slots": {
        # The exact metric coordinates are not available from dataset metadata in this repo.
        # These slots are chosen to preserve the requested topology:
        # storage box above, 3 cubes in a row below, robot approaching from below.
        "left": (-0.61, 0.095, 0.10),
        "center": (-0.535, 0.095, 0.10),
        "right": (-0.46, 0.095, 0.10),
    },
    "default_jitter_xy": (0.0, 0.0),
}
DEFAULT_CUBE_LAYOUT = "fixed_slots"
SETTLE_STEPS = 120
REAL_GRIPPER_CLOSED_RAD = math.radians(-15.0)
REAL_GRIPPER_OPEN_RAD = math.radians(-60.0)
SIM_GRIPPER_CLOSED = 0.0
SIM_GRIPPER_OPEN = 0.044
GRIPPER_POS_SCALE = (REAL_GRIPPER_OPEN_RAD - REAL_GRIPPER_CLOSED_RAD) / (
    SIM_GRIPPER_OPEN - SIM_GRIPPER_CLOSED
)


def get_3color_success_box_bounds() -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    storage_pos = REAL_3COLOR_LAYOUT["storage_box_world_pos"]
    half_x = STORAGE_BOX_SIZE[0] * 0.20 / 2.0
    half_y = STORAGE_BOX_SIZE[1] * 0.40 / 2.0
    x_bounds = (storage_pos[0] - half_x, storage_pos[0] + half_x)
    y_bounds = (storage_pos[1] - half_y, storage_pos[1] + half_y)
    z_bounds = (storage_pos[2] - 0.01, storage_pos[2] + STORAGE_BOX_SIZE[2] / 2.0)
    return x_bounds, y_bounds, z_bounds


def make_cube_cfg(name: str, color: str, default_pos: tuple[float, float, float]) -> RigidObjectCfg:
    return RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{name}",
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
        init_state=RigidObjectCfg.InitialStateCfg(pos=default_pos),
    )


@configclass
class ThreeColorSceneCfg(InteractiveSceneCfg):
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
        init_state=AssetBaseCfg.InitialStateCfg(pos=REAL_3COLOR_LAYOUT["storage_box_world_pos"]),
    )
    realsense = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Realsense",
        spawn=sim_utils.UsdFileCfg(
            usd_path=REALSENSE_USD,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=REAL_3COLOR_LAYOUT["front_camera_world_pos"],
            rot=REAL_3COLOR_LAYOUT["front_camera_world_rot"],
        ),
    )
    front_color = CameraCfg(
        prim_path="{ENV_REGEX_NS}/FrontColorCamera",
        update_period=0.0,
        height=480,
        width=640,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955
        ),
        offset=CameraCfg.OffsetCfg(
            pos=REAL_3COLOR_LAYOUT["front_camera_world_pos"],
            rot=REAL_3COLOR_LAYOUT["front_camera_world_rot"],
            convention="world",
        ),
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
            pos=REAL_3COLOR_LAYOUT["robot_world_pos"],
            rot=REAL_3COLOR_LAYOUT["robot_world_rot"],
            joint_pos=joint_positions_rad(),
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
    cube_red = make_cube_cfg("CubeRed", "red", REAL_3COLOR_LAYOUT["cube_slots"]["left"])
    cube_blue = make_cube_cfg("CubeBlue", "blue", REAL_3COLOR_LAYOUT["cube_slots"]["center"])
    cube_yellow = make_cube_cfg("CubeYellow", "yellow", REAL_3COLOR_LAYOUT["cube_slots"]["right"])


class OpenArmThreeColorEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}
    task = TASK_BY_COLOR[DEFAULT_TARGET_COLOR]
    task_description = task

    def __init__(
        self,
        render_mode: str = "rgb_array",
        max_episode_steps: int = 1000,
        target_color: str = DEFAULT_TARGET_COLOR,
        task: str | None = None,
        cube_layout: str = DEFAULT_CUBE_LAYOUT,
        cube_jitter: float = 0.0,
    ):
        super().__init__()
        if target_color not in COLOR_NAMES:
            raise ValueError(f"Unsupported target_color={target_color!r}")
        if cube_layout != "fixed_slots":
            raise ValueError(f"Unsupported cube_layout={cube_layout!r}")

        self.render_mode = render_mode
        self.max_episode_steps = max_episode_steps
        self.target_color = target_color
        self.task_str = task if task is not None else TASK_BY_COLOR[target_color]
        self.task = self.task_str
        self.task_description = self.task_str
        self.cube_layout = cube_layout
        self.cube_jitter = float(cube_jitter)
        self.current_step = 0
        self.reset_counter = 0
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.rng = np.random.default_rng(15)
        self.first_success_color: str | None = None
        self.last_reset_info: dict[str, object] = {}

        self.action_space = spaces.Box(low=-np.inf, high=np.inf, shape=(ACTION_DIM,), dtype=np.float32)
        self.observation_space = spaces.Dict(
            {
                "observation.state": spaces.Box(low=-np.inf, high=np.inf, shape=(STATE_DIM,), dtype=np.float32),
                "observation.images.front": spaces.Box(
                    low=0, high=255, shape=(480, 640, 3), dtype=np.uint8
                ),
                "observation.images.wrist": spaces.Box(
                    low=0, high=255, shape=(480, 640, 3), dtype=np.uint8
                ),
            }
        )

        self.sim = SimulationContext(sim_utils.SimulationCfg(device=self.device, dt=1.0 / 120.0))
        self.scene = InteractiveScene(ThreeColorSceneCfg(num_envs=1, env_spacing=1.0))
        wrist_root_path = attach_wrist_realsense(self.scene)
        self.sim.reset()

        env_root = self.scene.env_prim_paths[0]
        front_source_path = find_color_camera_path(Sdf.Path(f"{env_root}/Realsense"))
        wrist_source_path = find_color_camera_path(wrist_root_path)
        match_record_camera_to_source(front_source_path, Sdf.Path(f"{env_root}/FrontColorCamera"))

        self.robot: Articulation = self.scene["robot"]
        self.camera_front = self.scene["front_color"]
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

        self.cubes = {
            "red": self.scene["cube_red"],
            "blue": self.scene["cube_blue"],
            "yellow": self.scene["cube_yellow"],
        }
        self._cube_root_names = {
            "red": f"{env_root}/CubeRed",
            "blue": f"{env_root}/CubeBlue",
            "yellow": f"{env_root}/CubeYellow",
        }
        for color, root_path in self._cube_root_names.items():
            apply_cube_color_override(root_path, OBJECT_COLORS[color])

        self._cube_default_quat = {
            color: cube.data.default_root_state[0, 3:7].detach().cpu().numpy().astype(np.float32)
            for color, cube in self.cubes.items()
        }

        arm_names = [f"openarm_left_joint{i}" for i in range(1, 8)]
        self.arm_ids, _ = self.robot.find_joints(arm_names, preserve_order=True)
        gripper_names = ["openarm_left_finger_joint1", "openarm_left_finger_joint2"]
        self.gripper_ids, _ = self.robot.find_joints(gripper_names, preserve_order=True)
        self.record_gripper_id = self.gripper_ids[0]

    def feature_contract(self) -> dict[str, object]:
        return {
            "env_name": "real_3color",
            "layout_name": REAL_3COLOR_LAYOUT["layout_name"],
            "observation_keys": list(OBSERVATION_KEYS),
            "state_dimension": STATE_DIM,
            "action_dimension": ACTION_DIM,
            "robot_type": ROBOT_TYPE,
        }

    def describe_configuration(self) -> dict[str, object]:
        return {
            "task": self.task_str,
            "target_color": self.target_color,
            "cube_colors": list(COLOR_NAMES),
            "cube_size": CUBE_SIZE,
            "env_name": "real_3color",
            "layout_source": REAL_3COLOR_LAYOUT["source"],
            "layout_name": REAL_3COLOR_LAYOUT["layout_name"],
            "front_view_structure": REAL_3COLOR_LAYOUT["front_view_structure"],
            "slot_positions": {name: list(pos) for name, pos in REAL_3COLOR_LAYOUT["cube_slots"].items()},
            "cube_layout": self.cube_layout,
            "cube_jitter": self.cube_jitter,
            "storage_box_position": list(REAL_3COLOR_LAYOUT["storage_box_world_pos"]),
            "target_position": list(REAL_3COLOR_LAYOUT["storage_box_world_pos"]),
            "front_camera_resolution": [480, 640],
            "front_camera_pose": {
                "pos": list(REAL_3COLOR_LAYOUT["front_camera_world_pos"]),
                "rot_wxyz": list(REAL_3COLOR_LAYOUT["front_camera_world_rot"]),
            },
            "robot_base_pose": {
                "pos": list(REAL_3COLOR_LAYOUT["robot_world_pos"]),
                "rot_wxyz": list(REAL_3COLOR_LAYOUT["robot_world_rot"]),
            },
            "observation_keys": list(OBSERVATION_KEYS),
            "state_dimension": STATE_DIM,
            "action_dimension": ACTION_DIM,
        }

    def get_runtime_layout(self) -> dict[str, object]:
        cube_positions = {
            color: self.cubes[color].data.root_pos_w[0].detach().cpu().numpy().astype(np.float32).tolist()
            for color in COLOR_NAMES
        }
        return {
            "layout_name": REAL_3COLOR_LAYOUT["layout_name"],
            "layout_source": REAL_3COLOR_LAYOUT["source"],
            "red": {"slot": self._slot_for_color("red"), "position": cube_positions["red"]},
            "blue": {"slot": self._slot_for_color("blue"), "position": cube_positions["blue"]},
            "yellow": {"slot": self._slot_for_color("yellow"), "position": cube_positions["yellow"]},
            "storage": {"position": list(REAL_3COLOR_LAYOUT["storage_box_world_pos"])},
            "robot_base": {
                "position": list(REAL_3COLOR_LAYOUT["robot_world_pos"]),
                "orientation_wxyz": list(REAL_3COLOR_LAYOUT["robot_world_rot"]),
            },
            "front_camera": {
                "position": list(REAL_3COLOR_LAYOUT["front_camera_world_pos"]),
                "orientation_wxyz": list(REAL_3COLOR_LAYOUT["front_camera_world_rot"]),
            },
        }

    def _get_sim_gripper_state(self) -> tuple[float, float]:
        sim_pos = float(self.robot.data.joint_pos[0, self.record_gripper_id].item())
        sim_vel = float(self.robot.data.joint_vel[0, self.record_gripper_id].item())
        return sim_pos, sim_vel

    def _map_sim_gripper_to_real(self, sim_pos: float, sim_vel: float) -> tuple[float, float]:
        alpha = (sim_pos - SIM_GRIPPER_CLOSED) / (SIM_GRIPPER_OPEN - SIM_GRIPPER_CLOSED)
        alpha = min(max(alpha, 0.0), 1.0)
        real_pos = REAL_GRIPPER_CLOSED_RAD + alpha * (REAL_GRIPPER_OPEN_RAD - REAL_GRIPPER_CLOSED_RAD)
        real_vel = sim_vel * GRIPPER_POS_SCALE
        return real_pos, real_vel

    def _get_joint_torques(self) -> torch.Tensor | None:
        for attr_name in (
            "applied_torque",
            "applied_joint_torque",
            "computed_torque",
            "joint_torque",
        ):
            value = getattr(self.robot.data, attr_name, None)
            if value is not None:
                return value[0]
        return None

    def _build_state_vector(self) -> np.ndarray:
        arm_pos = self.robot.data.joint_pos[0, self.arm_ids]
        arm_vel = self.robot.data.joint_vel[0, self.arm_ids]
        sim_gripper_pos, sim_gripper_vel = self._get_sim_gripper_state()
        real_gripper_pos, real_gripper_vel = self._map_sim_gripper_to_real(sim_gripper_pos, sim_gripper_vel)

        torque_tensor = self._get_joint_torques()
        if torque_tensor is not None:
            arm_torque = torque_tensor[self.arm_ids].detach().cpu().numpy().astype(np.float32)
            gripper_torque = float(torque_tensor[self.record_gripper_id].detach().cpu().item())
        else:
            arm_torque = np.zeros(7, dtype=np.float32)
            gripper_torque = 0.0

        pos_deg = np.rad2deg(
            torch.cat((arm_pos, torch.tensor([real_gripper_pos], dtype=torch.float32, device=self.device)))
            .detach()
            .cpu()
            .numpy()
        ).astype(np.float32)
        vel_deg = np.rad2deg(
            torch.cat((arm_vel, torch.tensor([real_gripper_vel], dtype=torch.float32, device=self.device)))
            .detach()
            .cpu()
            .numpy()
        ).astype(np.float32)
        torque_vec = np.concatenate((arm_torque, np.array([gripper_torque], dtype=np.float32)), dtype=np.float32)

        state = np.empty(STATE_DIM, dtype=np.float32)
        for idx in range(len(MOTOR_NAMES)):
            base = idx * 3
            state[base] = pos_deg[idx]
            state[base + 1] = vel_deg[idx]
            state[base + 2] = torque_vec[idx]
        return state

    def _get_rgb(self, sensor) -> np.ndarray:
        rgb = sensor.data.output.get("rgb")
        if rgb is None:
            return np.zeros((480, 640, 3), dtype=np.uint8)
        if isinstance(rgb, torch.Tensor):
            arr = rgb[0].detach().cpu().numpy() if rgb.ndim == 4 else rgb.detach().cpu().numpy()
        else:
            arr = np.asarray(rgb)
        if arr.ndim == 4:
            arr = arr[0]
        if arr.dtype != np.uint8:
            if arr.size and float(np.nanmax(arr)) <= 1.0:
                arr = arr * 255.0
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        return np.ascontiguousarray(arr)

    def _get_obs(self) -> dict[str, np.ndarray]:
        self.camera_wrist.update(self.sim.get_physics_dt())
        return {
            "observation.state": self._build_state_vector(),
            "observation.images.front": self._get_rgb(self.camera_front),
            "observation.images.wrist": self._get_rgb(self.camera_wrist),
        }

    def _sample_cube_positions(self) -> dict[str, list[float]]:
        assigned_colors = self.rng.permutation(np.array(COLOR_NAMES, dtype=object)).tolist()
        positions: dict[str, list[float]] = {}
        self.slot_assignment = {}
        for slot_name, color in zip(SLOT_NAMES, assigned_colors, strict=True):
            base_x, base_y, base_z = REAL_3COLOR_LAYOUT["cube_slots"][slot_name]
            jitter_x = float(self.rng.uniform(-self.cube_jitter, self.cube_jitter))
            jitter_y = float(self.rng.uniform(-self.cube_jitter, self.cube_jitter))
            positions[color] = [base_x + jitter_x, base_y + jitter_y, base_z]
            self.slot_assignment[slot_name] = color
        return positions

    def _slot_for_color(self, color: str) -> str | None:
        for slot_name, slot_color in self.slot_assignment.items():
            if slot_color == color:
                return slot_name
        return None

    def _reset_robot(self) -> None:
        robot_state = self.robot.data.default_root_state.clone()
        robot_state[:, :3] += self.scene.env_origins
        self.robot.write_root_pose_to_sim(robot_state[:, :7])
        self.robot.write_root_velocity_to_sim(robot_state[:, 7:])
        joint_pos = self.robot.data.default_joint_pos.clone()
        self.robot.write_joint_state_to_sim(joint_pos, torch.zeros_like(self.robot.data.default_joint_vel))
        self.robot.set_joint_position_target(joint_pos)

    def _write_cube_state(self, color: str, position: list[float]) -> None:
        cube = self.cubes[color]
        cube_state = cube.data.default_root_state.clone()
        cube_state[:, 0] = float(position[0])
        cube_state[:, 1] = float(position[1])
        cube_state[:, 2] = float(position[2])
        quat = self._cube_default_quat[color]
        cube_state[:, 3:7] = torch.tensor(quat, dtype=cube_state.dtype, device=cube_state.device)
        cube_state[:, :3] += self.scene.env_origins
        cube.write_root_pose_to_sim(cube_state[:, :7])
        cube.write_root_velocity_to_sim(torch.zeros_like(cube_state[:, 7:]))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.first_success_color = None
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self._reset_robot()
        sampled_positions = self._sample_cube_positions()
        for color, position in sampled_positions.items():
            self._write_cube_state(color, position)

        self.scene.reset()
        for _ in range(SETTLE_STEPS):
            self.scene.write_data_to_sim()
            self.sim.step()
            self.scene.update(self.sim.get_physics_dt())

        initial_positions = {
            color: self.cubes[color].data.root_pos_w[0].detach().cpu().numpy().astype(np.float32).tolist()
            for color in COLOR_NAMES
        }
        self.last_reset_info = {
            "instruction": self.task_str,
            "target_color": self.target_color,
            "slot_assignment": dict(self.slot_assignment),
            "target_slot": self._slot_for_color(self.target_color),
            "red_slot": self._slot_for_color("red"),
            "blue_slot": self._slot_for_color("blue"),
            "yellow_slot": self._slot_for_color("yellow"),
            "red_initial_position": initial_positions["red"],
            "blue_initial_position": initial_positions["blue"],
            "yellow_initial_position": initial_positions["yellow"],
        }
        self.last_reset_info["runtime_layout"] = self.get_runtime_layout()
        self.reset_counter += 1
        return self._get_obs(), self.last_reset_info

    def _cube_success_state(self, color: str) -> tuple[bool, list[float], list[float]]:
        cube = self.cubes[color]
        cube_pos = cube.data.root_pos_w[0].detach().cpu().numpy().astype(np.float32)
        cube_vel = cube.data.root_vel_w[0].detach().cpu().numpy().astype(np.float32)
        (x_min, x_max), (y_min, y_max), (z_min, z_max) = get_3color_success_box_bounds()
        in_box = bool(
            x_min < cube_pos[0] < x_max
            and y_min < cube_pos[1] < y_max
            and z_min < cube_pos[2] < z_max
            and np.linalg.norm(cube_vel[:3]) < 0.05
        )
        return in_box, cube_pos.tolist(), cube_vel[:3].tolist()

    def _apply_action(self, action_deg: np.ndarray) -> None:
        position_triplets = action_deg.reshape(len(MOTOR_NAMES), 3)
        target_pos_deg = position_triplets[:, 0]
        arm_target = torch.tensor(
            np.deg2rad(target_pos_deg[:7]).astype(np.float32), dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        raw_gripper_action = torch.tensor(
            [math.radians(float(target_pos_deg[7]))], dtype=torch.float32, device=self.device
        )
        alpha = (raw_gripper_action - REAL_GRIPPER_CLOSED_RAD) / (REAL_GRIPPER_OPEN_RAD - REAL_GRIPPER_CLOSED_RAD)
        alpha = torch.clamp(alpha, min=0.0, max=1.0)
        sim_gripper_action = SIM_GRIPPER_CLOSED + alpha * (SIM_GRIPPER_OPEN - SIM_GRIPPER_CLOSED)
        joint1_target = sim_gripper_action
        joint2_target = SIM_GRIPPER_OPEN - sim_gripper_action
        gripper_target = torch.stack([joint1_target, joint2_target], dim=-1)

        self.robot.set_joint_position_target(arm_target, joint_ids=self.arm_ids)
        self.robot.set_joint_position_target(gripper_target, joint_ids=self.gripper_ids)
        for _ in range(4):
            self.scene.write_data_to_sim()
            self.sim.step()
            self.scene.update(self.sim.get_physics_dt())

    def step(self, action: np.ndarray):
        action_deg = np.asarray(action, dtype=np.float32)
        if action_deg.shape != (ACTION_DIM,):
            raise ValueError(f"Expected action shape {(ACTION_DIM,)}, got {action_deg.shape}")

        self.current_step += 1
        self._apply_action(action_deg)
        obs = self._get_obs()

        success_hits: list[str] = []
        final_positions: dict[str, list[float]] = {}
        final_velocities: dict[str, list[float]] = {}
        for color in COLOR_NAMES:
            is_success, cube_pos, cube_vel = self._cube_success_state(color)
            final_positions[color] = cube_pos
            final_velocities[color] = cube_vel
            if is_success:
                success_hits.append(color)

        if self.first_success_color is None and success_hits:
            self.first_success_color = success_hits[0]

        task_success = self.first_success_color is not None
        picked_color = self.first_success_color if self.first_success_color is not None else "failure"
        picked_slot = self._slot_for_color(picked_color) if picked_color in COLOR_NAMES else "failure"
        color_correct = bool(task_success and picked_color == self.target_color)
        reward = 1.0 if task_success else 0.0
        terminated = task_success
        truncated = self.current_step >= self.max_episode_steps
        info = {
            "instruction": self.task_str,
            "target_color": self.target_color,
            "target_slot": self._slot_for_color(self.target_color),
            "picked_color": picked_color,
            "picked_slot": picked_slot,
            "task_success": task_success,
            "color_correct": color_correct,
            "termination_reason": f"first_success:{picked_color}" if task_success else "max_steps",
            "simultaneous_success_colors": success_hits,
            "success_bounds": {
                "x": list(get_3color_success_box_bounds()[0]),
                "y": list(get_3color_success_box_bounds()[1]),
                "z": list(get_3color_success_box_bounds()[2]),
            },
            "final_positions": final_positions,
            "final_linear_velocities": final_velocities,
            "slot_assignment": dict(self.slot_assignment),
            "red_slot": self._slot_for_color("red"),
            "blue_slot": self._slot_for_color("blue"),
            "yellow_slot": self._slot_for_color("yellow"),
            "red_initial_position": self.last_reset_info["red_initial_position"],
            "blue_initial_position": self.last_reset_info["blue_initial_position"],
            "yellow_initial_position": self.last_reset_info["yellow_initial_position"],
        }
        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._get_obs()["observation.images.front"]
        return None

    def close(self):
        pass
