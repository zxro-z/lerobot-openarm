"""
Fixed-slot three-cube OpenArm Isaac Lab eval environment for SmolVLA.

This preserves the existing random eval environment and adds a separate,
controlled geometry aligned to the fixed-slot permutation dataset.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import gymnasium as gym
import numpy as np

import openarm_smolvla_env as random_env


TASK = random_env.TASK
TASKS_BY_COLOR = random_env.TASKS_BY_COLOR
ROBOT_TYPE = random_env.ROBOT_TYPE
OBSERVATION_KEYS = random_env.OBSERVATION_KEYS
ACTION_DIM = random_env.ACTION_DIM
COLOR_NAMES = random_env.COLOR_NAMES
OBJECT_COLORS = random_env.OBJECT_COLORS
simulation_app = random_env.simulation_app

FIXED_TCP_TILT_DEG = 50.0
FIXED_STORAGE_BOX_POS = (-0.54, -0.18, 0.04664)
TABLE_SURFACE_Z = -0.0211 + 0.04 / 2.0
CUBE_HALF_EXTENT = random_env.CUBE_SIZE / 2.0
RESTING_CUBE_CENTER_Z = TABLE_SURFACE_Z + CUBE_HALF_EXTENT
SLOT_A = (-0.55, 0.16, RESTING_CUBE_CENTER_Z)
SLOT_B = (-0.55, 0.08, RESTING_CUBE_CENTER_Z)
SLOT_C = (-0.55, 0.00, RESTING_CUBE_CENTER_Z)
SLOT_POSES = {"A": SLOT_A, "B": SLOT_B, "C": SLOT_C}
COLOR_PERMUTATIONS = (
    ("red", "blue", "yellow"),
    ("red", "yellow", "blue"),
    ("blue", "red", "yellow"),
    ("blue", "yellow", "red"),
    ("yellow", "red", "blue"),
    ("yellow", "blue", "red"),
)
LAYOUT_TARGET_ORDER = ("red", "blue", "yellow")


class OpenArmFixedSlotsEnv(random_env.OpenArmEnv):
    """Controlled eval environment aligned with the fixed-slot dataset geometry."""

    _instance = None

    def __init__(
        self,
        obs_type="pixels_agent_pos",
        render_mode="rgb_array",
        max_episode_steps=1000,
        seed: int | None = None,
        task: str | None = None,
        min_steps_before_success: int = random_env.DEFAULT_MIN_STEPS_BEFORE_SUCCESS,
        debug_success: bool = False,
        diagnostic_instrumentation: bool = False,
        dataset_root: str | Path | None = None,
    ):
        self.fixed_dataset_root = Path(dataset_root).expanduser().resolve() if dataset_root is not None else None
        super().__init__(
            obs_type=obs_type,
            render_mode=render_mode,
            max_episode_steps=max_episode_steps,
            seed=seed,
            task=task,
            min_steps_before_success=min_steps_before_success,
            debug_success=debug_success,
            diagnostic_instrumentation=diagnostic_instrumentation,
        )
        if getattr(self, "_fixed_slots_initialized", False):
            return
        self.layout_specs = self._load_layout_specs(self.fixed_dataset_root)
        self.layout_ids = sorted(self.layout_specs)
        self.current_layout: dict[str, object] | None = None
        self._fixed_slots_initialized = True

    @staticmethod
    def _load_layout_specs(dataset_root: Path | None) -> dict[int, dict[str, object]]:
        if dataset_root is None:
            raise ValueError("dataset_root is required for fixed-slot eval.")
        manifest_path = dataset_root / "triplet_manifest.csv"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Fixed-slot triplet manifest not found: {manifest_path}")

        specs: dict[int, dict[str, object]] = {}
        with manifest_path.open(newline="") as file:
            for row in csv.DictReader(file):
                layout_id = int(row["layout_id"])
                spec = specs.setdefault(
                    layout_id,
                    {
                        "layout_id": layout_id,
                        "permutation_id": int(row["permutation_id"]),
                        "repeat_index": int(row["repeat_index"]),
                        "slot_a_color": row["slot_a_color"],
                        "slot_b_color": row["slot_b_color"],
                        "slot_c_color": row["slot_c_color"],
                        "slot_a_xyz": (float(row["slot_a_x"]), float(row["slot_a_y"]), float(row["slot_a_z"])),
                        "slot_b_xyz": (float(row["slot_b_x"]), float(row["slot_b_y"]), float(row["slot_b_z"])),
                        "slot_c_xyz": (float(row["slot_c_x"]), float(row["slot_c_y"]), float(row["slot_c_z"])),
                        "storage_box_xyz": (
                            float(row["storage_box_x"]),
                            float(row["storage_box_y"]),
                            float(row["storage_box_z"]),
                        ),
                        "triplet_order": row.get("triplet_order", ""),
                        "valid_triplet": row.get("valid_triplet", ""),
                    },
                )

        expected_perm_ids = set(range(len(COLOR_PERMUTATIONS)))
        found_perm_ids = {int(spec["permutation_id"]) for spec in specs.values()}
        if found_perm_ids != expected_perm_ids:
            raise ValueError(f"Expected permutation ids {sorted(expected_perm_ids)}, got {sorted(found_perm_ids)}")

        for layout_id, spec in specs.items():
            permutation = (
                str(spec["slot_a_color"]),
                str(spec["slot_b_color"]),
                str(spec["slot_c_color"]),
            )
            if permutation != COLOR_PERMUTATIONS[int(spec["permutation_id"])]:
                raise ValueError(f"Layout {layout_id} permutation mismatch: {permutation}")
            if tuple(spec["slot_a_xyz"]) != SLOT_A:
                raise ValueError(f"Layout {layout_id} slot A mismatch: {spec['slot_a_xyz']}")
            if tuple(spec["slot_b_xyz"]) != SLOT_B:
                raise ValueError(f"Layout {layout_id} slot B mismatch: {spec['slot_b_xyz']}")
            if tuple(spec["slot_c_xyz"]) != SLOT_C:
                raise ValueError(f"Layout {layout_id} slot C mismatch: {spec['slot_c_xyz']}")
            if tuple(spec["storage_box_xyz"]) != FIXED_STORAGE_BOX_POS:
                raise ValueError(f"Layout {layout_id} storage pose mismatch: {spec['storage_box_xyz']}")
            if str(spec["triplet_order"]).replace(" ", "") != ",".join(LAYOUT_TARGET_ORDER):
                raise ValueError(f"Layout {layout_id} unexpected triplet_order: {spec['triplet_order']}")
            if str(spec["valid_triplet"]).lower() != "true":
                raise ValueError(f"Layout {layout_id} is not marked valid_triplet=true")
            del spec["triplet_order"]
            del spec["valid_triplet"]
        return specs

    def describe_configuration(self) -> dict[str, object]:
        cfg = super().describe_configuration()
        cfg.update(
            {
                "eval_geometry_mode": "fixed_slots",
                "fixed_tcp_tilt_deg": FIXED_TCP_TILT_DEG,
                "slot_A_xyz": list(SLOT_A),
                "slot_B_xyz": list(SLOT_B),
                "slot_C_xyz": list(SLOT_C),
                "storage_box_position": list(FIXED_STORAGE_BOX_POS),
                "color_permutations": [list(permutation) for permutation in COLOR_PERMUTATIONS],
                "paired_unit": "permutation_id + repeat_index",
                "layout_count": len(self.layout_specs),
                "dataset_root": str(self.fixed_dataset_root),
            }
        )
        return cfg

    def _select_layout(self, *, layout_id: int | None, seed: int | None) -> dict[str, object]:
        if layout_id is not None:
            if layout_id not in self.layout_specs:
                raise KeyError(f"Unknown layout_id={layout_id}")
            return self.layout_specs[layout_id]
        if seed is None:
            return self.layout_specs[self.layout_ids[0]]
        return self.layout_specs[self.layout_ids[seed % len(self.layout_ids)]]

    @staticmethod
    def _slot_for_color(layout: dict[str, object], color: str) -> str:
        mapping = {
            str(layout["slot_a_color"]): "A",
            str(layout["slot_b_color"]): "B",
            str(layout["slot_c_color"]): "C",
        }
        return mapping[color]

    def _pose_for_color(self, layout: dict[str, object], color: str) -> tuple[float, float, float]:
        return SLOT_POSES[self._slot_for_color(layout, color)]

    def _success_bounds(self) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
        outer_half_x = random_env.STORAGE_BOX_SIZE[0] / 2.0
        outer_half_y = random_env.STORAGE_BOX_SIZE[1] / 2.0
        inner_half_x = max(outer_half_x - random_env.SUCCESS_INNER_MARGIN_X, random_env.CUBE_SIZE / 2.0 + 0.02)
        inner_half_y = max(outer_half_y - random_env.SUCCESS_INNER_MARGIN_Y, random_env.CUBE_SIZE / 2.0 + 0.02)
        x_bounds = (FIXED_STORAGE_BOX_POS[0] - inner_half_x, FIXED_STORAGE_BOX_POS[0] + inner_half_x)
        y_bounds = (FIXED_STORAGE_BOX_POS[1] - inner_half_y, FIXED_STORAGE_BOX_POS[1] + inner_half_y)
        z_bounds = (
            FIXED_STORAGE_BOX_POS[2] + random_env.SUCCESS_Z_MARGIN_BOTTOM,
            FIXED_STORAGE_BOX_POS[2] + random_env.STORAGE_BOX_SIZE[2] / 2.0 - random_env.SUCCESS_Z_MARGIN_TOP,
        )
        return x_bounds, y_bounds, z_bounds

    def _cube_success(self, cube) -> tuple[bool, dict[str, object]]:
        position = cube.data.root_pos_w[0].detach().cpu().numpy()
        velocity = cube.data.root_vel_w[0].detach().cpu().numpy()
        (x_min, x_max), (y_min, y_max), (z_min, z_max) = self._success_bounds()
        x_inside = x_min < float(position[0]) < x_max
        y_inside = y_min < float(position[1]) < y_max
        z_inside = z_min < float(position[2]) < z_max
        in_box = x_inside and y_inside and z_inside
        speed = float(np.linalg.norm(velocity[:3]))
        stationary = speed < random_env.SUCCESS_MAX_LINEAR_SPEED
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

    def reset(self, seed=None, options=None):
        gym.Env.reset(self, seed=seed)
        self.current_step = 0
        requested_layout_id = None if options is None else options.get("layout_id")
        self.current_layout = self._select_layout(layout_id=requested_layout_id, seed=seed)

        robot_state = self.robot.data.default_root_state.clone()
        robot_state[:, :3] += self.scene.env_origins
        self.robot.write_root_pose_to_sim(robot_state[:, :7])
        self.robot.write_root_velocity_to_sim(robot_state[:, 7:])
        joint_pos = self.robot.data.default_joint_pos.clone()
        self.robot.write_joint_state_to_sim(joint_pos, random_env.torch.zeros_like(self.robot.data.default_joint_vel))
        self.robot.set_joint_position_target(joint_pos)

        self.initial_positions = {}
        for color, cube in self.cubes.items():
            cube_state = cube.data.default_root_state.clone()
            requested_position = np.asarray(self._pose_for_color(self.current_layout, color), dtype=np.float32)
            cube_state[:, :3] = random_env.torch.tensor(
                requested_position,
                dtype=cube_state.dtype,
                device=cube_state.device,
            )
            cube_state[:, 3:7] = random_env.torch.tensor(
                self._cube_default_quat[color],
                dtype=cube_state.dtype,
                device=cube_state.device,
            )
            cube_state[:, :3] += self.scene.env_origins
            cube.write_root_pose_to_sim(cube_state[:, :7])
            cube.write_root_velocity_to_sim(random_env.torch.zeros_like(cube_state[:, 7:]))

        self.scene.reset()
        for _ in range(120):
            self.scene.write_data_to_sim()
            self.sim.step()
            self.scene.update(self.sim.get_physics_dt())

        info: dict[str, object] = {
            "is_success": False,
            "task": self.task_str,
            "task_description": self.task_str,
            "layout_id": int(self.current_layout["layout_id"]),
            "permutation_id": int(self.current_layout["permutation_id"]),
            "repeat_index": int(self.current_layout["repeat_index"]),
            "slot_a_color": self.current_layout["slot_a_color"],
            "slot_b_color": self.current_layout["slot_b_color"],
            "slot_c_color": self.current_layout["slot_c_color"],
        }
        for color, cube in self.cubes.items():
            settled_position = cube.data.root_pos_w[0].detach().cpu().numpy().astype(np.float32)
            self.initial_positions[color] = settled_position.tolist()
            info[f"{color}_initial_x"] = float(settled_position[0])
            info[f"{color}_initial_y"] = float(settled_position[1])
            info[f"{color}_initial_z"] = float(settled_position[2])
        self.reset_counter += 1
        return self._get_obs(), info

    def _build_diagnostic_state(self, action_deg: np.ndarray) -> dict[str, object]:
        state = super()._build_diagnostic_state(action_deg)
        state["storage_box_position"] = list(FIXED_STORAGE_BOX_POS)
        if self.current_layout is not None:
            state["layout_id"] = int(self.current_layout["layout_id"])
            state["permutation_id"] = int(self.current_layout["permutation_id"])
            state["repeat_index"] = int(self.current_layout["repeat_index"])
            state["slot_a_color"] = str(self.current_layout["slot_a_color"])
            state["slot_b_color"] = str(self.current_layout["slot_b_color"])
            state["slot_c_color"] = str(self.current_layout["slot_c_color"])
        return state

    def step(self, action: np.ndarray):
        action_deg = np.asarray(action, dtype=np.float32)
        if action_deg.shape != (8,):
            raise ValueError(f"Expected action shape (8,), got {action_deg.shape}")
        action_rad = np.deg2rad(action_deg).astype(np.float32)
        self.current_step += 1
        action_tensor = random_env.torch.tensor(
            action_rad,
            dtype=random_env.torch.float32,
            device=self.device,
        ).unsqueeze(0)
        real_gripper_closed = math.radians(-15.0)
        real_gripper_open = math.radians(-60.0)
        sim_gripper_closed = 0.0
        sim_gripper_open = 0.044
        raw_gripper_action = action_tensor[:, 7]
        alpha = (raw_gripper_action - real_gripper_closed) / (real_gripper_open - real_gripper_closed)
        alpha = random_env.torch.clamp(alpha, min=0.0, max=1.0)
        sim_gripper_action = sim_gripper_closed + alpha * (sim_gripper_open - sim_gripper_closed)
        arm_target = action_tensor[:, :7]
        joint1_target = sim_gripper_action
        joint2_target = sim_gripper_open - sim_gripper_action
        gripper_target = random_env.torch.stack([joint1_target, joint2_target], dim=-1)
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
            success, debug = self._cube_success(cube)
            if self.current_step < self.min_steps_before_success:
                success = False
            success_by_color[color] = success
            cube_debug[color] = debug
            if success:
                picked_candidates.append(color)

        if len(picked_candidates) == 1:
            picked_color = picked_candidates[0]
            picked_slot = self._slot_for_color(self.current_layout, picked_color)
            termination_reason = "success"
        elif len(picked_candidates) > 1:
            picked_color = "ambiguous"
            picked_slot = "ambiguous"
            termination_reason = "ambiguous_multiple_cubes_in_box"
        else:
            picked_color = "failure"
            picked_slot = "failure"
            termination_reason = "max_steps" if self.current_step >= self.max_episode_steps else "running"

        is_success = len(picked_candidates) >= 1
        terminated = is_success
        truncated = self.current_step >= self.max_episode_steps
        info: dict[str, object] = {
            "is_success": is_success,
            "picked_color": picked_color,
            "picked_slot": picked_slot,
            "task": self.task_str,
            "task_description": self.task_str,
            "termination_reason": termination_reason,
            "success_by_color": success_by_color,
            "layout_id": int(self.current_layout["layout_id"]),
            "permutation_id": int(self.current_layout["permutation_id"]),
            "repeat_index": int(self.current_layout["repeat_index"]),
            "slot_a_color": self.current_layout["slot_a_color"],
            "slot_b_color": self.current_layout["slot_b_color"],
            "slot_c_color": self.current_layout["slot_c_color"],
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
        return obs, float(is_success), terminated, truncated, info
