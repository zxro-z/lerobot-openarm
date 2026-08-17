"""Generate a same-layout red/blue/yellow counterfactual triplet dataset.

This launcher preserves the existing three-color task semantics while changing
the data contract from:

    episode -> sample layout -> target-specific rollout

to:

    layout_id -> sample one layout once -> red/blue/yellow rollouts on the
    exact same initial scene geometry

Each layout is first recorded into its own temporary 3-episode staging dataset.
Only successful complete triplets are aggregated into the final dataset root.
This gives triplet-level atomicity without buffering RGB videos in RAM.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import shutil
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def _parse_counterfactual_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--num_layouts",
        type=int,
        default=50,
        help="Number of unique layouts; the final dataset stores 3 episodes per layout.",
    )
    parser.add_argument(
        "--layout_base_seed",
        type=int,
        default=1000,
        help="Deterministic base seed used to derive one layout seed per layout_id.",
    )
    parser.add_argument(
        "--cube_min_separation",
        type=float,
        default=0.09,
        help="Minimum XY center distance in metres between any two cubes.",
    )
    parser.add_argument(
        "--cube_sampling_max_attempts",
        type=int,
        default=1000,
        help="Maximum attempts per cube position before resampling the whole layout.",
    )
    parser.add_argument(
        "--layout_x_range",
        type=float,
        nargs=2,
        metavar=("MIN", "MAX"),
        default=(-0.60, -0.50),
        help="Default shared-layout cube x sampling range for counterfactual generation.",
    )
    parser.add_argument(
        "--layout_y_range",
        type=float,
        nargs=2,
        metavar=("MIN", "MAX"),
        default=(0.02, 0.15),
        help="Default shared-layout cube y sampling range for counterfactual generation.",
    )
    parser.add_argument(
        "--layout_tilt_deg_range",
        type=float,
        nargs=2,
        metavar=("MIN", "MAX"),
        default=(0.0, 45.0),
        help="Default shared TCP tilt range in degrees for each counterfactual layout.",
    )
    parser.add_argument(
        "--success_hold_time_s",
        type=float,
        default=0.25,
        help="Time the target must continuously satisfy the success criterion.",
    )
    parser.add_argument(
        "--success_timeout_s",
        type=float,
        default=2.0,
        help="Post-trajectory time allowed for the target cube to settle in the box.",
    )
    parser.add_argument(
        "--success_max_linear_speed",
        type=float,
        default=0.05,
        help="Maximum target-cube linear speed in m/s for success.",
    )
    parser.add_argument(
        "--episode_timeout_s",
        type=float,
        default=30.0,
        help="Discard and retry if any controller attempt exceeds this simulation time.",
    )
    parser.add_argument(
        "--max_triplet_attempts",
        type=int,
        default=100,
        help="Maximum full-layout retries before aborting a single layout_id.",
    )
    parser.add_argument(
        "--staging_root",
        type=str,
        default=None,
        help="Optional root for per-layout staging datasets. Defaults next to the final dataset root.",
    )
    parser.add_argument(
        "--keep_staging",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Keep successful per-layout staging datasets after final aggregation.",
    )
    parser.add_argument(
        "--layout_manifest_name",
        type=str,
        default="layout_manifest.csv",
    )
    parser.add_argument(
        "--triplet_manifest_name",
        type=str,
        default="triplet_manifest.csv",
    )
    parser.add_argument(
        "--workspace_audit_csv",
        type=str,
        default=None,
        help="Optional append-only CSV path for per-target grasp workspace audit rows.",
    )
    parser.add_argument(
        "--show_viewports",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Opt in/out of preview viewport windows. Defaults to disabled in workspace audit mode.",
    )
    return parser.parse_known_args()


launcher_args, wrapped_argv = _parse_counterfactual_args()
if launcher_args.num_layouts < 1:
    raise SystemExit("--num_layouts must be at least 1")
if launcher_args.cube_min_separation < math.sqrt(2.0) * 0.05:
    raise SystemExit("--cube_min_separation must be at least sqrt(2) * cube size (0.0708 m)")
if launcher_args.cube_sampling_max_attempts < 1:
    raise SystemExit("--cube_sampling_max_attempts must be at least 1")
if launcher_args.max_triplet_attempts < 1:
    raise SystemExit("--max_triplet_attempts must be at least 1")
if launcher_args.success_hold_time_s <= 0.0 or launcher_args.success_timeout_s <= 0.0:
    raise SystemExit("success hold/timeout values must be greater than zero")
if launcher_args.episode_timeout_s <= 0.0:
    raise SystemExit("--episode_timeout_s must be greater than zero")
if launcher_args.layout_x_range[0] > launcher_args.layout_x_range[1]:
    raise SystemExit("--layout_x_range requires MIN <= MAX")
if launcher_args.layout_y_range[0] > launcher_args.layout_y_range[1]:
    raise SystemExit("--layout_y_range requires MIN <= MAX")
if launcher_args.layout_tilt_deg_range[0] > launcher_args.layout_tilt_deg_range[1]:
    raise SystemExit("--layout_tilt_deg_range requires MIN <= MAX")

sys.argv = [sys.argv[0], *wrapped_argv]
import openarm_table_dual_realsense_ik_pick_place_make_dataset_random_cube_random_tilt_gripper_mapped_degree as degree  # noqa: E402
from lerobot.datasets.aggregate import aggregate_datasets  # noqa: E402


base = degree.base
torch = base.torch
tilt_launcher = degree.mapped.launcher_args
original_controller_init = degree.mapped._original_controller_init
original_fail = base.PickPlaceController.fail
original_enter = base.PickPlaceController.enter

LEROBOT_SRC_ROOT = Path(__file__).resolve().parents[1]
ASSET_DIR = LEROBOT_SRC_ROOT / "assets" / "openarm_use"
base.ASSET_DIR = ASSET_DIR
base.TABLE_USD = str(ASSET_DIR / "table.usd")
base.ROBOT_USD = str(ASSET_DIR / "openarm_half_tesollo_tactile.usd")
base.REALSENSE_USD = str(ASSET_DIR / "realsense.usd")
base.STORAGE_BOX_USD = str(ASSET_DIR / "storage_box.usd")

COLORS = ("red", "blue", "yellow")
CUBE_ASSET_NAMES = {"red": "cube", "blue": "blue_cube", "yellow": "yellow_cube"}
CUBE_RGB = {
    "red": (0.85, 0.05, 0.03),
    "blue": (0.03, 0.15, 0.90),
    "yellow": (0.95, 0.80, 0.03),
}
DEFAULT_X_RANGE = tuple(float(v) for v in launcher_args.layout_x_range)
DEFAULT_Y_RANGE = tuple(float(v) for v in launcher_args.layout_y_range)
DEFAULT_TILT_DEG_RANGE = tuple(float(v) for v in launcher_args.layout_tilt_deg_range)
SUCCESS_X_HALF_EXTENT = 0.20 - base.CUBE_SIZE / 2.0
SUCCESS_Y_HALF_EXTENT = 0.25 - base.CUBE_SIZE / 2.0
SUCCESS_Z_RANGE = (0.04, 0.12)
TARGET_SLOT_BY_COLOR = {"red": 0, "blue": 1, "yellow": 2}

ACTIVE_COLOR = "red"
ACTIVE_LAYOUT = None
ACTIVE_CANDIDATE_ATTEMPT = None
FINAL_DATASET_ROOT = Path(base.args_cli.dataset_root).expanduser().resolve()
FINAL_REPO_ID = base.args_cli.dataset_repo_id
FINAL_TOTAL_EPISODES = launcher_args.num_layouts * len(COLORS)
WORKSPACE_AUDIT_LOGGER = None
WORKSPACE_AUDIT_MODE = launcher_args.workspace_audit_csv is not None

WORKSPACE_AUDIT_FIELDNAMES = [
    "layout_id",
    "candidate_attempt",
    "layout_seed",
    "target_color",
    "cube_x",
    "cube_y",
    "cube_z",
    "tcp_tilt_deg",
    "grasp_success",
    "grasp_target_base_x",
    "grasp_target_base_y",
    "grasp_target_base_z",
    "current_ee_base_x",
    "current_ee_base_y",
    "current_ee_base_z",
    "xyz_error_x",
    "xyz_error_y",
    "xyz_error_z",
    "xyz_error_norm",
    "orientation_error_norm",
    "position_tolerance",
    "rotation_tolerance",
    "failure_reason",
    "descend_state_time_s",
    "trajectory_finished_expected",
]


@dataclass(frozen=True)
class CubePose:
    x: float
    y: float
    z: float
    qw: float
    qx: float
    qy: float
    qz: float


@dataclass(frozen=True)
class LayoutSpec:
    layout_id: int
    layout_seed: int
    tcp_tilt_deg: float
    red_pose: CubePose
    blue_pose: CubePose
    yellow_pose: CubePose
    robot_initial_state_deg: tuple[float, ...]


class WorkspaceAuditLogger:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        file_exists = self.path.exists()
        self._file = self.path.open("a", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=WORKSPACE_AUDIT_FIELDNAMES)
        if not file_exists or self.path.stat().st_size == 0:
            self._writer.writeheader()
            self._file.flush()

    def append_row(self, row: dict[str, object]) -> None:
        self._writer.writerow(row)
        self._file.flush()

    def close(self) -> None:
        self._file.close()


def _cube_cfg(color: str):
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
            collision_props=base.sim_utils.CollisionPropertiesCfg(contact_offset=0.002, rest_offset=0.0),
            physics_material=base.sim_utils.RigidBodyMaterialCfg(
                static_friction=1.0,
                dynamic_friction=0.8,
                restitution=0.0,
            ),
            visual_material=base.sim_utils.PreviewSurfaceCfg(diffuse_color=CUBE_RGB[color]),
        ),
        init_state=base.RigidObjectCfg.InitialStateCfg(pos=base.CUBE_POS),
    )


@base.configclass
class ThreeColorCounterfactualSceneCfg(base.DualRealSensePickPlaceSceneCfg):
    blue_cube = _cube_cfg("blue")
    yellow_cube = _cube_cfg("yellow")


def cube_pose_from_components(position: tuple[float, float, float], quat_wxyz: tuple[float, float, float, float]) -> CubePose:
    return CubePose(
        x=float(position[0]),
        y=float(position[1]),
        z=float(position[2]),
        qw=float(quat_wxyz[0]),
        qx=float(quat_wxyz[1]),
        qy=float(quat_wxyz[2]),
        qz=float(quat_wxyz[3]),
    )


def sample_layout(scene, layout_id: int, candidate_index: int = 0) -> LayoutSpec:
    # Retry attempts must consume a new deterministic candidate seed; otherwise
    # an IK-infeasible layout will repeat forever for the same layout_id.
    layout_seed = (
        launcher_args.layout_base_seed
        + layout_id * launcher_args.max_triplet_attempts
        + candidate_index
    )
    rng = np.random.default_rng(layout_seed)
    x_limits = base.args_cli.cube_x_range or DEFAULT_X_RANGE
    y_limits = base.args_cli.cube_y_range or DEFAULT_Y_RANGE
    default_quat = tuple(
        float(value)
        for value in scene[CUBE_ASSET_NAMES["red"]].data.default_root_state[0, 3:7].detach().cpu().tolist()
    )

    while True:
        sampled_positions: list[tuple[float, float, float]] = []
        valid = True
        for _ in range(len(COLORS)):
            for _ in range(launcher_args.cube_sampling_max_attempts):
                candidate = (
                    float(rng.uniform(*x_limits)),
                    float(rng.uniform(*y_limits)),
                    float(base.args_cli.cube_z),
                )
                if all(
                    math.hypot(candidate[0] - other[0], candidate[1] - other[1]) >= launcher_args.cube_min_separation
                    for other in sampled_positions
                ):
                    sampled_positions.append(candidate)
                    break
            else:
                valid = False
                break
        if valid:
            break

    shuffled_colors = list(COLORS)
    rng.shuffle(shuffled_colors)
    positions_by_color = dict(zip(shuffled_colors, sampled_positions, strict=True))
    tilt_deg = float(rng.uniform(DEFAULT_TILT_DEG_RANGE[0], DEFAULT_TILT_DEG_RANGE[1]))
    robot_initial_state = tuple(float(v) for v in np.rad2deg(scene["robot"].data.default_joint_pos[0].detach().cpu().numpy()))

    return LayoutSpec(
        layout_id=layout_id,
        layout_seed=layout_seed,
        tcp_tilt_deg=tilt_deg,
        red_pose=cube_pose_from_components(positions_by_color["red"], default_quat),
        blue_pose=cube_pose_from_components(positions_by_color["blue"], default_quat),
        yellow_pose=cube_pose_from_components(positions_by_color["yellow"], default_quat),
        robot_initial_state_deg=robot_initial_state,
    )


def pose_for_color(layout: LayoutSpec, color: str) -> CubePose:
    return {"red": layout.red_pose, "blue": layout.blue_pose, "yellow": layout.yellow_pose}[color]


def reset_scene_to_layout(scene, layout: LayoutSpec) -> None:
    robot = scene["robot"]
    robot_state = robot.data.default_root_state.clone()
    robot_state[:, :3] += scene.env_origins
    robot.write_root_pose_to_sim(robot_state[:, :7])
    robot.write_root_velocity_to_sim(robot_state[:, 7:])
    joint_pos = robot.data.default_joint_pos.clone()
    robot.write_joint_state_to_sim(joint_pos, torch.zeros_like(robot.data.default_joint_vel))
    robot.set_joint_position_target(joint_pos)

    for color, asset_name in CUBE_ASSET_NAMES.items():
        cube = scene[asset_name]
        pose = pose_for_color(layout, color)
        cube_state = cube.data.default_root_state.clone()
        cube_state[:, 0] = pose.x
        cube_state[:, 1] = pose.y
        cube_state[:, 2] = pose.z
        cube_state[:, 3] = pose.qw
        cube_state[:, 4] = pose.qx
        cube_state[:, 5] = pose.qy
        cube_state[:, 6] = pose.qz
        cube_state[:, :3] += scene.env_origins
        cube.write_root_pose_to_sim(cube_state[:, :7])
        cube.write_root_velocity_to_sim(torch.zeros_like(cube_state[:, 7:]))
    scene.reset()


def target_cube_success(cube) -> tuple[bool, str]:
    position = cube.data.root_pos_w[0].detach().cpu().numpy()
    velocity = cube.data.root_vel_w[0].detach().cpu().numpy()
    dx = abs(float(position[0]) - base.STORAGE_BOX_POS[0])
    dy = abs(float(position[1]) - base.STORAGE_BOX_POS[1])
    in_box = (
        dx < SUCCESS_X_HALF_EXTENT
        and dy < SUCCESS_Y_HALF_EXTENT
        and SUCCESS_Z_RANGE[0] < float(position[2]) < SUCCESS_Z_RANGE[1]
    )
    speed = float(np.linalg.norm(velocity[:3]))
    stationary = speed < launcher_args.success_max_linear_speed
    detail = f"xyz={position.tolist()}, speed={speed:.4f}, in_box={in_box}, stationary={stationary}"
    return bool(in_box and stationary), detail


def task_for_color(color: str) -> str:
    return f"Pick up the {color} cube and place it in the storage box."


def collect_grasp_pose_diagnostics(self) -> dict[str, object]:
    assert self.goal_pos_b is not None
    ee_pos_b, ee_quat_b = self.ee_pose_b()
    pos_error, rot_error = base.compute_pose_error(
        ee_pos_b,
        ee_quat_b,
        self.goal_pos_b.unsqueeze(0),
        self.grasp_quat_b,
        rot_error_type="axis_angle",
    )
    pos_norm = torch.linalg.norm(pos_error[0]).item()
    rot_norm = torch.linalg.norm(rot_error[0]).item()
    return {
        "target_color": getattr(self, "target_color", "unknown"),
        "target_cube_world_xyz": self.cube.data.root_pos_w[0].detach().cpu().tolist(),
        "grasp_target_base_xyz": self.goal_pos_b.detach().cpu().tolist(),
        "current_ee_base_xyz": ee_pos_b[0].detach().cpu().tolist(),
        "current_ee_base_quat_wxyz": ee_quat_b[0].detach().cpu().tolist(),
        "target_base_quat_wxyz": self.grasp_quat_b[0].detach().cpu().tolist(),
        "xyz_error_base": pos_error[0].detach().cpu().tolist(),
        "xyz_error_norm": pos_norm,
        "orientation_error_axis_angle": rot_error[0].detach().cpu().tolist(),
        "orientation_error_norm": rot_norm,
        "descend_state_time_s": float(self.state_time),
        "trajectory_finished_expected": bool(self.state_time >= base.args_cli.short_move_time_s),
        "grasp_position_tolerance": float(base.args_cli.grasp_position_tolerance),
        "grasp_rotation_tolerance": float(base.args_cli.grasp_rotation_tolerance),
        "ik_failure_timeout_s": float(base.args_cli.ik_failure_timeout_s),
        "short_move_time_s": float(base.args_cli.short_move_time_s),
    }


def maybe_append_workspace_audit_row(
    controller,
    *,
    grasp_success: bool,
    failure_reason: str,
    diagnostics: dict[str, object] | None = None,
) -> None:
    global WORKSPACE_AUDIT_LOGGER
    if WORKSPACE_AUDIT_LOGGER is None or ACTIVE_LAYOUT is None or ACTIVE_CANDIDATE_ATTEMPT is None:
        return
    if getattr(controller, "_workspace_audit_logged", False):
        return
    if diagnostics is None:
        diagnostics = collect_grasp_pose_diagnostics(controller)

    cube_xyz = diagnostics["target_cube_world_xyz"]
    grasp_target_xyz = diagnostics["grasp_target_base_xyz"]
    current_ee_xyz = diagnostics["current_ee_base_xyz"]
    xyz_error = diagnostics["xyz_error_base"]
    row = {
        "layout_id": ACTIVE_LAYOUT.layout_id,
        "candidate_attempt": ACTIVE_CANDIDATE_ATTEMPT,
        "layout_seed": ACTIVE_LAYOUT.layout_seed,
        "target_color": diagnostics["target_color"],
        "cube_x": float(cube_xyz[0]),
        "cube_y": float(cube_xyz[1]),
        "cube_z": float(cube_xyz[2]),
        "tcp_tilt_deg": float(ACTIVE_LAYOUT.tcp_tilt_deg),
        "grasp_success": int(grasp_success),
        "grasp_target_base_x": float(grasp_target_xyz[0]),
        "grasp_target_base_y": float(grasp_target_xyz[1]),
        "grasp_target_base_z": float(grasp_target_xyz[2]),
        "current_ee_base_x": float(current_ee_xyz[0]),
        "current_ee_base_y": float(current_ee_xyz[1]),
        "current_ee_base_z": float(current_ee_xyz[2]),
        "xyz_error_x": float(xyz_error[0]),
        "xyz_error_y": float(xyz_error[1]),
        "xyz_error_z": float(xyz_error[2]),
        "xyz_error_norm": float(diagnostics["xyz_error_norm"]),
        "orientation_error_norm": float(diagnostics["orientation_error_norm"]),
        "position_tolerance": float(diagnostics["grasp_position_tolerance"]),
        "rotation_tolerance": float(diagnostics["grasp_rotation_tolerance"]),
        "failure_reason": failure_reason,
        "descend_state_time_s": float(diagnostics["descend_state_time_s"]),
        "trajectory_finished_expected": int(bool(diagnostics["trajectory_finished_expected"])),
    }
    WORKSPACE_AUDIT_LOGGER.append_row(row)
    controller._workspace_audit_logged = True
    print(
        f"[WORKSPACE_AUDIT] seed={ACTIVE_LAYOUT.layout_seed} target={row['target_color']} "
        f"cube=({row['cube_x']:.4f},{row['cube_y']:.4f}) tilt={row['tcp_tilt_deg']:.2f} "
        f"success={row['grasp_success']} xyz_err={row['xyz_error_norm']:.5f} "
        f"rot_err={row['orientation_error_norm']:.5f}"
    )


def _counterfactual_controller_init(self, robot, scene) -> None:
    if ACTIVE_LAYOUT is None:
        raise RuntimeError("ACTIVE_LAYOUT must be set before creating the controller.")
    base.args_cli.tilt_deg = ACTIVE_LAYOUT.tcp_tilt_deg
    print(
        f"[EPISODE] Layout {ACTIVE_LAYOUT.layout_id:03d} target={ACTIVE_COLOR} "
        f"tcp_tilt={ACTIVE_LAYOUT.tcp_tilt_deg:.2f} deg"
    )
    original_controller_init(self, robot, scene)
    self.cube = scene[CUBE_ASSET_NAMES[ACTIVE_COLOR]]
    self.target_color = ACTIVE_COLOR
    self._last_grasp_debug = None
    self._last_descend_trajectory_finished = False
    self._workspace_audit_logged = False


base.PickPlaceController.__init__ = _counterfactual_controller_init


def _debug_grasp_pose_reached(self) -> bool:
    self._last_grasp_debug = collect_grasp_pose_diagnostics(self)
    return (
        float(self._last_grasp_debug["xyz_error_norm"]) <= base.args_cli.grasp_position_tolerance
        and float(self._last_grasp_debug["orientation_error_norm"]) <= base.args_cli.grasp_rotation_tolerance
    )


def _audit_enter(self, state: str, goal=None) -> None:
    if state == "close_gripper" and self.state == "descend_to_grasp":
        diagnostics = getattr(self, "_last_grasp_debug", None)
        maybe_append_workspace_audit_row(
            self,
            grasp_success=True,
            failure_reason="",
            diagnostics=diagnostics,
        )
    original_enter(self, state, goal)


def _debug_fail(self, reason: str) -> None:
    if reason.startswith("grasp IK target was not reached"):
        debug = getattr(self, "_last_grasp_debug", None)
        if debug is not None:
            print(f"[DIAG][GRASP_IK] {debug}")
        maybe_append_workspace_audit_row(
            self,
            grasp_success=False,
            failure_reason=reason,
            diagnostics=debug,
        )
    original_fail(self, reason)


base.PickPlaceController.grasp_pose_reached = _debug_grasp_pose_reached
base.PickPlaceController.enter = _audit_enter
base.PickPlaceController.fail = _debug_fail


@contextmanager
def temporary_dataset_args(*, dataset_root: Path, dataset_repo_id: str, num_episodes: int):
    old_root = base.args_cli.dataset_root
    old_repo_id = base.args_cli.dataset_repo_id
    old_num_episodes = base.args_cli.num_episodes
    base.args_cli.dataset_root = str(dataset_root)
    base.args_cli.dataset_repo_id = dataset_repo_id
    base.args_cli.num_episodes = num_episodes
    try:
        yield
    finally:
        base.args_cli.dataset_root = old_root
        base.args_cli.dataset_repo_id = old_repo_id
        base.args_cli.num_episodes = old_num_episodes


def stage_root_for_layout(staging_root: Path, layout_id: int) -> Path:
    return staging_root / f"layout_{layout_id:03d}"


def ensure_clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def remove_dir_if_exists(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)


def stage_repo_id(layout_id: int) -> str:
    return f"{FINAL_REPO_ID}_layout_{layout_id:03d}"


def close_preview_viewports(viewports: tuple[object, object] | None) -> None:
    if viewports is None:
        return
    for viewport in viewports:
        if viewport is None:
            continue
        if hasattr(viewport, "visible"):
            viewport.visible = False
        if hasattr(viewport, "destroy"):
            viewport.destroy()


def audit_mode_hard_exit() -> None:
    # Audit-only workaround: Isaac Kit teardown may hang inside simulation_app.close()
    # after all workspace audit rows and staging cleanup have already completed.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


def record_layout_triplet_v2(
    sim,
    scene,
    layout: LayoutSpec,
    staging_root: Path,
    *,
    candidate_attempt: int,
) -> tuple[bool, str, Path]:
    global ACTIVE_COLOR, ACTIVE_LAYOUT, ACTIVE_CANDIDATE_ATTEMPT
    dt = sim.get_physics_dt()
    layout_stage_root = stage_root_for_layout(staging_root, layout.layout_id)
    remove_dir_if_exists(layout_stage_root)
    repo_id = stage_repo_id(layout.layout_id)
    recorder = None
    try:
        ACTIVE_CANDIDATE_ATTEMPT = candidate_attempt
        with temporary_dataset_args(dataset_root=layout_stage_root, dataset_repo_id=repo_id, num_episodes=3):
            for target_idx, target_color in enumerate(COLORS):
                global ACTIVE_COLOR, ACTIVE_LAYOUT
                ACTIVE_COLOR = target_color
                ACTIVE_LAYOUT = layout
                base.args_cli.task = task_for_color(target_color)
                reset_scene_to_layout(scene, layout)
                print(
                    f"[LAYOUT] layout_id={layout.layout_id:03d}, seed={layout.layout_seed}, "
                    f"target={target_color}, positions={{'red': ({layout.red_pose.x:.4f}, {layout.red_pose.y:.4f}), "
                    f"'blue': ({layout.blue_pose.x:.4f}, {layout.blue_pose.y:.4f}), "
                    f"'yellow': ({layout.yellow_pose.x:.4f}, {layout.yellow_pose.y:.4f})}}"
                )
                scene.write_data_to_sim()
                sim.step()
                scene.update(dt)
                for _ in range(max(1, int(base.args_cli.warmup_time_s / dt))):
                    scene.write_data_to_sim()
                    sim.step()
                    scene.update(dt)

                controller = base.PickPlaceController(scene["robot"], scene)
                if recorder is None:
                    recorder = base.MultiEpisodeLeRobotRecorder(
                        controller,
                        scene["top_color"] if base.args_cli.record_camera else None,
                        scene["wrist_color"] if base.args_cli.record_camera else None,
                    )
                else:
                    recorder.begin_episode(controller)

                attempt_elapsed_s = 0.0
                while base.simulation_app.is_running() and controller.state not in ("done", "failed"):
                    controller.advance(dt)
                    scene.write_data_to_sim()
                    sim.step()
                    scene.update(dt)
                    recorder.record_if_needed(dt)
                    attempt_elapsed_s += dt
                    if attempt_elapsed_s >= launcher_args.episode_timeout_s:
                        controller.fail(
                            "episode controller exceeded safety timeout "
                            f"({launcher_args.episode_timeout_s:.2f}s) in state={controller.state}"
                        )

                if controller.state == "failed":
                    print("[TRACE] after_triplet_failure")
                    recorder.discard_episode(controller.failure_reason or "IK failure")
                    recorder.finalize()
                    print("[TRACE] before_staging_cleanup")
                    shutil.rmtree(layout_stage_root, ignore_errors=True)
                    print("[TRACE] after_staging_cleanup")
                    return False, controller.failure_reason or "IK failure", layout_stage_root
                if controller.state != "done":
                    recorder.discard_episode("simulation_stopped")
                    recorder.finalize()
                    print("[TRACE] before_staging_cleanup")
                    shutil.rmtree(layout_stage_root, ignore_errors=True)
                    print("[TRACE] after_staging_cleanup")
                    return False, "simulation_stopped", layout_stage_root

                success_time = 0.0
                check_time = 0.0
                last_detail = "not checked"
                while base.simulation_app.is_running() and check_time < launcher_args.success_timeout_s:
                    controller.advance(dt)
                    scene.write_data_to_sim()
                    sim.step()
                    scene.update(dt)
                    recorder.record_if_needed(dt)
                    passed, last_detail = target_cube_success(controller.cube)
                    success_time = success_time + dt if passed else 0.0
                    check_time += dt
                    if success_time >= launcher_args.success_hold_time_s:
                        break

                if success_time < launcher_args.success_hold_time_s:
                    recorder.discard_episode(
                        f"target-specific quality check failed for {target_color}: {last_detail}"
                    )
                    recorder.finalize()
                    print("[TRACE] before_staging_cleanup")
                    shutil.rmtree(layout_stage_root, ignore_errors=True)
                    print("[TRACE] after_staging_cleanup")
                    return False, f"quality_failed:{last_detail}", layout_stage_root

                recorder.record_if_needed(0.0, force=True)
                recorder.save_episode()
                print(
                    f"[TRIPLET][PASS] layout={layout.layout_id:03d} "
                    f"slot={target_idx} color={target_color}"
                )
    finally:
        ACTIVE_LAYOUT = None
        ACTIVE_CANDIDATE_ATTEMPT = None

    if recorder is not None:
        recorder.finalize()
    return True, "success", layout_stage_root


def resolve_staging_root() -> Path:
    if launcher_args.staging_root is not None:
        return Path(launcher_args.staging_root).expanduser().resolve()
    return FINAL_DATASET_ROOT.parent / f"{FINAL_DATASET_ROOT.name}__staging"


def layout_manifest_rows(layouts: list[LayoutSpec]) -> list[dict[str, object]]:
    rows = []
    for layout in layouts:
        for color in COLORS:
            pose = pose_for_color(layout, color)
            episode_index = layout.layout_id * 3 + TARGET_SLOT_BY_COLOR[color]
            rows.append(
                {
                    "episode_index": episode_index,
                    "layout_id": layout.layout_id,
                    "target_color": color,
                    "layout_seed": layout.layout_seed,
                    "tcp_tilt_deg": layout.tcp_tilt_deg,
                    "red_x": layout.red_pose.x,
                    "red_y": layout.red_pose.y,
                    "red_z": layout.red_pose.z,
                    "red_qw": layout.red_pose.qw,
                    "red_qx": layout.red_pose.qx,
                    "red_qy": layout.red_pose.qy,
                    "red_qz": layout.red_pose.qz,
                    "blue_x": layout.blue_pose.x,
                    "blue_y": layout.blue_pose.y,
                    "blue_z": layout.blue_pose.z,
                    "blue_qw": layout.blue_pose.qw,
                    "blue_qx": layout.blue_pose.qx,
                    "blue_qy": layout.blue_pose.qy,
                    "blue_qz": layout.blue_pose.qz,
                    "yellow_x": layout.yellow_pose.x,
                    "yellow_y": layout.yellow_pose.y,
                    "yellow_z": layout.yellow_pose.z,
                    "yellow_qw": layout.yellow_pose.qw,
                    "yellow_qx": layout.yellow_pose.qx,
                    "yellow_qy": layout.yellow_pose.qy,
                    "yellow_qz": layout.yellow_pose.qz,
                    **{
                        f"robot_initial_state_deg_{idx}": value
                        for idx, value in enumerate(layout.robot_initial_state_deg)
                    },
                    "target_pose_x": pose.x,
                    "target_pose_y": pose.y,
                    "target_pose_z": pose.z,
                    "target_pose_qw": pose.qw,
                    "target_pose_qx": pose.qx,
                    "target_pose_qy": pose.qy,
                    "target_pose_qz": pose.qz,
                }
            )
    return rows


def triplet_manifest_rows(layouts: list[LayoutSpec], stage_roots: list[Path]) -> list[dict[str, object]]:
    rows = []
    for layout, stage_root in zip(layouts, stage_roots, strict=True):
        rows.append(
            {
                "layout_id": layout.layout_id,
                "layout_seed": layout.layout_seed,
                "red_episode": layout.layout_id * 3 + 0,
                "blue_episode": layout.layout_id * 3 + 1,
                "yellow_episode": layout.layout_id * 3 + 2,
                "triplet_order": "red,blue,yellow",
                "valid_triplet": True,
                "stage_root": str(stage_root),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def aggregate_layout_datasets(stage_roots: list[Path]) -> None:
    if FINAL_DATASET_ROOT.exists():
        if base.args_cli.overwrite_dataset:
            shutil.rmtree(FINAL_DATASET_ROOT)
        else:
            raise RuntimeError(
                f"Dataset root already exists: {FINAL_DATASET_ROOT}. "
                "Use --overwrite_dataset or another --dataset_root."
            )

    aggregate_datasets(
        repo_ids=[stage_repo_id(layout_id) for layout_id in range(len(stage_roots))],
        aggr_repo_id=FINAL_REPO_ID,
        roots=stage_roots,
        aggr_root=FINAL_DATASET_ROOT,
        data_files_size_in_mb=base.DEFAULT_DATA_FILE_SIZE_IN_MB if hasattr(base, "DEFAULT_DATA_FILE_SIZE_IN_MB") else None,
        video_files_size_in_mb=base.DEFAULT_VIDEO_FILE_SIZE_IN_MB if hasattr(base, "DEFAULT_VIDEO_FILE_SIZE_IN_MB") else None,
        chunk_size=base.DEFAULT_CHUNK_SIZE if hasattr(base, "DEFAULT_CHUNK_SIZE") else None,
    )


def main() -> None:
    global WORKSPACE_AUDIT_LOGGER
    if launcher_args.workspace_audit_csv is not None:
        WORKSPACE_AUDIT_LOGGER = WorkspaceAuditLogger(Path(launcher_args.workspace_audit_csv).expanduser().resolve())
    show_viewports = launcher_args.show_viewports
    if show_viewports is None:
        show_viewports = not WORKSPACE_AUDIT_MODE
    sim = base.SimulationContext(base.sim_utils.SimulationCfg(device=base.args_cli.device, dt=1.0 / 120.0))
    sim.set_camera_view(eye=[1.0, -1.35, 0.85], target=[-0.35, -0.2, 0.15])
    scene_cfg = ThreeColorCounterfactualSceneCfg(num_envs=1, env_spacing=1.0)
    scene_cfg.table.spawn.usd_path = base.TABLE_USD
    scene_cfg.robot.spawn.usd_path = base.ROBOT_USD
    scene_cfg.storage_box.spawn.usd_path = base.STORAGE_BOX_USD
    scene_cfg.realsense.spawn.usd_path = base.REALSENSE_USD
    scene = base.InteractiveScene(scene_cfg)
    wrist_root_path = base.attach_wrist_realsense(scene)
    sim.reset()
    _, wrist_source_path = base.configure_dual_realsense_record_cameras(scene, wrist_root_path)
    base.log_wrist_camera_diagnostics(scene, wrist_root_path, wrist_source_path)
    preview_viewports = None
    if show_viewports:
        preview_viewports = base.create_dual_realsense_views(scene, wrist_root_path)
    print(
        f"[INFO] Counterfactual triplet dataset: {launcher_args.num_layouts} layouts x 3 colors = "
        f"{FINAL_TOTAL_EPISODES} episodes"
    )
    print(
        f"[INFO] Layout sampling x={base.args_cli.cube_x_range or DEFAULT_X_RANGE}, "
        f"y={base.args_cli.cube_y_range or DEFAULT_Y_RANGE}, min_separation={launcher_args.cube_min_separation:.3f} m"
    )
    print(
        f"[INFO] Tilt range reused per layout: "
        f"[{DEFAULT_TILT_DEG_RANGE[0]:.2f}, {DEFAULT_TILT_DEG_RANGE[1]:.2f}] deg"
    )

    staging_root = resolve_staging_root()
    ensure_clean_dir(staging_root)

    try:
        successful_layouts: list[LayoutSpec] = []
        successful_stage_roots: list[Path] = []
        for layout_id in range(launcher_args.num_layouts):
            success = False
            for attempt in range(1, launcher_args.max_triplet_attempts + 1):
                layout = sample_layout(scene, layout_id, candidate_index=attempt - 1)
                print(
                    f"\n[TRIPLET] layout_id={layout_id:03d} attempt={attempt} "
                    f"seed={layout.layout_seed} tilt={layout.tcp_tilt_deg:.2f}"
                )
                ok, reason, stage_root = record_layout_triplet_v2(
                    sim,
                    scene,
                    layout,
                    staging_root,
                    candidate_attempt=attempt,
                )
                if ok:
                    successful_layouts.append(layout)
                    successful_stage_roots.append(stage_root)
                    success = True
                    break
                print(f"[TRIPLET][RETRY] layout_id={layout_id:03d} reason={reason}")
            print("[TRACE] leaving_attempt_loop")
            if not success:
                print("[TRACE] layout_exhausted")
                if WORKSPACE_AUDIT_MODE:
                    print(
                        f"[RESULT] workspace_audit layout exhausted "
                        f"layout_id={layout_id:03d} attempts={launcher_args.max_triplet_attempts}"
                    )
                    continue
                raise RuntimeError(
                    f"Failed to record complete triplet for layout_id={layout_id} "
                    f"after {launcher_args.max_triplet_attempts} attempts."
                )
            if WORKSPACE_AUDIT_MODE:
                print(f"[RESULT] workspace_audit candidate complete layout_id={layout_id:03d}")

        if WORKSPACE_AUDIT_MODE:
            if not launcher_args.keep_staging:
                shutil.rmtree(staging_root, ignore_errors=True)
        else:
            aggregate_layout_datasets(successful_stage_roots)
            write_csv(FINAL_DATASET_ROOT / launcher_args.layout_manifest_name, layout_manifest_rows(successful_layouts))
            write_csv(FINAL_DATASET_ROOT / launcher_args.triplet_manifest_name, triplet_manifest_rows(successful_layouts, successful_stage_roots))

            if not launcher_args.keep_staging:
                shutil.rmtree(staging_root, ignore_errors=True)

            print(f"[RESULT] dataset_root={FINAL_DATASET_ROOT}")
            print(f"[RESULT] repo_id={FINAL_REPO_ID}")
            print(f"[RESULT] layout_manifest={FINAL_DATASET_ROOT / launcher_args.layout_manifest_name}")
            print(f"[RESULT] triplet_manifest={FINAL_DATASET_ROOT / launcher_args.triplet_manifest_name}")
    finally:
        if WORKSPACE_AUDIT_LOGGER is not None:
            WORKSPACE_AUDIT_LOGGER.close()
            WORKSPACE_AUDIT_LOGGER = None
        print("[TRACE] before_env_close")
        close_preview_viewports(preview_viewports)
        print("[TRACE] after_env_close")
        print("[TRACE] before_simulation_app_close")
        if WORKSPACE_AUDIT_MODE:
            audit_mode_hard_exit()
        base.simulation_app.close()
        print("[TRACE] after_simulation_app_close")


if __name__ == "__main__":
    main()
