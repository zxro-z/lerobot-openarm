"""Generate a fixed-geometry three-color counterfactual baseline dataset.

This baseline removes random workspace geometry and random tilt.  Each visual
layout uses:

    fixed cube slots
    fixed storage-box pose
    fixed TCP tilt = 50 deg
    one deterministic color permutation over the three slots

For each accepted layout, red/blue/yellow instructions are rolled out
atomically from the exact same scene geometry.  The only intended difference
within a triplet is the instruction / target cube identity.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def _parse_baseline_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--repeats_per_permutation",
        type=int,
        default=3,
        help="Number of accepted triplets to record for each of the 6 color permutations.",
    )
    parser.add_argument(
        "--max_triplet_attempts_per_layout",
        type=int,
        default=3,
        help="Maximum retries for one fixed permutation layout before failing hard.",
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
        "--show_viewports",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Opt in/out of preview viewport windows.",
    )
    parser.add_argument(
        "--smoke_test",
        action="store_true",
        help="Generate one deterministic pilot triplet using the first permutation only.",
    )
    return parser.parse_known_args()


launcher_args, wrapped_argv = _parse_baseline_args()
if launcher_args.repeats_per_permutation < 1:
    raise SystemExit("--repeats_per_permutation must be at least 1")
if launcher_args.max_triplet_attempts_per_layout < 1:
    raise SystemExit("--max_triplet_attempts_per_layout must be at least 1")

sys.argv = [sys.argv[0], *wrapped_argv]
import openarm_table_dual_realsense_ik_pick_place_make_dataset_random_cube_random_tilt_gripper_mapped_degree as degree  # noqa: E402
from lerobot.datasets.aggregate import aggregate_datasets  # noqa: E402


base = degree.base
torch = base.torch
original_controller_init = degree.mapped._original_controller_init

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
COLOR_PERMUTATIONS = (
    ("red", "blue", "yellow"),
    ("red", "yellow", "blue"),
    ("blue", "red", "yellow"),
    ("blue", "yellow", "red"),
    ("yellow", "red", "blue"),
    ("yellow", "blue", "red"),
)
FIXED_TCP_TILT_DEG = 50.0
FIXED_STORAGE_BOX_POS = (-0.54, -0.18, 0.04664)
base.STORAGE_BOX_POS = FIXED_STORAGE_BOX_POS
TABLE_SURFACE_Z = -0.0211 + 0.04 / 2.0
CUBE_HALF_EXTENT = base.CUBE_SIZE / 2.0
RESTING_CUBE_CENTER_Z = TABLE_SURFACE_Z + CUBE_HALF_EXTENT

# These three slots stay inside the existing proven random-layout bounds
# The robot body is at positive y and the storage box sits at negative y, so
# outward transport space is along decreasing y. Keep one straight slot row
# with constant x and widen spacing along y.
SLOT_A = (-0.55, 0.16, RESTING_CUBE_CENTER_Z)
SLOT_B = (-0.55, 0.08, RESTING_CUBE_CENTER_Z)
SLOT_C = (-0.55, 0.00, RESTING_CUBE_CENTER_Z)
SLOT_NAMES = ("slot_a", "slot_b", "slot_c")
SLOT_POSES = {
    "slot_a": SLOT_A,
    "slot_b": SLOT_B,
    "slot_c": SLOT_C,
}
TARGET_SLOT_BY_COLOR = {"red": 0, "blue": 1, "yellow": 2}
PRIM_NAME_BY_ASSET = {
    "red": "Cube",
    "blue": "BlueCube",
    "yellow": "YellowCube",
    "box": "StorageBox",
}

ACTIVE_COLOR = "red"
ACTIVE_LAYOUT = None
FINAL_DATASET_ROOT = Path(base.args_cli.dataset_root).expanduser().resolve()
FINAL_REPO_ID = base.args_cli.dataset_repo_id
LAYOUT_SPEC_FILENAME = "layout_spec.json"
GEOMETRY_LOGGED_LAYOUT_IDS: set[int] = set()


def _planned_layouts() -> list[tuple[int, int, tuple[str, str, str]]]:
    if launcher_args.smoke_test:
        return [(0, 0, COLOR_PERMUTATIONS[0])]
    layouts = []
    layout_id = 0
    for repeat_index in range(launcher_args.repeats_per_permutation):
        for permutation_id, permutation in enumerate(COLOR_PERMUTATIONS):
            layouts.append((layout_id, repeat_index, permutation))
            layout_id += 1
    return layouts


PLANNED_LAYOUTS = _planned_layouts()
FINAL_TOTAL_EPISODES = len(PLANNED_LAYOUTS) * len(COLORS)


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
    permutation_id: int
    repeat_index: int
    tcp_tilt_deg: float
    slot_a_color: str
    slot_b_color: str
    slot_c_color: str
    red_pose: CubePose
    blue_pose: CubePose
    yellow_pose: CubePose
    robot_initial_state_deg: tuple[float, ...]


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
class ThreeColorFixedSlotsSceneCfg(base.DualRealSensePickPlaceSceneCfg):
    blue_cube = _cube_cfg("blue")
    yellow_cube = _cube_cfg("yellow")


def cube_pose_from_components(
    position: tuple[float, float, float],
    quat_wxyz: tuple[float, float, float, float],
) -> CubePose:
    return CubePose(
        x=float(position[0]),
        y=float(position[1]),
        z=float(position[2]),
        qw=float(quat_wxyz[0]),
        qx=float(quat_wxyz[1]),
        qy=float(quat_wxyz[2]),
        qz=float(quat_wxyz[3]),
    )


def build_layout(scene, layout_id: int, repeat_index: int, permutation: tuple[str, str, str]) -> LayoutSpec:
    permutation_id = COLOR_PERMUTATIONS.index(permutation)
    default_quat = tuple(
        float(value)
        for value in scene[CUBE_ASSET_NAMES["red"]].data.default_root_state[0, 3:7].detach().cpu().tolist()
    )
    color_to_slot = {
        permutation[0]: "slot_a",
        permutation[1]: "slot_b",
        permutation[2]: "slot_c",
    }
    robot_initial_state = tuple(
        float(v) for v in np.rad2deg(scene["robot"].data.default_joint_pos[0].detach().cpu().numpy())
    )
    return LayoutSpec(
        layout_id=layout_id,
        permutation_id=permutation_id,
        repeat_index=repeat_index,
        tcp_tilt_deg=FIXED_TCP_TILT_DEG,
        slot_a_color=permutation[0],
        slot_b_color=permutation[1],
        slot_c_color=permutation[2],
        red_pose=cube_pose_from_components(SLOT_POSES[color_to_slot["red"]], default_quat),
        blue_pose=cube_pose_from_components(SLOT_POSES[color_to_slot["blue"]], default_quat),
        yellow_pose=cube_pose_from_components(SLOT_POSES[color_to_slot["yellow"]], default_quat),
        robot_initial_state_deg=robot_initial_state,
    )


def pose_for_color(layout: LayoutSpec, color: str) -> CubePose:
    return {"red": layout.red_pose, "blue": layout.blue_pose, "yellow": layout.yellow_pose}[color]


def task_for_color(color: str) -> str:
    return f"Pick up the {color} cube and place it in the storage box."


def _set_default_root_pose(asset, world_pose: CubePose | tuple[float, float, float], scene) -> None:
    state = asset.data.default_root_state.clone()
    if isinstance(world_pose, CubePose):
        state[:, 0] = world_pose.x
        state[:, 1] = world_pose.y
        state[:, 2] = world_pose.z
        state[:, 3] = world_pose.qw
        state[:, 4] = world_pose.qx
        state[:, 5] = world_pose.qy
        state[:, 6] = world_pose.qz
    else:
        state[:, 0] = float(world_pose[0])
        state[:, 1] = float(world_pose[1])
        state[:, 2] = float(world_pose[2])
    asset.data.default_root_state.copy_(state)
    state[:, :3] += scene.env_origins
    asset.write_root_pose_to_sim(state[:, :7])
    asset.write_root_velocity_to_sim(torch.zeros_like(state[:, 7:]))


def _set_box_pose(scene, world_pose: tuple[float, float, float]) -> None:
    stage = base.get_current_stage()
    prim = stage.GetPrimAtPath(prim_path_for_label(scene, "box"))
    if not prim.IsValid():
        raise RuntimeError(f"Storage box prim not found: {prim.GetPath()}")
    xformable = base.UsdGeom.Xformable(prim)
    local_pos = (
        float(world_pose[0] - scene.env_origins[0, 0].item()),
        float(world_pose[1] - scene.env_origins[0, 1].item()),
        float(world_pose[2] - scene.env_origins[0, 2].item()),
    )
    xformable.ClearXformOpOrder()
    xformable.AddTranslateOp().Set(base.Gf.Vec3d(*local_pos))


def _box_world_pose(scene) -> tuple[float, float, float]:
    world_xyz, _ = base._prim_world_pose(prim_path_for_label(scene, "box"))
    return tuple(float(v) for v in world_xyz)


def read_actual_geometry(scene) -> dict[str, tuple[float, float, float]]:
    return {
        "red": tuple(float(v) for v in scene[CUBE_ASSET_NAMES["red"]].data.root_pos_w[0].detach().cpu().tolist()),
        "blue": tuple(float(v) for v in scene[CUBE_ASSET_NAMES["blue"]].data.root_pos_w[0].detach().cpu().tolist()),
        "yellow": tuple(float(v) for v in scene[CUBE_ASSET_NAMES["yellow"]].data.root_pos_w[0].detach().cpu().tolist()),
        "box": _box_world_pose(scene),
    }


def prim_path_for_label(scene, label: str) -> str:
    return f"{scene.env_prim_paths[0]}/{PRIM_NAME_BY_ASSET[label]}"


def prim_visibility(scene, label: str) -> str:
    stage = base.get_current_stage()
    prim = stage.GetPrimAtPath(prim_path_for_label(scene, label))
    if not prim.IsValid():
        return "invalid_prim"
    imageable = base.UsdGeom.Imageable(prim)
    if not imageable:
        return "not_imageable"
    visibility = imageable.GetVisibilityAttr().Get()
    return str(visibility) if visibility is not None else "inherited"


def maybe_log_fixed_geometry(scene, layout: LayoutSpec) -> None:
    if layout.layout_id in GEOMETRY_LOGGED_LAYOUT_IDS:
        return
    actual = read_actual_geometry(scene)
    print(f"[FIXED_GEOMETRY] slot_A_expected={SLOT_A}")
    print(f"[FIXED_GEOMETRY] slot_B_expected={SLOT_B}")
    print(f"[FIXED_GEOMETRY] slot_C_expected={SLOT_C}")
    print(f"[FIXED_GEOMETRY] box_expected={FIXED_STORAGE_BOX_POS}")
    print(f"[FIXED_GEOMETRY] red_actual={actual['red']}")
    print(f"[FIXED_GEOMETRY] blue_actual={actual['blue']}")
    print(f"[FIXED_GEOMETRY] yellow_actual={actual['yellow']}")
    print(f"[FIXED_GEOMETRY] box_actual={actual['box']}")
    for color in COLORS:
        expected = pose_for_color(layout, color)
        actual_xyz = actual[color]
        delta = (
            actual_xyz[0] - expected.x,
            actual_xyz[1] - expected.y,
            actual_xyz[2] - expected.z,
        )
        print(f"[FIXED_GEOMETRY] {color}_delta_xyz={delta}")
    box_delta = (
        actual["box"][0] - FIXED_STORAGE_BOX_POS[0],
        actual["box"][1] - FIXED_STORAGE_BOX_POS[1],
        actual["box"][2] - FIXED_STORAGE_BOX_POS[2],
    )
    print(f"[FIXED_GEOMETRY] box_delta_xyz={box_delta}")
    for color in COLORS:
        expected = pose_for_color(layout, color)
        print(f"[FIXED_GEOMETRY_DIAG] {color}:")
        print(f"  prim={prim_path_for_label(scene, color)}")
        print(f"  visible={prim_visibility(scene, color)}")
        print(f"  expected_xyz=({expected.x}, {expected.y}, {expected.z})")
        print(f"  actual_xyz={actual[color]}")
    print("[FIXED_GEOMETRY_DIAG] box:")
    print(f"  prim={prim_path_for_label(scene, 'box')}")
    print(f"  visible={prim_visibility(scene, 'box')}")
    print(f"  expected_xyz={FIXED_STORAGE_BOX_POS}")
    print(f"  actual_xyz={actual['box']}")
    GEOMETRY_LOGGED_LAYOUT_IDS.add(layout.layout_id)


def restore_fixed_geometry(scene, layout: LayoutSpec) -> None:
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
        _set_default_root_pose(cube, pose_for_color(layout, color), scene)

    _set_box_pose(scene, FIXED_STORAGE_BOX_POS)
    scene.reset()
    _set_box_pose(scene, FIXED_STORAGE_BOX_POS)


def target_cube_success(cube) -> tuple[bool, str]:
    position = cube.data.root_pos_w[0].detach().cpu().numpy()
    velocity = cube.data.root_vel_w[0].detach().cpu().numpy()
    dx = abs(float(position[0]) - base.STORAGE_BOX_POS[0])
    dy = abs(float(position[1]) - base.STORAGE_BOX_POS[1])
    in_box = (
        dx < (0.20 - base.CUBE_SIZE / 2.0)
        and dy < (0.25 - base.CUBE_SIZE / 2.0)
        and 0.04 < float(position[2]) < 0.12
    )
    speed = float(np.linalg.norm(velocity[:3]))
    stationary = speed < 0.05
    detail = f"xyz={position.tolist()}, speed={speed:.4f}, in_box={in_box}, stationary={stationary}"
    return bool(in_box and stationary), detail


def _baseline_controller_init(self, robot, scene) -> None:
    if ACTIVE_LAYOUT is None:
        raise RuntimeError("ACTIVE_LAYOUT must be set before creating the controller.")
    base.args_cli.tilt_deg = FIXED_TCP_TILT_DEG
    print(
        f"[EPISODE] Layout {ACTIVE_LAYOUT.layout_id:03d} target={ACTIVE_COLOR} "
        f"tcp_tilt={FIXED_TCP_TILT_DEG:.1f} deg"
    )
    original_controller_init(self, robot, scene)
    self.cube = scene[CUBE_ASSET_NAMES[ACTIVE_COLOR]]
    self.target_color = ACTIVE_COLOR


base.PickPlaceController.__init__ = _baseline_controller_init


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


def resolve_staging_root() -> Path:
    if launcher_args.staging_root is not None:
        return Path(launcher_args.staging_root).expanduser().resolve()
    return FINAL_DATASET_ROOT.parent / f"{FINAL_DATASET_ROOT.name}__staging"


def ensure_clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def stage_repo_id(layout_id: int) -> str:
    return f"{FINAL_REPO_ID}_layout_{layout_id:03d}"


def slot_distances() -> tuple[float, float, float]:
    def _dist(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
        return math.dist(a[:2], b[:2])

    return (
        _dist(SLOT_A, SLOT_B),
        _dist(SLOT_B, SLOT_C),
        _dist(SLOT_A, SLOT_C),
    )


def serialize_layout_spec(layout: LayoutSpec) -> dict[str, object]:
    return {
        "layout_id": layout.layout_id,
        "permutation_id": layout.permutation_id,
        "repeat_index": layout.repeat_index,
        "tcp_tilt_deg": layout.tcp_tilt_deg,
        "slot_a_color": layout.slot_a_color,
        "slot_b_color": layout.slot_b_color,
        "slot_c_color": layout.slot_c_color,
        "slot_a_xyz": list(SLOT_A),
        "slot_b_xyz": list(SLOT_B),
        "slot_c_xyz": list(SLOT_C),
        "storage_box_xyz": list(FIXED_STORAGE_BOX_POS),
        "red_pose": layout.red_pose.__dict__,
        "blue_pose": layout.blue_pose.__dict__,
        "yellow_pose": layout.yellow_pose.__dict__,
        "robot_initial_state_deg": list(layout.robot_initial_state_deg),
    }


def write_layout_spec(stage_root: Path, layout: LayoutSpec) -> None:
    (stage_root / LAYOUT_SPEC_FILENAME).write_text(
        json.dumps(serialize_layout_spec(layout), indent=2),
        encoding="utf-8",
    )


def record_layout_triplet(
    sim,
    scene,
    layout: LayoutSpec,
    staging_root: Path,
    *,
    attempt_index: int,
) -> tuple[bool, str, Path, str | None]:
    global ACTIVE_COLOR, ACTIVE_LAYOUT
    dt = sim.get_physics_dt()
    layout_stage_root = stage_root_for_layout(staging_root, layout.layout_id)
    if layout_stage_root.exists():
        shutil.rmtree(layout_stage_root)
    repo_id = stage_repo_id(layout.layout_id)
    recorder = None
    try:
        with temporary_dataset_args(dataset_root=layout_stage_root, dataset_repo_id=repo_id, num_episodes=3):
            for target_idx, target_color in enumerate(COLORS):
                ACTIVE_COLOR = target_color
                ACTIVE_LAYOUT = layout
                base.args_cli.task = task_for_color(target_color)
                restore_fixed_geometry(scene, layout)
                print(
                    f"[LAYOUT] layout_id={layout.layout_id:03d} perm={layout.permutation_id} "
                    f"repeat={layout.repeat_index} attempt={attempt_index} target={target_color} "
                    f"slots=({layout.slot_a_color},{layout.slot_b_color},{layout.slot_c_color})"
                )
                scene.write_data_to_sim()
                sim.step()
                scene.update(dt)
                maybe_log_fixed_geometry(scene, layout)
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

                elapsed_s = 0.0
                while base.simulation_app.is_running() and controller.state not in ("done", "failed"):
                    controller.advance(dt)
                    scene.write_data_to_sim()
                    sim.step()
                    scene.update(dt)
                    recorder.record_if_needed(dt)
                    elapsed_s += dt
                    if elapsed_s >= 30.0:
                        controller.fail("episode controller exceeded safety timeout (30.00s)")

                if controller.state == "failed":
                    recorder.discard_episode(controller.failure_reason or "IK failure")
                    recorder.finalize()
                    shutil.rmtree(layout_stage_root, ignore_errors=True)
                    return False, controller.failure_reason or "IK failure", layout_stage_root, target_color
                if controller.state != "done":
                    recorder.discard_episode("simulation_stopped")
                    recorder.finalize()
                    shutil.rmtree(layout_stage_root, ignore_errors=True)
                    return False, "simulation_stopped", layout_stage_root, target_color

                success_time = 0.0
                check_time = 0.0
                last_detail = "not checked"
                while base.simulation_app.is_running() and check_time < 2.0:
                    controller.advance(dt)
                    scene.write_data_to_sim()
                    sim.step()
                    scene.update(dt)
                    recorder.record_if_needed(dt)
                    passed, last_detail = target_cube_success(controller.cube)
                    success_time = success_time + dt if passed else 0.0
                    check_time += dt
                    if success_time >= 0.25:
                        break

                if success_time < 0.25:
                    recorder.discard_episode(f"target-specific quality check failed for {target_color}: {last_detail}")
                    recorder.finalize()
                    shutil.rmtree(layout_stage_root, ignore_errors=True)
                    return False, f"quality_failed:{last_detail}", layout_stage_root, target_color

                recorder.record_if_needed(0.0, force=True)
                recorder.save_episode()
                print(f"[TRIPLET][PASS] layout={layout.layout_id:03d} slot={target_idx} color={target_color}")
    finally:
        ACTIVE_LAYOUT = None

    if recorder is not None:
        recorder.finalize()
    write_layout_spec(layout_stage_root, layout)
    return True, "success", layout_stage_root, None


def layout_manifest_rows(layouts: list[LayoutSpec]) -> list[dict[str, object]]:
    rows = []
    for layout in layouts:
        for color in COLORS:
            pose = pose_for_color(layout, color)
            rows.append(
                {
                    "episode_index": layout.layout_id * 3 + TARGET_SLOT_BY_COLOR[color],
                    "layout_id": layout.layout_id,
                    "permutation_id": layout.permutation_id,
                    "repeat_index": layout.repeat_index,
                    "target_color": color,
                    "tcp_tilt_deg": layout.tcp_tilt_deg,
                    "slot_a_color": layout.slot_a_color,
                    "slot_b_color": layout.slot_b_color,
                    "slot_c_color": layout.slot_c_color,
                    "slot_a_x": SLOT_A[0],
                    "slot_a_y": SLOT_A[1],
                    "slot_a_z": SLOT_A[2],
                    "slot_b_x": SLOT_B[0],
                    "slot_b_y": SLOT_B[1],
                    "slot_b_z": SLOT_B[2],
                    "slot_c_x": SLOT_C[0],
                    "slot_c_y": SLOT_C[1],
                    "slot_c_z": SLOT_C[2],
                    "storage_box_x": FIXED_STORAGE_BOX_POS[0],
                    "storage_box_y": FIXED_STORAGE_BOX_POS[1],
                    "storage_box_z": FIXED_STORAGE_BOX_POS[2],
                    **{f"robot_initial_state_deg_{idx}": value for idx, value in enumerate(layout.robot_initial_state_deg)},
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
                "permutation_id": layout.permutation_id,
                "repeat_index": layout.repeat_index,
                "slot_a_color": layout.slot_a_color,
                "slot_b_color": layout.slot_b_color,
                "slot_c_color": layout.slot_c_color,
                "tcp_tilt_deg": layout.tcp_tilt_deg,
                "slot_a_x": SLOT_A[0],
                "slot_a_y": SLOT_A[1],
                "slot_a_z": SLOT_A[2],
                "slot_b_x": SLOT_B[0],
                "slot_b_y": SLOT_B[1],
                "slot_b_z": SLOT_B[2],
                "slot_c_x": SLOT_C[0],
                "slot_c_y": SLOT_C[1],
                "slot_c_z": SLOT_C[2],
                "storage_box_x": FIXED_STORAGE_BOX_POS[0],
                "storage_box_y": FIXED_STORAGE_BOX_POS[1],
                "storage_box_z": FIXED_STORAGE_BOX_POS[2],
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


def main() -> None:
    show_viewports = launcher_args.show_viewports
    if show_viewports is None:
        show_viewports = True

    staging_root = resolve_staging_root()
    ensure_clean_dir(staging_root)

    sim = base.SimulationContext(base.sim_utils.SimulationCfg(device=base.args_cli.device, dt=1.0 / 120.0))
    sim.set_camera_view(eye=[1.0, -1.35, 0.85], target=[-0.35, -0.2, 0.15])
    scene_cfg = ThreeColorFixedSlotsSceneCfg(num_envs=1, env_spacing=1.0)
    scene_cfg.table.spawn.usd_path = base.TABLE_USD
    scene_cfg.robot.spawn.usd_path = base.ROBOT_USD
    scene_cfg.storage_box.spawn.usd_path = base.STORAGE_BOX_USD
    scene_cfg.storage_box.init_state.pos = FIXED_STORAGE_BOX_POS
    scene_cfg.realsense.spawn.usd_path = base.REALSENSE_USD
    scene = base.InteractiveScene(scene_cfg)
    wrist_root_path = base.attach_wrist_realsense(scene)
    sim.reset()
    _, wrist_source_path = base.configure_dual_realsense_record_cameras(scene, wrist_root_path)
    base.log_wrist_camera_diagnostics(scene, wrist_root_path, wrist_source_path)
    preview_layout = build_layout(scene, *PLANNED_LAYOUTS[0])
    restore_fixed_geometry(scene, preview_layout)
    scene.write_data_to_sim()
    sim.step()
    scene.update(sim.get_physics_dt())
    preview_viewports = None
    if show_viewports:
        preview_viewports = base.create_dual_realsense_views(scene, wrist_root_path)

    d_ab, d_bc, d_ac = slot_distances()
    print(f"[INFO] Fixed-slot baseline: {len(PLANNED_LAYOUTS)} layouts x 3 colors = {FINAL_TOTAL_EPISODES} episodes")
    print(f"[INFO] Fixed slots: A={SLOT_A}, B={SLOT_B}, C={SLOT_C}")
    print(f"[INFO] Inter-slot XY distances: AB={d_ab:.4f} m, BC={d_bc:.4f} m, AC={d_ac:.4f} m")
    print(f"[INFO] Storage center xyz={FIXED_STORAGE_BOX_POS}")
    print(f"[INFO] Fixed TCP tilt={FIXED_TCP_TILT_DEG:.1f} deg")
    print(
        f"[INFO] Working table/cube Z contract: table_surface_z={TABLE_SURFACE_Z:.4f}, "
        f"cube_half_extent={CUBE_HALF_EXTENT:.4f}, cube_center_z={RESTING_CUBE_CENTER_Z:.4f}"
    )
    print(f"[INFO] Color permutations={len(COLOR_PERMUTATIONS)}")
    if launcher_args.smoke_test:
        print("[INFO] Smoke test mode enabled: only the first permutation is recorded once.")

    successful_layouts: list[LayoutSpec] = []
    successful_stage_roots: list[Path] = []

    try:
        for layout_id, repeat_index, permutation in PLANNED_LAYOUTS:
            layout = build_layout(scene, layout_id, repeat_index, permutation)
            success = False
            print(
                f"\n[LAYOUT {layout_id:03d}] permutation_id={layout.permutation_id} "
                f"repeat={repeat_index} slots=({layout.slot_a_color},{layout.slot_b_color},{layout.slot_c_color})"
            )
            for attempt_index in range(1, launcher_args.max_triplet_attempts_per_layout + 1):
                ok, reason, stage_root, failed_color = record_layout_triplet(
                    sim,
                    scene,
                    layout,
                    staging_root,
                    attempt_index=attempt_index,
                )
                if ok:
                    successful_layouts.append(layout)
                    successful_stage_roots.append(stage_root)
                    success = True
                    print(f"[LAYOUT {layout_id:03d}] ACCEPT")
                    break
                print(
                    f"[LAYOUT {layout_id:03d}] FAIL attempt={attempt_index} "
                    f"color={failed_color or 'unknown'} reason={reason}"
                )
            if not success:
                raise RuntimeError(
                    f"Fixed-geometry layout failed after {launcher_args.max_triplet_attempts_per_layout} attempts: "
                    f"layout_id={layout_id}, permutation_id={layout.permutation_id}, "
                    f"slots=({layout.slot_a_color},{layout.slot_b_color},{layout.slot_c_color})"
                )

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
        close_preview_viewports(preview_viewports)
        base.simulation_app.close()


if __name__ == "__main__":
    main()
