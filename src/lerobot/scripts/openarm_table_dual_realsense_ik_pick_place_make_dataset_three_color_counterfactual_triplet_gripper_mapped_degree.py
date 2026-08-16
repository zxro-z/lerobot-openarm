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

sys.argv = [sys.argv[0], *wrapped_argv]
import openarm_table_dual_realsense_ik_pick_place_make_dataset_random_cube_random_tilt_gripper_mapped_degree as degree  # noqa: E402
from lerobot.datasets.aggregate import aggregate_datasets  # noqa: E402


base = degree.base
torch = base.torch
tilt_launcher = degree.mapped.launcher_args
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
DEFAULT_X_RANGE = (-0.65, -0.50)
DEFAULT_Y_RANGE = (0.02, 0.15)
SUCCESS_X_HALF_EXTENT = 0.20 - base.CUBE_SIZE / 2.0
SUCCESS_Y_HALF_EXTENT = 0.25 - base.CUBE_SIZE / 2.0
SUCCESS_Z_RANGE = (0.04, 0.12)
TARGET_SLOT_BY_COLOR = {"red": 0, "blue": 1, "yellow": 2}

ACTIVE_COLOR = "red"
ACTIVE_LAYOUT = None
FINAL_DATASET_ROOT = Path(base.args_cli.dataset_root).expanduser().resolve()
FINAL_REPO_ID = base.args_cli.dataset_repo_id
FINAL_TOTAL_EPISODES = launcher_args.num_layouts * len(COLORS)


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


def sample_layout(scene, layout_id: int) -> LayoutSpec:
    layout_seed = launcher_args.layout_base_seed + layout_id
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
    tilt_deg = float(rng.uniform(tilt_launcher.tilt_deg_range[0], tilt_launcher.tilt_deg_range[1]))
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


base.PickPlaceController.__init__ = _counterfactual_controller_init


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


def stage_repo_id(layout_id: int) -> str:
    return f"{FINAL_REPO_ID}_layout_{layout_id:03d}"


def record_layout_triplet_v2(sim, scene, layout: LayoutSpec, staging_root: Path) -> tuple[bool, str, Path]:
    global ACTIVE_COLOR, ACTIVE_LAYOUT
    dt = sim.get_physics_dt()
    layout_stage_root = stage_root_for_layout(staging_root, layout.layout_id)
    ensure_clean_dir(layout_stage_root)
    repo_id = stage_repo_id(layout.layout_id)
    recorder = None
    try:
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
                    recorder.discard_episode(controller.failure_reason or "IK failure")
                    recorder.finalize()
                    shutil.rmtree(layout_stage_root, ignore_errors=True)
                    return False, controller.failure_reason or "IK failure", layout_stage_root
                if controller.state != "done":
                    recorder.discard_episode("simulation_stopped")
                    recorder.finalize()
                    shutil.rmtree(layout_stage_root, ignore_errors=True)
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
                    shutil.rmtree(layout_stage_root, ignore_errors=True)
                    return False, f"quality_failed:{last_detail}", layout_stage_root

                recorder.record_if_needed(0.0, force=True)
                recorder.save_episode()
                print(
                    f"[TRIPLET][PASS] layout={layout.layout_id:03d} "
                    f"slot={target_idx} color={target_color}"
                )
    finally:
        ACTIVE_LAYOUT = None

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
    base.create_dual_realsense_views(scene, wrist_root_path)
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
        f"[{tilt_launcher.tilt_deg_range[0]:.2f}, {tilt_launcher.tilt_deg_range[1]:.2f}] deg"
    )

    staging_root = resolve_staging_root()
    ensure_clean_dir(staging_root)

    successful_layouts: list[LayoutSpec] = []
    successful_stage_roots: list[Path] = []
    for layout_id in range(launcher_args.num_layouts):
        success = False
        for attempt in range(1, launcher_args.max_triplet_attempts + 1):
            layout = sample_layout(scene, layout_id)
            print(
                f"\n[TRIPLET] layout_id={layout_id:03d} attempt={attempt} "
                f"seed={layout.layout_seed} tilt={layout.tcp_tilt_deg:.2f}"
            )
            ok, reason, stage_root = record_layout_triplet_v2(sim, scene, layout, staging_root)
            if ok:
                successful_layouts.append(layout)
                successful_stage_roots.append(stage_root)
                success = True
                break
            print(f"[TRIPLET][RETRY] layout_id={layout_id:03d} reason={reason}")
        if not success:
            raise RuntimeError(
                f"Failed to record complete triplet for layout_id={layout_id} "
                f"after {launcher_args.max_triplet_attempts} attempts."
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
    base.simulation_app.close()


if __name__ == "__main__":
    main()
