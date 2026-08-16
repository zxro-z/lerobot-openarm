"""Generate a quality-filtered, balanced red/blue/yellow cube dataset.

``--num_episodes`` means the number of *successful episodes per color* in
this launcher.  For example, ``--num_episodes 50`` saves 150 episodes in the
order red, blue, yellow, red, blue, yellow, ... .  A failed attempt is cleared
from LeRobot's episode buffer and retried with the same target color.

All three cubes are randomized for every attempt while maintaining a minimum
XY center distance.  An episode is saved only after the requested target cube
is inside the storage box and has remained nearly stationary for the required
hold time.  State/action values retain the wrapped launcher's degree units.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np


def _parse_three_color_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
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
        help=(
            "Maximum attempts per position before discarding the partial layout "
            "and restarting from position 1."
        ),
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
    return parser.parse_known_args()


launcher_args, wrapped_argv = _parse_three_color_args()
if launcher_args.cube_min_separation < math.sqrt(2.0) * 0.05:
    raise SystemExit("--cube_min_separation must be at least sqrt(2) * cube size (0.0708 m)")
if launcher_args.cube_sampling_max_attempts < 1:
    raise SystemExit("--cube_sampling_max_attempts must be at least 1")
if launcher_args.success_hold_time_s <= 0.0 or launcher_args.success_timeout_s <= 0.0:
    raise SystemExit("success hold/timeout values must be greater than zero")
if launcher_args.episode_timeout_s <= 0.0:
    raise SystemExit("--episode_timeout_s must be greater than zero")

# Let the existing random-tilt, real-gripper mapping and degree launchers own
# their arguments.  Importing this module also launches Isaac Sim, as in the
# existing degree launcher.
sys.argv = [sys.argv[0], *wrapped_argv]
import openarm_table_dual_realsense_ik_pick_place_make_dataset_random_cube_random_tilt_gripper_mapped_degree as degree  # noqa: E402


base = degree.base
torch = base.torch

# The wrapped script lives in scripts/make_dataset but calculates its asset
# directory relative to ``scripts``.  Resolve from this launcher's repository
# root and also replace the paths already captured by the base scene config.
REPO_ROOT = Path(__file__).resolve().parents[2]
ASSET_DIR = REPO_ROOT / "assets" / "openarm_use"
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
EPISODES_PER_COLOR = base.args_cli.num_episodes
TOTAL_EPISODES = EPISODES_PER_COLOR * len(COLORS)
base.args_cli.num_episodes = TOTAL_EPISODES

# Defaults provide a useful reachable sampling area even when the caller does
# not pass the original single-cube range arguments.
DEFAULT_X_RANGE = (-0.65, -0.50)
DEFAULT_Y_RANGE = (0.02, 0.15)
POSITION_RNG = np.random.default_rng(base.args_cli.cube_random_seed)
ACTIVE_COLOR = "red"

# Conservative center bounds already used by this repository's evaluation
# environment.  Requiring the whole 5 cm cube to remain inside in XY avoids
# accepting cubes balanced on a rim.
SUCCESS_X_HALF_EXTENT = 0.20 - base.CUBE_SIZE / 2.0
SUCCESS_Y_HALF_EXTENT = 0.25 - base.CUBE_SIZE / 2.0
SUCCESS_Z_RANGE = (0.04, 0.12)


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
            collision_props=base.sim_utils.CollisionPropertiesCfg(
                contact_offset=0.002, rest_offset=0.0
            ),
            physics_material=base.sim_utils.RigidBodyMaterialCfg(
                static_friction=1.0, dynamic_friction=0.8, restitution=0.0
            ),
            visual_material=base.sim_utils.PreviewSurfaceCfg(diffuse_color=CUBE_RGB[color]),
        ),
        init_state=base.RigidObjectCfg.InitialStateCfg(pos=base.CUBE_POS),
    )


@base.configclass
class ThreeColorPickPlaceSceneCfg(base.DualRealSensePickPlaceSceneCfg):
    # The inherited ``cube`` is red.  Two independent rigid objects are added.
    blue_cube = _cube_cfg("blue")
    yellow_cube = _cube_cfg("yellow")


def _sample_three_cube_positions() -> dict[str, tuple[float, float, float]]:
    x_limits = base.args_cli.cube_x_range or DEFAULT_X_RANGE
    y_limits = base.args_cli.cube_y_range or DEFAULT_Y_RANGE
    layout_attempt = 0
    while True:
        layout_attempt += 1
        sampled_positions: list[tuple[float, float, float]] = []
        layout_is_valid = True
        for position_index in range(len(COLORS)):
            for _ in range(launcher_args.cube_sampling_max_attempts):
                candidate = (
                    float(POSITION_RNG.uniform(*x_limits)),
                    float(POSITION_RNG.uniform(*y_limits)),
                    float(base.args_cli.cube_z),
                )
                if all(
                    math.hypot(candidate[0] - other[0], candidate[1] - other[1])
                    >= launcher_args.cube_min_separation
                    for other in sampled_positions
                ):
                    sampled_positions.append(candidate)
                    break
            else:
                layout_is_valid = False
                print(
                    f"[LAYOUT] Could not sample position {position_index + 1}; "
                    "discarding the partial layout and resampling from position 1 "
                    f"(layout attempt {layout_attempt + 1})."
                )
                break
        if layout_is_valid:
            # Positions are sampled without color identity.  Shuffle the three
            # colors only after a valid layout exists so no color is always the
            # unconstrained first sample or the most-constrained last sample.
            shuffled_colors = list(COLORS)
            POSITION_RNG.shuffle(shuffled_colors)
            positions_by_color = dict(zip(shuffled_colors, sampled_positions, strict=True))
            print(f"[LAYOUT] Random color assignment by position: {shuffled_colors}")
            return positions_by_color


def reset_three_color_scene(scene):
    robot = scene["robot"]
    robot_state = robot.data.default_root_state.clone()
    robot_state[:, :3] += scene.env_origins
    robot.write_root_pose_to_sim(robot_state[:, :7])
    robot.write_root_velocity_to_sim(robot_state[:, 7:])
    joint_pos = robot.data.default_joint_pos.clone()
    robot.write_joint_state_to_sim(joint_pos, torch.zeros_like(robot.data.default_joint_vel))
    robot.set_joint_position_target(joint_pos)

    positions = _sample_three_cube_positions()
    for color, asset_name in CUBE_ASSET_NAMES.items():
        cube = scene[asset_name]
        cube_state = cube.data.default_root_state.clone()
        cube_state[:, :3] = torch.tensor(
            positions[color], device=cube_state.device, dtype=cube_state.dtype
        )
        cube_state[:, :3] += scene.env_origins
        cube.write_root_pose_to_sim(cube_state[:, :7])
        cube.write_root_velocity_to_sim(torch.zeros_like(cube_state[:, 7:]))
    scene.reset()
    return positions


_wrapped_controller_init = base.PickPlaceController.__init__


def _target_controller_init(self, robot, scene) -> None:
    _wrapped_controller_init(self, robot, scene)
    self.cube = scene[CUBE_ASSET_NAMES[ACTIVE_COLOR]]
    self.target_color = ACTIVE_COLOR
    print(f"[EPISODE] Target cube: {self.target_color}")


base.PickPlaceController.__init__ = _target_controller_init


def target_cube_success(cube) -> tuple[bool, str]:
    """Target-specific criterion reusable by an evaluation environment."""
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


def run_three_color_simulator(sim, scene, top_rgb_sensor, wrist_rgb_sensor) -> None:
    global ACTIVE_COLOR
    dt = sim.get_physics_dt()
    recorder = None
    attempt_index = 0
    try:
        while recorder is None or recorder.saved_episodes < TOTAL_EPISODES:
            if not base.simulation_app.is_running():
                break
            saved = 0 if recorder is None else recorder.saved_episodes
            ACTIVE_COLOR = COLORS[saved % len(COLORS)]
            episode_number = saved + 1
            attempt_index += 1
            base.args_cli.task = (
                f"Pick up the {ACTIVE_COLOR} cube and place it in the storage box."
            )
            print(
                f"\n[EPISODE] Starting {episode_number}/{TOTAL_EPISODES} "
                f"target={ACTIVE_COLOR} (attempt {attempt_index})"
            )
            positions = reset_three_color_scene(scene)
            print(f"[EPISODE] Sampled cube xyz={positions}")
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
                    controller, top_rgb_sensor, wrist_rgb_sensor
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
                print(f"[EPISODE] Retrying target={ACTIVE_COLOR}")
                continue
            if controller.state != "done":
                break

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
                    f"target-specific quality check failed for {ACTIVE_COLOR}: {last_detail}"
                )
                print(f"[QUALITY][FAIL] target={ACTIVE_COLOR}; retrying same episode")
                continue

            recorder.record_if_needed(0.0, force=True)
            recorder.save_episode()
            per_color_count = (recorder.saved_episodes + 2) // 3
            print(
                f"[QUALITY][PASS] target={ACTIVE_COLOR}; {last_detail}; "
                f"saved {ACTIVE_COLOR}={per_color_count}/{EPISODES_PER_COLOR}"
            )
    finally:
        if recorder is not None:
            recorder.finalize()


def main() -> None:
    sim = base.SimulationContext(
        base.sim_utils.SimulationCfg(device=base.args_cli.device, dt=1.0 / 120.0)
    )
    sim.set_camera_view(eye=[1.0, -1.35, 0.85], target=[-0.35, -0.2, 0.15])
    scene_cfg = ThreeColorPickPlaceSceneCfg(num_envs=1, env_spacing=1.0)
    # Isaac Lab's @configclass fields are instance attributes, not mutable
    # class attributes.  Correct the inherited paths on this scene instance.
    scene_cfg.table.spawn.usd_path = base.TABLE_USD
    scene_cfg.robot.spawn.usd_path = base.ROBOT_USD
    scene_cfg.storage_box.spawn.usd_path = base.STORAGE_BOX_USD
    scene_cfg.realsense.spawn.usd_path = base.REALSENSE_USD
    scene = base.InteractiveScene(scene_cfg)
    wrist_root_path = base.attach_wrist_realsense(scene)
    sim.reset()
    camera_views = base.create_dual_realsense_views(scene, wrist_root_path)
    top_rgb_sensor = scene["top_color"] if base.args_cli.record_camera else None
    wrist_rgb_sensor = scene["wrist_color"] if base.args_cli.record_camera else None
    print(
        f"[INFO] Three-color balanced dataset: {EPISODES_PER_COLOR} successful episodes/color, "
        f"{TOTAL_EPISODES} total"
    )
    print(
        f"[INFO] Cube ranges: x={base.args_cli.cube_x_range or DEFAULT_X_RANGE}, "
        f"y={base.args_cli.cube_y_range or DEFAULT_Y_RANGE}, "
        f"minimum separation={launcher_args.cube_min_separation:.3f} m"
    )
    run_three_color_simulator(sim, scene, top_rgb_sensor, wrist_rgb_sensor)


if __name__ == "__main__":
    main()
    base.simulation_app.close()
