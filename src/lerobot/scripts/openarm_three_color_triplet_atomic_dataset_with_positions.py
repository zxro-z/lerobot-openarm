#!/usr/bin/env python3
"""Collect red -> blue -> yellow episodes atomically from one shared scene.

One successful layout produces three distinct LeRobot episodes.  The cube
coordinates are sampled once.  Before red, blue, and yellow respectively, the
robot and all three cubes are restored to that same fixed layout.  Therefore
each color starts from an identical scene instead of seeing an earlier target
already left inside the storage box.

Each scene is first written to a temporary three-episode LeRobot dataset.  If
any color fails, the complete temporary dataset is deleted, so a partially
successful red/blue/yellow group can never leak into the final dataset.  Once
all requested groups have succeeded, the temporary datasets are merged into
the requested final dataset.

Cube positions are inspection metadata only.  They are written beside the
dataset as CSV and JSON and are not added to the training feature schema.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
from pathlib import Path


def _parse_triplet_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--position_log_dir",
        type=Path,
        default=None,
        help="Default: DATASET_ROOT_cube_positions.",
    )
    parser.add_argument("--overwrite_position_logs", action="store_true")
    parser.add_argument(
        "--keep_triplet_staging",
        action="store_true",
        help="Keep the accepted temporary triplet datasets after the final merge.",
    )
    return parser.parse_known_args()


triplet_args, generator_argv = _parse_triplet_args()
sys.argv = [sys.argv[0], *generator_argv]

import openarm_table_dual_realsense_ik_pick_place_make_dataset_three_color_random_cube_random_tilt_gripper_mapped_degree as generator  # noqa: E402


base = generator.base
torch = base.torch

# The imported three-color launcher has already converted --num_episodes from
# groups-per-color to total episodes.  Recover the intended number of scenes.
TRIPLETS_TO_SAVE = generator.EPISODES_PER_COLOR
TOTAL_EPISODES = TRIPLETS_TO_SAVE * len(generator.COLORS)
FINAL_ROOT = Path(base.args_cli.dataset_root).expanduser().resolve()
FINAL_REPO_ID = base.args_cli.dataset_repo_id
FINAL_PUSH_TO_HUB = base.args_cli.push_to_hub

POSITION_LOG_DIR = (
    triplet_args.position_log_dir.expanduser().resolve()
    if triplet_args.position_log_dir is not None
    else FINAL_ROOT.with_name(FINAL_ROOT.name + "_cube_positions")
)
STAGING_ROOT = FINAL_ROOT.with_name(FINAL_ROOT.name + "_triplet_staging")


def _prepare_output_path(path: Path, overwrite: bool, description: str) -> None:
    if path.exists():
        if not overwrite:
            raise RuntimeError(
                f"{description} already exists: {path}. "
                "Use --overwrite_dataset (and, for logs, --overwrite_position_logs) "
                "or select another path."
            )
        shutil.rmtree(path)


_prepare_output_path(FINAL_ROOT, base.args_cli.overwrite_dataset, "Final dataset root")
_prepare_output_path(STAGING_ROOT, base.args_cli.overwrite_dataset, "Triplet staging root")
_prepare_output_path(
    POSITION_LOG_DIR,
    base.args_cli.overwrite_dataset or triplet_args.overwrite_position_logs,
    "Position log directory",
)
STAGING_ROOT.mkdir(parents=True)
POSITION_LOG_DIR.mkdir(parents=True)
(POSITION_LOG_DIR / "attempts").mkdir()


TRIPLET_FIELDS = [
    "attempt_index",
    "accepted_triplet_index",
    "status",
    "failed_color",
    "failure_reason",
    "red_sampled_x", "red_sampled_y", "red_sampled_z",
    "blue_sampled_x", "blue_sampled_y", "blue_sampled_z",
    "yellow_sampled_x", "yellow_sampled_y", "yellow_sampled_z",
    "red_initial_x", "red_initial_y", "red_initial_z",
    "blue_initial_x", "blue_initial_y", "blue_initial_z",
    "yellow_initial_x", "yellow_initial_y", "yellow_initial_z",
]
EPISODE_FIELDS = [
    "episode_index",
    "triplet_index",
    "triplet_episode_index",
    "target_color",
    "task",
    "red_pre_x", "red_pre_y", "red_pre_z",
    "blue_pre_x", "blue_pre_y", "blue_pre_z",
    "yellow_pre_x", "yellow_pre_y", "yellow_pre_z",
    "red_post_x", "red_post_y", "red_post_z",
    "blue_post_x", "blue_post_y", "blue_post_z",
    "yellow_post_x", "yellow_post_y", "yellow_post_z",
]
TRIPLET_CSV = POSITION_LOG_DIR / "triplets.csv"
EPISODE_CSV = POSITION_LOG_DIR / "saved_episodes.csv"


def _initialize_csv(path: Path, fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        csv.DictWriter(handle, fieldnames=fields).writeheader()


def _append_csv(path: Path, fields: list[str], row: dict[str, object]) -> None:
    with path.open("a", newline="", encoding="utf-8") as handle:
        csv.DictWriter(handle, fieldnames=fields).writerow(
            {field: row.get(field, "") for field in fields}
        )
        handle.flush()
        os.fsync(handle.fileno())


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _flat_positions(prefix: str, positions: dict[str, list[float]]) -> dict[str, float]:
    result: dict[str, float] = {}
    for color in generator.COLORS:
        for axis, value in zip("xyz", positions[color], strict=True):
            result[f"{color}_{prefix}_{axis}"] = float(value)
    return result


def _read_positions(scene) -> dict[str, list[float]]:
    return {
        color: [float(value) for value in scene[asset].data.root_pos_w[0].detach().cpu().tolist()]
        for color, asset in generator.CUBE_ASSET_NAMES.items()
    }


def _reset_fixed_layout(scene, positions: dict[str, list[float]]) -> None:
    """Reset the robot and restore all cubes to one previously sampled layout."""
    robot = scene["robot"]
    root_state = robot.data.default_root_state.clone()
    root_state[:, :3] += scene.env_origins
    robot.write_root_pose_to_sim(root_state[:, :7])
    robot.write_root_velocity_to_sim(root_state[:, 7:])
    joint_pos = robot.data.default_joint_pos.clone()
    robot.write_joint_state_to_sim(joint_pos, torch.zeros_like(robot.data.default_joint_vel))
    robot.set_joint_position_target(joint_pos)

    for color, asset_name in generator.CUBE_ASSET_NAMES.items():
        cube = scene[asset_name]
        cube_state = cube.data.default_root_state.clone()
        cube_state[:, :3] = torch.tensor(
            positions[color], device=cube_state.device, dtype=cube_state.dtype
        )
        cube_state[:, :3] += scene.env_origins
        cube.write_root_pose_to_sim(cube_state[:, :7])
        cube.write_root_velocity_to_sim(torch.zeros_like(cube_state[:, 7:]))
    scene.reset()


def _step_for(sim, scene, seconds: float) -> None:
    dt = sim.get_physics_dt()
    for _ in range(max(1, int(seconds / dt))):
        scene.write_data_to_sim()
        sim.step()
        scene.update(dt)


def _run_one_color(
    sim,
    scene,
    recorder,
    color: str,
    fixed_positions: dict[str, list[float]],
    top_rgb_sensor,
    wrist_rgb_sensor,
) -> tuple[bool, str, object]:
    generator.ACTIVE_COLOR = color
    base.args_cli.task = f"Pick up the {color} cube and place it in the storage box."
    # Every color episode starts from the exact same sampled drop coordinates.
    # The settling phase is repeated so its observation trajectory is complete.
    _reset_fixed_layout(scene, fixed_positions)
    scene.write_data_to_sim()
    sim.step()
    scene.update(sim.get_physics_dt())
    _step_for(sim, scene, base.args_cli.warmup_time_s)
    pre_positions = _read_positions(scene)
    print(f"[TRIPLET] restored fixed layout for {color}; target_xyz={pre_positions[color]}")

    controller = base.PickPlaceController(scene["robot"], scene)
    controller.triplet_pre_positions = pre_positions
    if recorder is None:
        recorder = base.MultiEpisodeLeRobotRecorder(
            controller, top_rgb_sensor, wrist_rgb_sensor
        )
    else:
        recorder.begin_episode(controller)

    dt = sim.get_physics_dt()
    elapsed = 0.0
    while base.simulation_app.is_running() and controller.state not in ("done", "failed"):
        controller.advance(dt)
        scene.write_data_to_sim()
        sim.step()
        scene.update(dt)
        recorder.record_if_needed(dt)
        elapsed += dt
        if elapsed >= generator.launcher_args.episode_timeout_s:
            controller.fail(
                f"episode exceeded {generator.launcher_args.episode_timeout_s:.2f}s "
                f"in state={controller.state}"
            )

    if controller.state == "failed":
        reason = controller.failure_reason or "IK/controller failure"
        recorder.discard_episode(reason)
        return False, reason, recorder
    if controller.state != "done":
        recorder.discard_episode("simulation stopped before controller completed")
        return False, "simulation stopped before controller completed", recorder

    held_s = 0.0
    checked_s = 0.0
    detail = "not checked"
    while base.simulation_app.is_running() and checked_s < generator.launcher_args.success_timeout_s:
        controller.advance(dt)
        scene.write_data_to_sim()
        sim.step()
        scene.update(dt)
        recorder.record_if_needed(dt)
        passed, detail = generator.target_cube_success(controller.cube)
        held_s = held_s + dt if passed else 0.0
        checked_s += dt
        if held_s >= generator.launcher_args.success_hold_time_s:
            break

    if held_s < generator.launcher_args.success_hold_time_s:
        reason = f"target-specific quality check failed for {color}: {detail}"
        recorder.discard_episode(reason)
        return False, reason, recorder

    recorder.record_if_needed(0.0, force=True)
    recorder.save_episode()
    return True, detail, recorder


def _make_staging_recorder_args(attempt_index: int) -> tuple[Path, str]:
    root = STAGING_ROOT / f"attempt_{attempt_index:06d}"
    repo_id = f"local/openarm_three_color_triplet_{attempt_index:06d}"
    base.args_cli.dataset_root = str(root)
    base.args_cli.dataset_repo_id = repo_id
    base.args_cli.overwrite_dataset = True
    base.args_cli.push_to_hub = False
    return root, repo_id


def _collect(sim, scene, top_rgb_sensor, wrist_rgb_sensor) -> tuple[list[Path], list[str]]:
    accepted_roots: list[Path] = []
    accepted_repo_ids: list[str] = []
    attempt_index = 0
    accepted = 0

    while accepted < TRIPLETS_TO_SAVE and base.simulation_app.is_running():
        attempt_index += 1
        print(
            f"\n[TRIPLET] attempt={attempt_index} accepted={accepted}/{TRIPLETS_TO_SAVE}; "
            "targets=red -> blue -> yellow"
        )
        # Sample this triplet's fixed layout once.  The same coordinates are
        # restored before each of the red, blue and yellow episodes below.
        sampled_raw = generator.reset_three_color_scene(scene)
        sampled = {color: [float(v) for v in sampled_raw[color]] for color in generator.COLORS}
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim.get_physics_dt())
        _step_for(sim, scene, base.args_cli.warmup_time_s)
        initial = _read_positions(scene)

        staging_path, staging_repo_id = _make_staging_recorder_args(attempt_index)
        recorder = None
        episode_logs: list[dict[str, object]] = []
        failed_color = ""
        failure_reason = ""

        try:
            for local_index, color in enumerate(generator.COLORS):
                ok, detail, recorder = _run_one_color(
                    sim,
                    scene,
                    recorder,
                    color,
                    sampled,
                    top_rgb_sensor,
                    wrist_rgb_sensor,
                )
                # Read after reset + settling + trajectory.  Controller setup
                # has already captured the target from this restored layout.
                post = _read_positions(scene)
                pre = getattr(recorder.controller, "triplet_pre_positions", None)
                if pre is None:
                    # Keep a useful fallback for older controller variants.
                    pre = initial
                if not ok:
                    failed_color = color
                    failure_reason = detail
                    print(
                        f"[TRIPLET][REJECT] color={color}; all episodes from this scene "
                        f"will be discarded; reason={detail}"
                    )
                    break
                episode_logs.append(
                    {
                        "triplet_episode_index": local_index,
                        "target_color": color,
                        "task": base.args_cli.task,
                        "pre_positions": pre,
                        "post_positions": post,
                        "quality_detail": detail,
                    }
                )
                print(f"[TRIPLET][PASS] color={color}; {detail}")
        finally:
            if recorder is not None:
                recorder.finalize()

        is_accepted = len(episode_logs) == len(generator.COLORS)
        triplet_row = {
            "attempt_index": attempt_index,
            "accepted_triplet_index": accepted if is_accepted else "",
            "status": "accepted" if is_accepted else "rejected",
            "failed_color": failed_color,
            "failure_reason": failure_reason,
            **_flat_positions("sampled", sampled),
            **_flat_positions("initial", initial),
        }
        _append_csv(TRIPLET_CSV, TRIPLET_FIELDS, triplet_row)
        _write_json(
            POSITION_LOG_DIR / "attempts" / f"attempt_{attempt_index:06d}.json",
            {
                **triplet_row,
                "sampled_positions": sampled,
                "initial_settled_positions": initial,
                "episodes": episode_logs,
            },
        )

        if not is_accepted:
            if staging_path.exists():
                shutil.rmtree(staging_path)
            continue

        accepted_roots.append(staging_path)
        accepted_repo_ids.append(staging_repo_id)
        for local_index, episode_log in enumerate(episode_logs):
            final_episode_index = accepted * 3 + local_index
            row = {
                "episode_index": final_episode_index,
                "triplet_index": accepted,
                "triplet_episode_index": local_index,
                "target_color": episode_log["target_color"],
                "task": episode_log["task"],
                **_flat_positions("pre", episode_log["pre_positions"]),
                **_flat_positions("post", episode_log["post_positions"]),
            }
            _append_csv(EPISODE_CSV, EPISODE_FIELDS, row)
        accepted += 1
        print(
            f"[TRIPLET][ACCEPT] scene saved atomically: {accepted}/{TRIPLETS_TO_SAVE} "
            f"({accepted * 3}/{TOTAL_EPISODES} episodes staged)"
        )

    return accepted_roots, accepted_repo_ids


def _merge_final(roots: list[Path], repo_ids: list[str]) -> None:
    if len(roots) != TRIPLETS_TO_SAVE:
        raise RuntimeError(
            f"Collection stopped with {len(roots)}/{TRIPLETS_TO_SAVE} complete triplets; "
            "the final dataset was not created. Accepted staging data was retained."
        )
    from lerobot.datasets.aggregate import aggregate_datasets

    print(f"[MERGE] Combining {len(roots)} accepted triplets into {FINAL_ROOT}")
    aggregate_datasets(
        repo_ids=repo_ids,
        aggr_repo_id=FINAL_REPO_ID,
        roots=roots,
        aggr_root=FINAL_ROOT,
    )
    if FINAL_PUSH_TO_HUB:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        LeRobotDataset(repo_id=FINAL_REPO_ID, root=FINAL_ROOT).push_to_hub(
            private=base.args_cli.private
        )
        print(f"[MERGE] Pushed final dataset to Hub: {FINAL_REPO_ID}")
    if not triplet_args.keep_triplet_staging:
        shutil.rmtree(STAGING_ROOT)


def main() -> None:
    _initialize_csv(TRIPLET_CSV, TRIPLET_FIELDS)
    _initialize_csv(EPISODE_CSV, EPISODE_FIELDS)
    _write_json(
        POSITION_LOG_DIR / "manifest.json",
        {
            "dataset_root": str(FINAL_ROOT),
            "dataset_repo_id": FINAL_REPO_ID,
            "triplets_requested": TRIPLETS_TO_SAVE,
            "episodes_expected": TOTAL_EPISODES,
            "target_order": list(generator.COLORS),
            "atomic_policy": "Reject all three episodes if any target fails.",
            "scene_policy": (
                "Sample one layout per triplet; restore robot and all three cubes "
                "to that fixed layout before every color episode."
            ),
            "coordinate_frame": "world",
            "units": "metres",
            "training_schema_includes_positions": False,
        },
    )

    sim = base.SimulationContext(
        base.sim_utils.SimulationCfg(device=base.args_cli.device, dt=1.0 / 120.0)
    )
    sim.set_camera_view(eye=[1.0, -1.35, 0.85], target=[-0.35, -0.2, 0.15])
    scene_cfg = generator.ThreeColorPickPlaceSceneCfg(num_envs=1, env_spacing=1.0)
    scene_cfg.table.spawn.usd_path = base.TABLE_USD
    scene_cfg.robot.spawn.usd_path = base.ROBOT_USD
    scene_cfg.storage_box.spawn.usd_path = base.STORAGE_BOX_USD
    scene_cfg.realsense.spawn.usd_path = base.REALSENSE_USD
    scene = base.InteractiveScene(scene_cfg)
    wrist_root_path = base.attach_wrist_realsense(scene)
    sim.reset()
    base.create_dual_realsense_views(scene, wrist_root_path)
    top_rgb_sensor = scene["top_color"] if base.args_cli.record_camera else None
    wrist_rgb_sensor = scene["wrist_color"] if base.args_cli.record_camera else None

    print(
        f"[INFO] Atomic shared-scene collection: {TRIPLETS_TO_SAVE} triplets, "
        f"{TOTAL_EPISODES} final episodes"
    )
    print(f"[INFO] Position logs: {POSITION_LOG_DIR}")
    accepted_roots, accepted_repo_ids = _collect(
        sim, scene, top_rgb_sensor, wrist_rgb_sensor
    )
    _merge_final(accepted_roots, accepted_repo_ids)
    print(f"[RESULT] Final dataset: {FINAL_ROOT}")
    print(f"[RESULT] Cube-position logs: {POSITION_LOG_DIR}")


if __name__ == "__main__":
    try:
        main()
    finally:
        base.simulation_app.close()
