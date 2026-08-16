#!/usr/bin/env python3
"""Run three-color closed-loop SmolVLA evaluation in the OpenArm Isaac Lab scene."""

from __future__ import annotations

import argparse
import csv
import os
import re
import traceback
from collections import Counter
from pathlib import Path

from diagnostic_metrics import compute_episode_diagnostics
from lerobot.utils.random_utils import set_seed


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_ROOT = PROJECT_ROOT / "src/lerobot/datasets/openarm_three_color_transit_tilt_50"
COLOR_NAMES = ("red", "blue", "yellow")
TASKS_BY_COLOR = {
    "red": "Pick up the red cube and place it in the storage box.",
    "blue": "Pick up the blue cube and place it in the storage box.",
    "yellow": "Pick up the yellow cube and place it in the storage box.",
}
APPROACH_DISTANCE_M = 0.08
SELECTION_MARGIN_M = 0.02
SUSTAINED_APPROACH_FRAMES = 5
GRASP_DISTANCE_M = 0.065
LIFT_Z_THRESHOLD_M = 0.03
FOLLOW_DISTANCE_M = 0.08
FOLLOW_DISPLACEMENT_M = 0.02
TRANSPORT_DISTANCE_DELTA_M = 0.08


def infer_instruction_color(task_text: str) -> str | None:
    match = re.search(r"\b(red|blue|yellow)\b cube", task_text.lower())
    return None if match is None else match.group(1)


def get_video_frame(observation: dict[str, object], video_camera: str):
    if video_camera == "top":
        return observation["observation.images.top"]
    if video_camera == "wrist":
        return observation["observation.images.wrist"]
    raise ValueError(f"Unsupported video_camera={video_camera!r}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-path", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--dataset-repo-id", default="local/openarm_three_color_transit_tilt_50")
    parser.add_argument("--num-episodes-per-color", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--use-amp", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--progress-every", type=int, default=1)
    parser.add_argument(
        "--record-video",
        "--save-video",
        dest="record_video",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--video-dir", type=Path, default=Path("outputs/eval/openarm_three_color_smolvla/videos"))
    parser.add_argument("--video-camera", choices=["top", "wrist"], default="top")
    parser.add_argument("--video-fps", type=int, default=None)
    parser.add_argument("--success-video-tail-seconds", type=float, default=0.0)
    parser.add_argument("--instruction-order", choices=["grouped", "cycle"], default="grouped")
    parser.add_argument("--min-steps-before-success", type=int, default=50)
    parser.add_argument("--debug-success", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--target-color", choices=list(COLOR_NAMES), default=None)
    parser.add_argument("--eval-seeds", default=None)
    parser.add_argument("--diagnostic-instrumentation", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--diagnostic-output-dir", type=Path, default=None)
    parser.add_argument("--reseed-per-episode", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/eval/openarm_three_color_smolvla/results.csv"))
    return parser.parse_args()


def build_instruction_schedule(num_episodes_per_color: int, mode: str) -> list[str]:
    if mode == "grouped":
        return [color for color in COLOR_NAMES for _ in range(num_episodes_per_color)]
    schedule: list[str] = []
    for episode_idx in range(num_episodes_per_color * len(COLOR_NAMES)):
        schedule.append(COLOR_NAMES[episode_idx % len(COLOR_NAMES)])
    return schedule


def parse_eval_seeds(raw_value: str | None) -> list[int] | None:
    if raw_value is None:
        return None
    seeds = [int(part.strip()) for part in raw_value.split(",") if part.strip()]
    if not seeds:
        raise ValueError("--eval-seeds was provided but no seeds were parsed.")
    return seeds


def select_episode_schedule(args: argparse.Namespace) -> list[tuple[int, str]]:
    explicit_seeds = parse_eval_seeds(args.eval_seeds)
    if explicit_seeds is not None:
        if args.target_color is None:
            raise ValueError("--target-color is required when --eval-seeds is used.")
        return [(seed, args.target_color) for seed in explicit_seeds]

    episode_colors = build_instruction_schedule(args.num_episodes_per_color, args.instruction_order)
    if args.target_color is not None:
        episode_colors = [args.target_color for _ in range(args.num_episodes_per_color)]
    return [(args.seed + episode_id, color) for episode_id, color in enumerate(episode_colors)]


def infer_approach(distances: dict[str, float]) -> tuple[str, bool]:
    ordered = sorted(distances.items(), key=lambda item: item[1])
    nearest_color, nearest_distance = ordered[0]
    second_distance = ordered[1][1]
    is_clear = (
        nearest_distance <= APPROACH_DISTANCE_M
        and nearest_distance + SELECTION_MARGIN_M < second_distance
    )
    return (nearest_color if is_clear else "none"), is_clear


def summarize_episode_diagnostics(
    episode_id: int,
    seed: int,
    target_color: str,
    task_text: str,
    step_rows: list[dict[str, object]],
    result_row: dict[str, object],
    max_steps: int,
) -> dict[str, object]:
    return compute_episode_diagnostics(
        episode_id=episode_id,
        seed=seed,
        target_color=target_color,
        task_text=task_text,
        step_rows=step_rows,
        result_row=result_row,
        max_steps=max_steps,
        approach_distance_m=APPROACH_DISTANCE_M,
        selection_margin_m=SELECTION_MARGIN_M,
        sustained_approach_frames=SUSTAINED_APPROACH_FRAMES,
        grasp_distance_m=GRASP_DISTANCE_M,
        lift_z_threshold_m=LIFT_Z_THRESHOLD_M,
        follow_distance_m=FOLLOW_DISTANCE_M,
        follow_displacement_m=FOLLOW_DISPLACEMENT_M,
        transport_distance_delta_m=TRANSPORT_DISTANCE_DELTA_M,
    )


def main() -> None:
    args = parse_args()
    os.environ["OPENARM_SMOLVLA_HEADLESS"] = "1" if args.headless else "0"
    output = args.output.expanduser().resolve()
    print(f"[EVAL_STAGE] args_parsed headless={args.headless} device={args.device}", flush=True)

    print("[EVAL_STAGE] importing_openarm_env", flush=True)
    from openarm_smolvla_env import (
        ACTION_DIM,
        CUBE_SIZE,
        OBSERVATION_KEYS,
        ROBOT_TYPE,
        STORAGE_BOX_POS,
        OpenArmEnv,
        simulation_app,
    )
    print("[EVAL_STAGE] imported_openarm_env", flush=True)

    import numpy as np
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
    from lerobot.policies.factory import make_policy, make_pre_post_processors
    from lerobot.utils.control_utils import predict_action
    from lerobot.utils.io_utils import write_video

    policy_path = args.policy_path.expanduser().resolve()
    if not (policy_path / "config.json").is_file():
        raise FileNotFoundError(f"Checkpoint config.json not found: {policy_path}")

    dataset_root = args.dataset_root.expanduser().resolve()
    print(f"[EVAL_STAGE] loading_dataset_metadata root={dataset_root}", flush=True)
    metadata = LeRobotDatasetMetadata(args.dataset_repo_id, root=dataset_root)
    print("[EVAL_STAGE] loading_policy_config", flush=True)
    policy_cfg = PreTrainedConfig.from_pretrained(str(policy_path))
    policy_cfg.pretrained_path = policy_path
    policy_cfg.device = args.device
    policy_cfg.use_amp = args.use_amp

    print("[EVAL_STAGE] building_policy", flush=True)
    policy = make_policy(policy_cfg, ds_meta=metadata)
    policy.eval()
    device = next(policy.parameters()).device
    preprocessor, postprocessor = make_pre_post_processors(policy_cfg=policy_cfg, pretrained_path=str(policy_path))

    print(f"[EVAL DEBUG] checkpoint_path={policy_path}", flush=True)
    print(f"[EVAL DEBUG] observation_feature_keys={list(OBSERVATION_KEYS)}", flush=True)
    print("[EVAL DEBUG] state_dimension=8", flush=True)
    print(f"[EVAL DEBUG] action_dimension={ACTION_DIM}", flush=True)
    print(f"[EVAL DEBUG] min_steps_before_success={args.min_steps_before_success}", flush=True)
    video_fps = metadata.fps if args.video_fps is None else args.video_fps
    print(f"[EVAL DEBUG] video_camera={args.video_camera} video_fps={video_fps}", flush=True)

    print("[EVAL_STAGE] creating_env", flush=True)
    env = OpenArmEnv(
        max_episode_steps=args.max_steps,
        seed=args.seed,
        min_steps_before_success=args.min_steps_before_success,
        debug_success=args.debug_success,
        diagnostic_instrumentation=args.diagnostic_instrumentation,
    )
    print("[EVAL_STAGE] env_ready", flush=True)
    print(f"[EVAL DEBUG] env_config={env.describe_configuration()}", flush=True)

    episode_schedule = select_episode_schedule(args)
    video_dir = args.video_dir.expanduser().resolve()
    if args.record_video:
        video_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, object]] = []
    diagnostic_step_rows: list[dict[str, object]] = []
    diagnostic_summary_rows: list[dict[str, object]] = []
    diagnostic_output_dir = (
        args.diagnostic_output_dir.expanduser().resolve()
        if args.diagnostic_output_dir is not None
        else None
    )
    if args.diagnostic_instrumentation and diagnostic_output_dir is None:
        raise ValueError("--diagnostic-output-dir is required with --diagnostic-instrumentation")
    if diagnostic_output_dir is not None:
        diagnostic_output_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"[EVAL DEBUG] approach_distance_m={APPROACH_DISTANCE_M} "
        f"selection_margin_m={SELECTION_MARGIN_M} sustained_frames={SUSTAINED_APPROACH_FRAMES} "
        f"grasp_distance_m={GRASP_DISTANCE_M} lift_z_threshold_m={LIFT_Z_THRESHOLD_M} "
        f"transport_delta_m={TRANSPORT_DISTANCE_DELTA_M}",
        flush=True,
    )

    try:
        for episode_id, (seed, target_color) in enumerate(episode_schedule):
            task_text = TASKS_BY_COLOR[target_color]
            instruction_color = infer_instruction_color(task_text)
            if args.reseed_per_episode:
                set_seed(seed)
                print(f"[EVAL_STAGE] reseeded_global_rng seed={seed}", flush=True)
            print(f"[EVAL_STAGE] reset episode={episode_id} target_color={target_color} seed={seed}", flush=True)
            obs, reset_info = env.reset(seed=seed)
            policy.reset()
            preprocessor.reset()
            postprocessor.reset()
            success = False
            termination_reason = "max_steps"
            picked_color = "failure"
            step = -1
            frames: list[np.ndarray] = []
            video_relpath = ""
            episode_step_rows: list[dict[str, object]] = []
            if args.record_video:
                frames.append(np.asarray(get_video_frame(obs, args.video_camera)).copy())

            try:
                for step in range(args.max_steps):
                    if step % max(args.progress_every, 1) == 0:
                        print(f"[EVAL_STEP] episode={episode_id} step={step}", flush=True)
                    action = predict_action(
                        observation=obs,
                        policy=policy,
                        device=device,
                        preprocessor=preprocessor,
                        postprocessor=postprocessor,
                        use_amp=args.use_amp,
                        task=task_text,
                        robot_type=ROBOT_TYPE,
                    )
                    action_np = np.asarray(action.squeeze(0).cpu(), dtype=np.float32)
                    if action_np.shape != (8,):
                        raise RuntimeError(f"Expected policy action (8,), got {action_np.shape}")
                    obs, _, terminated, truncated, info = env.step(action_np)
                    if args.record_video:
                        frames.append(np.asarray(get_video_frame(obs, args.video_camera)).copy())
                    success = bool(info.get("is_success", False))
                    picked_color = str(info.get("picked_color", "failure"))
                    if args.diagnostic_instrumentation:
                        diagnostic_state = info.get("diagnostic_state")
                        if diagnostic_state is None:
                            raise RuntimeError("Diagnostic instrumentation enabled but env did not return diagnostic_state")
                        ee_position = diagnostic_state["ee_position"]
                        cube_positions = diagnostic_state["cube_positions"]
                        distances = {
                            color: float(np.linalg.norm(np.asarray(ee_position) - np.asarray(cube_positions[color])))
                            for color in COLOR_NAMES
                        }
                        nearest_cube = min(distances, key=distances.get)
                        approach_color, approach_is_clear = infer_approach(distances)
                        cube_displacements = {
                            color: float(
                                np.linalg.norm(
                                    np.asarray(cube_positions[color]) - np.asarray(diagnostic_state["initial_positions"][color])
                                )
                            )
                            for color in COLOR_NAMES
                        }
                        cube_delta_z = {
                            color: float(cube_positions[color][2] - diagnostic_state["initial_positions"][color][2])
                            for color in COLOR_NAMES
                        }
                        cube_box_distances = {
                            color: float(
                                np.linalg.norm(
                                    np.asarray(cube_positions[color][:2]) - np.asarray(diagnostic_state["storage_box_position"][:2])
                                )
                            )
                            for color in COLOR_NAMES
                        }
                        step_row = {
                            "episode_id": episode_id,
                            "seed": seed,
                            "step": step,
                            "target_color": target_color,
                            "task_text": task_text,
                            "ee_frame": diagnostic_state["frame"],
                            "ee_x": ee_position[0],
                            "ee_y": ee_position[1],
                            "ee_z": ee_position[2],
                            "red_x": cube_positions["red"][0],
                            "red_y": cube_positions["red"][1],
                            "red_z": cube_positions["red"][2],
                            "blue_x": cube_positions["blue"][0],
                            "blue_y": cube_positions["blue"][1],
                            "blue_z": cube_positions["blue"][2],
                            "yellow_x": cube_positions["yellow"][0],
                            "yellow_y": cube_positions["yellow"][1],
                            "yellow_z": cube_positions["yellow"][2],
                            "dist_ee_red": distances["red"],
                            "dist_ee_blue": distances["blue"],
                            "dist_ee_yellow": distances["yellow"],
                            "nearest_cube": nearest_cube,
                            "nearest_cube_distance": distances[nearest_cube],
                            "approach_color": approach_color,
                            "approach_is_clear": approach_is_clear,
                            "gripper_action_deg": diagnostic_state["gripper_action_deg"],
                            "gripper_state_deg": diagnostic_state["gripper_state_deg"],
                            "gripper_sim_joint_deg": diagnostic_state["gripper_sim_joint_deg"],
                            "policy_action_j0_deg": diagnostic_state["policy_action_deg"][0],
                            "policy_action_j1_deg": diagnostic_state["policy_action_deg"][1],
                            "policy_action_j2_deg": diagnostic_state["policy_action_deg"][2],
                            "policy_action_j3_deg": diagnostic_state["policy_action_deg"][3],
                            "policy_action_j4_deg": diagnostic_state["policy_action_deg"][4],
                            "policy_action_j5_deg": diagnostic_state["policy_action_deg"][5],
                            "policy_action_j6_deg": diagnostic_state["policy_action_deg"][6],
                            "policy_action_gripper_deg": diagnostic_state["policy_action_deg"][7],
                            "disp_red": cube_displacements["red"],
                            "disp_blue": cube_displacements["blue"],
                            "disp_yellow": cube_displacements["yellow"],
                            "red_delta_z": cube_delta_z["red"],
                            "blue_delta_z": cube_delta_z["blue"],
                            "yellow_delta_z": cube_delta_z["yellow"],
                            "red_box_distance": cube_box_distances["red"],
                            "blue_box_distance": cube_box_distances["blue"],
                            "yellow_box_distance": cube_box_distances["yellow"],
                        }
                        episode_step_rows.append(step_row)
                    if terminated or truncated:
                        termination_reason = str(info.get("termination_reason", "max_steps"))
                        if truncated and not success:
                            termination_reason = "max_steps"
                        if success and args.record_video and args.success_video_tail_seconds > 0:
                            tail_steps = max(int(round(args.success_video_tail_seconds * video_fps)), 0)
                            for _ in range(tail_steps):
                                obs, _, _, _, _ = env.step(action_np)
                                frames.append(np.asarray(get_video_frame(obs, args.video_camera)).copy())
                        break
            finally:
                if args.record_video and frames:
                    status_label = "success" if success else termination_reason
                    video_name = (
                        f"episode_{episode_id:03d}_target_{target_color}_picked_{picked_color}_{status_label}.mp4"
                    )
                    video_path = video_dir / video_name
                    write_video(video_path, frames, fps=video_fps)
                    video_relpath = os.path.relpath(video_path, start=output.parent)

            color_correct = bool(success and picked_color == instruction_color)
            row: dict[str, object] = {
                "episode_id": episode_id,
                "seed": seed,
                "instruction": task_text,
                "target_color": target_color,
                "picked_color": picked_color,
                "task_success": success,
                "color_correct": color_correct,
                "termination_reason": termination_reason,
                "steps": step + 1,
                "video_path": video_relpath,
            }
            for color in COLOR_NAMES:
                row[f"{color}_initial_x"] = reset_info.get(f"{color}_initial_x")
                row[f"{color}_initial_y"] = reset_info.get(f"{color}_initial_y")
                row[f"{color}_initial_z"] = reset_info.get(f"{color}_initial_z")
                row[f"{color}_final_x"] = info.get(f"{color}_final_x")
                row[f"{color}_final_y"] = info.get(f"{color}_final_y")
                row[f"{color}_final_z"] = info.get(f"{color}_final_z")
            results.append(row)
            if args.diagnostic_instrumentation:
                diagnostic_summary_rows.append(
                    summarize_episode_diagnostics(
                        episode_id=episode_id,
                        seed=seed,
                        target_color=target_color,
                        task_text=task_text,
                        step_rows=episode_step_rows,
                        result_row=row,
                        max_steps=args.max_steps,
                    )
                )
                diagnostic_step_rows.extend(episode_step_rows)
            print(
                f"[EVAL] episode={episode_id} target={target_color} picked={picked_color} "
                f"success={success} color_correct={color_correct}",
                flush=True,
            )
    except Exception as exc:
        print(f"[EVAL_ERROR] {type(exc).__name__}: {exc}", flush=True)
        traceback.print_exc()
        raise
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "episode_id",
            "seed",
            "instruction",
            "target_color",
            "picked_color",
            "task_success",
            "color_correct",
            "termination_reason",
            "steps",
            "video_path",
            "red_initial_x",
            "red_initial_y",
            "red_initial_z",
            "blue_initial_x",
            "blue_initial_y",
            "blue_initial_z",
            "yellow_initial_x",
            "yellow_initial_y",
            "yellow_initial_z",
            "red_final_x",
            "red_final_y",
            "red_final_z",
            "blue_final_x",
            "blue_final_y",
            "blue_final_z",
            "yellow_final_x",
            "yellow_final_y",
            "yellow_final_z",
        ]
        with output.open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        if args.diagnostic_instrumentation:
            steps_path = diagnostic_output_dir / "instrumentation_steps.csv"
            summary_path = diagnostic_output_dir / "instrumentation_summary.csv"
            with steps_path.open("w", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=list(diagnostic_step_rows[0].keys()))
                writer.writeheader()
                writer.writerows(diagnostic_step_rows)
            with summary_path.open("w", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=list(diagnostic_summary_rows[0].keys()))
                writer.writeheader()
                writer.writerows(diagnostic_summary_rows)
            blue_rows = [row for row in diagnostic_summary_rows if row["target_color"] == "blue"]
            if blue_rows:
                selection_correct = sum(bool(row["selection_correct"]) for row in blue_rows)
                wrong_color_selection = sum(row["auto_failure_stage"] == "wrong_color_selection" for row in blue_rows)
                no_clear_selection = sum(row["auto_failure_stage"] == "no_clear_selection" for row in blue_rows)
                close_near_blue = sum(bool(row["close_near_target"]) for row in blue_rows)
                blue_grasp = sum(bool(row["grasp_candidate"]) and row["grasped_color_candidate"] == "blue" for row in blue_rows)
                blue_lift = sum(bool(row["blue_lift_detected"]) for row in blue_rows)
                blue_transport = sum(bool(row["blue_transport_detected"]) for row in blue_rows)
                blue_successes = sum(bool(row["task_success"]) for row in blue_rows)
                print("[BLUE TARGET DIAGNOSTIC]", flush=True)
                print(f"episodes {len(blue_rows)}", flush=True)
                print(f"clear correct selection {selection_correct}", flush=True)
                print(f"wrong-color selection {wrong_color_selection}", flush=True)
                print(f"no-clear selection {no_clear_selection}", flush=True)
                print(f"correct blue approach {sum(bool(row['blue_approach_detected']) for row in blue_rows)}", flush=True)
                print(f"close near blue {close_near_blue}", flush=True)
                print(f"blue grasp candidate {blue_grasp}", flush=True)
                print(f"blue lift {blue_lift}", flush=True)
                print(f"blue transport {blue_transport}", flush=True)
                print(f"task success {blue_successes}", flush=True)
                grasp_given_selection = (blue_grasp / selection_correct) if selection_correct else 0.0
                lift_given_grasp = (blue_lift / blue_grasp) if blue_grasp else 0.0
                transport_given_lift = (blue_transport / blue_lift) if blue_lift else 0.0
                print(f"selection_accuracy={selection_correct}/{len(blue_rows)}", flush=True)
                print(f"grasp_given_correct_selection={blue_grasp}/{selection_correct if selection_correct else 1} ({grasp_given_selection:.3f})", flush=True)
                print(f"lift_given_grasp={blue_lift}/{blue_grasp if blue_grasp else 1} ({lift_given_grasp:.3f})", flush=True)
                print(f"transport_given_lift={blue_transport}/{blue_lift if blue_lift else 1} ({transport_given_lift:.3f})", flush=True)
            print(f"[RESULT] diagnostic_steps_csv={steps_path}", flush=True)
            print(f"[RESULT] diagnostic_summary_csv={summary_path}", flush=True)
        successes = sum(bool(row["task_success"]) for row in results)
        color_correct = sum(bool(row["color_correct"]) for row in results)
        print(f"[RESULT] task_successes={successes}/{len(results)} ({successes / len(results):.1%})")
        print(f"[RESULT] color_correct={color_correct}/{len(results)} ({color_correct / len(results):.1%})")
        print(f"[RESULT] csv={output}")
    finally:
        print("[EVAL_STAGE] closing_env", flush=True)
        env.close()
        simulation_app.close()
        print("[EVAL_STAGE] closed_env", flush=True)


if __name__ == "__main__":
    main()
