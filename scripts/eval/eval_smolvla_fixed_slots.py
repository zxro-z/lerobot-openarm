#!/usr/bin/env python3
"""Run fixed-slot closed-loop SmolVLA evaluation in the OpenArm Isaac Lab scene."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import traceback
from pathlib import Path

from diagnostic_metrics import compute_episode_diagnostics
from lerobot.utils.random_utils import set_seed


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_ROOT = PROJECT_ROOT / "src/lerobot/datasets/openarm_three_color_fixed_slots_perm_tilt50_r3"
DEFAULT_OUTPUT = Path("outputs/eval/openarm_three_color_fixed_slots_perm_tilt50_r3/results.csv")
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
REQUIRED_OBS_KEYS = ("observation.images.top", "observation.images.wrist", "observation.state")


def load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text())


def summarize_value(value) -> dict[str, object]:
    import numpy as np

    shape = tuple(getattr(value, "shape", ()))
    dtype = str(getattr(value, "dtype", type(value).__name__))
    device = str(getattr(value, "device", "cpu"))
    if hasattr(value, "detach"):
        arr = value.detach().float().cpu().numpy()
    else:
        arr = np.asarray(value)
    if arr.size == 0:
        return {
            "shape": shape,
            "dtype": dtype,
            "device": device,
            "min": None,
            "max": None,
            "mean": None,
            "std": None,
        }
    arr = arr.astype(np.float32, copy=False)
    return {
        "shape": shape,
        "dtype": dtype,
        "device": device,
        "min": float(arr.min()),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
    }


def print_value_summary(header: str, value) -> None:
    summary = summarize_value(value)
    print(header, flush=True)
    print(f"  shape={summary['shape']}", flush=True)
    print(f"  dtype={summary['dtype']}", flush=True)
    print(f"  device={summary['device']}", flush=True)
    print(f"  min={summary['min']}", flush=True)
    print(f"  max={summary['max']}", flush=True)
    print(f"  mean={summary['mean']}", flush=True)
    print(f"  std={summary['std']}", flush=True)


def tensor_to_uint8_image(value) -> object:
    import numpy as np

    arr = value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)
    while arr.ndim > 3:
        arr = arr[0]
    if arr.ndim != 3:
        raise ValueError(f"Expected image tensor with 3 dims after squeezing, got shape {arr.shape}")
    if arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
        arr = np.transpose(arr, (1, 2, 0))
    if arr.dtype != np.uint8:
        arr = arr.astype(np.float32)
        if arr.size and float(arr.max()) <= 1.0:
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def save_debug_image(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = tensor_to_uint8_image(value)
    try:
        import imageio.v3 as iio

        iio.imwrite(path, image)
    except Exception:
        from PIL import Image

        Image.fromarray(image).save(path)


def print_feature_block(title: str, feature: dict[str, object] | None) -> None:
    print(title, flush=True)
    if feature is None:
        print("  MISSING", flush=True)
        return
    print(f"  type={feature.get('type', feature.get('dtype'))}", flush=True)
    print(f"  shape={feature.get('shape')}", flush=True)


def print_checkpoint_and_dataset_contract(policy_path: Path, dataset_root: Path) -> None:
    cfg = load_json(policy_path / "config.json")
    dataset_info = load_json(dataset_root / "meta" / "info.json")
    preprocessor_cfg = load_json(policy_path / "policy_preprocessor.json")
    input_features = cfg.get("input_features", {})
    output_features = cfg.get("output_features", {})
    dataset_features = dataset_info.get("features", {})

    print("CHECKPOINT EXPECTED FEATURES", flush=True)
    print_feature_block("top:", input_features.get("observation.images.top"))
    print_feature_block("wrist:", input_features.get("observation.images.wrist"))
    print_feature_block("state:", input_features.get("observation.state"))
    print_feature_block("action:", output_features.get("action"))

    print("[TRAIN_DATASET_CONTRACT]", flush=True)
    print_feature_block("top:", dataset_features.get("observation.images.top"))
    print_feature_block("wrist:", dataset_features.get("observation.images.wrist"))
    print_feature_block("state:", dataset_features.get("observation.state"))
    print_feature_block("action:", dataset_features.get("action"))
    print("[PREPROCESSOR_STEPS]", flush=True)
    for step in preprocessor_cfg.get("steps", []):
        print(f"  {step['registry_name']}", flush=True)
        if step["registry_name"] == "rename_observations_processor":
            print(f"    rename_map={step.get('config', {}).get('rename_map', {})}", flush=True)


def print_runtime_obs_keys(tag: str, observation: dict[str, object]) -> None:
    print(f"[{tag}]", flush=True)
    for key in observation:
        print(key, flush=True)
    for key in REQUIRED_OBS_KEYS:
        if key in observation:
            print_value_summary(f"{tag}:{key}", observation[key])
        else:
            print(f"{tag}:{key}\n  MISSING", flush=True)


def mean_abs_diff(current, previous) -> float:
    import numpy as np

    if hasattr(current, "detach"):
        current = current.detach().float().cpu().numpy()
    else:
        current = np.asarray(current, dtype=np.float32)
    if hasattr(previous, "detach"):
        previous = previous.detach().float().cpu().numpy()
    else:
        previous = np.asarray(previous, dtype=np.float32)
    return float(np.mean(np.abs(current - previous)))


def print_camera_flow(env) -> None:
    print("[CAMERA_FLOW]", flush=True)
    camera_paths = env.describe_camera_paths()
    for key in ("observation.images.top", "observation.images.wrist"):
        mapping = camera_paths[key]
        print(key, flush=True)
        print(f"  wrapper camera object={type(env.camera_top).__name__ if key.endswith('top') else type(env.camera_wrist).__name__}", flush=True)
        print(f"  wrapper prim path={mapping['wrapper_prim_path']}", flush=True)
        print(f"  record/source prim path={mapping['record_camera_prim_path']}", flush=True)
        print(f"  source prim path={mapping['source_camera_prim_path']}", flush=True)


def infer_instruction_color(task_text: str) -> str | None:
    match = re.search(r"\b(red|blue|yellow)\b cube", task_text.lower())
    return None if match is None else match.group(1)


def get_video_frame(observation: dict[str, object], video_camera: str):
    if video_camera == "top":
        return observation["observation.images.top"]
    if video_camera == "wrist":
        return observation["observation.images.wrist"]
    raise ValueError(f"Unsupported video_camera={video_camera!r}")


def parse_comma_ints(raw_value: str | None) -> list[int] | None:
    if raw_value is None:
        return None
    values = [int(part.strip()) for part in raw_value.split(",") if part.strip()]
    if not values:
        raise ValueError("Expected at least one integer.")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-path", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--dataset-repo-id", default="local/openarm_three_color_fixed_slots_perm_tilt50_r3")
    parser.add_argument("--repeats-per-permutation", type=int, default=3)
    parser.add_argument("--permutation-ids", default=None)
    parser.add_argument("--repeat-indices", default=None)
    parser.add_argument("--layout-ids", default=None)
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
    parser.add_argument("--video-dir", type=Path, default=Path("outputs/eval/openarm_three_color_fixed_slots/videos"))
    parser.add_argument("--video-camera", choices=["top", "wrist"], default="top")
    parser.add_argument("--video-fps", type=int, default=None)
    parser.add_argument("--success-video-tail-seconds", type=float, default=0.0)
    parser.add_argument("--min-steps-before-success", type=int, default=50)
    parser.add_argument("--debug-success", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--diagnostic-instrumentation", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--diagnostic-output-dir", type=Path, default=None)
    parser.add_argument("--camera-runtime-diagnostic", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--dump-camera-debug", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--camera-debug-dir", type=Path, default=Path("/tmp/smolvla_eval_camera_debug"))
    parser.add_argument("--reseed-per-episode", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def load_layout_specs(dataset_root: Path) -> list[dict[str, object]]:
    manifest_path = dataset_root / "triplet_manifest.csv"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Fixed-slot triplet manifest not found: {manifest_path}")

    layouts: dict[int, dict[str, object]] = {}
    with manifest_path.open(newline="") as file:
        for row in csv.DictReader(file):
            layout_id = int(row["layout_id"])
            layout = layouts.setdefault(
                layout_id,
                {
                    "layout_id": layout_id,
                    "permutation_id": int(row["permutation_id"]),
                    "repeat_index": int(row["repeat_index"]),
                    "slot_a_color": row["slot_a_color"],
                    "slot_b_color": row["slot_b_color"],
                    "slot_c_color": row["slot_c_color"],
                    "triplet_order": row.get("triplet_order", ""),
                    "valid_triplet": row.get("valid_triplet", ""),
                },
            )

    result = sorted(layouts.values(), key=lambda row: (int(row["permutation_id"]), int(row["repeat_index"]), int(row["layout_id"])))
    for layout in result:
        if str(layout["triplet_order"]).replace(" ", "") != ",".join(COLOR_NAMES):
            raise ValueError(
                f"layout_id={layout['layout_id']} expected triplet_order {','.join(COLOR_NAMES)}, "
                f"got {layout['triplet_order']}"
            )
        if str(layout["valid_triplet"]).lower() != "true":
            raise ValueError(f"layout_id={layout['layout_id']} is not marked valid_triplet=true")
        del layout["triplet_order"]
        del layout["valid_triplet"]
    return result


def select_layouts(args: argparse.Namespace, layouts: list[dict[str, object]]) -> list[dict[str, object]]:
    permutation_filter = parse_comma_ints(args.permutation_ids)
    repeat_filter = parse_comma_ints(args.repeat_indices)
    layout_filter = parse_comma_ints(args.layout_ids)

    selected = []
    for layout in layouts:
        if layout_filter is not None and int(layout["layout_id"]) not in layout_filter:
            continue
        if permutation_filter is not None and int(layout["permutation_id"]) not in permutation_filter:
            continue
        if repeat_filter is not None and int(layout["repeat_index"]) not in repeat_filter:
            continue
        if int(layout["repeat_index"]) >= args.repeats_per_permutation:
            continue
        selected.append(layout)

    if not selected:
        raise ValueError("No layouts selected for fixed-slot eval.")
    return selected


def build_episode_schedule(layouts: list[dict[str, object]]) -> list[dict[str, object]]:
    schedule: list[dict[str, object]] = []
    for layout in layouts:
        for target_color in COLOR_NAMES:
            schedule.append(
                {
                    "layout_id": int(layout["layout_id"]),
                    "permutation_id": int(layout["permutation_id"]),
                    "repeat_index": int(layout["repeat_index"]),
                    "target_color": target_color,
                    "slot_a_color": str(layout["slot_a_color"]),
                    "slot_b_color": str(layout["slot_b_color"]),
                    "slot_c_color": str(layout["slot_c_color"]),
                }
            )
    return schedule


def infer_approach(distances: dict[str, float]) -> tuple[str, bool]:
    ordered = sorted(distances.items(), key=lambda item: item[1])
    nearest_color, nearest_distance = ordered[0]
    second_distance = ordered[1][1]
    is_clear = nearest_distance <= APPROACH_DISTANCE_M and nearest_distance + SELECTION_MARGIN_M < second_distance
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

    from openarm_smolvla_fixed_slots_env import (
        ACTION_DIM,
        OBSERVATION_KEYS,
        ROBOT_TYPE,
        OpenArmFixedSlotsEnv,
        simulation_app,
    )

    from contextlib import nullcontext

    import numpy as np
    import torch
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
    from lerobot.policies.factory import make_policy, make_pre_post_processors
    from lerobot.utils.io_utils import write_video
    from lerobot.policies.utils import prepare_observation_for_inference

    policy_path = args.policy_path.expanduser().resolve()
    if not (policy_path / "config.json").is_file():
        raise FileNotFoundError(f"Checkpoint config.json not found: {policy_path}")

    dataset_root = args.dataset_root.expanduser().resolve()
    layouts = select_layouts(args, load_layout_specs(dataset_root))
    episode_schedule = build_episode_schedule(layouts)
    if args.camera_runtime_diagnostic:
        print_checkpoint_and_dataset_contract(policy_path, dataset_root)

    metadata = LeRobotDatasetMetadata(args.dataset_repo_id, root=dataset_root)
    policy_cfg = PreTrainedConfig.from_pretrained(str(policy_path))
    policy_cfg.pretrained_path = policy_path
    policy_cfg.device = args.device
    policy_cfg.use_amp = args.use_amp

    policy = make_policy(policy_cfg, ds_meta=metadata)
    policy.eval()
    device = next(policy.parameters()).device
    preprocessor, postprocessor = make_pre_post_processors(policy_cfg=policy_cfg, pretrained_path=str(policy_path))

    video_fps = metadata.fps if args.video_fps is None else args.video_fps
    env = OpenArmFixedSlotsEnv(
        max_episode_steps=args.max_steps,
        seed=args.seed,
        min_steps_before_success=args.min_steps_before_success,
        debug_success=args.debug_success,
        diagnostic_instrumentation=args.diagnostic_instrumentation,
        dataset_root=dataset_root,
    )

    print(f"[EVAL DEBUG] checkpoint_path={policy_path}", flush=True)
    print(f"[EVAL DEBUG] observation_feature_keys={list(OBSERVATION_KEYS)}", flush=True)
    print(f"[EVAL DEBUG] action_dimension={ACTION_DIM}", flush=True)
    print(f"[EVAL DEBUG] env_config={env.describe_configuration()}", flush=True)
    print(f"[EVAL DEBUG] selected_layouts={len(layouts)} total_episodes={len(episode_schedule)}", flush=True)
    if args.camera_runtime_diagnostic:
        print_camera_flow(env)

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
    camera_debug_dir = args.camera_debug_dir.expanduser().resolve()
    runtime_reset_logged = False
    runtime_post_step_logged = False
    policy_input_logged = False

    try:
        for episode_id, episode in enumerate(episode_schedule):
            layout_id = int(episode["layout_id"])
            permutation_id = int(episode["permutation_id"])
            repeat_index = int(episode["repeat_index"])
            target_color = str(episode["target_color"])
            task_text = TASKS_BY_COLOR[target_color]
            instruction_color = infer_instruction_color(task_text)
            group_seed = args.seed + layout_id

            env.task_str = task_text
            env.task = task_text
            env.task_description = task_text
            if args.reseed_per_episode:
                set_seed(group_seed)

            print(
                f"[EVAL] episode={episode_id} layout_id={layout_id} permutation_id={permutation_id} "
                f"repeat_index={repeat_index} target={target_color} seed={group_seed}",
                flush=True,
            )
            obs, reset_info = env.reset(seed=group_seed, options={"layout_id": layout_id})
            policy.reset()
            preprocessor.reset()
            postprocessor.reset()
            success = False
            termination_reason = "max_steps"
            picked_color = "failure"
            picked_slot = "failure"
            step = -1
            frames: list[np.ndarray] = []
            video_relpath = ""
            episode_step_rows: list[dict[str, object]] = []
            previous_camera_frames: dict[str, object] = {}
            if args.record_video:
                frames.append(np.asarray(get_video_frame(obs, args.video_camera)).copy())
            if args.camera_runtime_diagnostic and not runtime_reset_logged:
                print_runtime_obs_keys("EVAL_OBS_KEYS_RESET", obs)
                runtime_reset_logged = True

            try:
                for step in range(args.max_steps):
                    if step % max(args.progress_every, 1) == 0:
                        print(f"[EVAL_STEP] episode={episode_id} step={step}", flush=True)
                    raw_policy_obs = {key: np.asarray(value).copy() for key, value in obs.items()}
                    with (
                        torch.inference_mode(),
                        torch.autocast(device_type=device.type) if device.type == "cuda" and args.use_amp else nullcontext(),
                    ):
                        prepared_obs = prepare_observation_for_inference(
                            raw_policy_obs,
                            device=device,
                            task=task_text,
                            robot_type=ROBOT_TYPE,
                        )
                        processed_obs = preprocessor(prepared_obs)
                        if args.camera_runtime_diagnostic:
                            if not policy_input_logged:
                                print("[POLICY_INPUT]", flush=True)
                                print(f"keys = {list(processed_obs.keys())}", flush=True)
                                for key in REQUIRED_OBS_KEYS:
                                    if key in processed_obs:
                                        print_value_summary(f"POLICY_INPUT:{key}", processed_obs[key])
                                    else:
                                        print(f"POLICY_INPUT:{key}\n  MISSING", flush=True)
                                policy_input_logged = True
                            for key in ("observation.images.top", "observation.images.wrist"):
                                current_frame = processed_obs[key]
                                if key in previous_camera_frames and step < 10:
                                    diff = mean_abs_diff(current_frame, previous_camera_frames[key])
                                    print(f"[CAMERA_FRESHNESS] step={step} key={key} mean_abs_diff={diff}", flush=True)
                                previous_camera_frames[key] = current_frame.detach().cpu().clone()
                        if args.dump_camera_debug and episode_id == 0 and step == 0:
                            save_debug_image(camera_debug_dir / "top_raw.png", obs["observation.images.top"])
                            save_debug_image(camera_debug_dir / "wrist_raw.png", obs["observation.images.wrist"])
                            save_debug_image(camera_debug_dir / "top_prepared.png", prepared_obs["observation.images.top"])
                            save_debug_image(camera_debug_dir / "wrist_prepared.png", prepared_obs["observation.images.wrist"])
                            save_debug_image(camera_debug_dir / "top_policy_input.png", processed_obs["observation.images.top"])
                            save_debug_image(camera_debug_dir / "wrist_policy_input.png", processed_obs["observation.images.wrist"])
                            print(f"[CAMERA_DEBUG_FILES] dir={camera_debug_dir}", flush=True)
                        action = policy.select_action(processed_obs)
                        action = postprocessor(action)
                    action_np = np.asarray(action.squeeze(0).cpu(), dtype=np.float32)
                    if action_np.shape != (8,):
                        raise RuntimeError(f"Expected policy action (8,), got {action_np.shape}")
                    obs, _, terminated, truncated, info = env.step(action_np)
                    if args.camera_runtime_diagnostic and not runtime_post_step_logged:
                        print_runtime_obs_keys("EVAL_OBS_KEYS_STEP1", obs)
                        runtime_post_step_logged = True
                    if args.record_video:
                        frames.append(np.asarray(get_video_frame(obs, args.video_camera)).copy())
                    success = bool(info.get("is_success", False))
                    picked_color = str(info.get("picked_color", "failure"))
                    picked_slot = str(info.get("picked_slot", "failure"))
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
                        episode_step_rows.append(
                            {
                                "episode_id": episode_id,
                                "seed": group_seed,
                                "layout_id": layout_id,
                                "permutation_id": permutation_id,
                                "repeat_index": repeat_index,
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
                        )
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
                        f"episode_{episode_id:03d}_layout_{layout_id:03d}_target_{target_color}_"
                        f"picked_{picked_color}_{status_label}.mp4"
                    )
                    video_path = video_dir / video_name
                    write_video(video_path, frames, fps=video_fps)
                    video_relpath = os.path.relpath(video_path, start=output.parent)

            color_correct = bool(success and picked_color == instruction_color)
            row: dict[str, object] = {
                "episode_id": episode_id,
                "seed": group_seed,
                "instruction": task_text,
                "target_color": target_color,
                "picked_color": picked_color,
                "picked_slot": picked_slot,
                "task_success": success,
                "color_correct": color_correct,
                "termination_reason": termination_reason,
                "steps": step + 1,
                "video_path": video_relpath,
                "layout_id": layout_id,
                "permutation_id": permutation_id,
                "repeat_index": repeat_index,
                "slot_a_color": reset_info.get("slot_a_color"),
                "slot_b_color": reset_info.get("slot_b_color"),
                "slot_c_color": reset_info.get("slot_c_color"),
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
                        seed=group_seed,
                        target_color=target_color,
                        task_text=task_text,
                        step_rows=episode_step_rows,
                        result_row=row,
                        max_steps=args.max_steps,
                    )
                )
                diagnostic_step_rows.extend(episode_step_rows)

            print(
                f"[EVAL RESULT] episode={episode_id} target={target_color} picked={picked_color} "
                f"picked_slot={picked_slot} success={success} color_correct={color_correct}",
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
            "picked_slot",
            "task_success",
            "color_correct",
            "termination_reason",
            "steps",
            "video_path",
            "layout_id",
            "permutation_id",
            "repeat_index",
            "slot_a_color",
            "slot_b_color",
            "slot_c_color",
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

        if args.diagnostic_instrumentation and diagnostic_step_rows and diagnostic_summary_rows:
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
            print(f"[RESULT] diagnostic_steps_csv={steps_path}", flush=True)
            print(f"[RESULT] diagnostic_summary_csv={summary_path}", flush=True)

        successes = sum(bool(row["task_success"]) for row in results)
        color_correct = sum(bool(row["color_correct"]) for row in results)
        print(f"[RESULT] task_successes={successes}/{len(results)} ({successes / len(results):.1%})")
        print(f"[RESULT] color_correct={color_correct}/{len(results)} ({color_correct / len(results):.1%})")
        print(f"[RESULT] csv={output}", flush=True)
    finally:
        env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
