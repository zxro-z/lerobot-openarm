#!/usr/bin/env python3
"""Run 3-color closed-loop SmolVLA evaluation in the OpenArm Isaac Lab scene."""

from __future__ import annotations

import argparse
import csv
import os
import re
import traceback
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
COLOR_NAMES = ("red", "blue", "yellow")


def infer_instruction_color(task_text: str) -> str | None:
    match = re.search(r"\b(red|blue|yellow)\b cube", task_text.lower())
    return match.group(1) if match else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-path", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument(
        "--dataset-repo-id",
        default="a126-kitech/openarm_pickcube_3colors_no_ep10_12",
    )
    parser.add_argument("--num-episodes", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--use-amp", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--progress-every", type=int, default=1)
    parser.add_argument("--save-video", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--video-dir", type=Path, default=Path("outputs/eval/openarm_smolvla_3color/videos"))
    parser.add_argument("--video-fps", type=int, default=30)
    parser.add_argument("--success-video-tail-seconds", type=float, default=3.0)
    parser.add_argument("--target-color", choices=list(COLOR_NAMES), required=True)
    parser.add_argument("--cube-layout", choices=["fixed_slots"], default="fixed_slots")
    parser.add_argument("--cube-jitter", type=float, default=0.0)
    parser.add_argument("--render-scene-only", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--save-front-frame", type=Path, default=None)
    parser.add_argument("--task", default=None)
    parser.add_argument("--output", type=Path, default=Path("outputs/eval/openarm_smolvla_3color/results.csv"))
    return parser.parse_args()


def validate_feature_contract(policy_cfg, env_contract: dict[str, object]) -> None:
    policy_input_features = policy_cfg.input_features or {}
    policy_output_features = policy_cfg.output_features or {}
    policy_visual_keys = [key for key, ft in policy_input_features.items() if str(ft.type).endswith("VISUAL")]
    policy_state_keys = [key for key, ft in policy_input_features.items() if str(ft.type).endswith("STATE")]
    env_visual_keys = [key for key in env_contract["observation_keys"] if key.startswith("observation.images.")]
    env_state_keys = [key for key in env_contract["observation_keys"] if key == "observation.state"]

    print("[POLICY INPUT FEATURES]", flush=True)
    print({key: {"type": str(ft.type), "shape": ft.shape} for key, ft in policy_input_features.items()}, flush=True)
    print("[POLICY OUTPUT FEATURES]", flush=True)
    print({key: {"type": str(ft.type), "shape": ft.shape} for key, ft in policy_output_features.items()}, flush=True)
    print("[ENV INPUT FEATURES]", flush=True)
    print(env_contract, flush=True)
    print("[POLICY VISUAL FEATURES]", flush=True)
    print(policy_visual_keys, flush=True)
    print("[ENV VISUAL FEATURES]", flush=True)
    print(env_visual_keys, flush=True)

    missing = sorted(set(policy_input_features) - set(env_contract["observation_keys"]))
    extra = sorted(set(env_contract["observation_keys"]) - set(policy_input_features))
    action_feature = policy_output_features.get("action")
    action_shape = tuple(action_feature.shape) if action_feature is not None else None

    if missing or extra:
        raise ValueError(
            "Feature mismatch between policy config and 3-color environment.\n"
            f"- Missing features: {missing}\n"
            f"- Extra features: {extra}"
        )
    if tuple(action_shape or ()) != (env_contract["action_dimension"],):
        raise ValueError(
            "Action mismatch between policy config and 3-color environment.\n"
            f"- Policy action shape: {action_shape}\n"
            f"- Env action shape: {(env_contract['action_dimension'],)}"
        )
    if len(policy_state_keys) != 1 or env_contract["state_dimension"] != policy_input_features["observation.state"].shape[0]:
        raise ValueError(
            "State mismatch between policy config and 3-color environment.\n"
            f"- Policy state shape: {policy_input_features['observation.state'].shape}\n"
            f"- Env state shape: {(env_contract['state_dimension'],)}"
        )
    print("[FEATURE MATCH] PASS", flush=True)


def main() -> None:
    args = parse_args()
    os.environ["OPENARM_SMOLVLA_HEADLESS"] = "1" if args.headless else "0"

    from openarm_smolvla_3color_env import (
        ACTION_DIM,
        COLOR_NAMES as ENV_COLORS,
        OBSERVATION_KEYS,
        ROBOT_TYPE,
        TASK_BY_COLOR,
        OpenArmThreeColorEnv,
        STATE_DIM,
        simulation_app,
    )

    import numpy as np
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
    from lerobot.policies.factory import make_policy, make_pre_post_processors
    from lerobot.utils.control_utils import predict_action
    from lerobot.utils.io_utils import write_video

    policy_path = args.policy_path.expanduser().resolve()
    dataset_root = args.dataset_root.expanduser().resolve() if args.dataset_root is not None else None
    task_text = args.task if args.task is not None else TASK_BY_COLOR[args.target_color]
    instruction_color = infer_instruction_color(task_text)

    metadata = LeRobotDatasetMetadata(args.dataset_repo_id, root=dataset_root)
    policy_cfg = PreTrainedConfig.from_pretrained(str(policy_path))
    policy_cfg.pretrained_path = policy_path
    policy_cfg.device = args.device
    policy_cfg.use_amp = args.use_amp

    env = OpenArmThreeColorEnv(
        max_episode_steps=args.max_steps,
        target_color=args.target_color,
        task=task_text,
        cube_layout=args.cube_layout,
        cube_jitter=args.cube_jitter,
    )
    env_contract = env.feature_contract()
    print("[EVAL ENV]", flush=True)
    print(env_contract["env_name"], flush=True)
    print("[CAMERAS]", flush=True)
    print("front", flush=True)
    print("wrist", flush=True)
    print("[LAYOUT]", flush=True)
    print(env_contract["layout_name"], flush=True)
    validate_feature_contract(policy_cfg, env_contract)

    policy = make_policy(policy_cfg, ds_meta=metadata)
    policy.eval()
    device = next(policy.parameters()).device
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=str(policy_path),
    )

    print(f"[CHECKPOINT] {policy_path}", flush=True)
    print(f"[TRAINING DATASET] repo_id={args.dataset_repo_id} root={dataset_root}", flush=True)
    print(f"[TARGET COLOR] {args.target_color}", flush=True)
    print(f"[TASK] {task_text}", flush=True)
    print(f"[SCENE LAYOUT] {env.describe_configuration()}", flush=True)

    if args.render_scene_only:
        obs, reset_info = env.reset(seed=args.seed)
        print("[3COLOR SCENE LAYOUT]", flush=True)
        print(reset_info["runtime_layout"], flush=True)
        if args.save_front_frame is not None:
            frame_path = args.save_front_frame.expanduser().resolve()
            frame_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                import imageio.v2 as imageio

                imageio.imwrite(frame_path, obs["observation.images.front"])
            except Exception as exc:
                raise RuntimeError(f"Failed to save scene-only front frame to {frame_path}: {exc}") from exc
            print(f"[SCENE_ONLY] saved_front_frame={frame_path}", flush=True)
        print(f"[SCENE_ONLY] reset_info={reset_info}", flush=True)
        return

    video_dir = args.video_dir.expanduser().resolve()
    if args.save_video:
        video_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, object]] = []
    try:
        for episode in range(args.num_episodes):
            obs, reset_info = env.reset(seed=args.seed + episode)
            print("[3COLOR SCENE LAYOUT]", flush=True)
            print(reset_info["runtime_layout"], flush=True)
            policy.reset()
            preprocessor.reset()
            postprocessor.reset()
            success = False
            termination_reason = "max_steps"
            step = -1
            frames: list[np.ndarray] = []
            if args.save_video:
                frames.append(np.asarray(env.render()).copy())

            for step in range(args.max_steps):
                if step % max(args.progress_every, 1) == 0:
                    print(f"[EVAL_STEP] episode={episode} step={step}", flush=True)
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
                if action_np.shape != (ACTION_DIM,):
                    raise RuntimeError(f"Expected policy action {(ACTION_DIM,)}, got {action_np.shape}")

                obs, _, terminated, truncated, info = env.step(action_np)
                if args.save_video:
                    frames.append(np.asarray(env.render()).copy())

                success = bool(info.get("task_success", False))
                if terminated or truncated:
                    termination_reason = str(info.get("termination_reason", "max_steps"))
                    if success and args.save_video and args.success_video_tail_seconds > 0:
                        tail_steps = max(int(round(args.success_video_tail_seconds * args.video_fps)), 0)
                        for _ in range(tail_steps):
                            obs, _, _, _, _ = env.step(action_np)
                            frames.append(np.asarray(env.render()).copy())
                    break

            results.append(
                {
                    "episode_id": episode,
                    "seed": args.seed + episode,
                    "instruction": task_text,
                    "instruction_color": instruction_color,
                    "target_color": args.target_color,
                    "target_slot": info.get("target_slot"),
                    "red_slot": info.get("red_slot"),
                    "blue_slot": info.get("blue_slot"),
                    "yellow_slot": info.get("yellow_slot"),
                    "picked_color": info.get("picked_color", "failure"),
                    "picked_slot": info.get("picked_slot", "failure"),
                    "task_success": bool(info.get("task_success", False)),
                    "color_correct": bool(info.get("color_correct", False)),
                    "red_initial_x": reset_info["red_initial_position"][0],
                    "red_initial_y": reset_info["red_initial_position"][1],
                    "red_initial_z": reset_info["red_initial_position"][2],
                    "blue_initial_x": reset_info["blue_initial_position"][0],
                    "blue_initial_y": reset_info["blue_initial_position"][1],
                    "blue_initial_z": reset_info["blue_initial_position"][2],
                    "yellow_initial_x": reset_info["yellow_initial_position"][0],
                    "yellow_initial_y": reset_info["yellow_initial_position"][1],
                    "yellow_initial_z": reset_info["yellow_initial_position"][2],
                    "final_red_position": info.get("final_positions", {}).get("red"),
                    "final_blue_position": info.get("final_positions", {}).get("blue"),
                    "final_yellow_position": info.get("final_positions", {}).get("yellow"),
                    "slot_assignment": reset_info.get("slot_assignment"),
                    "termination_reason": termination_reason,
                    "steps": step + 1,
                }
            )
            print(
                f"[EVAL] episode={episode} target={args.target_color} "
                f"picked={info.get('picked_color', 'failure')} success={success}",
                flush=True,
            )
            if args.save_video and frames:
                video_path = video_dir / f"episode_{episode:03d}_{args.target_color}_{info.get('picked_color', 'failure')}.mp4"
                write_video(video_path, frames, fps=args.video_fps)
    except Exception as exc:
        print(f"[EVAL_ERROR] {type(exc).__name__}: {exc}", flush=True)
        traceback.print_exc()
        raise
    else:
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", newline="") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=[
                    "episode_id",
                    "seed",
                    "instruction",
                    "instruction_color",
                    "target_color",
                    "target_slot",
                    "red_slot",
                    "blue_slot",
                    "yellow_slot",
                    "picked_color",
                    "picked_slot",
                    "task_success",
                    "color_correct",
                    "red_initial_x",
                    "red_initial_y",
                    "red_initial_z",
                    "blue_initial_x",
                    "blue_initial_y",
                    "blue_initial_z",
                    "yellow_initial_x",
                    "yellow_initial_y",
                    "yellow_initial_z",
                    "final_red_position",
                    "final_blue_position",
                    "final_yellow_position",
                    "slot_assignment",
                    "termination_reason",
                    "steps",
                ],
            )
            writer.writeheader()
            writer.writerows(results)
        successes = sum(bool(row["task_success"]) for row in results)
        color_correct = sum(bool(row["color_correct"]) for row in results)
        print(f"[RESULT] task_successes={successes}/{len(results)} ({successes / len(results):.1%})", flush=True)
        print(
            f"[RESULT] color_correct={color_correct}/{len(results)} ({color_correct / len(results):.1%})",
            flush=True,
        )
        print(f"[RESULT] csv={output}", flush=True)
    finally:
        env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
