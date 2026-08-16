#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.utils import prepare_observation_for_inference
from lerobot.utils.constants import ACTION
from lerobot.utils.utils import get_safe_torch_device

TASK = "Pick up the red cube and place it in the storage box."
ROBOT_TYPE = "openarm_isaaclab"
DEFAULT_DATASET_REPO_ID = "a126-kitech/openarm_dual_realsense_pick_place_random_cube_tilt_30_box_blue"
DEFAULT_DATASET_ROOT = Path(
    "/home/zxro/snap/codex/34/.cache/huggingface/lerobot/"
    "a126-kitech/openarm_dual_realsense_pick_place_random_cube_tilt_30_box_blue"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record a step-by-step OpenArm rollout diagnostic.")
    parser.add_argument("--policy-path", type=Path, required=True)
    parser.add_argument("--dataset-repo-id", default=DEFAULT_DATASET_REPO_ID)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--use-amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--contract", choices=["radian", "degree"], default="degree")
    parser.add_argument("--gripper-override-mode", choices=["normal", "dataset_schedule"], default="normal")
    return parser.parse_args()


def make_gripper_targets(gripper_action_value: float, contract: str) -> tuple[float, float]:
    real_gripper_closed = math.radians(-15.0)
    real_gripper_open = math.radians(-60.0)
    sim_gripper_closed = 0.0
    sim_gripper_open = 0.044
    if contract == "degree":
        gripper_action_rad = math.radians(gripper_action_value)
    else:
        gripper_action_rad = gripper_action_value
    alpha = (gripper_action_rad - real_gripper_closed) / (real_gripper_open - real_gripper_closed)
    alpha = float(np.clip(alpha, 0.0, 1.0))
    sim_gripper_action = sim_gripper_closed + alpha * (sim_gripper_open - sim_gripper_closed)
    return sim_gripper_action, sim_gripper_open - sim_gripper_action


def tensor_to_list(t: torch.Tensor) -> list[float]:
    return t.detach().cpu().numpy().astype(np.float32).tolist()


def maybe_get_cube_pos(env) -> list[float] | None:
    try:
        return env.cube.data.root_pos_w[0].detach().cpu().numpy().astype(np.float32).tolist()
    except Exception:
        return None


def maybe_get_ee_pos(env) -> list[float] | None:
    try:
        if getattr(env, "ee_body_id", None) is None:
            return None
        return env.robot.data.body_pos_w[0, env.ee_body_id].detach().cpu().numpy().astype(np.float32).tolist()
    except Exception:
        return None


def main() -> None:
    args = parse_args()

    # Import after CLI parse because this starts Isaac Sim.
    from gym_openarm.openarm_env import OpenArmEnv

    policy_path = args.policy_path.expanduser().resolve()
    dataset_root = args.dataset_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = LeRobotDatasetMetadata(args.dataset_repo_id, root=dataset_root)
    policy_cfg = PreTrainedConfig.from_pretrained(str(policy_path))
    policy_cfg.pretrained_path = policy_path
    policy_cfg.device = args.device
    policy_cfg.use_amp = args.use_amp

    requested_device = get_safe_torch_device(args.device)
    policy = make_policy(policy_cfg, ds_meta=metadata)
    policy.eval()
    policy_device = next(policy.parameters()).device
    vlm_device = torch.device(policy.model.vlm_with_expert.vlm.device)
    inference_device = vlm_device
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=str(policy_path),
        preprocessor_overrides={"device_processor": {"device": str(inference_device)}},
    )

    if not isinstance(policy, SmolVLAPolicy):
        raise TypeError(f"Expected SmolVLAPolicy, got {type(policy)}")

    env = OpenArmEnv(max_episode_steps=args.max_steps, gripper_override_mode=args.gripper_override_mode)
    records: list[dict[str, object]] = []
    chunk_events: list[dict[str, object]] = []

    try:
        obs, _ = env.reset(seed=args.seed)
        policy.reset()
        preprocessor.reset()
        postprocessor.reset()

        for step_idx in range(args.max_steps):
            obs_state = np.asarray(obs["observation.state"], dtype=np.float32)
            sim_state_arm_rad = env.robot.data.joint_pos[0, env.arm_ids].detach().cpu().numpy().astype(np.float32)

            prepared_obs = prepare_observation_for_inference(
                observation={k: np.asarray(v).copy() for k, v in obs.items()},
                device=inference_device,
                task=TASK,
                robot_type=ROBOT_TYPE,
            )
            prepared_obs = preprocessor(prepared_obs)

            queue_before = len(policy._queues[ACTION])
            with (
                torch.inference_mode(),
                torch.autocast(device_type=inference_device.type)
                if inference_device.type == "cuda" and args.use_amp
                else torch.no_grad(),
            ):
                raw_action = policy.select_action(prepared_obs)

            if queue_before == 0:
                raw_chunk = torch.stack([raw_action, *list(policy._queues[ACTION])], dim=0).squeeze(1)
                final_chunk = postprocessor(raw_chunk.clone())
                chunk_events.append(
                    {
                        "step": step_idx,
                        "chunk_len": int(final_chunk.shape[0]),
                        "chunk_mean": tensor_to_list(final_chunk.mean(dim=0)),
                        "chunk_std": tensor_to_list(final_chunk.std(dim=0, unbiased=False)),
                        "chunk_first5": [tensor_to_list(x) for x in final_chunk[:5]],
                    }
                )

            final_action = postprocessor(raw_action.clone())
            action_np = np.asarray(final_action.squeeze(0).cpu(), dtype=np.float32)
            if args.contract == "degree":
                sim_arm_target_rad = np.deg2rad(action_np[:7]).astype(np.float32)
                gripper_policy_key = "gripper_policy_value_deg"
            else:
                sim_arm_target_rad = action_np[:7].copy()
                gripper_policy_key = "gripper_policy_value_rad"
            sim_gripper_target = list(make_gripper_targets(float(action_np[7]), args.contract))

            next_obs, reward, terminated, truncated, info = env.step(action_np)
            success = bool(info.get("is_success", False))
            debug_info = getattr(env, "_last_debug_info", {})
            ee_pos_w = debug_info.get("ee_pos_w", maybe_get_ee_pos(env))
            cube_pos_w = debug_info.get("cube_pos_w", maybe_get_cube_pos(env))
            ee_to_cube_distance = None
            if ee_pos_w is not None and cube_pos_w is not None:
                ee_to_cube_distance = float(
                    np.linalg.norm(np.asarray(ee_pos_w, dtype=np.float32) - np.asarray(cube_pos_w, dtype=np.float32))
                )

            records.append(
                {
                    "step": step_idx,
                    "policy_state": obs_state.tolist(),
                    "sim_state_arm_rad": sim_state_arm_rad.tolist(),
                    "policy_final_action": action_np.tolist(),
                    "policy_arm_action": action_np[:7].tolist(),
                    "sim_arm_target_rad": sim_arm_target_rad.tolist(),
                    gripper_policy_key: float(action_np[7]),
                    "override_gripper_action_deg": debug_info.get("override_gripper_action_deg"),
                    "gripper_sim_target": sim_gripper_target,
                    "ee_pos_w": ee_pos_w,
                    "cube_pos_w": cube_pos_w,
                    "ee_to_cube_distance": ee_to_cube_distance,
                    "reward": float(reward),
                    "success": success,
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                    "queue_before": int(queue_before),
                    "queue_after": int(len(policy._queues[ACTION])),
                }
            )

            obs = next_obs
            if terminated or truncated:
                break
    finally:
        env.close()

    json_path = output_dir / "rollout_diagnostic.json"
    with json_path.open("w") as f:
        json.dump({"records": records, "chunk_events": chunk_events}, f, indent=2)

    csv_path = output_dir / "rollout_diagnostic.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "step",
                "policy_state",
                "sim_state_arm_rad",
                "policy_final_action",
                "policy_arm_action",
                "sim_arm_target_rad",
                "gripper_policy_value",
                "override_gripper_action_deg",
                "gripper_sim_target",
                "ee_pos_w",
                "cube_pos_w",
                "ee_to_cube_distance",
                "reward",
                "success",
                "terminated",
                "truncated",
                "queue_before",
                "queue_after",
            ]
        )
        for record in records:
            writer.writerow(
                [
                    record["step"],
                    json.dumps(record["policy_state"]),
                    json.dumps(record["sim_state_arm_rad"]),
                    json.dumps(record["policy_final_action"]),
                    json.dumps(record["policy_arm_action"]),
                    json.dumps(record["sim_arm_target_rad"]),
                    record.get("gripper_policy_value_deg", record.get("gripper_policy_value_rad")),
                    record["override_gripper_action_deg"],
                    json.dumps(record["gripper_sim_target"]),
                    json.dumps(record["ee_pos_w"]),
                    json.dumps(record["cube_pos_w"]),
                    record["ee_to_cube_distance"],
                    record["reward"],
                    record["success"],
                    record["terminated"],
                    record["truncated"],
                    record["queue_before"],
                    record["queue_after"],
                ]
            )

    print(f"[DIAG] wrote {json_path}")
    print(f"[DIAG] wrote {csv_path}")
    print(f"[DIAG] steps={len(records)} chunk_events={len(chunk_events)}")
    print(
        f"[DIAG] requested_device={requested_device} policy_device={policy_device} "
        f"vlm_device={vlm_device} inference_device={inference_device}"
    )
    print(f"[DIAG] gripper_override_mode={args.gripper_override_mode}")


if __name__ == "__main__":
    main()
