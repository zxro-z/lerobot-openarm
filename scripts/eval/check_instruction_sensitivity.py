#!/usr/bin/env python3
"""Measure fixed-observation instruction sensitivity for a SmolVLA checkpoint."""

from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.utils import prepare_observation_for_inference
from lerobot.utils.constants import ACTION


TASKS = {
    "red": "Pick up the red cube and place it in the storage box.",
    "blue": "Pick up the blue cube and place it in the storage box.",
    "yellow": "Pick up the yellow cube and place it in the storage box.",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-path", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--dataset-repo-id", required=True)
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--frame-index", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--use-amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--robot-type", default="openarm_isaaclab")
    parser.add_argument("--video-backend", choices=["pyav", "torchcodec", "video_reader"], default="pyav")
    return parser.parse_args()


def select_frame(dataset, episode_index: int, frame_index: int) -> dict[str, np.ndarray]:
    episode_column = dataset.hf_dataset["episode_index"]
    relative_indices = [
        idx for idx, ep_idx in enumerate(episode_column) if int(ep_idx) == episode_index
    ]
    if not relative_indices:
        raise ValueError(f"Episode index not found: {episode_index}")
    if frame_index >= len(relative_indices):
        raise ValueError(f"Frame index {frame_index} out of range for episode {episode_index}")

    item = dataset[relative_indices[frame_index]]
    observation = {}
    observation["observation.state"] = np.asarray(item["observation.state"], dtype=np.float32).copy()
    for key in ("observation.images.top", "observation.images.wrist"):
        image = item[key]
        if isinstance(image, torch.Tensor):
            image = image.detach().cpu().numpy()
        image = np.asarray(image)
        if image.ndim != 3:
            raise ValueError(f"Expected image tensor with 3 dims for {key}, got shape {image.shape}")
        if image.shape[0] in (1, 3) and image.shape[-1] not in (1, 3):
            image = np.transpose(image, (1, 2, 0))
        observation[key] = image.copy()
    return observation


def collect_action_chunk(policy, preprocessor, postprocessor, observation, task, device, use_amp, robot_type):
    prepared = prepare_observation_for_inference(
        observation={k: np.asarray(v).copy() for k, v in observation.items()},
        device=device,
        task=task,
        robot_type=robot_type,
    )
    prepared = preprocessor(prepared)
    policy.reset()
    preprocessor.reset()
    postprocessor.reset()
    with (
        torch.inference_mode(),
        torch.autocast(device_type=device.type) if device.type == "cuda" and use_amp else torch.no_grad(),
    ):
        first_action = policy.select_action(prepared)
        queue_tail = list(policy._queues[ACTION])
    raw_chunk = torch.stack([first_action, *queue_tail], dim=0).squeeze(1)
    final_chunk = postprocessor(raw_chunk.clone())
    return final_chunk.detach().cpu().to(torch.float32)


def main() -> None:
    args = parse_args()
    policy_path = args.policy_path.expanduser().resolve()
    dataset_root = args.dataset_root.expanduser().resolve()

    metadata = LeRobotDatasetMetadata(args.dataset_repo_id, root=dataset_root)
    policy_cfg = PreTrainedConfig.from_pretrained(str(policy_path))
    policy_cfg.pretrained_path = policy_path
    policy_cfg.device = args.device
    policy_cfg.use_amp = args.use_amp

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset(
        args.dataset_repo_id,
        root=dataset_root,
        video_backend=args.video_backend,
    )
    observation = select_frame(dataset, args.episode_index, args.frame_index)

    policy = make_policy(policy_cfg, ds_meta=metadata)
    policy.eval()
    device = next(policy.parameters()).device
    preprocessor, postprocessor = make_pre_post_processors(policy_cfg=policy_cfg, pretrained_path=str(policy_path))

    chunks = {
        color: collect_action_chunk(
            policy=policy,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            observation=observation,
            task=task,
            device=device,
            use_amp=args.use_amp,
            robot_type=args.robot_type,
        )
        for color, task in TASKS.items()
    }

    pairwise_l2 = {}
    pairwise_abs = {}
    for a, b in combinations(("red", "blue", "yellow"), 2):
        diff = chunks[a] - chunks[b]
        pairwise_l2[f"{a[0].upper()}{b[0].upper()}"] = float(torch.norm(diff).item())
        pairwise_abs[f"{a[0].upper()}{b[0].upper()}"] = float(torch.mean(torch.abs(diff)).item())

    mean_pairwise_l2 = float(np.mean(list(pairwise_l2.values())))
    mean_pairwise_abs = float(np.mean(list(pairwise_abs.values())))
    chunk_scale = float(np.mean([torch.mean(torch.abs(chunk)).item() for chunk in chunks.values()]))
    relative_instruction_effect = mean_pairwise_abs / max(chunk_scale, 1e-12)

    print(f"[CHECKPOINT] {policy_path}", flush=True)
    print(f"[OBSERVATION] episode_index={args.episode_index} frame_index={args.frame_index}", flush=True)
    print(f"RB L2: {pairwise_l2['RB']:.6f}", flush=True)
    print(f"RY L2: {pairwise_l2['RY']:.6f}", flush=True)
    print(f"BY L2: {pairwise_l2['BY']:.6f}", flush=True)
    print(f"mean pairwise L2: {mean_pairwise_l2:.6f}", flush=True)
    print(f"mean pairwise abs: {mean_pairwise_abs:.6f}", flush=True)
    print(f"relative_instruction_effect: {relative_instruction_effect:.6f}", flush=True)


if __name__ == "__main__":
    main()
