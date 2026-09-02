#!/usr/bin/env python3
"""Paired counterfactual validation for SmolVLA fixed-slot triplets."""

from __future__ import annotations

import argparse
import csv
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.counterfactual import load_counterfactual_triplet_metadata
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.target_grounding import TARGET_SLOT_LABEL_KEY, load_target_grounding_metadata
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.utils import prepare_observation_for_inference

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT = PROJECT_ROOT / "outputs/train/latest/checkpoints/last/pretrained_model"
DEFAULT_DATASET_ROOT = PROJECT_ROOT / "src/lerobot/datasets/openarm_three_color_fixed_slots_perm_tilt50_r3"
DEFAULT_DATASET_REPO_ID = "local/openarm_three_color_fixed_slots_perm_tilt50_r3"
COLORS = ("red", "blue", "yellow")
PAIR_ORDER = (("red", "blue"), ("red", "yellow"), ("blue", "yellow"))
TASK_FALLBACK = {
    "red": "Pick up the red cube and place it in the storage box.",
    "blue": "Pick up the blue cube and place it in the storage box.",
    "yellow": "Pick up the yellow cube and place it in the storage box.",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--dataset-repo-id", default=DEFAULT_DATASET_REPO_ID)
    parser.add_argument("--triplet-manifest", type=Path, default=None)
    parser.add_argument("--num-groups", type=int, default=16)
    parser.add_argument("--local-frame", type=int, default=36)
    parser.add_argument("--noise-seed", type=int, default=1234)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--use-amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--eval-csv", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


def resolve_tasks(dataset: LeRobotDataset) -> dict[str, str]:
    found: dict[str, str] = {}
    for task in dataset.meta.tasks.index.tolist():
        lower = task.lower()
        for color in COLORS:
            if f"{color} cube" in lower and color not in found:
                found[color] = task
    for color in COLORS:
        found.setdefault(color, TASK_FALLBACK[color])
    return found


def make_generator(device: torch.device, seed: int) -> torch.Generator:
    if device.type == "cuda":
        return torch.Generator(device=device).manual_seed(seed)
    return torch.Generator().manual_seed(seed)


def pairwise_l2(values: dict[str, np.ndarray]) -> tuple[dict[str, float], float]:
    distances: dict[str, float] = {}
    scalars: list[float] = []
    for left, right in combinations(COLORS, 2):
        key = f"{left}_vs_{right}"
        dist = float(np.linalg.norm(values[left].reshape(-1) - values[right].reshape(-1)))
        distances[key] = dist
        scalars.append(dist)
    return distances, float(np.mean(scalars)) if scalars else 0.0


def pairwise_deltas(values: dict[str, np.ndarray]) -> dict[str, list[float]]:
    deltas: dict[str, list[float]] = {}
    for left, right in PAIR_ORDER:
        key = f"{left}_vs_{right}"
        delta = values[left].reshape(-1) - values[right].reshape(-1)
        deltas[key] = delta.astype(np.float32).tolist()
    return deltas


def leading_action_vector(value: np.ndarray) -> np.ndarray:
    flat = np.asarray(value, dtype=np.float32)
    if flat.ndim <= 1:
        return flat.reshape(-1)
    return flat[0].reshape(-1)


def pairwise_cosine(gt_values: dict[str, np.ndarray], pred_values: dict[str, np.ndarray]) -> dict[str, float]:
    result: dict[str, float] = {}
    for left, right in PAIR_ORDER:
        key = f"{left}_vs_{right}"
        gt_delta = leading_action_vector(gt_values[left]) - leading_action_vector(gt_values[right])
        pred_delta = leading_action_vector(pred_values[left]) - leading_action_vector(pred_values[right])
        gt_norm = float(np.linalg.norm(gt_delta))
        pred_norm = float(np.linalg.norm(pred_delta))
        if gt_norm == 0.0 or pred_norm == 0.0:
            result[key] = 0.0
        else:
            result[key] = float(np.dot(gt_delta, pred_delta) / (gt_norm * pred_norm))
    return result


def to_numpy_image(image: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()
    image = np.asarray(image)
    if image.ndim == 3 and image.shape[0] in (1, 3, 4) and image.shape[-1] not in (1, 3, 4):
        image = np.transpose(image, (1, 2, 0))
    return np.asarray(image).copy()


def summarize_eval_csv(path: Path) -> dict[str, object]:
    rows = list(csv.DictReader(path.open()))
    if not rows:
        return {}
    task_success = sum(str(row.get("task_success", "")).lower() == "true" for row in rows) / len(rows)
    color_accuracy = sum(str(row.get("color_correct", "")).lower() == "true" for row in rows) / len(rows)
    slot_counts: dict[str, dict[str, int]] = {color: {} for color in COLORS}
    slot_c_valid = []
    for row in rows:
        target = str(row.get("target_color", ""))
        picked_slot = str(row.get("picked_slot", "failure"))
        slot_counts.setdefault(target, {})
        slot_counts[target][picked_slot] = slot_counts[target].get(picked_slot, 0) + 1
        if target in COLORS:
            slot_c_valid.append(int(picked_slot == "c"))
    return {
        "task_success": task_success,
        "color_accuracy": color_accuracy,
        "target_color_x_picked_slot": slot_counts,
        "slot_c_valid_pick_ratio": float(np.mean(slot_c_valid)) if slot_c_valid else 0.0,
    }


def main() -> None:
    args = parse_args()
    checkpoint = args.checkpoint.expanduser().resolve()
    dataset_root = args.dataset_root.expanduser().resolve()
    device = torch.device(args.device)

    dataset = LeRobotDataset(
        args.dataset_repo_id,
        root=dataset_root,
        video_backend="pyav",
    )
    metadata = load_counterfactual_triplet_metadata(dataset=dataset, manifest_path=args.triplet_manifest)
    grounding_metadata = load_target_grounding_metadata(dataset)
    tasks = resolve_tasks(dataset)

    policy_cfg = PreTrainedConfig.from_pretrained(str(checkpoint))
    policy_cfg.pretrained_path = checkpoint
    policy_cfg.device = args.device
    policy = make_policy(policy_cfg, ds_meta=dataset.meta)
    policy.eval().to(device)
    processor_overrides = {"device_processor": {"device": args.device}}
    preprocessor, _ = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=str(checkpoint),
        preprocessor_overrides=processor_overrides,
        postprocessor_overrides=processor_overrides,
    )

    generator = make_generator(device, args.noise_seed)
    shared_noise = torch.randn(
        (1, policy.config.chunk_size, policy.config.max_action_dim),
        generator=generator,
        device=device,
        dtype=torch.float32,
    )

    grouped_rows: list[dict[str, object]] = []
    used_groups = 0
    for triplet_id in range(metadata.summary.triplet_count):
        if used_groups >= args.num_groups:
            break
        candidates = []
        for color_index, color in enumerate(COLORS):
            episode = next(
                ep
                for ep, tid in metadata.episode_to_triplet_id.items()
                if tid == triplet_id and metadata.episode_to_color_id[ep] == color_index
            )
            episode_row = dataset.meta.episodes[episode]
            frame_index = min(args.local_frame, int(episode_row["length"]) - 1)
            global_index = int(episode_row["dataset_from_index"]) + frame_index
            item = dataset[global_index]
            candidates.append((color, item))

        observation = {
            key: candidates[0][1][key].unsqueeze(0).to(device)
            for key in candidates[0][1]
            if key.startswith("observation.images.") or key == "observation.state"
        }
        gt_chunks: dict[str, np.ndarray] = {}
        pred_chunks: dict[str, np.ndarray] = {}
        grounding_predictions: dict[str, int] = {}
        grounding_targets: dict[str, int] = {}
        grounding_logits: dict[str, list[float]] = {}
        for color, item in candidates:
            gt_chunks[color] = item["action"].detach().cpu().to(torch.float32).numpy()
            raw_observation = {
                "observation.state": observation["observation.state"][0].detach().cpu().numpy().copy(),
                "observation.images.top": to_numpy_image(observation["observation.images.top"][0]),
                "observation.images.wrist": to_numpy_image(observation["observation.images.wrist"][0]),
            }
            probe_batch = prepare_observation_for_inference(
                observation=raw_observation,
                device=device,
                task=tasks[color],
                robot_type=dataset.meta.robot_type,
            )
            probe_batch = preprocessor(probe_batch)
            if getattr(policy.config, "target_grounding_enabled", False):
                target_slot = grounding_metadata.episode_labels[int(item["episode_index"])].target_slot_label
                if getattr(policy.config, "target_conditioning_enabled", False):
                    probe_batch[TARGET_SLOT_LABEL_KEY] = torch.tensor(
                        [target_slot],
                        device=device,
                        dtype=torch.long,
                    )
                slot_logits = policy.predict_target_slot_logits(probe_batch)[0].float().cpu()
                grounding_logits[color] = slot_logits.tolist()
                grounding_predictions[color] = int(slot_logits.argmax(-1).item())
                grounding_targets[color] = target_slot
            with torch.inference_mode(), (
                torch.autocast(device_type=device.type) if device.type == "cuda" and args.use_amp else torch.no_grad()
            ):
                pred = policy.predict_action_chunk(
                    probe_batch, noise=shared_noise.clone()
                )
            pred_chunks[color] = pred[0].detach().cpu().to(torch.float32).numpy()

        gt_pairwise, gt_sep = pairwise_l2(gt_chunks)
        pred_pairwise, pred_sep = pairwise_l2(pred_chunks)
        gt_delta_pairwise = pairwise_deltas(gt_chunks)
        pred_delta_pairwise = pairwise_deltas(pred_chunks)
        cosine_pairwise = pairwise_cosine(gt_chunks, pred_chunks)
        example_episode = triplet_to_episode = next(
            ep for ep, tid in metadata.episode_to_triplet_id.items() if tid == triplet_id
        )
        grouped_rows.append(
            {
                "triplet_id": triplet_id,
                "layout_id": metadata.episode_to_layout_id[example_episode],
                "permutation_id": metadata.episode_to_permutation_id[example_episode],
                "repeat_index": metadata.episode_to_repeat_index[example_episode],
                "gt_pairwise": gt_pairwise,
                "pred_pairwise": pred_pairwise,
                "gt_delta_pairwise": gt_delta_pairwise,
                "pred_delta_pairwise": pred_delta_pairwise,
                "cosine_pairwise": cosine_pairwise,
                "gt_separation": gt_sep,
                "prediction_separation": pred_sep,
                "ratio": pred_sep / max(gt_sep, 1e-6),
                "grounding_predictions": grounding_predictions,
                "grounding_targets": grounding_targets,
                "grounding_logits": grounding_logits,
            }
        )
        used_groups += 1

    gt_mean = float(np.mean([row["gt_separation"] for row in grouped_rows])) if grouped_rows else 0.0
    pred_mean = float(np.mean([row["prediction_separation"] for row in grouped_rows])) if grouped_rows else 0.0
    ratio_mean = float(np.mean([row["ratio"] for row in grouped_rows])) if grouped_rows else 0.0

    result: dict[str, object] = {
        "num_groups": used_groups,
        "local_frame": args.local_frame,
        "gt_separation": gt_mean,
        "prediction_separation": pred_mean,
        "prediction_gt_ratio": ratio_mean,
        "groups": grouped_rows,
    }
    if args.eval_csv is not None:
        result["eval_csv_summary"] = summarize_eval_csv(args.eval_csv.expanduser().resolve())

    print(json.dumps(result, indent=2))
    if args.output_json is not None:
        args.output_json.expanduser().resolve().write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
