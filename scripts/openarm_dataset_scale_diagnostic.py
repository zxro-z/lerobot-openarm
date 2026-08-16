#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from safetensors.torch import load_file

from lerobot.utils.constants import HF_LEROBOT_HOME


DEFAULT_DATASET_REPO_ID = "a126-kitech/openarm_dual_realsense_pick_place_random_cube_tilt_30_box_blue"
DEFAULT_CHECKPOINT = (
    "outputs/train/smolvla_openarm_v11_refalign/checkpoints/020000/pretrained_model"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read-only diagnostic for OpenArm dataset scale and checkpoint normalization stats."
    )
    parser.add_argument("--dataset-repo-id", default=DEFAULT_DATASET_REPO_ID)
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=Path(DEFAULT_CHECKPOINT))
    parser.add_argument("--num-samples", type=int, default=10)
    return parser.parse_args()


def resolve_dataset_root(repo_id: str, dataset_root: Path | None) -> Path:
    root = dataset_root if dataset_root is not None else HF_LEROBOT_HOME / repo_id
    root = Path(root)
    if not (root / "meta" / "stats.json").exists():
        raise FileNotFoundError(f"Dataset metadata not found under: {root}")
    return root


def load_all_vectors(dataset_root: Path, column: str) -> np.ndarray:
    parquet_files = sorted((dataset_root / "data").glob("chunk-*/file-*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found under: {dataset_root / 'data'}")

    batches: list[np.ndarray] = []
    for parquet_file in parquet_files:
        frame = pd.read_parquet(parquet_file, columns=[column])
        values = np.stack(frame[column].to_list()).astype(np.float32)
        batches.append(values)
    return np.concatenate(batches, axis=0)


def load_first_samples(dataset_root: Path, num_samples: int) -> tuple[np.ndarray, np.ndarray]:
    parquet_files = sorted((dataset_root / "data").glob("chunk-*/file-*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found under: {dataset_root / 'data'}")

    state_rows: list[np.ndarray] = []
    action_rows: list[np.ndarray] = []
    remaining = num_samples

    for parquet_file in parquet_files:
        frame = pd.read_parquet(parquet_file, columns=["observation.state", "action"])
        for state, action in zip(frame["observation.state"], frame["action"], strict=False):
            state_rows.append(np.asarray(state, dtype=np.float32))
            action_rows.append(np.asarray(action, dtype=np.float32))
            remaining -= 1
            if remaining == 0:
                return np.stack(state_rows), np.stack(action_rows)

    return np.stack(state_rows), np.stack(action_rows)


def summarize_vectors(values: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "min": values.min(axis=0),
        "max": values.max(axis=0),
        "mean": values.mean(axis=0),
        "std": values.std(axis=0),
    }


def format_vec(vec: np.ndarray) -> str:
    return np.array2string(vec, precision=6, floatmode="fixed", suppress_small=False)


def print_feature_section(name: str, values: np.ndarray, first_samples: np.ndarray) -> None:
    stats = summarize_vectors(values)
    arm = values[:, :7]
    gripper = values[:, 7]

    print(f"[{name}]")
    print(f"shape: {values.shape}")
    print("first_10_samples:")
    for idx, sample in enumerate(first_samples):
        print(f"  {idx:02d}: {format_vec(sample)}")
    print("arm_dims:")
    print(f"  min : {format_vec(stats['min'][:7])}")
    print(f"  max : {format_vec(stats['max'][:7])}")
    print(f"  mean: {format_vec(stats['mean'][:7])}")
    print(f"  std : {format_vec(stats['std'][:7])}")
    print("gripper_dim:")
    print(f"  min : {gripper.min():.6f}")
    print(f"  max : {gripper.max():.6f}")
    print(f"  mean: {gripper.mean():.6f}")
    print(f"  std : {gripper.std():.6f}")
    print("all_dims:")
    print(f"  min : {format_vec(stats['min'])}")
    print(f"  max : {format_vec(stats['max'])}")
    print(f"  mean: {format_vec(stats['mean'])}")
    print(f"  std : {format_vec(stats['std'])}")
    print()


def judge_arm_unit(values: np.ndarray) -> tuple[str, str]:
    arm = values[:, :7]
    arm_abs_max = float(np.abs(arm).max())
    arm_mean_abs = float(np.abs(arm).mean())
    rad_limit = 2 * np.pi + 0.25
    deg_limit = 10.0

    if arm_abs_max <= rad_limit and arm_mean_abs < deg_limit:
        rationale = (
            "arm values stay in the low single-digit range and peak around ~2.03, "
            "which matches radians. If they were degrees, common poses like 20 or 60 "
            "would appear directly instead of 0.349 or 1.047."
        )
        return "radian", rationale

    rationale = (
        "value range does not fit the expected radian heuristic cleanly; manual inspection is needed."
    )
    return "unknown", rationale


def load_checkpoint_stats(checkpoint: Path) -> dict[str, dict[str, np.ndarray]]:
    pre_state = load_file(
        checkpoint / "policy_preprocessor_step_5_normalizer_processor.safetensors"
    )
    stats: dict[str, dict[str, np.ndarray]] = {
        "observation.state": {},
        "action": {},
    }
    for feature in stats:
        for stat_name in ("min", "max", "mean", "std"):
            key = f"{feature}.{stat_name}"
            stats[feature][stat_name] = pre_state[key].cpu().numpy()
    return stats


def print_checkpoint_section(
    dataset_stats: dict[str, dict[str, np.ndarray]],
    checkpoint_stats: dict[str, dict[str, np.ndarray]],
) -> None:
    print("[checkpoint_normalization_stats]")
    for feature in ("observation.state", "action"):
        print(f"{feature}:")
        for stat_name in ("min", "max", "mean", "std"):
            ds = dataset_stats[feature][stat_name]
            ckpt = checkpoint_stats[feature][stat_name]
            max_abs_diff = float(np.max(np.abs(ds - ckpt)))
            print(f"  {stat_name}:")
            print(f"    dataset   : {format_vec(ds)}")
            print(f"    checkpoint: {format_vec(ckpt)}")
            print(f"    max_abs_diff: {max_abs_diff:.9f}")
    print()


def load_dataset_stats_json(dataset_root: Path) -> dict:
    with (dataset_root / "meta" / "stats.json").open() as f:
        return json.load(f)


def main() -> None:
    args = parse_args()
    dataset_root = resolve_dataset_root(args.dataset_repo_id, args.dataset_root)
    checkpoint = args.checkpoint.resolve()

    state_values = load_all_vectors(dataset_root, "observation.state")
    action_values = load_all_vectors(dataset_root, "action")
    first_states, first_actions = load_first_samples(dataset_root, args.num_samples)

    print("=== OpenArm Dataset Scale Diagnostic ===")
    print(f"dataset_repo_id: {args.dataset_repo_id}")
    print(f"dataset_root: {dataset_root}")
    print(f"checkpoint: {checkpoint}")
    print()

    print_feature_section("observation.state", state_values, first_states)
    print_feature_section("action", action_values, first_actions)

    state_unit, state_reason = judge_arm_unit(state_values)
    action_unit, action_reason = judge_arm_unit(action_values)
    print("[unit_judgment]")
    print(f"observation.state arm unit: {state_unit}")
    print(f"reason: {state_reason}")
    print(f"action arm unit: {action_unit}")
    print(f"reason: {action_reason}")
    print()

    dataset_stats_json = load_dataset_stats_json(dataset_root)
    dataset_stats = {
        "observation.state": {
            key: np.asarray(dataset_stats_json["observation.state"][key], dtype=np.float32)
            for key in ("min", "max", "mean", "std")
        },
        "action": {
            key: np.asarray(dataset_stats_json["action"][key], dtype=np.float32)
            for key in ("min", "max", "mean", "std")
        },
    }
    checkpoint_stats = load_checkpoint_stats(checkpoint)
    print_checkpoint_section(dataset_stats, checkpoint_stats)


if __name__ == "__main__":
    main()
