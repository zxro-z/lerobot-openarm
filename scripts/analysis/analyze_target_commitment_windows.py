#!/usr/bin/env python3
"""Analyze target-dependent action divergence onset and build commitment windows."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


COLOR_BY_TASK_INDEX = {0: "red", 1: "blue", 2: "yellow"}
TARGET_COLORS = ("red", "blue", "yellow")
PAIR_NAMES = (("red", "blue"), ("red", "yellow"), ("blue", "yellow"))
CHUNK_SIZE = 50
PROGRESS_BIN_COUNT = 101
SUSTAINED_BINS = 5
EARLY_BASELINE_BIN_COUNT = 6
OLD_SELECTION_RATIO = 0.0665
GRASP_RATIO = 0.2198
PHASES = (
    ("open_gripper", 0.7),
    ("raise_to_safe", 1.4),
    ("transit_to_cube", 2.5),
    ("pregrasp", 1.4),
    ("descend_to_grasp", 1.65),
    ("close_gripper", 1.0),
    ("lift", 1.4),
    ("transit_to_storage", 2.5),
    ("lower_into_storage", 1.4),
    ("release", 0.7),
)


@dataclass
class CommitmentWindow:
    episode_index: int
    target_color: str
    episode_length: int
    onset_frame: int
    onset_progress: float
    window_start: int
    window_end: int
    anchor_start: int
    anchor_end: int
    confidence: str
    source: str
    between_within_at_onset: float
    absolute_divergence_at_onset: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("src/lerobot/datasets/openarm_three_color_transit_tilt_50"),
    )
    parser.add_argument(
        "--grasp-manifest",
        type=Path,
        default=Path("outputs/analysis/openarm_three_color_grasp_windows/grasp_positive_windows.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/analysis/openarm_three_color_target_commitment"),
    )
    return parser.parse_args()


def load_dataset(dataset_root: Path) -> tuple[dict[int, dict[str, object]], list[dict[str, object]]]:
    data_table = pq.read_table(
        dataset_root / "data/chunk-000/file-000.parquet",
        columns=["action", "episode_index", "frame_index", "task_index"],
    )
    meta_table = pq.read_table(dataset_root / "meta/episodes/chunk-000/file-000.parquet")
    data = data_table.to_pydict()
    meta = meta_table.to_pydict()
    episodes: dict[int, dict[str, object]] = defaultdict(lambda: {"action": [], "frame_index": []})
    for action, episode_index, frame_index, task_index in zip(
        data["action"],
        data["episode_index"],
        data["frame_index"],
        data["task_index"],
        strict=True,
    ):
        ep = episodes[int(episode_index)]
        ep["action"].append(np.asarray(action, dtype=np.float32))
        ep["frame_index"].append(int(frame_index))
        ep["target_color"] = COLOR_BY_TASK_INDEX[int(task_index)]
    meta_rows: list[dict[str, object]] = []
    for idx in range(len(meta["episode_index"])):
        meta_rows.append({key: meta[key][idx] for key in meta})
    return episodes, meta_rows


def load_grasp_manifest(path: Path) -> dict[int, dict[str, object]]:
    rows = {}
    with path.open(newline="") as file:
        for row in csv.DictReader(file):
            rows[int(row["episode_index"])] = row
    return rows


def compute_phase_boundaries() -> list[dict[str, float]]:
    total = sum(duration for _, duration in PHASES)
    output = []
    elapsed = 0.0
    for name, duration in PHASES:
        start = elapsed / total
        elapsed += duration
        end = elapsed / total
        output.append(
            {
                "phase": name,
                "duration_s": duration,
                "start_progress": start,
                "end_progress": end,
            }
        )
    return output


def phase_for_progress(progress: float, phase_boundaries: list[dict[str, float]]) -> str:
    for row in phase_boundaries:
        if progress <= row["end_progress"]:
            return str(row["phase"])
    return str(phase_boundaries[-1]["phase"])


def progress_grid() -> np.ndarray:
    return np.linspace(0.0, 1.0, PROGRESS_BIN_COUNT, dtype=np.float32)


def episode_action_matrix(episode: dict[str, object]) -> np.ndarray:
    return np.stack(episode["action"]).astype(np.float32)


def normalized_frame_samples(action: np.ndarray, grid: np.ndarray) -> np.ndarray:
    if len(action) == 1:
        return np.repeat(action, len(grid), axis=0)
    positions = np.linspace(0.0, 1.0, len(action), dtype=np.float32)
    output = np.empty((len(grid), action.shape[1]), dtype=np.float32)
    for dim in range(action.shape[1]):
        output[:, dim] = np.interp(grid, positions, action[:, dim])
    return output


def chunk_anchor_indices(length: int, grid: np.ndarray) -> np.ndarray:
    max_anchor = max(length - CHUNK_SIZE, 0)
    if max_anchor == 0:
        return np.zeros(len(grid), dtype=np.int32)
    return np.clip(np.rint(grid * max_anchor), 0, max_anchor).astype(np.int32)


def normalized_chunk_samples(action: np.ndarray, grid: np.ndarray, dims: slice) -> np.ndarray:
    anchors = chunk_anchor_indices(len(action), grid)
    chunks = []
    for anchor in anchors:
        chunk = action[anchor : anchor + CHUNK_SIZE, dims]
        if len(chunk) < CHUNK_SIZE:
            pad = np.repeat(chunk[-1:], CHUNK_SIZE - len(chunk), axis=0)
            chunk = np.concatenate([chunk, pad], axis=0)
        chunks.append(chunk.reshape(-1))
    return np.stack(chunks).astype(np.float32)


def aggregate_samples(
    episodes: dict[int, dict[str, object]],
    extractor,
) -> dict[str, dict[str, list[np.ndarray]]]:
    grouped = {
        "red": defaultdict(list),
        "blue": defaultdict(list),
        "yellow": defaultdict(list),
    }
    grid = progress_grid()
    for episode_index, episode in episodes.items():
        samples = extractor(episode_action_matrix(episode), grid)
        color = str(episode["target_color"])
        for bin_idx, sample in enumerate(samples):
            grouped[color][str(bin_idx)].append(sample)
    return grouped


def summarize_bin(color_rows: dict[str, list[np.ndarray]]) -> dict[str, float]:
    means = {color: np.stack(rows).mean(axis=0) for color, rows in color_rows.items()}
    within = {}
    for color, rows in color_rows.items():
        arr = np.stack(rows)
        within[color] = math.sqrt(float(np.mean(np.sum((arr - means[color]) ** 2, axis=1))))
    pair_values = {}
    pair_ratios = {}
    for left, right in PAIR_NAMES:
        distance = float(np.linalg.norm(means[left] - means[right]))
        pair_values[f"{left}_{right}"] = distance
        pair_ratios[f"{left}_{right}"] = distance / (0.5 * (within[left] + within[right]) + 1e-8)
    between = float(np.mean(list(pair_values.values())))
    within_mean = float(np.mean(list(within.values())))
    return {
        "between": between,
        "within": within_mean,
        "between_within_ratio": between / (within_mean + 1e-8),
        "rb_between": pair_values["red_blue"],
        "ry_between": pair_values["red_yellow"],
        "by_between": pair_values["blue_yellow"],
        "rb_ratio": pair_ratios["red_blue"],
        "ry_ratio": pair_ratios["red_yellow"],
        "by_ratio": pair_ratios["blue_yellow"],
    }


def divergence_curve(
    grouped_samples: dict[str, dict[str, list[np.ndarray]]],
    grid: np.ndarray,
) -> list[dict[str, float]]:
    rows = []
    for bin_idx, progress in enumerate(grid):
        color_rows = {color: grouped_samples[color][str(bin_idx)] for color in TARGET_COLORS}
        if any(not rows_for_color for rows_for_color in color_rows.values()):
            continue
        summary = summarize_bin(color_rows)
        rows.append({"bin": int(bin_idx), "progress": float(progress), **summary})
    return rows


def smooth_curve(values: list[float], radius: int = 1) -> list[float]:
    smoothed = []
    for idx in range(len(values)):
        start = max(0, idx - radius)
        end = min(len(values), idx + radius + 1)
        smoothed.append(float(np.mean(values[start:end])))
    return smoothed


def detect_onset(
    rows: list[dict[str, float]],
    *,
    between_key: str,
    ratio_key: str,
    sustained_bins: int,
) -> dict[str, float] | None:
    between_values = [float(row[between_key]) for row in rows]
    ratio_values = [float(row[ratio_key]) for row in rows]
    smoothed_between = smooth_curve(between_values, radius=1)
    smoothed_ratio = smooth_curve(ratio_values, radius=1)
    early_between = between_values[:EARLY_BASELINE_BIN_COUNT]
    early_ratio = ratio_values[:EARLY_BASELINE_BIN_COUNT]
    baseline_between = float(np.median(early_between))
    baseline_ratio = float(np.median(early_ratio))
    between_threshold = max(0.25, 10.0 * baseline_between)
    ratio_threshold = max(0.12, 0.8 * baseline_ratio)
    for start_idx in range(0, len(rows) - sustained_bins + 1):
        window = rows[start_idx : start_idx + sustained_bins]
        if all(
            smoothed_between[start_idx + offset] >= between_threshold
            and smoothed_ratio[start_idx + offset] >= ratio_threshold
            for offset, _ in enumerate(window)
        ):
            onset = dict(window[0])
            onset["between_threshold"] = between_threshold
            onset["ratio_threshold"] = ratio_threshold
            return onset
    return None


def ratio_from_rows(rows_by_color: dict[str, list[np.ndarray]]) -> tuple[float, dict[str, float]]:
    means = {color: np.stack(rows).mean(axis=0) for color, rows in rows_by_color.items()}
    within = {
        color: math.sqrt(float(np.mean(np.sum((np.stack(rows) - means[color]) ** 2, axis=1))))
        for color, rows in rows_by_color.items()
    }
    pair_distances = {
        f"{left}_{right}": float(np.linalg.norm(means[left] - means[right]))
        for left, right in PAIR_NAMES
    }
    between_mean = float(np.mean(list(pair_distances.values())))
    return between_mean / (float(np.mean(list(within.values()))) + 1e-8), pair_distances


def sweep_global_windows(episodes: dict[int, dict[str, object]]) -> list[dict[str, object]]:
    rows = []
    for center_progress in np.arange(0.05, 0.31, 0.01):
        rows_by_color: dict[str, list[np.ndarray]] = defaultdict(list)
        anchor_rows_by_color: dict[str, list[np.ndarray]] = defaultdict(list)
        for episode in episodes.values():
            action = episode_action_matrix(episode)
            length = len(action)
            center_frame = int(round(center_progress * (length - 1)))
            window_start = max(0, center_frame - 20)
            window_end = min(length - 1, center_frame + 30)
            anchor_start = max(0, center_frame - 30)
            anchor_end = min(length - CHUNK_SIZE, center_frame - 10)
            color = str(episode["target_color"])
            rows_by_color[color].extend(list(action[window_start : window_end + 1]))
            if anchor_end >= anchor_start:
                anchor_rows_by_color[color].extend(list(action[anchor_start : anchor_end + 1]))
        window_ratio, pair_distances = ratio_from_rows(rows_by_color)
        anchor_ratio, _ = ratio_from_rows(anchor_rows_by_color)
        rows.append(
            {
                "center_progress": float(round(center_progress, 2)),
                "window_ratio": window_ratio,
                "anchor_ratio": anchor_ratio,
                "rb_distance": pair_distances["red_blue"],
                "ry_distance": pair_distances["red_yellow"],
                "by_distance": pair_distances["blue_yellow"],
            }
        )
    return rows


def choose_commitment_center(sweep_rows: list[dict[str, object]], onset_progress: float) -> dict[str, object]:
    for row in sweep_rows:
        center_progress = float(row["center_progress"])
        if center_progress < max(0.18, onset_progress):
            continue
        if float(row["window_ratio"]) > OLD_SELECTION_RATIO:
            return row
    return max(sweep_rows, key=lambda row: float(row["window_ratio"]))


def build_windows(
    episodes: dict[int, dict[str, object]],
    center_progress: float,
    window_ratio: float,
    pair_distances: dict[str, float],
) -> list[CommitmentWindow]:
    windows = []
    for episode_index, episode in episodes.items():
        action = episode_action_matrix(episode)
        length = len(action)
        color = str(episode["target_color"])
        onset_frame = int(round(center_progress * (length - 1)))
        window_start = max(0, onset_frame - 20)
        window_end = min(length - 1, onset_frame + 30)
        anchor_start = max(0, onset_frame - 30)
        anchor_end = min(length - CHUNK_SIZE, onset_frame - 10)
        windows.append(
            CommitmentWindow(
                episode_index=episode_index,
                target_color=color,
                episode_length=length,
                onset_frame=onset_frame,
                onset_progress=center_progress,
                window_start=window_start,
                window_end=window_end,
                anchor_start=max(0, anchor_start),
                anchor_end=max(max(0, anchor_start), anchor_end),
                confidence="medium",
                source="global_progress_window_sweep_after_chunk_onset",
                between_within_at_onset=window_ratio,
                absolute_divergence_at_onset=float(np.mean(list(pair_distances.values()))),
            )
        )
    return windows


def rows_to_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def commitment_indices_from_windows(
    meta_rows: list[dict[str, object]],
    windows: list[CommitmentWindow],
) -> tuple[set[int], Counter[str]]:
    indices: set[int] = set()
    colors = Counter()
    meta_by_episode = {int(row["episode_index"]): row for row in meta_rows}
    for window in windows:
        if window.confidence not in {"high", "medium"}:
            continue
        episode_meta = meta_by_episode[window.episode_index]
        episode_start = int(episode_meta["dataset_from_index"])
        for frame in range(window.anchor_start, window.anchor_end + 1):
            indices.add(episode_start + frame)
            colors[window.target_color] += 1
    return indices, colors


def commitment_window_frames(windows: list[CommitmentWindow]) -> int:
    return sum(
        max(0, window.window_end - window.window_start + 1)
        for window in windows
        if window.confidence in {"high", "medium"}
    )


def compute_overlap(windows: list[CommitmentWindow], grasp_manifest: dict[int, dict[str, object]], total_frames: int) -> dict[str, float]:
    selection_only = 0
    grasp_only = 0
    overlap = 0
    neither = 0
    for window in windows:
        length = window.episode_length
        commitment_mask = np.zeros(length, dtype=bool)
        grasp_mask = np.zeros(length, dtype=bool)
        if window.confidence in {"high", "medium"}:
            commitment_mask[window.anchor_start : window.anchor_end + 1] = True
        grasp = grasp_manifest.get(window.episode_index)
        if grasp is not None:
            start = int(grasp["anchor_start"])
            end = int(grasp["anchor_end"])
            grasp_mask[start : end + 1] = True
        selection_only += int(np.logical_and(commitment_mask, ~grasp_mask).sum())
        grasp_only += int(np.logical_and(grasp_mask, ~commitment_mask).sum())
        overlap += int(np.logical_and(commitment_mask, grasp_mask).sum())
        neither += int(np.logical_and(~commitment_mask, ~grasp_mask).sum())
    return {
        "selection_only_fraction": selection_only / total_frames,
        "grasp_only_fraction": grasp_only / total_frames,
        "overlap_fraction": overlap / total_frames,
        "neither_fraction": neither / total_frames,
    }


def simulate_sampling(
    total_frames: int,
    grasp_indices: set[int],
    commitment_indices: set[int],
    color_by_index: dict[int, str],
) -> list[dict[str, object]]:
    scenarios = [
        ("baseline", 1.0, 1.0),
        ("grasp3x", 3.0, 1.0),
        ("grasp3x_commitment1.5x", 3.0, 1.5),
        ("grasp3x_commitment2x", 3.0, 2.0),
        ("grasp3x_commitment3x", 3.0, 3.0),
    ]
    rows = []
    all_indices = range(total_frames)
    for name, grasp_weight, commitment_weight in scenarios:
        total_weight = 0.0
        grasp_weight_sum = 0.0
        commitment_weight_sum = 0.0
        union_weight_sum = 0.0
        color_weight = Counter()
        for idx in all_indices:
            weight = 1.0
            in_grasp = idx in grasp_indices
            in_commitment = idx in commitment_indices
            if in_grasp:
                weight = max(weight, grasp_weight)
                grasp_weight_sum += weight
            if in_commitment:
                weight = max(weight, commitment_weight)
                commitment_weight_sum += weight
            if in_grasp or in_commitment:
                union_weight_sum += weight
            total_weight += weight
            if in_commitment:
                color_weight[color_by_index[idx]] += weight
        rows.append(
            {
                "scenario": name,
                "grasp_weight": grasp_weight,
                "commitment_weight": commitment_weight,
                "effective_grasp_share": grasp_weight_sum / total_weight,
                "effective_commitment_share": commitment_weight_sum / total_weight,
                "union_positive_share": union_weight_sum / total_weight,
                "non_positive_share": 1.0 - union_weight_sum / total_weight,
                "red_share": color_weight["red"] / max(sum(color_weight.values()), 1e-8),
                "blue_share": color_weight["blue"] / max(sum(color_weight.values()), 1e-8),
                "yellow_share": color_weight["yellow"] / max(sum(color_weight.values()), 1e-8),
            }
        )
    return rows


def future_training_command(commitment_manifest: Path) -> str:
    return (
        "cd /home/zxro/arena/lerobot && \\\n"
        "HF_HOME=/home/zxro/.cache/hf_lerobot \\\n"
        "HF_DATASETS_CACHE=/home/zxro/.cache/hf_lerobot/datasets \\\n"
        "/home/zxro/miniforge3/envs/lab-isaac5-py311/bin/python \\\n"
        "  /home/zxro/arena/lerobot/scripts/train/train_smolvla.py \\\n"
        "  --policy-path lerobot/smolvla_base \\\n"
        "  --dataset-source local \\\n"
        "  --dataset-root /home/zxro/arena/lerobot/src/lerobot/datasets/openarm_three_color_transit_tilt_50 \\\n"
        "  --dataset-repo-id local/openarm_three_color_transit_tilt_50 \\\n"
        "  --output-dir /home/zxro/arena/lerobot/outputs/train/openarm_three_color_transit_tilt_50_vlm_unfreeze_grasp3x_commitment2x \\\n"
        "  --batch-size 2 \\\n"
        "  --steps 20000 \\\n"
        "  --seed 1000 \\\n"
        "  --device cuda \\\n"
        "  --use-amp \\\n"
        "  --no-train-expert-only \\\n"
        "  --freeze-vision-encoder \\\n"
        "  --train-state-proj \\\n"
        "  --attention-mode cross_attn \\\n"
        "  --grasp-positive-manifest /home/zxro/arena/lerobot/outputs/analysis/openarm_three_color_grasp_windows/grasp_positive_windows.csv \\\n"
        "  --grasp-positive-weight 3.0 \\\n"
        f"  --target-commitment-manifest {commitment_manifest} \\\n"
        "  --target-commitment-weight 2.0"
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    episodes, meta_rows = load_dataset(args.dataset_root)
    grasp_manifest = load_grasp_manifest(args.grasp_manifest)
    total_frames = sum(len(episode["action"]) for episode in episodes.values())
    grid = progress_grid()

    frame_all = aggregate_samples(episodes, lambda action, grid: normalized_frame_samples(action, grid))
    frame_arm = aggregate_samples(episodes, lambda action, grid: normalized_frame_samples(action[:, :7], grid))
    chunk_all = aggregate_samples(episodes, lambda action, grid: normalized_chunk_samples(action, grid, slice(0, 8)))
    chunk_arm = aggregate_samples(episodes, lambda action, grid: normalized_chunk_samples(action, grid, slice(0, 7)))

    frame_all_rows = divergence_curve(frame_all, grid)
    frame_arm_rows = divergence_curve(frame_arm, grid)
    chunk_all_rows = divergence_curve(chunk_all, grid)
    chunk_arm_rows = divergence_curve(chunk_arm, grid)

    aggregate_onset = detect_onset(chunk_all_rows, between_key="between", ratio_key="between_within_ratio", sustained_bins=SUSTAINED_BINS)
    rb_onset = detect_onset(chunk_all_rows, between_key="rb_between", ratio_key="rb_ratio", sustained_bins=SUSTAINED_BINS)
    ry_onset = detect_onset(chunk_all_rows, between_key="ry_between", ratio_key="ry_ratio", sustained_bins=SUSTAINED_BINS)
    by_onset = detect_onset(chunk_all_rows, between_key="by_between", ratio_key="by_ratio", sustained_bins=SUSTAINED_BINS)
    sweep_rows = sweep_global_windows(episodes)
    recommended_center = choose_commitment_center(sweep_rows, float(aggregate_onset["progress"]) if aggregate_onset else 0.0)
    pair_distances = {
        "red_blue": float(recommended_center["rb_distance"]),
        "red_yellow": float(recommended_center["ry_distance"]),
        "blue_yellow": float(recommended_center["by_distance"]),
    }
    windows = build_windows(
        episodes,
        center_progress=float(recommended_center["center_progress"]),
        window_ratio=float(recommended_center["window_ratio"]),
        pair_distances=pair_distances,
    )

    commitment_indices, commitment_color_counts = commitment_indices_from_windows(meta_rows, windows)
    grasp_indices, _ = commitment_indices_from_windows(
        meta_rows,
        [
            CommitmentWindow(
                episode_index=episode_index,
                target_color=str(row["target_color"]),
                episode_length=int(row["episode_length"]),
                onset_frame=0,
                onset_progress=0.0,
                window_start=int(row["window_start"]),
                window_end=int(row["window_end"]),
                anchor_start=int(row["anchor_start"]),
                anchor_end=int(row["anchor_end"]),
                confidence=str(row["confidence"]),
                source=str(row["source"]),
                between_within_at_onset=0.0,
                absolute_divergence_at_onset=0.0,
            )
            for episode_index, row in grasp_manifest.items()
        ],
    )
    color_by_index = {}
    meta_by_episode = {int(row["episode_index"]): row for row in meta_rows}
    for episode_index, episode in episodes.items():
        episode_start = int(meta_by_episode[episode_index]["dataset_from_index"])
        color = str(episode["target_color"])
        for local_idx in range(len(episode["action"])):
            color_by_index[episode_start + local_idx] = color
    overlap = compute_overlap(windows, grasp_manifest, total_frames)
    sampling_rows = simulate_sampling(total_frames, grasp_indices, commitment_indices, color_by_index)

    onset_progress = float(aggregate_onset["progress"]) if aggregate_onset is not None else -1.0
    onset_frame = int(round(onset_progress * np.median([len(ep["action"]) - 1 for ep in episodes.values()]))) if aggregate_onset is not None else -1
    phase_boundaries = compute_phase_boundaries()
    phase_name = phase_for_progress(onset_progress, phase_boundaries) if aggregate_onset is not None else "unknown"

    stats = {
        "dataset_root": str(args.dataset_root.resolve()),
        "grasp_manifest": str(args.grasp_manifest.resolve()),
        "action_layout": {
            "all_dims": ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6", "joint_7", "gripper"],
            "arm_only_dims": ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6", "joint_7"],
        },
        "frame_divergence_rows": frame_all_rows,
        "frame_arm_divergence_rows": frame_arm_rows,
        "chunk_divergence_rows": chunk_all_rows,
        "chunk_arm_divergence_rows": chunk_arm_rows,
        "aggregate_onset": aggregate_onset,
        "pairwise_onsets": {
            "red_blue": rb_onset,
            "red_yellow": ry_onset,
            "blue_yellow": by_onset,
        },
        "global_window_sweep": sweep_rows,
        "recommended_center": recommended_center,
        "generator_phase_boundaries": phase_boundaries,
        "aggregate_phase_mapping": {
            "onset_progress": onset_progress,
            "onset_frame": onset_frame,
            "phase": phase_name,
        },
        "episode_level_onset_variation": {
            "count_high_confidence": 0,
            "status": "not_estimated_per_episode",
            "reason": "global progress sweep was strong enough for window design, but per-episode onset estimation was not robust enough to trust for weighting",
        },
        "window_comparison": {
            "full_dataset": 0.0640,
            "old_selection_window": OLD_SELECTION_RATIO,
            "new_commitment_window": float(recommended_center["window_ratio"]),
            "grasp_window": GRASP_RATIO,
        },
        "recommended_window": {
            "window_start_offset": -20,
            "window_end_offset": 30,
            "anchor_start_offset": -30,
            "anchor_end_offset": -10,
            "chunk_size": CHUNK_SIZE,
            "center_progress": float(recommended_center["center_progress"]),
        },
        "overlap_with_grasp": overlap,
        "commitment_color_counts": dict(commitment_color_counts),
        "sampling_simulation": sampling_rows,
        "future_training_command": future_training_command(args.output_dir / "target_commitment_windows.csv"),
    }

    curve_rows = []
    for label, rows in (
        ("frame_all", frame_all_rows),
        ("frame_arm", frame_arm_rows),
        ("chunk_all", chunk_all_rows),
        ("chunk_arm", chunk_arm_rows),
    ):
        for row in rows:
            curve_rows.append({"representation": label, **row})

    rows_to_csv(args.output_dir / "divergence_curves.csv", curve_rows)
    rows_to_csv(
        args.output_dir / "target_commitment_windows.csv",
        [window.__dict__ for window in windows],
    )
    rows_to_csv(args.output_dir / "sampling_simulation.csv", sampling_rows)
    (args.output_dir / "target_commitment_stats.json").write_text(json.dumps(stats, indent=2))

    report_lines = [
        "E. Target-Dependent Divergence Curve",
        f"aggregate chunk onset progress = {onset_progress:.4f}",
        f"aggregate chunk onset frame = {onset_frame}",
        "",
        "F. Divergence Onset",
        f"aggregate onset = {aggregate_onset}",
        f"RB onset = {rb_onset}",
        f"RY onset = {ry_onset}",
        f"BY onset = {by_onset}",
        "",
        "G. Generator Phase Mapping",
        f"aggregate onset phase = {phase_name}",
        "",
        "H. Old Selection vs New Commitment Window",
        f"full dataset = 0.0640",
        f"old selection = {OLD_SELECTION_RATIO:.4f}",
        f"new commitment = {stats['window_comparison']['new_commitment_window']:.4f}",
        f"grasp = {GRASP_RATIO:.4f}",
        "",
        "I. Recommended Commitment Window",
        f"global center progress {float(recommended_center['center_progress']):.2f} with window center-20 .. center+30",
        "",
        "J. Recommended Anchor Band",
        "center-30 .. center-10",
        "",
        "K. Overlap With Grasp Anchors",
        str(overlap),
        "",
        "L. Sampling Simulation",
    ]
    for row in sampling_rows:
        report_lines.append(str(row))
    (args.output_dir / "report.txt").write_text("\n".join(report_lines) + "\n")

    print(f"[RESULT] curves_csv={args.output_dir / 'divergence_curves.csv'}")
    print(f"[RESULT] commitment_manifest={args.output_dir / 'target_commitment_windows.csv'}")
    print(f"[RESULT] stats_json={args.output_dir / 'target_commitment_stats.json'}")
    print(f"[RESULT] sampling_csv={args.output_dir / 'sampling_simulation.csv'}")
    print(f"[RESULT] report={args.output_dir / 'report.txt'}")


if __name__ == "__main__":
    main()
