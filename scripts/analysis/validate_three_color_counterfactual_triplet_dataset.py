#!/usr/bin/env python3
"""Validate a same-layout OpenArm three-color counterfactual triplet dataset."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path

import av
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.optimize import minimize


COLORS = ("red", "blue", "yellow")
COLOR_TO_LABEL = {color: idx for idx, color in enumerate(COLORS)}
PAIRS = (("red", "blue"), ("red", "yellow"), ("blue", "yellow"))
CHUNK_SIZE = 50
PROGRESS_GRID = np.linspace(0.0, 1.0, 101, dtype=np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("src/lerobot/datasets/openarm_three_color_counterfactual_triplet_50"),
    )
    parser.add_argument(
        "--layout-manifest",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--triplet-manifest",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/analysis/openarm_three_color_counterfactual_triplet_validation"),
    )
    return parser.parse_args()


class Standardizer:
    def fit(self, x: np.ndarray) -> "Standardizer":
        self.mean_ = x.mean(axis=0)
        self.std_ = x.std(axis=0)
        self.std_[self.std_ < 1e-8] = 1.0
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean_) / self.std_


class MultinomialLogReg:
    def __init__(self, reg: float = 1e-2, maxiter: int = 300):
        self.reg = reg
        self.maxiter = maxiter

    def fit(self, x: np.ndarray, y: np.ndarray, num_classes: int) -> "MultinomialLogReg":
        y_one_hot = np.eye(num_classes, dtype=np.float64)[y]
        x_aug = np.concatenate([x, np.ones((len(x), 1), dtype=np.float64)], axis=1)

        def softmax(logits: np.ndarray) -> np.ndarray:
            shifted = logits - logits.max(axis=1, keepdims=True)
            exp = np.exp(shifted)
            return exp / exp.sum(axis=1, keepdims=True)

        def objective(flat_w: np.ndarray) -> tuple[float, np.ndarray]:
            w = flat_w.reshape(x_aug.shape[1], num_classes)
            probs = softmax(x_aug @ w)
            loss = -np.sum(y_one_hot * np.log(probs + 1e-12)) / len(x)
            loss += 0.5 * self.reg * np.sum(w[:-1] ** 2)
            grad = (x_aug.T @ (probs - y_one_hot)) / len(x)
            grad[:-1] += self.reg * w[:-1]
            return float(loss), grad.ravel()

        result = minimize(
            fun=lambda flat: objective(flat)[0],
            x0=np.zeros((x_aug.shape[1], num_classes), dtype=np.float64).ravel(),
            jac=lambda flat: objective(flat)[1],
            method="L-BFGS-B",
            options={"maxiter": self.maxiter},
        )
        self.weights_ = result.x.reshape(x_aug.shape[1], num_classes)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        x_aug = np.concatenate([x, np.ones((len(x), 1), dtype=np.float64)], axis=1)
        return np.argmax(x_aug @ self.weights_, axis=1)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def load_inputs(dataset_root: Path, layout_manifest: Path, triplet_manifest: Path):
    episodes = pq.read_table(dataset_root / "meta/episodes/chunk-000/file-000.parquet").to_pandas()
    data = pq.read_table(
        dataset_root / "data/chunk-000/file-000.parquet",
        columns=["observation.state", "action", "episode_index", "frame_index", "task_index"],
    ).to_pydict()
    layout_df = pd.read_csv(layout_manifest)
    triplet_df = pd.read_csv(triplet_manifest)
    per_episode: dict[int, dict[str, object]] = {}
    for state, action, episode_index, frame_index, task_index in zip(
        data["observation.state"],
        data["action"],
        data["episode_index"],
        data["frame_index"],
        data["task_index"],
        strict=True,
    ):
        ep = per_episode.setdefault(
            int(episode_index),
            {"state": [], "action": [], "frame_index": [], "task_index": int(task_index)},
        )
        ep["state"].append(np.asarray(state, dtype=np.float32))
        ep["action"].append(np.asarray(action, dtype=np.float32))
        ep["frame_index"].append(int(frame_index))
    for episode_index, ep in per_episode.items():
        ep["state"] = np.stack(ep["state"]).astype(np.float32)
        ep["action"] = np.stack(ep["action"]).astype(np.float32)
        ep["initial_state"] = ep["state"][0]
        ep["target_color"] = COLORS[int(ep["task_index"])]
    return episodes, per_episode, layout_df, triplet_df


def decode_first_frames(dataset_root: Path, episodes_df: pd.DataFrame) -> dict[int, np.ndarray]:
    requests: dict[tuple[int, int], dict[int, int]] = {}
    for row in episodes_df.to_dict("records"):
        key = (
            int(row["videos/observation.images.top/chunk_index"]),
            int(row["videos/observation.images.top/file_index"]),
        )
        requests.setdefault(key, {})[int(row["episode_index"])] = int(
            round(float(row["videos/observation.images.top/from_timestamp"]) * 30.0)
        )
    frames = {}
    for (chunk_idx, file_idx), mapping in requests.items():
        needed = {}
        for episode_index, frame_idx in mapping.items():
            needed.setdefault(frame_idx, []).append(episode_index)
        path = dataset_root / f"videos/observation.images.top/chunk-{chunk_idx:03d}/file-{file_idx:03d}.mp4"
        container = av.open(str(path))
        stream = container.streams.video[0]
        for current_idx, frame in enumerate(container.decode(stream)):
            if current_idx not in needed:
                continue
            image = frame.to_ndarray(format="rgb24")
            for episode_index in needed[current_idx]:
                frames[episode_index] = image.copy()
            if len(frames) == len(episodes_df):
                break
        container.close()
    return frames


def normalized_frame_samples(action: np.ndarray) -> np.ndarray:
    if len(action) == 1:
        return np.repeat(action, len(PROGRESS_GRID), axis=0)
    positions = np.linspace(0.0, 1.0, len(action), dtype=np.float32)
    out = np.empty((len(PROGRESS_GRID), action.shape[1]), dtype=np.float32)
    for dim in range(action.shape[1]):
        out[:, dim] = np.interp(PROGRESS_GRID, positions, action[:, dim])
    return out


def normalized_chunk_samples(action: np.ndarray, dims: slice) -> np.ndarray:
    max_anchor = max(len(action) - CHUNK_SIZE, 0)
    anchors = np.clip(np.rint(PROGRESS_GRID * max_anchor), 0, max_anchor).astype(np.int32)
    samples = []
    for anchor in anchors:
        chunk = action[anchor : anchor + CHUNK_SIZE, dims]
        if len(chunk) < CHUNK_SIZE:
            pad = np.repeat(chunk[-1:], CHUNK_SIZE - len(chunk), axis=0)
            chunk = np.concatenate([chunk, pad], axis=0)
        samples.append(chunk.reshape(-1))
    return np.stack(samples).astype(np.float32)


def grouped_folds(layout_ids: np.ndarray, n_splits: int = 5) -> list[np.ndarray]:
    unique = np.array(sorted(np.unique(layout_ids)), dtype=np.int64)
    folds = np.array_split(unique, min(n_splits, len(unique)))
    return [fold.astype(np.int64) for fold in folds if len(fold) > 0]


def geometry_group_accuracy(layout_df: pd.DataFrame) -> tuple[float, list[float]]:
    feature_cols = [
        "tcp_tilt_deg",
        "red_x",
        "red_y",
        "red_z",
        "red_qw",
        "red_qx",
        "red_qy",
        "red_qz",
        "blue_x",
        "blue_y",
        "blue_z",
        "blue_qw",
        "blue_qx",
        "blue_qy",
        "blue_qz",
        "yellow_x",
        "yellow_y",
        "yellow_z",
        "yellow_qw",
        "yellow_qx",
        "yellow_qy",
        "yellow_qz",
    ] + [col for col in layout_df.columns if col.startswith("robot_initial_state_deg_")]
    x = layout_df[feature_cols].to_numpy(dtype=np.float64)
    y = layout_df["target_color"].map(COLOR_TO_LABEL).to_numpy(dtype=np.int64)
    groups = layout_df["layout_id"].to_numpy(dtype=np.int64)
    folds = grouped_folds(groups)
    accuracies = []
    for test_groups in folds:
        test_mask = np.isin(groups, test_groups)
        train_mask = ~test_mask
        scaler = Standardizer().fit(x[train_mask])
        x_train = scaler.transform(x[train_mask])
        x_test = scaler.transform(x[test_mask])
        model = MultinomialLogReg().fit(x_train, y[train_mask], 3)
        pred = model.predict(x_test)
        accuracies.append(float((pred == y[test_mask]).mean()))
    return float(np.mean(accuracies)), accuracies


def color_position_balance(layout_df: pd.DataFrame) -> list[dict[str, object]]:
    one_per_layout = layout_df.sort_values(["layout_id", "target_color"]).groupby("layout_id", as_index=False).first()
    rows = []
    rank_counts = {color: Counter() for color in COLORS}
    for row in one_per_layout.to_dict("records"):
        x_positions = {"red": row["red_x"], "blue": row["blue_x"], "yellow": row["yellow_x"]}
        ranked = sorted(x_positions.items(), key=lambda item: item[1])
        for rank, (color, _) in enumerate(ranked):
            rank_counts[color][("left", "middle", "right")[rank]] += 1
    total = len(one_per_layout)
    for color in COLORS:
        rows.append(
            {
                "color": color,
                "left_fraction": rank_counts[color]["left"] / total,
                "middle_fraction": rank_counts[color]["middle"] / total,
                "right_fraction": rank_counts[color]["right"] / total,
            }
        )
    return rows


def triplet_consistency(
    episodes_df: pd.DataFrame,
    per_episode: dict[int, dict[str, object]],
    layout_df: pd.DataFrame,
    triplet_df: pd.DataFrame,
    first_frames: dict[int, np.ndarray],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    rows = []
    pose_diffs = []
    state_diffs = []
    image_maes = []
    for triplet in triplet_df.to_dict("records"):
        layout_id = int(triplet["layout_id"])
        group = layout_df.loc[layout_df["layout_id"] == layout_id].sort_values("target_color")
        target_colors = tuple(group["target_color"].tolist())
        episode_ids = group["episode_index"].to_numpy(dtype=np.int64)
        pose_cols = [
            "red_x", "red_y", "red_z", "red_qw", "red_qx", "red_qy", "red_qz",
            "blue_x", "blue_y", "blue_z", "blue_qw", "blue_qx", "blue_qy", "blue_qz",
            "yellow_x", "yellow_y", "yellow_z", "yellow_qw", "yellow_qx", "yellow_qy", "yellow_qz",
            "tcp_tilt_deg",
        ]
        pose_matrix = group[pose_cols].to_numpy(dtype=np.float64)
        pose_max_abs_diff = float(np.max(np.abs(pose_matrix - pose_matrix[0:1])))
        pose_diffs.append(pose_max_abs_diff)

        initial_states = np.stack([per_episode[int(ep)]["initial_state"] for ep in episode_ids]).astype(np.float64)
        state_max_abs_diff = float(np.max(np.abs(initial_states - initial_states[0:1])))
        state_l2_max = float(np.max(np.linalg.norm(initial_states - initial_states[0:1], axis=1)))
        state_diffs.append(state_max_abs_diff)

        frame_triplet = [first_frames[int(ep)] for ep in episode_ids]
        maes = []
        for i in range(len(frame_triplet)):
            for j in range(i + 1, len(frame_triplet)):
                maes.append(float(np.mean(np.abs(frame_triplet[i].astype(np.float32) - frame_triplet[j].astype(np.float32)))))
        mean_mae = float(np.mean(maes))
        max_mae = float(np.max(maes))
        image_maes.append(mean_mae)

        rows.append(
            {
                "layout_id": layout_id,
                "episode_count": int(len(group)),
                "target_colors": ",".join(target_colors),
                "red_episode": int(triplet["red_episode"]),
                "blue_episode": int(triplet["blue_episode"]),
                "yellow_episode": int(triplet["yellow_episode"]),
                "pose_max_abs_diff": pose_max_abs_diff,
                "initial_state_max_abs_diff": state_max_abs_diff,
                "initial_state_l2_max": state_l2_max,
                "first_frame_pairwise_mae_mean": mean_mae,
                "first_frame_pairwise_mae_max": max_mae,
            }
        )
    summary = {
        "pose_max_abs_diff_overall": float(max(pose_diffs) if pose_diffs else 0.0),
        "initial_state_max_abs_diff_overall": float(max(state_diffs) if state_diffs else 0.0),
        "first_frame_mae_mean_overall": float(np.mean(image_maes) if image_maes else 0.0),
        "first_frame_mae_max_overall": float(np.max(image_maes) if image_maes else 0.0),
    }
    return rows, summary


def exact_counterfactual_divergence(layout_df: pd.DataFrame, per_episode: dict[int, dict[str, object]]) -> tuple[list[dict[str, object]], dict[str, object]]:
    rows = []
    onset_summary = {}
    for left_color, right_color in PAIRS:
        pair_key = f"{left_color}_{right_color}"
        frame_curves = []
        chunk_curves = []
        for layout_id, group in layout_df.groupby("layout_id"):
            left_episode = int(group.loc[group["target_color"] == left_color, "episode_index"].iloc[0])
            right_episode = int(group.loc[group["target_color"] == right_color, "episode_index"].iloc[0])
            left_action = per_episode[left_episode]["action"]
            right_action = per_episode[right_episode]["action"]
            left_frame = normalized_frame_samples(left_action)[:, :7]
            right_frame = normalized_frame_samples(right_action)[:, :7]
            left_chunk = normalized_chunk_samples(left_action, slice(0, 7))
            right_chunk = normalized_chunk_samples(right_action, slice(0, 7))
            frame_curves.append(np.linalg.norm(left_frame - right_frame, axis=1))
            chunk_curves.append(np.linalg.norm(left_chunk - right_chunk, axis=1))
        frame_curves_np = np.stack(frame_curves).astype(np.float32)
        chunk_curves_np = np.stack(chunk_curves).astype(np.float32)
        chunk_mean = chunk_curves_np.mean(axis=0)
        baseline = float(np.median(chunk_mean[:6]))
        threshold = max(0.25, baseline + 0.15)
        onset = None
        for start_idx in range(len(PROGRESS_GRID) - 4):
            if all(chunk_mean[start_idx + offset] >= threshold for offset in range(5)):
                onset = float(PROGRESS_GRID[start_idx])
                break
        onset_summary[pair_key] = onset
        for bin_idx, progress in enumerate(PROGRESS_GRID):
            rows.append(
                {
                    "pair_type": pair_key,
                    "progress": float(progress),
                    "frame_arm_mean": float(frame_curves_np[:, bin_idx].mean()),
                    "frame_arm_median": float(np.median(frame_curves_np[:, bin_idx])),
                    "chunk_arm_mean": float(chunk_curves_np[:, bin_idx].mean()),
                    "chunk_arm_median": float(np.median(chunk_curves_np[:, bin_idx])),
                    "support_layouts": int(len(frame_curves_np)),
                }
            )
    valid_onsets = [value for value in onset_summary.values() if value is not None]
    onset_summary["aggregate"] = float(np.median(valid_onsets)) if valid_onsets else None
    return rows, onset_summary


def build_report(
    dataset_root: Path,
    layout_df: pd.DataFrame,
    triplet_df: pd.DataFrame,
    geometry_acc: float,
    geometry_folds: list[float],
    consistency_summary: dict[str, object],
    divergence_summary: dict[str, object],
    balance_rows: list[dict[str, object]],
) -> tuple[str, dict[str, object]]:
    episode_count = int(len(layout_df))
    layout_count = int(layout_df["layout_id"].nunique())
    target_counts = layout_df["target_color"].value_counts().to_dict()
    triplet_complete = bool(
        layout_count == len(triplet_df)
        and all(value == 3 for value in layout_df.groupby("layout_id")["episode_index"].count().tolist())
        and all(
            set(group["target_color"]) == set(COLORS)
            for _, group in layout_df.groupby("layout_id")
        )
    )

    hard_failures = []
    warnings = []
    if episode_count != 150:
        hard_failures.append(f"episode_count={episode_count} expected 150")
    if layout_count != 50:
        hard_failures.append(f"layout_count={layout_count} expected 50")
    if any(target_counts.get(color, 0) != 50 for color in COLORS):
        hard_failures.append(f"target_balance={target_counts} expected 50/50/50")
    if not triplet_complete:
        hard_failures.append("triplet completeness failed")
    if consistency_summary["pose_max_abs_diff_overall"] > 1e-8:
        hard_failures.append(f"pose mismatch within triplet: {consistency_summary['pose_max_abs_diff_overall']}")
    if consistency_summary["initial_state_max_abs_diff_overall"] > 1e-5:
        warnings.append(
            f"initial state mismatch above tolerance: {consistency_summary['initial_state_max_abs_diff_overall']:.6g}"
        )
    if geometry_acc > 0.45:
        warnings.append(f"group-aware geometry-only accuracy too high: {geometry_acc:.3f}")
    if divergence_summary["aggregate"] is None or divergence_summary["aggregate"] > 0.20:
        warnings.append(f"early divergence onset too late or missing: {divergence_summary['aggregate']}")
    for row in balance_rows:
        if max(row["left_fraction"], row["middle_fraction"], row["right_fraction"]) > 0.6:
            warnings.append(f"{row['color']} x-rank distribution skewed: {row}")

    if hard_failures:
        verdict = "FAIL"
    elif warnings:
        verdict = "PASS WITH WARNINGS"
    else:
        verdict = "PASS"

    summary = {
        "dataset_root": str(dataset_root.resolve()),
        "episode_count": episode_count,
        "layout_count": layout_count,
        "target_balance": target_counts,
        "exact_triplet_coverage": triplet_complete,
        "group_aware_geometry_only_accuracy": geometry_acc,
        "group_aware_geometry_fold_accuracies": geometry_folds,
        "consistency_summary": consistency_summary,
        "divergence_onset_summary": divergence_summary,
        "hard_failures": hard_failures,
        "warnings": warnings,
        "verdict": verdict,
    }

    lines = [
        "A. Dataset Counts",
        f"episodes = {episode_count}",
        f"layouts = {layout_count}",
        f"target balance = {target_counts}",
        "",
        "B. Triplet Completeness",
        f"exact triplet coverage = {triplet_complete}",
        "",
        "C. Same-Layout Consistency",
        f"pose max abs diff = {consistency_summary['pose_max_abs_diff_overall']}",
        f"initial state max abs diff = {consistency_summary['initial_state_max_abs_diff_overall']}",
        f"first-frame pairwise MAE mean = {consistency_summary['first_frame_mae_mean_overall']:.4f}",
        f"first-frame pairwise MAE max = {consistency_summary['first_frame_mae_max_overall']:.4f}",
        "",
        "D. Geometry-Only Group-Aware Accuracy",
        f"mean accuracy = {geometry_acc:.4f}",
        f"folds = {geometry_folds}",
        f"chance = {1.0 / 3.0:.4f}",
        "",
        "E. Exact Counterfactual Action Divergence",
        f"RB onset = {divergence_summary['red_blue']}",
        f"RY onset = {divergence_summary['red_yellow']}",
        f"BY onset = {divergence_summary['blue_yellow']}",
        f"aggregate onset = {divergence_summary['aggregate']}",
        "",
        "F. Color Position Balance",
    ]
    for row in balance_rows:
        lines.append(
            f"{row['color']}: left={row['left_fraction']:.3f}, middle={row['middle_fraction']:.3f}, right={row['right_fraction']:.3f}"
        )
    lines.extend(["", "G. Hard Failures"])
    lines.extend(hard_failures or ["none"])
    lines.extend(["", "H. Warnings"])
    lines.extend(warnings or ["none"])
    lines.extend(["", "I. Verdict", verdict])
    return "\n".join(lines) + "\n", summary


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.expanduser().resolve()
    layout_manifest = (args.layout_manifest or (dataset_root / "layout_manifest.csv")).expanduser().resolve()
    triplet_manifest = (args.triplet_manifest or (dataset_root / "triplet_manifest.csv")).expanduser().resolve()
    ensure_dir(args.output_dir)

    episodes_df, per_episode, layout_df, triplet_df = load_inputs(dataset_root, layout_manifest, triplet_manifest)
    first_frames = decode_first_frames(dataset_root, episodes_df)

    geometry_acc, geometry_folds = geometry_group_accuracy(layout_df)
    balance_rows = color_position_balance(layout_df)
    consistency_rows, consistency_summary = triplet_consistency(
        episodes_df,
        per_episode,
        layout_df,
        triplet_df,
        first_frames,
    )
    divergence_rows, divergence_summary = exact_counterfactual_divergence(layout_df, per_episode)

    write_csv(args.output_dir / "triplet_consistency.csv", consistency_rows)
    write_csv(args.output_dir / "color_position_balance.csv", balance_rows)
    write_csv(args.output_dir / "counterfactual_action_divergence.csv", divergence_rows)
    layout_df.to_csv(args.output_dir / "layout_manifest_audit.csv", index=False)

    report_text, summary = build_report(
        dataset_root,
        layout_df,
        triplet_df,
        geometry_acc,
        geometry_folds,
        consistency_summary,
        divergence_summary,
        balance_rows,
    )
    (args.output_dir / "dataset_validation_report.txt").write_text(report_text)
    (args.output_dir / "validation_summary.json").write_text(json.dumps(summary, indent=2))
    print(args.output_dir.resolve())


if __name__ == "__main__":
    main()
