#!/usr/bin/env python3
"""Offline conditional-identifiability audit for the OpenArm three-color dataset."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import av
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.optimize import linear_sum_assignment, minimize


matplotlib.use("Agg")

COLOR_BY_TASK_INDEX = {0: "red", 1: "blue", 2: "yellow"}
COLORS = ("red", "blue", "yellow")
PAIRS = (("red", "blue"), ("red", "yellow"), ("blue", "yellow"))
CHUNK_SIZE = 50
PROGRESS_GRID = np.linspace(0.0, 1.0, 101, dtype=np.float32)
EARLY_WINDOWS = (
    ("0_10", 0.00, 0.10),
    ("10_20", 0.10, 0.20),
    ("20_30", 0.20, 0.30),
    ("30_40", 0.30, 0.40),
)


@dataclass
class MatchPair:
    pair_type: str
    left_episode: int
    left_color: str
    right_episode: int
    right_color: str
    distance: float
    threshold_label: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("src/lerobot/datasets/openarm_three_color_transit_tilt_50"),
    )
    parser.add_argument(
        "--commitment-manifest",
        type=Path,
        default=Path("outputs/analysis/openarm_three_color_target_commitment/target_commitment_windows.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/analysis/openarm_three_color_conditional_identifiability"),
    )
    parser.add_argument("--pca-dim", type=int, default=16)
    parser.add_argument("--tree-depth", type=int, default=3)
    parser.add_argument("--min-leaf", type=int, default=8)
    return parser.parse_args()


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


class Standardizer:
    def fit(self, x: np.ndarray) -> "Standardizer":
        self.mean_ = x.mean(axis=0)
        self.std_ = x.std(axis=0)
        self.std_[self.std_ < 1e-8] = 1.0
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean_) / self.std_


class PCAProjector:
    def __init__(self, n_components: int):
        self.n_components = n_components

    def fit(self, x: np.ndarray) -> "PCAProjector":
        u, s, vt = np.linalg.svd(x, full_matrices=False)
        del u, s
        self.components_ = vt[: min(self.n_components, vt.shape[0])]
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        return x @ self.components_.T


class MultinomialLogReg:
    def __init__(self, reg: float = 1e-2, maxiter: int = 300):
        self.reg = reg
        self.maxiter = maxiter

    def fit(self, x: np.ndarray, y: np.ndarray, num_classes: int) -> "MultinomialLogReg":
        n, d = x.shape
        y_one_hot = np.eye(num_classes, dtype=np.float64)[y]
        x_aug = np.concatenate([x, np.ones((n, 1), dtype=np.float64)], axis=1)

        def objective(flat_w: np.ndarray) -> tuple[float, np.ndarray]:
            w = flat_w.reshape(x_aug.shape[1], num_classes)
            probs = softmax(x_aug @ w)
            loss = -np.sum(y_one_hot * np.log(probs + 1e-12)) / n
            loss += 0.5 * self.reg * np.sum(w[:-1] ** 2)
            grad = (x_aug.T @ (probs - y_one_hot)) / n
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


class DecisionTreeNode:
    def __init__(self, prediction: int):
        self.prediction = prediction
        self.feature = None
        self.threshold = None
        self.left = None
        self.right = None


class ShallowDecisionTree:
    def __init__(self, max_depth: int = 3, min_leaf: int = 8):
        self.max_depth = max_depth
        self.min_leaf = min_leaf

    @staticmethod
    def gini(y: np.ndarray, num_classes: int) -> float:
        if len(y) == 0:
            return 0.0
        probs = np.bincount(y, minlength=num_classes) / len(y)
        return float(1.0 - np.sum(probs**2))

    def _build(self, x: np.ndarray, y: np.ndarray, depth: int, num_classes: int) -> DecisionTreeNode:
        counts = np.bincount(y, minlength=num_classes)
        node = DecisionTreeNode(int(np.argmax(counts)))
        if depth >= self.max_depth or len(y) < 2 * self.min_leaf or np.count_nonzero(counts) == 1:
            return node

        best_score = math.inf
        best = None
        for feature_idx in range(x.shape[1]):
            values = x[:, feature_idx]
            unique = np.unique(values)
            if len(unique) < 2:
                continue
            thresholds = 0.5 * (unique[:-1] + unique[1:])
            for threshold in thresholds:
                left_mask = values <= threshold
                right_mask = ~left_mask
                if left_mask.sum() < self.min_leaf or right_mask.sum() < self.min_leaf:
                    continue
                score = (
                    left_mask.mean() * self.gini(y[left_mask], num_classes)
                    + right_mask.mean() * self.gini(y[right_mask], num_classes)
                )
                if score < best_score:
                    best_score = score
                    best = (feature_idx, threshold, left_mask, right_mask)
        if best is None:
            return node
        feature_idx, threshold, left_mask, right_mask = best
        node.feature = feature_idx
        node.threshold = float(threshold)
        node.left = self._build(x[left_mask], y[left_mask], depth + 1, num_classes)
        node.right = self._build(x[right_mask], y[right_mask], depth + 1, num_classes)
        return node

    def fit(self, x: np.ndarray, y: np.ndarray, num_classes: int) -> "ShallowDecisionTree":
        self.root_ = self._build(x, y, 0, num_classes)
        return self

    def _predict_one(self, row: np.ndarray, node: DecisionTreeNode) -> int:
        while node.feature is not None:
            node = node.left if row[node.feature] <= node.threshold else node.right
        return node.prediction

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.asarray([self._predict_one(row, self.root_) for row in x], dtype=np.int64)


def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def load_dataset(dataset_root: Path) -> tuple[pd.DataFrame, dict[int, dict[str, object]], pd.DataFrame]:
    episodes = pq.read_table(dataset_root / "meta/episodes/chunk-000/file-000.parquet").to_pandas()
    tasks = pq.read_table(dataset_root / "meta/tasks.parquet").to_pandas()
    data = pq.read_table(
        dataset_root / "data/chunk-000/file-000.parquet",
        columns=["observation.state", "action", "episode_index", "frame_index", "task_index"],
    ).to_pydict()
    per_episode: dict[int, dict[str, object]] = defaultdict(
        lambda: {"state": [], "action": [], "frame_index": [], "task_index": None}
    )
    for state, action, episode_index, frame_index, task_index in zip(
        data["observation.state"],
        data["action"],
        data["episode_index"],
        data["frame_index"],
        data["task_index"],
        strict=True,
    ):
        episode = per_episode[int(episode_index)]
        episode["state"].append(np.asarray(state, dtype=np.float32))
        episode["action"].append(np.asarray(action, dtype=np.float32))
        episode["frame_index"].append(int(frame_index))
        episode["task_index"] = int(task_index)
    for episode_index, episode in per_episode.items():
        episode["target_color"] = COLOR_BY_TASK_INDEX[int(episode["task_index"])]
        episode["state"] = np.stack(episode["state"]).astype(np.float32)
        episode["action"] = np.stack(episode["action"]).astype(np.float32)
        episode["initial_state"] = episode["state"][0].astype(np.float32)
    return episodes, per_episode, tasks


def decode_first_frames(dataset_root: Path, episodes: pd.DataFrame) -> dict[int, np.ndarray]:
    start_requests: dict[tuple[int, int], dict[int, int]] = defaultdict(dict)
    for row in episodes.to_dict("records"):
        key = (
            int(row["videos/observation.images.top/chunk_index"]),
            int(row["videos/observation.images.top/file_index"]),
        )
        start_frame = int(round(float(row["videos/observation.images.top/from_timestamp"]) * 30.0))
        start_requests[key][int(row["episode_index"])] = start_frame

    frames_by_episode: dict[int, np.ndarray] = {}
    for (chunk_index, file_index), episode_to_frame in start_requests.items():
        path = dataset_root / f"videos/observation.images.top/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
        needed = defaultdict(list)
        for episode_index, frame_number in episode_to_frame.items():
            needed[frame_number].append(episode_index)
        container = av.open(str(path))
        stream = container.streams.video[0]
        for frame_idx, frame in enumerate(container.decode(stream)):
            if frame_idx not in needed:
                continue
            image = frame.to_ndarray(format="rgb24")
            for episode_index in needed[frame_idx]:
                frames_by_episode[episode_index] = image.copy()
            if len(frames_by_episode) >= len(episodes):
                break
        container.close()
    missing = sorted(set(episodes["episode_index"]) - set(frames_by_episode))
    if missing:
        raise RuntimeError(f"Missing first frames for episodes: {missing[:10]}")
    return frames_by_episode


def average_pool_image(image: np.ndarray, block_h: int = 10, block_w: int = 10) -> np.ndarray:
    crop = image[80:440, 40:600]
    h, w, c = crop.shape
    h2 = (h // block_h) * block_h
    w2 = (w // block_w) * block_w
    crop = crop[:h2, :w2]
    pooled = crop.reshape(h2 // block_h, block_h, w2 // block_w, block_w, c).mean(axis=(1, 3))
    return pooled.astype(np.float32) / 255.0


def build_scene_proxy_features(
    episodes_df: pd.DataFrame,
    per_episode: dict[int, dict[str, object]],
    first_frames: dict[int, np.ndarray],
    pca_dim: int,
) -> tuple[pd.DataFrame, np.ndarray]:
    rows = []
    image_features = []
    for row in episodes_df.to_dict("records"):
        episode_index = int(row["episode_index"])
        pooled = average_pool_image(first_frames[episode_index])
        image_feature = pooled.reshape(-1)
        image_features.append(image_feature)
        initial_state = per_episode[episode_index]["initial_state"]
        rows.append(
            {
                "episode_index": episode_index,
                "target_color": per_episode[episode_index]["target_color"],
                "task_string": row["tasks"][0],
                "episode_length": int(row["length"]),
                "dataset_from_index": int(row["dataset_from_index"]),
                "dataset_to_index": int(row["dataset_to_index"]),
                "top_video_file_index": int(row["videos/observation.images.top/file_index"]),
                "top_video_from_timestamp": float(row["videos/observation.images.top/from_timestamp"]),
                "top_video_to_timestamp": float(row["videos/observation.images.top/to_timestamp"]),
                **{f"robot_state_{i}": float(initial_state[i]) for i in range(len(initial_state))},
            }
        )
    image_matrix = np.stack(image_features).astype(np.float32)
    state_matrix = np.stack([per_episode[int(ep)]["initial_state"] for ep in episodes_df["episode_index"]]).astype(np.float32)
    standardizer = Standardizer().fit(image_matrix)
    image_std = standardizer.transform(image_matrix)
    projector = PCAProjector(pca_dim).fit(image_std)
    image_pca = projector.transform(image_std).astype(np.float32)
    feature_matrix = np.concatenate([image_pca, state_matrix], axis=1).astype(np.float32)
    scene_df = pd.DataFrame(rows)
    for idx in range(image_pca.shape[1]):
        scene_df[f"scene_proxy_pca_{idx+1}"] = image_pca[:, idx]
    return scene_df, feature_matrix


def stratified_folds(labels: np.ndarray, n_splits: int = 5) -> list[np.ndarray]:
    by_class = {cls: np.where(labels == cls)[0].tolist() for cls in sorted(np.unique(labels))}
    for indices in by_class.values():
        indices.sort()
    folds = [[] for _ in range(n_splits)]
    for _, indices in sorted(by_class.items()):
        for idx, sample in enumerate(indices):
            folds[idx % n_splits].append(sample)
    return [np.asarray(sorted(fold), dtype=np.int64) for fold in folds]


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for truth, pred in zip(y_true, y_pred, strict=True):
        matrix[int(truth), int(pred)] += 1
    return matrix


def prepare_fold_features(x_train: np.ndarray, x_test: np.ndarray, pca_dim: int) -> tuple[np.ndarray, np.ndarray]:
    standardizer = Standardizer().fit(x_train)
    x_train_std = standardizer.transform(x_train)
    x_test_std = standardizer.transform(x_test)
    dim = min(pca_dim, x_train_std.shape[0] - 1, x_train_std.shape[1])
    if dim >= 1:
        projector = PCAProjector(dim).fit(x_train_std)
        x_train_use = projector.transform(x_train_std)
        x_test_use = projector.transform(x_test_std)
    else:
        x_train_use = x_train_std
        x_test_use = x_test_std
    return x_train_use.astype(np.float64), x_test_use.astype(np.float64)


def run_geometry_classifier_cv(
    x: np.ndarray,
    labels: np.ndarray,
    pca_dim: int,
    tree_depth: int,
    min_leaf: int,
) -> tuple[list[dict[str, object]], pd.DataFrame]:
    folds = stratified_folds(labels, 5)
    rows = []
    confusion_rows = []
    models = {
        "logistic_regression": lambda: MultinomialLogReg(),
        "decision_tree": lambda: ShallowDecisionTree(max_depth=tree_depth, min_leaf=min_leaf),
        "nearest_centroid": None,
    }
    for model_name, factory in models.items():
        per_fold = []
        total_conf = np.zeros((3, 3), dtype=np.int64)
        for fold_idx, test_idx in enumerate(folds):
            train_mask = np.ones(len(labels), dtype=bool)
            train_mask[test_idx] = False
            train_idx = np.where(train_mask)[0]
            x_train, x_test = prepare_fold_features(x[train_idx], x[test_idx], pca_dim)
            y_train, y_test = labels[train_idx], labels[test_idx]
            if model_name == "nearest_centroid":
                centroids = np.stack([x_train[y_train == cls].mean(axis=0) for cls in range(3)])
                distances = ((x_test[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
                pred = np.argmin(distances, axis=1)
            else:
                model = factory().fit(x_train, y_train, 3)
                pred = model.predict(x_test)
            acc = float((pred == y_test).mean())
            per_fold.append(acc)
            total_conf += confusion_matrix(y_test, pred, 3)
        rows.append(
            {
                "model": model_name,
                "mean_accuracy": float(np.mean(per_fold)),
                "std_accuracy": float(np.std(per_fold)),
                "fold_accuracies": ",".join(f"{value:.4f}" for value in per_fold),
            }
        )
        for truth_idx, truth in enumerate(COLORS):
            for pred_idx, pred in enumerate(COLORS):
                confusion_rows.append(
                    {
                        "model": model_name,
                        "true_color": truth,
                        "pred_color": pred,
                        "count": int(total_conf[truth_idx, pred_idx]),
                    }
                )
    shuffled = np.random.default_rng(0).permutation(x)
    shuffled_rows = run_geometry_classifier_cv(shuffled, labels, pca_dim, tree_depth, min_leaf)[0] if False else None
    del shuffled_rows
    return rows, pd.DataFrame(confusion_rows)


def shuffled_sanity_accuracy(x: np.ndarray, labels: np.ndarray, pca_dim: int) -> float:
    shuffled_x = np.random.default_rng(0).permutation(x)
    folds = stratified_folds(labels, 5)
    accuracies = []
    for test_idx in folds:
        train_mask = np.ones(len(labels), dtype=bool)
        train_mask[test_idx] = False
        train_idx = np.where(train_mask)[0]
        x_train, x_test = prepare_fold_features(shuffled_x[train_idx], shuffled_x[test_idx], pca_dim)
        y_train, y_test = labels[train_idx], labels[test_idx]
        model = MultinomialLogReg().fit(x_train, y_train, 3)
        pred = model.predict(x_test)
        accuracies.append(float((pred == y_test).mean()))
    return float(np.mean(accuracies))


def pairwise_distance_matrix(x: np.ndarray) -> np.ndarray:
    diffs = x[:, None, :] - x[None, :, :]
    return np.sqrt(np.sum(diffs**2, axis=2))


def color_indices(scene_df: pd.DataFrame) -> dict[str, np.ndarray]:
    return {
        color: scene_df.loc[scene_df["target_color"] == color, "episode_index"].to_numpy(dtype=np.int64)
        for color in COLORS
    }


def one_to_one_match(
    distance_matrix: np.ndarray,
    left_ids: np.ndarray,
    right_ids: np.ndarray,
    threshold: float | None,
    pair_type: str,
    left_color: str,
    right_color: str,
    threshold_label: str,
) -> list[MatchPair]:
    sub = distance_matrix[np.ix_(left_ids, right_ids)]
    row_ind, col_ind = linear_sum_assignment(sub)
    matches = []
    for row_idx, col_idx in zip(row_ind, col_ind, strict=True):
        distance = float(sub[row_idx, col_idx])
        if threshold is not None and distance > threshold:
            continue
        matches.append(
            MatchPair(
                pair_type=pair_type,
                left_episode=int(left_ids[row_idx]),
                left_color=left_color,
                right_episode=int(right_ids[col_idx]),
                right_color=right_color,
                distance=distance,
                threshold_label=threshold_label,
            )
        )
    return matches


def compute_counterfactual_coverage(scene_df: pd.DataFrame, feature_matrix: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float], list[MatchPair]]:
    ids_by_color = color_indices(scene_df)
    distance_matrix = pairwise_distance_matrix(feature_matrix)
    nearest_rows = []
    all_cross_nearest = []
    for left_color, right_color in PAIRS:
        left_ids = ids_by_color[left_color]
        right_ids = ids_by_color[right_color]
        sub = distance_matrix[np.ix_(left_ids, right_ids)]
        for left_local, left_episode in enumerate(left_ids):
            nearest = float(sub[left_local].min())
            nearest_rows.append(
                {
                    "source_color": left_color,
                    "target_color": right_color,
                    "episode_index": int(left_episode),
                    "nearest_distance": nearest,
                }
            )
            all_cross_nearest.append(nearest)
        for right_local, right_episode in enumerate(right_ids):
            nearest = float(sub[:, right_local].min())
            nearest_rows.append(
                {
                    "source_color": right_color,
                    "target_color": left_color,
                    "episode_index": int(right_episode),
                    "nearest_distance": nearest,
                }
            )
            all_cross_nearest.append(nearest)
    thresholds = {
        "strict": float(np.quantile(all_cross_nearest, 0.10)),
        "moderate": float(np.quantile(all_cross_nearest, 0.25)),
        "loose": float(np.quantile(all_cross_nearest, 0.50)),
    }

    coverage_rows = []
    moderate_pairs: list[MatchPair] = []
    for label, threshold in thresholds.items():
        pair_counts = {}
        matches_by_pair = {}
        for left_color, right_color in PAIRS:
            matches = one_to_one_match(
                distance_matrix,
                ids_by_color[left_color],
                ids_by_color[right_color],
                threshold,
                f"{left_color}_{right_color}",
                left_color,
                right_color,
                label,
            )
            matches_by_pair[(left_color, right_color)] = matches
            pair_counts[f"{left_color}_{right_color}"] = len(matches)
            if label == "moderate":
                moderate_pairs.extend(matches)
        coverage_flags = []
        rgb_candidate_count = 0
        for episode_index in scene_df["episode_index"]:
            color = scene_df.loc[scene_df["episode_index"] == episode_index, "target_color"].iloc[0]
            others = [other for other in COLORS if other != color]
            has_all = True
            row = {"episode_index": int(episode_index), "target_color": color}
            for other in others:
                left = color
                right = other
                if (left, right) in matches_by_pair:
                    used = matches_by_pair[(left, right)]
                    flag = any(match.left_episode == episode_index for match in used)
                else:
                    used = matches_by_pair[(right, left)]
                    flag = any(match.right_episode == episode_index for match in used)
                row[f"has_{other}_counterpart"] = bool(flag)
                has_all = has_all and flag
            row["has_both_other_colors"] = bool(has_all)
            coverage_flags.append(row)
        coverage_df = pd.DataFrame(coverage_flags)
        rgb_candidate_count = int(coverage_df["has_both_other_colors"].sum())
        any_cross_count = int(
            coverage_df[
                [col for col in coverage_df.columns if col.startswith("has_") and col.endswith("_counterpart")]
            ].any(axis=1).sum()
        )
        coverage_rows.append(
            {
                "scene_tolerance": label,
                "threshold_value": threshold,
                "threshold_units": "scene_proxy_l2",
                "rgb_triplet_candidates": rgb_candidate_count,
                "rb_pairs": pair_counts["red_blue"],
                "ry_pairs": pair_counts["red_yellow"],
                "by_pairs": pair_counts["blue_yellow"],
                "cross_color_covered_episodes": any_cross_count,
                "fraction_with_any_cross_color_counterpart": float(
                    coverage_df[[col for col in coverage_df.columns if col.startswith("has_") and col.endswith("_counterpart")]].any(axis=1).mean()
                ),
            }
        )
    return pd.DataFrame(nearest_rows), pd.DataFrame(coverage_rows), thresholds, moderate_pairs


def normalized_frame_samples(action: np.ndarray) -> np.ndarray:
    if len(action) == 1:
        return np.repeat(action, len(PROGRESS_GRID), axis=0)
    positions = np.linspace(0.0, 1.0, len(action), dtype=np.float32)
    output = np.empty((len(PROGRESS_GRID), action.shape[1]), dtype=np.float32)
    for dim in range(action.shape[1]):
        output[:, dim] = np.interp(PROGRESS_GRID, positions, action[:, dim])
    return output


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


def matched_within_color_pairs(
    scene_df: pd.DataFrame,
    distance_matrix: np.ndarray,
    pair_name: str,
    count: int,
    threshold_label: str,
) -> list[MatchPair]:
    color = pair_name.split("_")[0]
    episode_ids = scene_df.loc[scene_df["target_color"] == color, "episode_index"].to_numpy(dtype=np.int64)
    if len(episode_ids) < 2:
        return []
    sub = distance_matrix[np.ix_(episode_ids, episode_ids)].copy()
    np.fill_diagonal(sub, np.inf)
    available = set(range(len(episode_ids)))
    pairs = []
    while len(available) >= 2 and len(pairs) < count:
        best = None
        best_dist = np.inf
        available_list = sorted(available)
        for i_pos, i in enumerate(available_list):
            for j in available_list[i_pos + 1 :]:
                if sub[i, j] < best_dist:
                    best_dist = float(sub[i, j])
                    best = (i, j)
        if best is None or not np.isfinite(best_dist):
            break
        i, j = best
        pairs.append(
            MatchPair(
                pair_type=f"{color}_{color}",
                left_episode=int(episode_ids[i]),
                left_color=color,
                right_episode=int(episode_ids[j]),
                right_color=color,
                distance=best_dist,
                threshold_label=threshold_label,
            )
        )
        available.remove(i)
        available.remove(j)
    return pairs


def bootstrap_ci(values: np.ndarray, repeats: int = 200) -> tuple[float, float]:
    if len(values) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(0)
    means = []
    for _ in range(repeats):
        sample = rng.choice(values, size=len(values), replace=True)
        means.append(float(sample.mean()))
    return float(np.quantile(means, 0.05)), float(np.quantile(means, 0.95))


def summarize_divergence(values: np.ndarray) -> dict[str, float]:
    ci_low, ci_high = bootstrap_ci(values)
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "q25": float(np.quantile(values, 0.25)),
        "q75": float(np.quantile(values, 0.75)),
        "ci05": ci_low,
        "ci95": ci_high,
    }


def analyze_matched_divergence(
    scene_df: pd.DataFrame,
    per_episode: dict[int, dict[str, object]],
    feature_matrix: np.ndarray,
    thresholds: dict[str, float],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    distance_matrix = pairwise_distance_matrix(feature_matrix)
    ids_by_color = color_indices(scene_df)
    frame_rows = []
    chunk_rows = []
    match_rows = []
    onset_results = {}
    moderate_threshold = thresholds["moderate"]

    for left_color, right_color in PAIRS:
        cross_pairs = one_to_one_match(
            distance_matrix,
            ids_by_color[left_color],
            ids_by_color[right_color],
            moderate_threshold,
            f"{left_color}_{right_color}",
            left_color,
            right_color,
            "moderate",
        )
        within_left = matched_within_color_pairs(scene_df, distance_matrix, f"{left_color}_{left_color}", len(cross_pairs), "moderate")
        within_right = matched_within_color_pairs(scene_df, distance_matrix, f"{right_color}_{right_color}", len(cross_pairs), "moderate")
        for match in cross_pairs + within_left + within_right:
            match_rows.append(match.__dict__)

        def collect_diffs(pairs: list[MatchPair], chunk: bool, dims: slice) -> np.ndarray:
            curves = []
            for match in pairs:
                left_action = per_episode[match.left_episode]["action"]
                right_action = per_episode[match.right_episode]["action"]
                left = normalized_chunk_samples(left_action, dims) if chunk else normalized_frame_samples(left_action)[:, dims]
                right = normalized_chunk_samples(right_action, dims) if chunk else normalized_frame_samples(right_action)[:, dims]
                curves.append(np.linalg.norm(left - right, axis=1))
            return np.stack(curves) if curves else np.zeros((0, len(PROGRESS_GRID)), dtype=np.float32)

        cross_frame_all = collect_diffs(cross_pairs, False, slice(0, 8))
        cross_frame_arm = collect_diffs(cross_pairs, False, slice(0, 7))
        cross_chunk_all = collect_diffs(cross_pairs, True, slice(0, 8))
        cross_chunk_arm = collect_diffs(cross_pairs, True, slice(0, 7))
        within_frame_arm = np.concatenate(
            [collect_diffs(within_left, False, slice(0, 7)), collect_diffs(within_right, False, slice(0, 7))],
            axis=0,
        )
        within_chunk_arm = np.concatenate(
            [collect_diffs(within_left, True, slice(0, 7)), collect_diffs(within_right, True, slice(0, 7))],
            axis=0,
        )

        pair_key = f"{left_color}_{right_color}"
        for bin_idx, progress in enumerate(PROGRESS_GRID):
            if len(cross_frame_all):
                for rep_name, values in (
                    ("frame_all", cross_frame_all[:, bin_idx]),
                    ("frame_arm", cross_frame_arm[:, bin_idx]),
                ):
                    stats = summarize_divergence(values)
                    frame_rows.append(
                        {
                            "pair_type": pair_key,
                            "representation": rep_name,
                            "progress": float(progress),
                            "match_type": "cross_color",
                            "support": int(len(values)),
                            **stats,
                        }
                    )
            if len(cross_chunk_all):
                for rep_name, values in (
                    ("chunk_all", cross_chunk_all[:, bin_idx]),
                    ("chunk_arm", cross_chunk_arm[:, bin_idx]),
                ):
                    stats = summarize_divergence(values)
                    chunk_rows.append(
                        {
                            "pair_type": pair_key,
                            "representation": rep_name,
                            "progress": float(progress),
                            "match_type": "cross_color",
                            "support": int(len(values)),
                            **stats,
                        }
                    )
            if len(within_frame_arm):
                stats = summarize_divergence(within_frame_arm[:, bin_idx])
                frame_rows.append(
                    {
                        "pair_type": pair_key,
                        "representation": "frame_arm",
                        "progress": float(progress),
                        "match_type": "within_color_control",
                        "support": int(len(within_frame_arm)),
                        **stats,
                    }
                )
            if len(within_chunk_arm):
                stats = summarize_divergence(within_chunk_arm[:, bin_idx])
                chunk_rows.append(
                    {
                        "pair_type": pair_key,
                        "representation": "chunk_arm",
                        "progress": float(progress),
                        "match_type": "within_color_control",
                        "support": int(len(within_chunk_arm)),
                        **stats,
                    }
                )

        if len(cross_chunk_arm) and len(within_chunk_arm):
            cross_mean = cross_chunk_arm.mean(axis=0)
            within_mean = within_chunk_arm.mean(axis=0)
            ratio = cross_mean / (within_mean + 1e-8)
            delta = cross_mean - within_mean
            ratio_threshold = max(1.15, float(np.median(ratio[:6]) + 0.10))
            delta_threshold = max(0.15, float(np.median(delta[:6]) + 0.05))
            onset = None
            for start_idx in range(len(PROGRESS_GRID) - 4):
                if all(ratio[start_idx + offset] >= ratio_threshold and delta[start_idx + offset] >= delta_threshold for offset in range(5)):
                    onset = {
                        "pair_type": pair_key,
                        "progress": float(PROGRESS_GRID[start_idx]),
                        "ratio_threshold": ratio_threshold,
                        "delta_threshold": delta_threshold,
                        "ratio": float(ratio[start_idx]),
                        "delta": float(delta[start_idx]),
                    }
                    break
            onset_results[pair_key] = onset
        else:
            onset_results[pair_key] = None

    frame_df = pd.DataFrame(frame_rows)
    chunk_df = pd.DataFrame(chunk_rows)
    match_df = pd.DataFrame(match_rows)
    aggregate_onsets = [row["progress"] for row in onset_results.values() if row is not None]
    onset_results["aggregate"] = float(np.median(aggregate_onsets)) if aggregate_onsets else None
    return frame_df, chunk_df, match_df, onset_results


def action_to_target_predictability(
    chunk_df: pd.DataFrame,
    match_df: pd.DataFrame,
    per_episode: dict[int, dict[str, object]],
    feature_matrix: np.ndarray,
    scene_df: pd.DataFrame,
) -> pd.DataFrame:
    feature_by_episode = {
        int(ep): feature_matrix[idx] for idx, ep in enumerate(scene_df["episode_index"].to_numpy(dtype=np.int64))
    }
    results = []
    for label, start, end in EARLY_WINDOWS:
        mask = (PROGRESS_GRID >= start) & (PROGRESS_GRID < end)
        for left_color, right_color in PAIRS:
            pair_key = f"{left_color}_{right_color}"
            pairs = match_df.loc[match_df["pair_type"] == pair_key]
            if pairs.empty:
                continue
            x_rows = []
            x_res_rows = []
            y_rows = []
            groups = []
            for group_id, row in enumerate(pairs.itertuples(index=False)):
                left_chunk = normalized_chunk_samples(per_episode[int(row.left_episode)]["action"], slice(0, 7))[mask].mean(axis=0)
                right_chunk = normalized_chunk_samples(per_episode[int(row.right_episode)]["action"], slice(0, 7))[mask].mean(axis=0)
                left_scene = feature_by_episode[int(row.left_episode)]
                right_scene = feature_by_episode[int(row.right_episode)]
                x_rows.extend([left_chunk, right_chunk])
                y_rows.extend([0, 1])
                groups.extend([group_id, group_id])
                x_scene = np.stack([left_scene, right_scene]).astype(np.float64)
                x_chunk = np.stack([left_chunk, right_chunk]).astype(np.float64)
                scene_aug = np.concatenate([x_scene, np.ones((len(x_scene), 1))], axis=1)
                ridge = 1e-3 * np.eye(scene_aug.shape[1], dtype=np.float64)
                w = np.linalg.solve(scene_aug.T @ scene_aug + ridge, scene_aug.T @ x_chunk)
                residual = x_chunk - scene_aug @ w
                x_res_rows.extend(list(residual))
            x = np.stack(x_rows).astype(np.float64)
            x_res = np.stack(x_res_rows).astype(np.float64)
            y = np.asarray(y_rows, dtype=np.int64)
            groups = np.asarray(groups, dtype=np.int64)
            unique_groups = np.unique(groups)
            group_folds = np.array_split(unique_groups, min(5, len(unique_groups)))

            def grouped_accuracy(features: np.ndarray) -> float:
                accuracies = []
                for test_groups in group_folds:
                    test_mask = np.isin(groups, test_groups)
                    train_mask = ~test_mask
                    if train_mask.sum() == 0 or test_mask.sum() == 0 or len(np.unique(y[train_mask])) < 2:
                        continue
                    x_train, x_test = prepare_fold_features(features[train_mask], features[test_mask], 8)
                    model = MultinomialLogReg().fit(x_train, y[train_mask], 2)
                    pred = model.predict(x_test)
                    accuracies.append(float((pred == y[test_mask]).mean()))
                return float(np.mean(accuracies)) if accuracies else float("nan")

            results.append(
                {
                    "pair_type": pair_key,
                    "progress_window": label,
                    "raw_chunk_accuracy": grouped_accuracy(x),
                    "residual_chunk_accuracy": grouped_accuracy(x_res),
                    "chance_accuracy": 0.5,
                    "support_pairs": int(len(pairs)),
                }
            )
    return pd.DataFrame(results)


def episode_length_summary(scene_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for color in COLORS:
        lengths = scene_df.loc[scene_df["target_color"] == color, "episode_length"].to_numpy(dtype=np.int64)
        rows.append(
            {
                "target_color": color,
                "count": int(len(lengths)),
                "mean": float(lengths.mean()),
                "std": float(lengths.std()),
                "median": float(np.median(lengths)),
                "min": int(lengths.min()),
                "max": int(lengths.max()),
            }
        )
    return pd.DataFrame(rows)


def commitment_alignment(commitment_manifest: Path, scene_df: pd.DataFrame, onsets: dict[str, object]) -> tuple[pd.DataFrame, dict[str, object]]:
    manifest = pd.read_csv(commitment_manifest)
    current_onset = float(manifest["onset_progress"].median())
    matched_onset = onsets.get("aggregate")
    deltas = manifest["onset_progress"] - matched_onset if matched_onset is not None else pd.Series(dtype=float)
    alignment_rows = []
    for row in manifest.itertuples(index=False):
        if matched_onset is None:
            relation = "unknown"
            delta = float("nan")
        else:
            delta = float(row.onset_progress - matched_onset)
            if row.anchor_end / max(row.episode_length - 1, 1) < matched_onset:
                relation = "anchor_before_matched_onset"
            elif row.anchor_start / max(row.episode_length - 1, 1) > matched_onset:
                relation = "anchor_after_matched_onset"
            else:
                relation = "anchor_overlaps_matched_onset"
        alignment_rows.append(
            {
                "episode_index": int(row.episode_index),
                "target_color": row.target_color,
                "existing_onset_progress": float(row.onset_progress),
                "matched_onset_progress": matched_onset,
                "delta_progress": delta,
                "relation": relation,
            }
        )
    summary = {
        "existing_median_onset_progress": current_onset,
        "matched_aggregate_onset_progress": matched_onset,
        "median_delta_progress": float(deltas.median()) if matched_onset is not None else None,
        "iqr_delta_progress": float(deltas.quantile(0.75) - deltas.quantile(0.25)) if matched_onset is not None else None,
        "relation_fractions": pd.Series([row["relation"] for row in alignment_rows]).value_counts(normalize=True).to_dict(),
    }
    return pd.DataFrame(alignment_rows), summary


def plot_confusion(confusion_df: pd.DataFrame, output_path: Path) -> None:
    subset = confusion_df.loc[confusion_df["model"] == "logistic_regression"].copy()
    matrix = np.zeros((3, 3), dtype=np.float32)
    for row in subset.itertuples(index=False):
        matrix[COLORS.index(row.true_color), COLORS.index(row.pred_color)] = row.count
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(3), COLORS)
    ax.set_yticks(range(3), COLORS)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Geometry-Only CV Confusion")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, int(matrix[i, j]), ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_scene_distance_distribution(nearest_df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    for pair_name, group in nearest_df.groupby(["source_color", "target_color"]):
        values = np.sort(group["nearest_distance"].to_numpy(dtype=np.float64))
        ax.plot(values, label=f"{pair_name[0]}->{pair_name[1]}")
    ax.set_xlabel("Episode rank")
    ax.set_ylabel("Nearest cross-color scene distance")
    ax.set_title("Cross-Color Nearest Scene Distance")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_divergence(chunk_df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    for pair_type in sorted(chunk_df["pair_type"].unique()):
        cross = chunk_df.loc[
            (chunk_df["pair_type"] == pair_type)
            & (chunk_df["representation"] == "chunk_arm")
            & (chunk_df["match_type"] == "cross_color")
        ].sort_values("progress")
        within = chunk_df.loc[
            (chunk_df["pair_type"] == pair_type)
            & (chunk_df["representation"] == "chunk_arm")
            & (chunk_df["match_type"] == "within_color_control")
        ].sort_values("progress")
        if not cross.empty:
            ax.plot(cross["progress"], cross["mean"], label=f"{pair_type} cross")
        if not within.empty:
            ax.plot(within["progress"], within["mean"], linestyle="--", label=f"{pair_type} within")
    ax.set_xlabel("Progress")
    ax.set_ylabel("Mean chunk-arm divergence")
    ax.set_title("Matched Chunk Divergence")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def verdict_from_accuracy(acc: float) -> str:
    if acc >= 0.60:
        return "STRONG SHORTCUT"
    if acc >= 0.45:
        return "MODERATE SHORTCUT"
    return "WEAK / NO CLEAR SHORTCUT"


def coverage_verdict(fraction: float) -> str:
    if fraction >= 0.75:
        return "STRONG"
    if fraction >= 0.45:
        return "PARTIAL"
    if fraction >= 0.20:
        return "WEAK"
    return "VERY WEAK"


def signal_verdict(ratio_rows: pd.DataFrame) -> str:
    if ratio_rows.empty:
        return "NO CLEAR TARGET SIGNAL"
    peak = float(ratio_rows["cross_within_ratio"].max())
    if peak >= 1.6:
        return "STRONG"
    if peak >= 1.3:
        return "MODERATE"
    if peak >= 1.1:
        return "WEAK"
    return "NO CLEAR TARGET SIGNAL"


def build_report(
    output_dir: Path,
    recoverability: str,
    generator_audit: dict[str, object],
    scene_df: pd.DataFrame,
    geometry_cv: pd.DataFrame,
    shuffle_acc: float,
    length_df: pd.DataFrame,
    nearest_df: pd.DataFrame,
    coverage_df: pd.DataFrame,
    chunk_df: pd.DataFrame,
    action_predict_df: pd.DataFrame,
    onsets: dict[str, object],
    alignment_summary: dict[str, object],
) -> tuple[str, dict[str, object]]:
    best_geometry_acc = float(geometry_cv.loc[geometry_cv["model"] == "logistic_regression", "mean_accuracy"].iloc[0])
    geometry_verdict = verdict_from_accuracy(best_geometry_acc)
    moderate_coverage = float(coverage_df.loc[coverage_df["scene_tolerance"] == "moderate", "fraction_with_any_cross_color_counterpart"].iloc[0])
    counterfactual_verdict = coverage_verdict(moderate_coverage)

    ratio_rows = []
    for pair_type in sorted(chunk_df["pair_type"].unique()):
        cross = chunk_df.loc[
            (chunk_df["pair_type"] == pair_type)
            & (chunk_df["representation"] == "chunk_arm")
            & (chunk_df["match_type"] == "cross_color")
        ].sort_values("progress")
        within = chunk_df.loc[
            (chunk_df["pair_type"] == pair_type)
            & (chunk_df["representation"] == "chunk_arm")
            & (chunk_df["match_type"] == "within_color_control")
        ].sort_values("progress")
        if cross.empty or within.empty:
            continue
        merged = cross[["progress", "mean"]].merge(within[["progress", "mean"]], on="progress", suffixes=("_cross", "_within"))
        merged["pair_type"] = pair_type
        merged["cross_within_ratio"] = merged["mean_cross"] / (merged["mean_within"] + 1e-8)
        ratio_rows.append(merged)
    ratio_df = pd.concat(ratio_rows, ignore_index=True) if ratio_rows else pd.DataFrame()
    matched_signal_verdict = signal_verdict(ratio_df)

    if geometry_verdict == "STRONG SHORTCUT":
        overall = "VERY WEAK"
        redesign = "YES"
        next_step = "REDESIGN DATASET WITH COUNTERFACTUAL TRIPLETS"
    elif counterfactual_verdict in {"WEAK", "VERY WEAK"}:
        overall = "WEAK"
        redesign = "YES"
        next_step = "REDESIGN DATASET WITH COUNTERFACTUAL TRIPLETS"
    elif matched_signal_verdict == "NO CLEAR TARGET SIGNAL":
        overall = "WEAK"
        redesign = "YES"
        next_step = "REDESIGN DATASET WITH COUNTERFACTUAL TRIPLETS"
    elif matched_signal_verdict == "WEAK":
        overall = "ADEQUATE BUT IMPERFECT"
        redesign = "PARTIALLY"
        next_step = "IMPROVE DATASET BUT KEEP CURRENT POLICY ANALYSIS"
    else:
        overall = "STRONG"
        redesign = "NO"
        next_step = "PROCEED TO POLICY-SIDE WRONG-COLOR COMMITMENT ANALYSIS"

    evidence_table = [
        {
            "evidence": "geometry-only target prediction",
            "result": f"logreg CV acc={best_geometry_acc:.3f}, shuffle={shuffle_acc:.3f}, chance=0.333",
            "interpretation": geometry_verdict,
        },
        {
            "evidence": "counterfactual scene coverage",
            "result": f"moderate coverage fraction={moderate_coverage:.3f}",
            "interpretation": counterfactual_verdict,
        },
        {
            "evidence": "matched future-chunk divergence",
            "result": f"aggregate onset={onsets.get('aggregate')}",
            "interpretation": matched_signal_verdict,
        },
        {
            "evidence": "action->target predictability under matched geometry",
            "result": f"max raw pairwise accuracy={action_predict_df['raw_chunk_accuracy'].max():.3f}" if not action_predict_df.empty else "not estimable",
            "interpretation": "supports readability" if not action_predict_df.empty and action_predict_df["raw_chunk_accuracy"].max() > 0.6 else "weak readability",
        },
        {
            "evidence": "commitment-window alignment",
            "result": f"existing median={alignment_summary['existing_median_onset_progress']:.3f}, matched aggregate={alignment_summary['matched_aggregate_onset_progress']}",
            "interpretation": "alignment likely late" if alignment_summary["matched_aggregate_onset_progress"] is not None and alignment_summary["existing_median_onset_progress"] > alignment_summary["matched_aggregate_onset_progress"] else "alignment roughly consistent",
        },
    ]

    summary = {
        "dataset_geometry_recoverability": recoverability,
        "generator_randomization_audit": generator_audit,
        "geometry_shortcut_verdict": geometry_verdict.replace(" SHORTCUT", ""),
        "counterfactual_coverage_verdict": counterfactual_verdict,
        "matched_action_signal_verdict": matched_signal_verdict,
        "overall_conditional_identifiability_verdict": overall,
        "dataset_redesign_next_priority": redesign,
        "recommended_next_step": next_step,
        "evidence_table": evidence_table,
    }

    lines = [
        "A. Dataset / Geometry Recoverability",
        recoverability,
        "",
        "B. Generator Randomization Audit",
        f"target cycling order = {generator_audit['target_cycle']}",
        f"position sampling independent of target before color assignment = {generator_audit['positions_sampled_before_color_assignment']}",
        f"shared RNG used for layout and color permutation = {generator_audit['shared_position_rng']}",
        f"robot initial state fixed across episodes = {generator_audit['robot_initial_state_fixed']}",
        f"tilt randomized independently of target = {generator_audit['tilt_randomized_independently']}",
        "",
        "C. Geometry Shortcut Classification",
    ]
    for row in geometry_cv.itertuples(index=False):
        lines.append(f"{row.model}: mean={row.mean_accuracy:.4f}, std={row.std_accuracy:.4f}, folds={row.fold_accuracies}")
    lines.extend(
        [
            f"chance accuracy = 0.3333",
            f"shuffle sanity accuracy = {shuffle_acc:.4f}",
            f"verdict = {geometry_verdict}",
            "",
            "D. Target/Color Balance and Episode Length",
        ]
    )
    for row in length_df.itertuples(index=False):
        lines.append(f"{row.target_color}: count={row.count}, mean={row.mean:.2f}, std={row.std:.2f}, median={row.median:.1f}, min={row.min}, max={row.max}")
    lines.extend(["", "E. Counterfactual Coverage"])
    for row in coverage_df.itertuples(index=False):
        lines.append(
            f"{row.scene_tolerance}: threshold={row.threshold_value:.4f} scene-proxy L2, "
            f"RB={row.rb_pairs}, RY={row.ry_pairs}, BY={row.by_pairs}, "
            f"RGB candidates={row.rgb_triplet_candidates}, any-cross coverage={row.fraction_with_any_cross_color_counterpart:.3f}"
        )
    lines.extend(
        [
            f"verdict = {counterfactual_verdict}",
            "",
            "F. Matched Action Divergence",
            f"pairwise onsets = RB {onsets.get('red_blue')}, RY {onsets.get('red_yellow')}, BY {onsets.get('blue_yellow')}",
            f"aggregate onset = {onsets.get('aggregate')}",
            f"verdict = {matched_signal_verdict}",
            "",
            "G. Action-to-Target Predictability Under Matched Geometry",
        ]
    )
    if action_predict_df.empty:
        lines.append("insufficient matched support")
    else:
        for row in action_predict_df.itertuples(index=False):
            lines.append(
                f"{row.pair_type} {row.progress_window}: raw={row.raw_chunk_accuracy:.3f}, residual={row.residual_chunk_accuracy:.3f}, chance={row.chance_accuracy:.3f}, pairs={row.support_pairs}"
            )
    lines.extend(
        [
            "",
            "H. Existing Commitment Window Alignment",
            f"existing onset progress median = {alignment_summary['existing_median_onset_progress']:.4f}",
            f"matched aggregate onset progress = {alignment_summary['matched_aggregate_onset_progress']}",
            f"median delta progress = {alignment_summary['median_delta_progress']}",
            f"IQR delta progress = {alignment_summary['iqr_delta_progress']}",
            f"relation fractions = {alignment_summary['relation_fractions']}",
            "",
            "I. Evidence Table",
        ]
    )
    for row in evidence_table:
        lines.append(f"{row['evidence']}: {row['result']} -> {row['interpretation']}")
    lines.extend(
        [
            "",
            "J. Overall Conditional-Identifiability Verdict",
            overall,
            "",
            "K. Is Dataset Redesign the Next Priority?",
            redesign,
            "",
            "L. Recommended Next Step",
            next_step,
            "",
            "M. If Redesign Is Recommended",
            "Use exact same-layout triplets: one shared cube layout and robot initial state, with separate red/blue/yellow instruction-demonstration episodes." if redesign != "NO" else "Current audit does not make redesign the primary next step.",
            "",
            "N. Output Paths",
            str(output_dir.resolve()),
            "",
            "O. One-Sentence Conclusion",
            (
                "The dataset does not expose exact geometry, so this audit relies on first-frame scene proxies; within that limit it asks whether scene-only cues, same-context coverage, and geometry-controlled action divergence together make language use statistically necessary."
            ),
        ]
    )
    return "\n".join(lines) + "\n", summary


def main() -> None:
    args = parse_args()
    ensure_output_dir(args.output_dir)
    episodes_df, per_episode, tasks_df = load_dataset(args.dataset_root)
    first_frames = decode_first_frames(args.dataset_root, episodes_df)
    scene_df, feature_matrix = build_scene_proxy_features(episodes_df, per_episode, first_frames, args.pca_dim)
    scene_df.to_csv(args.output_dir / "episode_scene_table.csv", index=False)

    labels = scene_df["target_color"].map({color: idx for idx, color in enumerate(COLORS)}).to_numpy(dtype=np.int64)
    geometry_rows, confusion_df = run_geometry_classifier_cv(
        feature_matrix,
        labels,
        args.pca_dim,
        args.tree_depth,
        args.min_leaf,
    )
    geometry_cv_df = pd.DataFrame(geometry_rows)
    geometry_cv_df["chance_accuracy"] = 1.0 / 3.0
    geometry_cv_df.to_csv(args.output_dir / "geometry_shortcut_cv.csv", index=False)
    confusion_df.to_csv(args.output_dir / "geometry_shortcut_confusion.csv", index=False)
    shuffle_acc = shuffled_sanity_accuracy(feature_matrix, labels, args.pca_dim)

    nearest_df, coverage_df, thresholds, moderate_pairs = compute_counterfactual_coverage(scene_df, feature_matrix)
    nearest_df.to_csv(args.output_dir / "scene_match_pairs.csv", index=False)
    coverage_df.to_csv(args.output_dir / "counterfactual_coverage.csv", index=False)
    del moderate_pairs

    frame_df, chunk_df, match_df, onsets = analyze_matched_divergence(scene_df, per_episode, feature_matrix, thresholds)
    frame_df.to_csv(args.output_dir / "matched_action_divergence.csv", index=False)
    chunk_df.to_csv(args.output_dir / "matched_chunk_divergence.csv", index=False)
    match_df.to_csv(args.output_dir / "scene_match_pairs_moderate.csv", index=False)

    action_predict_df = action_to_target_predictability(chunk_df, match_df, per_episode, feature_matrix, scene_df)
    action_predict_df.to_csv(args.output_dir / "action_target_predictability.csv", index=False)

    length_df = episode_length_summary(scene_df)
    length_df.to_csv(args.output_dir / "episode_length_by_color.csv", index=False)

    alignment_df, alignment_summary = commitment_alignment(args.commitment_manifest, scene_df, onsets)
    alignment_df.to_csv(args.output_dir / "commitment_window_alignment.csv", index=False)

    plot_confusion(confusion_df, args.output_dir / "geometry_shortcut_confusion.png")
    plot_scene_distance_distribution(nearest_df, args.output_dir / "cross_color_nearest_scene_distance.png")
    plot_divergence(chunk_df, args.output_dir / "matched_chunk_divergence.png")

    generator_audit = {
        "target_cycle": "red -> blue -> yellow repeating by saved_episodes % 3",
        "positions_sampled_before_color_assignment": True,
        "shared_position_rng": True,
        "robot_initial_state_fixed": True,
        "tilt_randomized_independently": True,
        "geometry_recoverability_limit": "dataset/meta do not store cube xyz or seed per episode; generator prints positions but no retained sidecar/log was found",
        "episode_index_modulo_leakage_risk": "target order is deterministic, but episode_index/task_index were excluded from all classifier features",
    }
    recoverability = "IMAGE-BASED PROXY ONLY"

    report_text, summary = build_report(
        args.output_dir,
        recoverability,
        generator_audit,
        scene_df,
        geometry_cv_df,
        shuffle_acc,
        length_df,
        nearest_df,
        coverage_df,
        chunk_df,
        action_predict_df,
        onsets,
        alignment_summary,
    )
    (args.output_dir / "conditional_identifiability_report.txt").write_text(report_text)
    (args.output_dir / "conditional_identifiability_summary.json").write_text(json.dumps(summary, indent=2))

    print(args.output_dir.resolve())


if __name__ == "__main__":
    main()
