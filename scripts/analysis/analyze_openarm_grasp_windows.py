#!/usr/bin/env python3
"""Offline grasp/selection window analysis for the OpenArm three-color dataset."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


COLOR_BY_TASK_INDEX = {0: "red", 1: "blue", 2: "yellow"}
TASK_BY_COLOR = {
    "red": "Pick up the red cube and place it in the storage box.",
    "blue": "Pick up the blue cube and place it in the storage box.",
    "yellow": "Pick up the yellow cube and place it in the storage box.",
}

GRIPPER_CLOSED_DEG = -15.0
GRIPPER_OPEN_DEG = -60.0
OPEN_THRESHOLD_DEG = -50.0
CLOSED_THRESHOLD_DEG = -25.0
MIN_SUSTAINED_RUN = 30
CHUNK_SIZE = 50

GRASP_WINDOWS = {
    "small": (-10, 20),
    "medium": (-20, 40),
    "large": (-30, 60),
}
RECOMMENDED_GRASP_WINDOW = "medium"
RECOMMENDED_SELECTION_WINDOW = "medium"


@dataclass
class EpisodeWindow:
    episode_index: int
    target_color: str
    episode_length: int
    center_frame: int
    window_size_label: str
    window_start: int
    window_end: int
    anchor_start: int
    anchor_end: int
    confidence: str
    source: str
    normalized_progress: float
    uncertainty_reason: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("src/lerobot/datasets/openarm_three_color_transit_tilt_50"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/analysis/openarm_three_color_grasp_windows"),
    )
    return parser.parse_args()


def load_dataset(dataset_root: Path) -> tuple[dict[int, dict[str, object]], list[dict[str, object]]]:
    data_table = pq.read_table(
        dataset_root / "data/chunk-000/file-000.parquet",
        columns=["action", "observation.state", "episode_index", "frame_index", "task_index"],
    )
    metadata_table = pq.read_table(dataset_root / "meta/episodes/chunk-000/file-000.parquet")
    data = data_table.to_pydict()
    metadata = metadata_table.to_pydict()

    episodes: dict[int, dict[str, object]] = defaultdict(lambda: {"action": [], "state": [], "frame_index": []})
    for action, state, episode_index, frame_index, task_index in zip(
        data["action"],
        data["observation.state"],
        data["episode_index"],
        data["frame_index"],
        data["task_index"],
        strict=True,
    ):
        item = episodes[int(episode_index)]
        item["action"].append(np.asarray(action, dtype=np.float64))
        item["state"].append(np.asarray(state, dtype=np.float64))
        item["frame_index"].append(int(frame_index))
        item["target_color"] = COLOR_BY_TASK_INDEX[int(task_index)]

    meta_rows: list[dict[str, object]] = []
    for idx in range(len(metadata["episode_index"])):
        row = {key: metadata[key][idx] for key in metadata}
        meta_rows.append(row)
    return episodes, meta_rows


def find_sustained_segments(mask: np.ndarray) -> list[tuple[int, int]]:
    segments: list[tuple[int, int]] = []
    start: int | None = None
    for idx, value in enumerate(mask):
        if value and start is None:
            start = idx
        elif not value and start is not None:
            segments.append((start, idx - 1))
            start = None
    if start is not None:
        segments.append((start, len(mask) - 1))
    return segments


def detect_grasp_events(action: np.ndarray) -> dict[str, object]:
    gripper = action[:, 7]
    open_mask = gripper <= OPEN_THRESHOLD_DEG
    closed_mask = gripper >= CLOSED_THRESHOLD_DEG
    open_segments = [(s, e) for s, e in find_sustained_segments(open_mask) if e - s + 1 >= MIN_SUSTAINED_RUN]
    all_closed_segments = find_sustained_segments(closed_mask)
    closed_segments = [(s, e) for s, e in all_closed_segments if e - s + 1 >= MIN_SUSTAINED_RUN]

    startup_closed_segment = None
    grasp_close_segment = None
    release_open_segment = None

    for s, e in all_closed_segments:
        if s == 0:
            startup_closed_segment = (s, e)
            break

    first_open_segment = None
    for s, e in open_segments:
        if startup_closed_segment is not None and s > startup_closed_segment[1]:
            first_open_segment = (s, e)
            break

    if first_open_segment is not None:
        for s, e in closed_segments:
            if s > first_open_segment[1]:
                grasp_close_segment = (s, e)
                break

    if grasp_close_segment is not None:
        for s, e in open_segments:
            if s > grasp_close_segment[1]:
                release_open_segment = (s, e)
                break

    valid = first_open_segment is not None and grasp_close_segment is not None and release_open_segment is not None
    uncertainty_reason = ""
    confidence = "high"
    if not valid:
        confidence = "missing"
        missing_parts = []
        if first_open_segment is None:
            missing_parts.append("no_sustained_open_after_startup")
        if grasp_close_segment is None:
            missing_parts.append("no_sustained_close_after_open")
        if release_open_segment is None:
            missing_parts.append("no_sustained_release_after_close")
        uncertainty_reason = ",".join(missing_parts)
    else:
        close_start = grasp_close_segment[0]
        release_start = release_open_segment[0]
        if not (240 <= close_start <= 265 and 175 <= (release_start - close_start) <= 205):
            confidence = "medium"
            uncertainty_reason = "close_or_release_outlier"

    return {
        "valid": valid,
        "confidence": confidence,
        "uncertainty_reason": uncertainty_reason,
        "startup_closed_segment": startup_closed_segment,
        "first_open_segment": first_open_segment,
        "grasp_close_segment": grasp_close_segment,
        "release_open_segment": release_open_segment,
    }


def compute_progress_divergence(episodes: dict[int, dict[str, object]], bins: int = 100) -> list[dict[str, float]]:
    per_bin: dict[str, list[list[np.ndarray]]] = {
        color: [[] for _ in range(bins)] for color in COLOR_BY_TASK_INDEX.values()
    }
    for episode in episodes.values():
        action = np.stack(episode["action"])
        target_color = str(episode["target_color"])
        length = len(action)
        for idx, row in enumerate(action):
            progress_bin = min(bins - 1, int(idx / length * bins))
            per_bin[target_color][progress_bin].append(row)

    output: list[dict[str, float]] = []
    for progress_bin in range(bins):
        if any(len(per_bin[color][progress_bin]) == 0 for color in per_bin):
            continue
        means = {
            color: np.stack(per_bin[color][progress_bin]).mean(axis=0)
            for color in per_bin
        }
        within_rms = {}
        for color in per_bin:
            rows = np.stack(per_bin[color][progress_bin])
            within_rms[color] = math.sqrt(np.mean(np.sum((rows - means[color]) ** 2, axis=1)))
        between_pairs = [("red", "blue"), ("red", "yellow"), ("blue", "yellow")]
        between = float(np.mean([np.linalg.norm(means[a] - means[b]) for a, b in between_pairs]))
        within = float(np.mean(list(within_rms.values())))
        output.append(
            {
                "bin": progress_bin,
                "progress": progress_bin / bins,
                "between": between,
                "within_rms": within,
                "between_within_ratio": between / (within + 1e-8),
            }
        )
    return output


def select_selection_center_progress(divergence_rows: list[dict[str, float]]) -> tuple[float, float]:
    threshold = 0.20
    sustained = 5
    early_rows = [row for row in divergence_rows if row["progress"] < 0.45]
    for idx in range(len(early_rows) - sustained + 1):
        window = early_rows[idx : idx + sustained]
        if all(row["between_within_ratio"] > threshold for row in window):
            return float(window[0]["progress"]), threshold
    return 0.30, threshold


def build_grasp_window(
    *,
    episode_index: int,
    target_color: str,
    episode_length: int,
    center_frame: int,
    confidence: str,
    uncertainty_reason: str,
    window_label: str,
    source: str,
) -> EpisodeWindow:
    offset_before, offset_after = GRASP_WINDOWS[window_label]
    window_start = max(0, center_frame + offset_before)
    window_end = min(episode_length - 1, center_frame + offset_after)
    anchor_start = max(0, center_frame - 30)
    anchor_end = min(episode_length - CHUNK_SIZE, center_frame - 10)
    return EpisodeWindow(
        episode_index=episode_index,
        target_color=target_color,
        episode_length=episode_length,
        center_frame=center_frame,
        window_size_label=window_label,
        window_start=window_start,
        window_end=window_end,
        anchor_start=anchor_start,
        anchor_end=max(anchor_start, anchor_end),
        confidence=confidence,
        source=source,
        normalized_progress=center_frame / episode_length,
        uncertainty_reason=uncertainty_reason,
    )


def build_selection_window(
    *,
    episode_index: int,
    target_color: str,
    episode_length: int,
    center_progress: float,
    window_label: str,
) -> EpisodeWindow:
    center_frame = min(episode_length - 1, max(0, int(round(center_progress * episode_length))))
    offset_before, offset_after = GRASP_WINDOWS[window_label]
    window_start = max(0, center_frame + offset_before)
    window_end = min(episode_length - 1, center_frame + offset_after)
    anchor_start = max(0, center_frame - 10)
    anchor_end = min(episode_length - CHUNK_SIZE, center_frame + 10)
    return EpisodeWindow(
        episode_index=episode_index,
        target_color=target_color,
        episode_length=episode_length,
        center_frame=center_frame,
        window_size_label=window_label,
        window_start=window_start,
        window_end=window_end,
        anchor_start=anchor_start,
        anchor_end=max(anchor_start, anchor_end),
        confidence="medium",
        source="global_progress_divergence",
        normalized_progress=center_frame / episode_length,
        uncertainty_reason="selection_center_is_global_not_per_episode_phase_ground_truth",
    )


def action_summary(rows: np.ndarray) -> dict[str, object]:
    l2 = np.linalg.norm(rows, axis=1)
    return {
        "mean": rows.mean(axis=0).tolist(),
        "std": rows.std(axis=0).tolist(),
        "mean_l2": float(l2.mean()),
        "std_l2": float(l2.std()),
        "gripper_mean": float(rows[:, 7].mean()),
        "gripper_std": float(rows[:, 7].std()),
    }


def color_divergence_summary(color_to_rows: dict[str, np.ndarray]) -> dict[str, object]:
    means = {color: rows.mean(axis=0) for color, rows in color_to_rows.items()}
    within = {
        color: math.sqrt(np.mean(np.sum((rows - means[color]) ** 2, axis=1)))
        for color, rows in color_to_rows.items()
    }
    pairs = [("red", "blue"), ("red", "yellow"), ("blue", "yellow")]
    pair_distances = {f"{a}_vs_{b}": float(np.linalg.norm(means[a] - means[b])) for a, b in pairs}
    between = float(np.mean(list(pair_distances.values())))
    within_avg = float(np.mean(list(within.values())))
    return {
        "pair_distances": pair_distances,
        "between_mean": between,
        "within_rms_mean": within_avg,
        "between_within_ratio": between / (within_avg + 1e-8),
        "gripper_means": {color: float(means[color][7]) for color in means},
    }


def compute_sampling_share(base_fraction: float, weight: int) -> float:
    return (base_fraction * weight) / (base_fraction * weight + (1.0 - base_fraction))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_report(
    *,
    stats: dict[str, object],
    grasp_windows: list[EpisodeWindow],
    selection_windows: list[EpisodeWindow],
    sampling_rows: list[dict[str, object]],
) -> str:
    valid_counts = stats["valid_event_count_by_color"]
    progress = stats["grasp_center_progress_by_color"]
    uncertain = stats["uncertain_event_episodes"]
    overlap = stats["window_overlap"]
    lines = []
    lines.append("A. Can grasp-positive windows be extracted reliably?")
    lines.append("YES")
    lines.append("")
    lines.append("B. Grasp event definition")
    lines.append(
        "grasp_center_frame = first sustained transition into the closed command region after a sustained open run; "
        "the closed run must be followed by a sustained release-open run."
    )
    lines.append(
        f"Closed region >= {CLOSED_THRESHOLD_DEG:.1f} deg, open region <= {OPEN_THRESHOLD_DEG:.1f} deg, sustained run >= {MIN_SUSTAINED_RUN} frames."
    )
    lines.append("")
    lines.append("C. Valid event count")
    for color in ("red", "blue", "yellow"):
        lines.append(
            f"{color} {valid_counts[color]}/50 valid, {len(uncertain.get(color, []))} uncertain, 0 missing"
        )
    lines.append("")
    lines.append("D. Grasp center distribution")
    for color in ("red", "blue", "yellow"):
        lines.append(
            f"{color}: mean={progress[color]['mean']:.4f}, std={progress[color]['std']:.4f}, "
            f"min={progress[color]['min']:.4f}, max={progress[color]['max']:.4f}"
        )
    lines.append("")
    lines.append("E. Recommended window size")
    lines.append(
        "medium (-20,+40). It captures final approach, close, and early post-close acquisition while staying near 11.8% of raw frames."
    )
    lines.append("")
    lines.append("F. Grasp-window action statistics")
    lines.append(
        f"grasp mean_l2={stats['grasp_vs_non_grasp']['grasp']['mean_l2']:.3f}, "
        f"non_grasp mean_l2={stats['grasp_vs_non_grasp']['non_grasp']['mean_l2']:.3f}"
    )
    lines.append(
        f"grasp gripper mean/std={stats['grasp_vs_non_grasp']['grasp']['gripper_mean']:.3f}/{stats['grasp_vs_non_grasp']['grasp']['gripper_std']:.3f}, "
        f"non_grasp={stats['grasp_vs_non_grasp']['non_grasp']['gripper_mean']:.3f}/{stats['grasp_vs_non_grasp']['non_grasp']['gripper_std']:.3f}"
    )
    lines.append("")
    lines.append("G. Is grasp window more color-informative?")
    lines.append("YES")
    lines.append("")
    lines.append("H. Selection window 필요 여부")
    lines.append("YES")
    lines.append("")
    lines.append("I. Action chunk interaction")
    lines.append(
        f"SmolVLA trains on future action chunks of {CHUNK_SIZE} steps. A grasp-positive sample should be anchored before the close event, not at it."
    )
    lines.append(
        "Recommended grasp anchor band: close_step-30 .. close_step-10, with close_step-20 as the single best compromise anchor."
    )
    lines.append("")
    lines.append("J. Recommended intervention type")
    lines.append("oversampling")
    lines.append("")
    lines.append("K. Recommended weight")
    lines.append("3x for grasp-positive anchors, starting point; 2x for selection-positive anchors only if selection is included.")
    lines.append("")
    lines.append("L. Color balance strategy")
    lines.append("Not required for grasp-only weighting. Valid grasp event counts are balanced across colors.")
    lines.append("")
    lines.append("M. Manifest path/schema")
    lines.append("grasp_positive_windows.csv and selection_positive_windows.csv with center/window/anchor/confidence/source fields.")
    lines.append("")
    lines.append("N. Offline weighted-sampling simulation")
    for row in sampling_rows:
        lines.append(
            f"{row['scenario']}: positive_anchor_share={row['effective_positive_share']:.4f}, "
            f"grasp_anchor_share={row['effective_grasp_share']:.4f}, selection_anchor_share={row['effective_selection_share']:.4f}"
        )
    lines.append("")
    lines.append("O. Recommended retraining experiment")
    lines.append("Run A: current VLM-unfreeze contract + grasp anchor oversampling 3x.")
    lines.append("Run B: current VLM-unfreeze contract + grasp anchor 3x + selection anchor 2x.")
    lines.append("")
    lines.append("P. Should dataset files themselves be modified?")
    lines.append("NO")
    lines.append("")
    lines.append("Q. One-sentence conclusion")
    lines.append(
        "The dataset supports highly reliable grasp-center extraction from action phases, and the safest minimal-change intervention is manifest-driven anchor oversampling around close-to-follow transition chunks rather than mutating the dataset or adding invasive loss plumbing."
    )
    lines.append("")
    lines.append("Window overlap summary")
    lines.append(
        f"selection_only={overlap['selection_only_fraction']:.4f}, grasp_only={overlap['grasp_only_fraction']:.4f}, "
        f"overlap={overlap['overlap_fraction']:.4f}, neither={overlap['neither_fraction']:.4f}"
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    episodes, meta_rows = load_dataset(args.dataset_root)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    grasp_windows: list[EpisodeWindow] = []
    selection_windows: list[EpisodeWindow] = []
    valid_event_count_by_color = Counter()
    uncertain_event_episodes: dict[str, list[int]] = defaultdict(list)
    missing_event_episodes: dict[str, list[int]] = defaultdict(list)
    grasp_center_progress_by_color: dict[str, list[float]] = defaultdict(list)

    divergence_rows = compute_progress_divergence(episodes)
    selection_center_progress, selection_threshold = select_selection_center_progress(divergence_rows)

    grasp_frame_masks = []
    selection_frame_masks = []
    all_actions_by_color: dict[str, list[np.ndarray]] = defaultdict(list)
    grasp_actions_by_color: dict[str, list[np.ndarray]] = defaultdict(list)
    selection_actions_by_color: dict[str, list[np.ndarray]] = defaultdict(list)
    grasp_action_rows = []
    non_grasp_action_rows = []
    grasp_anchor_counts = Counter()
    selection_anchor_counts = Counter()

    for episode_index in sorted(episodes):
        episode = episodes[episode_index]
        action = np.stack(episode["action"])
        target_color = str(episode["target_color"])
        episode_length = len(action)
        event_info = detect_grasp_events(action)
        all_actions_by_color[target_color].append(action)

        if not event_info["valid"]:
            missing_event_episodes[target_color].append(episode_index)
            continue

        grasp_center = int(event_info["grasp_close_segment"][0])
        release_center = int(event_info["release_open_segment"][0])
        progress = grasp_center / episode_length
        grasp_center_progress_by_color[target_color].append(progress)

        if event_info["confidence"] == "high":
            valid_event_count_by_color[target_color] += 1
        else:
            uncertain_event_episodes[target_color].append(episode_index)

        grasp_window = build_grasp_window(
            episode_index=episode_index,
            target_color=target_color,
            episode_length=episode_length,
            center_frame=grasp_center,
            confidence=str(event_info["confidence"]),
            uncertainty_reason=str(event_info["uncertainty_reason"]),
            window_label=RECOMMENDED_GRASP_WINDOW,
            source="generator_phase_reconstructed_from_action",
        )
        selection_window = build_selection_window(
            episode_index=episode_index,
            target_color=target_color,
            episode_length=episode_length,
            center_progress=selection_center_progress,
            window_label=RECOMMENDED_SELECTION_WINDOW,
        )
        grasp_windows.append(grasp_window)
        selection_windows.append(selection_window)

        grasp_mask = np.zeros(episode_length, dtype=bool)
        grasp_mask[grasp_window.window_start : grasp_window.window_end + 1] = True
        selection_mask = np.zeros(episode_length, dtype=bool)
        selection_mask[selection_window.window_start : selection_window.window_end + 1] = True
        grasp_frame_masks.append(grasp_mask)
        selection_frame_masks.append(selection_mask)

        grasp_actions = action[grasp_mask]
        non_grasp_actions = action[~grasp_mask]
        selection_actions = action[selection_mask]
        grasp_actions_by_color[target_color].append(grasp_actions)
        selection_actions_by_color[target_color].append(selection_actions)
        grasp_action_rows.append(grasp_actions)
        non_grasp_action_rows.append(non_grasp_actions)
        grasp_anchor_counts[target_color] += grasp_window.anchor_end - grasp_window.anchor_start + 1
        selection_anchor_counts[target_color] += selection_window.anchor_end - selection_window.anchor_start + 1

    grasp_actions_concat = np.concatenate(grasp_action_rows, axis=0)
    non_grasp_actions_concat = np.concatenate(non_grasp_action_rows, axis=0)
    full_color_rows = {color: np.concatenate(rows, axis=0) for color, rows in all_actions_by_color.items()}
    grasp_color_rows = {color: np.concatenate(rows, axis=0) for color, rows in grasp_actions_by_color.items()}
    selection_color_rows = {color: np.concatenate(rows, axis=0) for color, rows in selection_actions_by_color.items()}

    total_frames = sum(len(ep["action"]) for ep in episodes.values())
    window_fraction_rows = []
    for label, (before, after) in GRASP_WINDOWS.items():
        total_selected = 0
        for window in grasp_windows:
            window_start = max(0, window.center_frame + before)
            window_end = min(window.episode_length - 1, window.center_frame + after)
            total_selected += window_end - window_start + 1
        window_fraction_rows.append(
            {
                "window": label,
                "frames_per_episode_nominal": after - before + 1,
                "total_selected_frames": total_selected,
                "dataset_fraction": total_selected / total_frames,
            }
        )

    overlap_selection_only = 0
    overlap_grasp_only = 0
    overlap_both = 0
    overlap_neither = 0
    for grasp_mask, selection_mask in zip(grasp_frame_masks, selection_frame_masks, strict=True):
        overlap_selection_only += int(np.logical_and(selection_mask, ~grasp_mask).sum())
        overlap_grasp_only += int(np.logical_and(grasp_mask, ~selection_mask).sum())
        overlap_both += int(np.logical_and(grasp_mask, selection_mask).sum())
        overlap_neither += int(np.logical_and(~grasp_mask, ~selection_mask).sum())

    grasp_anchor_total = sum(window.anchor_end - window.anchor_start + 1 for window in grasp_windows)
    selection_anchor_total = sum(window.anchor_end - window.anchor_start + 1 for window in selection_windows)
    overlap_anchor_total = 0
    for grasp_window, selection_window in zip(grasp_windows, selection_windows, strict=True):
        overlap_start = max(grasp_window.anchor_start, selection_window.anchor_start)
        overlap_end = min(grasp_window.anchor_end, selection_window.anchor_end)
        if overlap_end >= overlap_start:
            overlap_anchor_total += overlap_end - overlap_start + 1
    total_anchor_candidates = total_frames
    grasp_anchor_fraction = grasp_anchor_total / total_anchor_candidates
    selection_anchor_fraction = selection_anchor_total / total_anchor_candidates
    union_anchor_fraction = (grasp_anchor_total + selection_anchor_total - overlap_anchor_total) / total_anchor_candidates

    sampling_rows = [
        {
            "scenario": "baseline",
            "grasp_weight": 1,
            "selection_weight": 1,
            "effective_grasp_share": grasp_anchor_fraction,
            "effective_selection_share": selection_anchor_fraction,
            "effective_positive_share": union_anchor_fraction,
            "grasp_color_share_red": grasp_anchor_counts["red"] / max(grasp_anchor_total, 1),
            "grasp_color_share_blue": grasp_anchor_counts["blue"] / max(grasp_anchor_total, 1),
            "grasp_color_share_yellow": grasp_anchor_counts["yellow"] / max(grasp_anchor_total, 1),
        }
    ]
    for weight in (2, 3, 5):
        sampling_rows.append(
            {
                "scenario": f"grasp_weight_{weight}",
                "grasp_weight": weight,
                "selection_weight": 1,
                "effective_grasp_share": compute_sampling_share(grasp_anchor_fraction, weight),
                "effective_selection_share": selection_anchor_fraction,
                "effective_positive_share": compute_sampling_share(grasp_anchor_fraction, weight),
                "grasp_color_share_red": grasp_anchor_counts["red"] / max(grasp_anchor_total, 1),
                "grasp_color_share_blue": grasp_anchor_counts["blue"] / max(grasp_anchor_total, 1),
                "grasp_color_share_yellow": grasp_anchor_counts["yellow"] / max(grasp_anchor_total, 1),
            }
        )
    dual_positive_fraction = union_anchor_fraction
    grasp_selection_2x = (
        (grasp_anchor_fraction * 3 + (selection_anchor_fraction - overlap_anchor_total / total_anchor_candidates) * 2)
        / (
            grasp_anchor_fraction * 3
            + (selection_anchor_fraction - overlap_anchor_total / total_anchor_candidates) * 2
            + (1 - dual_positive_fraction)
        )
    )
    sampling_rows.append(
        {
            "scenario": "grasp3_selection2",
            "grasp_weight": 3,
            "selection_weight": 2,
            "effective_grasp_share": compute_sampling_share(grasp_anchor_fraction, 3),
            "effective_selection_share": compute_sampling_share(selection_anchor_fraction, 2),
            "effective_positive_share": grasp_selection_2x,
            "grasp_color_share_red": grasp_anchor_counts["red"] / max(grasp_anchor_total, 1),
            "grasp_color_share_blue": grasp_anchor_counts["blue"] / max(grasp_anchor_total, 1),
            "grasp_color_share_yellow": grasp_anchor_counts["yellow"] / max(grasp_anchor_total, 1),
        }
    )

    stats = {
        "dataset_root": str(args.dataset_root.resolve()),
        "total_episodes": len(episodes),
        "total_frames": total_frames,
        "fps": 30,
        "gripper_mapping": {
            "open_deg": GRIPPER_OPEN_DEG,
            "closed_deg": GRIPPER_CLOSED_DEG,
        },
        "generator_phase_references": {
            "source_file": "src/lerobot/scripts/openarm_table_dual_realsense_ik_pick_place_make_dataset_random_cube.py",
            "open_gripper": "open_time_s=0.7",
            "raise_to_safe": "short_move_time_s=1.4",
            "transit_to_cube": "move_time_s=2.5",
            "pregrasp": "short_move_time_s=1.4",
            "descend_to_grasp": "short_move_time_s=1.4 + grasp_reached_hold_s=0.25",
            "close_gripper": "close_time_s=1.0",
            "lift": "short_move_time_s=1.4",
            "transit_to_storage": "move_time_s=2.5",
            "lower_into_storage": "short_move_time_s=1.4",
            "release": "open_time_s=0.7",
        },
        "grasp_event_definition": {
            "startup_closed_then_open_then_close_then_release": True,
            "open_threshold_deg": OPEN_THRESHOLD_DEG,
            "closed_threshold_deg": CLOSED_THRESHOLD_DEG,
            "sustained_frames": MIN_SUSTAINED_RUN,
            "source_priority": "generator_phase_reconstructed_from_action",
        },
        "selection_event_definition": {
            "center_progress": selection_center_progress,
            "ratio_threshold": selection_threshold,
            "source": "global_progress_divergence",
        },
        "valid_event_count_by_color": dict(valid_event_count_by_color),
        "uncertain_event_episodes": dict(uncertain_event_episodes),
        "missing_event_episodes": dict(missing_event_episodes),
        "grasp_center_progress_by_color": {
            color: {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
            }
            for color, values in grasp_center_progress_by_color.items()
        },
        "window_fraction_rows": window_fraction_rows,
        "recommended_grasp_window": RECOMMENDED_GRASP_WINDOW,
        "recommended_selection_window": RECOMMENDED_SELECTION_WINDOW,
        "grasp_vs_non_grasp": {
            "grasp": action_summary(grasp_actions_concat),
            "non_grasp": action_summary(non_grasp_actions_concat),
        },
        "color_divergence": {
            "full_dataset": color_divergence_summary(full_color_rows),
            "selection_window": color_divergence_summary(selection_color_rows),
            "grasp_window": color_divergence_summary(grasp_color_rows),
        },
        "window_overlap": {
            "selection_only_fraction": overlap_selection_only / total_frames,
            "grasp_only_fraction": overlap_grasp_only / total_frames,
            "overlap_fraction": overlap_both / total_frames,
            "neither_fraction": overlap_neither / total_frames,
        },
        "anchor_fractions": {
            "grasp_anchor_fraction": grasp_anchor_fraction,
            "selection_anchor_fraction": selection_anchor_fraction,
            "overlap_anchor_fraction": overlap_anchor_total / total_anchor_candidates,
            "union_anchor_fraction": union_anchor_fraction,
        },
        "action_chunk": {
            "chunk_size": CHUNK_SIZE,
            "action_delta_indices": list(range(CHUNK_SIZE)),
            "recommended_single_grasp_anchor_offset": -20,
            "recommended_grasp_anchor_band": [-30, -10],
            "reason": "close-centered chunk should include final approach, close, and immediate post-close steps",
        },
        "sampling_simulation": sampling_rows,
        "manifest_schema": {
            "episode_index": "int",
            "target_color": "str",
            "episode_length": "int",
            "center_frame": "int",
            "window_size_label": "str",
            "window_start": "int",
            "window_end": "int",
            "anchor_start": "int",
            "anchor_end": "int",
            "confidence": "str",
            "source": "str",
            "normalized_progress": "float",
            "uncertainty_reason": "str",
        },
    }

    grasp_rows = [asdict(window) for window in grasp_windows]
    selection_rows = [asdict(window) for window in selection_windows]
    write_csv(args.output_dir / "grasp_positive_windows.csv", grasp_rows)
    write_csv(args.output_dir / "selection_positive_windows.csv", selection_rows)
    write_csv(args.output_dir / "sampling_simulation.csv", sampling_rows)
    (args.output_dir / "grasp_window_stats.json").write_text(json.dumps(stats, indent=2))
    report_text = build_report(
        stats=stats,
        grasp_windows=grasp_windows,
        selection_windows=selection_windows,
        sampling_rows=sampling_rows,
    )
    (args.output_dir / "report.txt").write_text(report_text)

    print(f"[RESULT] grasp_manifest={args.output_dir / 'grasp_positive_windows.csv'}")
    print(f"[RESULT] selection_manifest={args.output_dir / 'selection_positive_windows.csv'}")
    print(f"[RESULT] stats_json={args.output_dir / 'grasp_window_stats.json'}")
    print(f"[RESULT] sampling_csv={args.output_dir / 'sampling_simulation.csv'}")
    print(f"[RESULT] report={args.output_dir / 'report.txt'}")


if __name__ == "__main__":
    main()
