#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import torch


class TripletAwareBatchSampler(torch.utils.data.Sampler[list[int]]):
    def __init__(
        self,
        *,
        base_sampler,
        triplet_frame_indices: list[tuple[int, int, int]],
        batch_size: int,
        triplets_per_batch: int = 1,
        shuffle: bool = True,
    ):
        if batch_size < 3:
            raise ValueError(f"batch_size must be >= 3 for triplet-aware batching, got {batch_size}")
        if triplets_per_batch < 1:
            raise ValueError(f"triplets_per_batch must be >= 1, got {triplets_per_batch}")
        if triplets_per_batch * 3 > batch_size:
            raise ValueError(
                f"triplets_per_batch={triplets_per_batch} exceeds batch capacity for batch_size={batch_size}"
            )
        if len(triplet_frame_indices) == 0:
            raise ValueError("triplet_frame_indices must not be empty")

        self.base_sampler = base_sampler
        self.triplet_frame_indices = triplet_frame_indices
        self.batch_size = batch_size
        self.triplets_per_batch = triplets_per_batch
        self.shuffle = shuffle

    def __len__(self) -> int:
        return max(1, len(self.base_sampler) // self.batch_size)

    def __iter__(self) -> Iterator[list[int]]:
        base_indices = list(iter(self.base_sampler))
        if len(base_indices) == 0:
            return

        triplet_order = torch.arange(len(self.triplet_frame_indices))
        if self.shuffle:
            triplet_order = triplet_order[torch.randperm(len(triplet_order))]

        base_ptr = 0
        triplet_ptr = 0
        for _ in range(len(self)):
            batch: list[int] = []

            for _ in range(self.triplets_per_batch):
                triplet = self.triplet_frame_indices[int(triplet_order[triplet_ptr % len(triplet_order)])]
                batch.extend(int(idx) for idx in triplet)
                triplet_ptr += 1

            while len(batch) < self.batch_size:
                batch.append(int(base_indices[base_ptr % len(base_indices)]))
                base_ptr += 1

            if self.shuffle:
                permutation = torch.randperm(len(batch)).tolist()
                batch = [batch[i] for i in permutation]

            yield batch


def build_episode_indices(
    dataset_from_indices: list[int],
    dataset_to_indices: list[int],
    episode_indices_to_use: list | None = None,
    drop_n_first_frames: int = 0,
    drop_n_last_frames: int = 0,
) -> list[int]:
    indices = []
    for episode_idx, (start_index, end_index) in enumerate(
        zip(dataset_from_indices, dataset_to_indices, strict=True)
    ):
        if episode_indices_to_use is None or episode_idx in episode_indices_to_use:
            indices.extend(range(start_index + drop_n_first_frames, end_index - drop_n_last_frames))
    return indices


class EpisodeAwareSampler:
    def __init__(
        self,
        dataset_from_indices: list[int],
        dataset_to_indices: list[int],
        episode_indices_to_use: list | None = None,
        drop_n_first_frames: int = 0,
        drop_n_last_frames: int = 0,
        shuffle: bool = False,
    ):
        """Sampler that optionally incorporates episode boundary information."""
        self.indices = build_episode_indices(
            dataset_from_indices,
            dataset_to_indices,
            episode_indices_to_use=episode_indices_to_use,
            drop_n_first_frames=drop_n_first_frames,
            drop_n_last_frames=drop_n_last_frames,
        )
        self.shuffle = shuffle

    def __iter__(self) -> Iterator[int]:
        if self.shuffle:
            for i in torch.randperm(len(self.indices)):
                yield self.indices[i]
        else:
            for i in self.indices:
                yield i

    def __len__(self) -> int:
        return len(self.indices)


@dataclass
class GraspOversamplingSummary:
    manifest_path: str
    dataset_root: str
    sampler_type: str
    replacement: bool
    positive_weight: float
    dataset_num_frames: int
    eligible_num_frames: int
    positive_num_frames: int
    positive_episode_count: int
    uncertain_episode_count: int
    uncertain_episode_indices: list[int]
    manifest_row_count: int
    valid_row_count: int
    uncertain_row_count: int
    excluded_row_count: int
    positive_rows_by_color: dict[str, int]
    positive_frames_by_color: dict[str, int]
    expected_sample_share: float
    raw_positive_share: float
    expected_sample_share_by_color: dict[str, float]
    distinct_chunk_count_by_episode: dict[int, int]


@dataclass
class TargetCommitmentOversamplingSummary:
    manifest_path: str
    dataset_root: str
    positive_weight: float
    positive_num_frames: int
    positive_episode_count: int
    uncertain_episode_count: int
    uncertain_episode_indices: list[int]
    manifest_row_count: int
    valid_row_count: int
    uncertain_row_count: int
    excluded_row_count: int
    positive_rows_by_color: dict[str, int]
    positive_frames_by_color: dict[str, int]
    raw_positive_share: float
    expected_sample_share: float
    expected_sample_share_by_color: dict[str, float]
    distinct_chunk_count_by_episode: dict[int, int]


@dataclass
class CombinedOversamplingSummary:
    sampler_type: str
    replacement: bool
    dataset_num_frames: int
    eligible_num_frames: int
    grasp_positive_num_frames: int
    commitment_positive_num_frames: int
    union_positive_num_frames: int
    raw_union_positive_share: float
    expected_union_positive_share: float
    expected_grasp_share: float
    expected_commitment_share: float
    expected_sampling_share_by_color: dict[str, float]
    grasp: GraspOversamplingSummary | None
    target_commitment: TargetCommitmentOversamplingSummary | None


def _normalize_confidence(value: Any) -> str:
    return "" if pd.isna(value) else str(value).strip().lower()


def _collect_manifest_positive_indices(
    *,
    dataset,
    manifest_path: str | Path,
    valid_confidence: set[str],
    episode_indices_to_use: list | None,
    drop_n_first_frames: int,
    drop_n_last_frames: int,
) -> dict[str, Any]:
    manifest_path = Path(manifest_path).expanduser().resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    manifest = pd.read_csv(manifest_path)
    required_columns = {
        "episode_index",
        "target_color",
        "episode_length",
        "anchor_start",
        "anchor_end",
        "confidence",
    }
    missing_columns = sorted(required_columns - set(manifest.columns))
    if missing_columns:
        raise ValueError(f"Manifest is missing required columns: {missing_columns}")

    episodes = dataset.meta.episodes
    num_episodes = len(episodes)
    selected_episode_set = set(range(num_episodes) if episode_indices_to_use is None else episode_indices_to_use)
    eligible_indices = build_episode_indices(
        list(episodes["dataset_from_index"]),
        list(episodes["dataset_to_index"]),
        episode_indices_to_use=episode_indices_to_use,
        drop_n_first_frames=drop_n_first_frames,
        drop_n_last_frames=drop_n_last_frames,
    )
    eligible_index_set = set(eligible_indices)

    valid_rows = []
    uncertain_rows = []
    positive_indices: set[int] = set()
    positive_frames_by_color: Counter[str] = Counter()
    positive_rows_by_color: Counter[str] = Counter()
    positive_episodes: set[int] = set()
    uncertain_episodes: set[int] = set()
    distinct_chunk_count_by_episode: dict[int, int] = {}

    for row in manifest.to_dict("records"):
        episode_index = int(row["episode_index"])
        if episode_index < 0 or episode_index >= num_episodes:
            raise ValueError(f"Manifest episode_index {episode_index} is outside dataset range 0..{num_episodes - 1}")

        episode_meta = episodes[episode_index]
        episode_length = int(episode_meta["length"])
        if int(row["episode_length"]) != episode_length:
            raise ValueError(
                f"Manifest episode_length mismatch for episode {episode_index}: "
                f"manifest={int(row['episode_length'])}, dataset={episode_length}"
            )

        anchor_start = int(row["anchor_start"])
        anchor_end = int(row["anchor_end"])
        if anchor_start < 0 or anchor_end < 0:
            raise ValueError(f"Negative anchor index detected in manifest for episode {episode_index}")
        if anchor_start > anchor_end:
            raise ValueError(f"anchor_start > anchor_end for episode {episode_index}")
        if anchor_end >= episode_length:
            raise ValueError(
                f"Anchor outside episode bounds for episode {episode_index}: "
                f"anchor=({anchor_start}, {anchor_end}), length={episode_length}"
            )

        confidence = _normalize_confidence(row["confidence"])
        if confidence in valid_confidence:
            valid_rows.append(row)
        else:
            uncertain_rows.append(row)

    for row in valid_rows:
        episode_index = int(row["episode_index"])
        if episode_index not in selected_episode_set:
            continue
        episode_meta = episodes[episode_index]
        episode_start = int(episode_meta["dataset_from_index"])
        row_positive_count = 0
        for local_frame in range(int(row["anchor_start"]), int(row["anchor_end"]) + 1):
            global_index = episode_start + local_frame
            if global_index in eligible_index_set:
                positive_indices.add(global_index)
                positive_frames_by_color[str(row["target_color"])] += 1
                row_positive_count += 1
        if row_positive_count > 0:
            positive_rows_by_color[str(row["target_color"])] += 1
            positive_episodes.add(episode_index)
            distinct_chunk_count_by_episode[episode_index] = row_positive_count

    for row in uncertain_rows:
        episode_index = int(row["episode_index"])
        if episode_index in selected_episode_set:
            uncertain_episodes.add(episode_index)

    return {
        "manifest_path": manifest_path,
        "manifest_row_count": len(manifest),
        "valid_row_count": len(valid_rows),
        "uncertain_row_count": len(uncertain_rows),
        "excluded_row_count": len(uncertain_rows),
        "eligible_indices": eligible_indices,
        "positive_indices": positive_indices,
        "positive_frames_by_color": dict(sorted(positive_frames_by_color.items())),
        "positive_rows_by_color": dict(sorted(positive_rows_by_color.items())),
        "positive_episode_count": len(positive_episodes),
        "uncertain_episode_count": len(uncertain_episodes),
        "uncertain_episode_indices": sorted(uncertain_episodes),
        "distinct_chunk_count_by_episode": distinct_chunk_count_by_episode,
    }


def build_grasp_oversampling_sampler(
    *,
    dataset,
    manifest_path: str | Path,
    positive_weight: float,
    seed: int | None,
    episode_indices_to_use: list | None = None,
    drop_n_first_frames: int = 0,
    drop_n_last_frames: int = 0,
) -> tuple[torch.utils.data.WeightedRandomSampler, GraspOversamplingSummary]:
    if positive_weight < 1.0:
        raise ValueError(f"grasp_positive_weight must be >= 1.0, got {positive_weight}")

    dataset_root = str(dataset.root.resolve())
    dataset_num_frames = dataset.num_frames
    manifest_stats = _collect_manifest_positive_indices(
        dataset=dataset,
        manifest_path=manifest_path,
        valid_confidence={"high", "valid"},
        episode_indices_to_use=episode_indices_to_use,
        drop_n_first_frames=drop_n_first_frames,
        drop_n_last_frames=drop_n_last_frames,
    )
    eligible_indices = manifest_stats["eligible_indices"]
    positive_indices = manifest_stats["positive_indices"]

    if not eligible_indices:
        raise ValueError("No eligible dataset indices available for grasp oversampling.")

    weights = torch.zeros(dataset_num_frames, dtype=torch.double)
    weights[eligible_indices] = 1.0
    if positive_indices:
        positive_idx_tensor = torch.tensor(sorted(positive_indices), dtype=torch.long)
        weights[positive_idx_tensor] = positive_weight

    positive_num_frames = len(positive_indices)
    eligible_num_frames = len(eligible_indices)
    raw_positive_share = positive_num_frames / eligible_num_frames
    effective_denominator = (eligible_num_frames - positive_num_frames) + positive_weight * positive_num_frames
    expected_sample_share = (positive_weight * positive_num_frames) / effective_denominator
    expected_sample_share_by_color = {
        color: (positive_weight * count) / effective_denominator for color, count in sorted(positive_frames_by_color.items())
    }

    generator = torch.Generator()
    if seed is not None:
        generator.manual_seed(seed)

    sampler = torch.utils.data.WeightedRandomSampler(
        weights=weights,
        num_samples=eligible_num_frames,
        replacement=True,
        generator=generator,
    )
    summary = GraspOversamplingSummary(
        manifest_path=str(manifest_stats["manifest_path"]),
        dataset_root=dataset_root,
        sampler_type="WeightedRandomSampler",
        replacement=True,
        positive_weight=positive_weight,
        dataset_num_frames=dataset_num_frames,
        eligible_num_frames=eligible_num_frames,
        positive_num_frames=positive_num_frames,
        positive_episode_count=manifest_stats["positive_episode_count"],
        uncertain_episode_count=manifest_stats["uncertain_episode_count"],
        uncertain_episode_indices=manifest_stats["uncertain_episode_indices"],
        manifest_row_count=manifest_stats["manifest_row_count"],
        valid_row_count=manifest_stats["valid_row_count"],
        uncertain_row_count=manifest_stats["uncertain_row_count"],
        excluded_row_count=manifest_stats["excluded_row_count"],
        positive_rows_by_color=manifest_stats["positive_rows_by_color"],
        positive_frames_by_color=manifest_stats["positive_frames_by_color"],
        expected_sample_share=expected_sample_share,
        raw_positive_share=raw_positive_share,
        expected_sample_share_by_color=expected_sample_share_by_color,
        distinct_chunk_count_by_episode=manifest_stats["distinct_chunk_count_by_episode"],
    )
    return sampler, summary


def build_combined_oversampling_sampler(
    *,
    dataset,
    grasp_manifest_path: str | Path | None,
    grasp_positive_weight: float,
    target_commitment_manifest_path: str | Path | None,
    target_commitment_weight: float,
    seed: int | None,
    episode_indices_to_use: list | None = None,
    drop_n_first_frames: int = 0,
    drop_n_last_frames: int = 0,
) -> tuple[torch.utils.data.WeightedRandomSampler, CombinedOversamplingSummary]:
    if grasp_manifest_path is None and target_commitment_manifest_path is None:
        raise ValueError("At least one oversampling manifest must be provided.")
    if grasp_manifest_path is not None and grasp_positive_weight < 1.0:
        raise ValueError(f"grasp_positive_weight must be >= 1.0, got {grasp_positive_weight}")
    if target_commitment_manifest_path is not None and target_commitment_weight < 1.0:
        raise ValueError(f"target_commitment_weight must be >= 1.0, got {target_commitment_weight}")

    dataset_root = str(dataset.root.resolve())
    dataset_num_frames = dataset.num_frames
    episodes = dataset.meta.episodes
    eligible_indices = build_episode_indices(
        list(episodes["dataset_from_index"]),
        list(episodes["dataset_to_index"]),
        episode_indices_to_use=episode_indices_to_use,
        drop_n_first_frames=drop_n_first_frames,
        drop_n_last_frames=drop_n_last_frames,
    )
    if not eligible_indices:
        raise ValueError("No eligible dataset indices available for oversampling.")

    grasp_stats = None
    grasp_indices: set[int] = set()
    if grasp_manifest_path is not None:
        grasp_stats = _collect_manifest_positive_indices(
            dataset=dataset,
            manifest_path=grasp_manifest_path,
            valid_confidence={"high", "valid"},
            episode_indices_to_use=episode_indices_to_use,
            drop_n_first_frames=drop_n_first_frames,
            drop_n_last_frames=drop_n_last_frames,
        )
        grasp_indices = grasp_stats["positive_indices"]

    commitment_stats = None
    commitment_indices: set[int] = set()
    if target_commitment_manifest_path is not None:
        commitment_stats = _collect_manifest_positive_indices(
            dataset=dataset,
            manifest_path=target_commitment_manifest_path,
            valid_confidence={"high", "valid", "medium"},
            episode_indices_to_use=episode_indices_to_use,
            drop_n_first_frames=drop_n_first_frames,
            drop_n_last_frames=drop_n_last_frames,
        )
        commitment_indices = commitment_stats["positive_indices"]

    eligible_num_frames = len(eligible_indices)
    weights = torch.zeros(dataset_num_frames, dtype=torch.double)
    weights[eligible_indices] = 1.0
    union_indices = grasp_indices | commitment_indices
    color_weight_frames: Counter[str] = Counter()
    for idx in union_indices:
        weight = 1.0
        if idx in grasp_indices:
            weight = max(weight, grasp_positive_weight)
        if idx in commitment_indices:
            weight = max(weight, target_commitment_weight)
        weights[idx] = weight
    color_task_map: dict[int, str] = {}
    for episode_index in range(len(episodes)):
        task = episodes[episode_index]["tasks"][0]
        if "red cube" in task:
            color_task_map[episode_index] = "red"
        elif "blue cube" in task:
            color_task_map[episode_index] = "blue"
        elif "yellow cube" in task:
            color_task_map[episode_index] = "yellow"
    for idx in union_indices:
        episode_idx = int(dataset.hf_dataset[int(idx)]["episode_index"])
        color_weight_frames[color_task_map[episode_idx]] += float(weights[idx].item())

    effective_denominator = float(weights[eligible_indices].sum().item())
    grasp_weighted_mass = sum(float(weights[idx].item()) for idx in grasp_indices)
    commitment_weighted_mass = sum(float(weights[idx].item()) for idx in commitment_indices)
    union_weighted_mass = sum(float(weights[idx].item()) for idx in union_indices)
    expected_sampling_share_by_color = {
        color: value / effective_denominator for color, value in sorted(color_weight_frames.items())
    }

    generator = torch.Generator()
    if seed is not None:
        generator.manual_seed(seed)
    sampler = torch.utils.data.WeightedRandomSampler(
        weights=weights,
        num_samples=eligible_num_frames,
        replacement=True,
        generator=generator,
    )

    grasp_summary = None
    if grasp_stats is not None:
        grasp_summary = GraspOversamplingSummary(
            manifest_path=str(grasp_stats["manifest_path"]),
            dataset_root=dataset_root,
            sampler_type="WeightedRandomSampler",
            replacement=True,
            positive_weight=grasp_positive_weight,
            dataset_num_frames=dataset_num_frames,
            eligible_num_frames=eligible_num_frames,
            positive_num_frames=len(grasp_indices),
            positive_episode_count=grasp_stats["positive_episode_count"],
            uncertain_episode_count=grasp_stats["uncertain_episode_count"],
            uncertain_episode_indices=grasp_stats["uncertain_episode_indices"],
            manifest_row_count=grasp_stats["manifest_row_count"],
            valid_row_count=grasp_stats["valid_row_count"],
            uncertain_row_count=grasp_stats["uncertain_row_count"],
            excluded_row_count=grasp_stats["excluded_row_count"],
            positive_rows_by_color=grasp_stats["positive_rows_by_color"],
            positive_frames_by_color=grasp_stats["positive_frames_by_color"],
            expected_sample_share=grasp_weighted_mass / effective_denominator,
            raw_positive_share=len(grasp_indices) / eligible_num_frames,
            expected_sample_share_by_color={
                color: (grasp_positive_weight * count) / effective_denominator
                for color, count in grasp_stats["positive_frames_by_color"].items()
            },
            distinct_chunk_count_by_episode=grasp_stats["distinct_chunk_count_by_episode"],
        )

    commitment_summary = None
    if commitment_stats is not None:
        commitment_summary = TargetCommitmentOversamplingSummary(
            manifest_path=str(commitment_stats["manifest_path"]),
            dataset_root=dataset_root,
            positive_weight=target_commitment_weight,
            positive_num_frames=len(commitment_indices),
            positive_episode_count=commitment_stats["positive_episode_count"],
            uncertain_episode_count=commitment_stats["uncertain_episode_count"],
            uncertain_episode_indices=commitment_stats["uncertain_episode_indices"],
            manifest_row_count=commitment_stats["manifest_row_count"],
            valid_row_count=commitment_stats["valid_row_count"],
            uncertain_row_count=commitment_stats["uncertain_row_count"],
            excluded_row_count=commitment_stats["excluded_row_count"],
            positive_rows_by_color=commitment_stats["positive_rows_by_color"],
            positive_frames_by_color=commitment_stats["positive_frames_by_color"],
            raw_positive_share=len(commitment_indices) / eligible_num_frames,
            expected_sample_share=commitment_weighted_mass / effective_denominator,
            expected_sample_share_by_color={
                color: (target_commitment_weight * count) / effective_denominator
                for color, count in commitment_stats["positive_frames_by_color"].items()
            },
            distinct_chunk_count_by_episode=commitment_stats["distinct_chunk_count_by_episode"],
        )

    combined_summary = CombinedOversamplingSummary(
        sampler_type="WeightedRandomSampler",
        replacement=True,
        dataset_num_frames=dataset_num_frames,
        eligible_num_frames=eligible_num_frames,
        grasp_positive_num_frames=len(grasp_indices),
        commitment_positive_num_frames=len(commitment_indices),
        union_positive_num_frames=len(union_indices),
        raw_union_positive_share=len(union_indices) / eligible_num_frames,
        expected_union_positive_share=union_weighted_mass / effective_denominator,
        expected_grasp_share=grasp_weighted_mass / effective_denominator,
        expected_commitment_share=commitment_weighted_mass / effective_denominator,
        expected_sampling_share_by_color=expected_sampling_share_by_color,
        grasp=grasp_summary,
        target_commitment=commitment_summary,
    )
    return sampler, combined_summary
