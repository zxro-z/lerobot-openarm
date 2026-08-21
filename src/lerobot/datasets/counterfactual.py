from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


COLOR_TO_ID = {"red": 0, "blue": 1, "yellow": 2}
EXPECTED_COLOR_IDS = tuple(sorted(COLOR_TO_ID.values()))


@dataclass
class CounterfactualTripletSummary:
    manifest_path: str
    triplet_count: int
    triplet_frame_count: int
    covered_episode_count: int
    min_triplet_frames: int
    max_triplet_frames: int


@dataclass
class CounterfactualTripletMetadata:
    manifest_path: Path
    summary: CounterfactualTripletSummary
    episode_to_triplet_id: dict[int, int]
    episode_to_color_id: dict[int, int]
    episode_to_layout_id: dict[int, int]
    episode_to_permutation_id: dict[int, int]
    episode_to_repeat_index: dict[int, int]
    triplet_frame_indices: list[tuple[int, int, int]]


def resolve_counterfactual_triplet_manifest(dataset_root: Path, manifest_path: str | Path | None) -> Path:
    if manifest_path is not None:
        path = Path(manifest_path).expanduser().resolve()
    else:
        path = (dataset_root / "triplet_manifest.csv").resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Counterfactual triplet manifest not found: {path}")
    return path


def load_counterfactual_triplet_metadata(
    *,
    dataset,
    manifest_path: str | Path | None = None,
    drop_n_first_frames: int = 0,
    drop_n_last_frames: int = 0,
    episode_indices_to_use: list[int] | None = None,
) -> CounterfactualTripletMetadata:
    manifest = resolve_counterfactual_triplet_manifest(dataset.root, manifest_path)
    selected_episodes = None if episode_indices_to_use is None else set(int(i) for i in episode_indices_to_use)
    episode_meta = dataset.meta.episodes

    episode_to_triplet_id: dict[int, int] = {}
    episode_to_color_id: dict[int, int] = {}
    episode_to_layout_id: dict[int, int] = {}
    episode_to_permutation_id: dict[int, int] = {}
    episode_to_repeat_index: dict[int, int] = {}
    triplet_frame_indices: list[tuple[int, int, int]] = []
    triplet_lengths: list[int] = []
    triplet_id = 0

    with manifest.open(newline="") as f:
        reader = csv.DictReader(f)
        required = {"layout_id", "permutation_id", "repeat_index", "red_episode", "blue_episode", "yellow_episode"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Counterfactual triplet manifest missing required columns: {sorted(missing)}")

        for row in reader:
            red_episode = int(row["red_episode"])
            blue_episode = int(row["blue_episode"])
            yellow_episode = int(row["yellow_episode"])
            triplet_episodes = (red_episode, blue_episode, yellow_episode)

            if selected_episodes is not None and any(ep not in selected_episodes for ep in triplet_episodes):
                continue

            local_starts: list[int] = []
            local_ends: list[int] = []
            for ep in triplet_episodes:
                ep_row = episode_meta[ep]
                ep_len = int(ep_row["length"])
                local_starts.append(drop_n_first_frames)
                local_ends.append(ep_len - drop_n_last_frames)

            local_start = max(local_starts)
            local_end = min(local_ends)
            if local_end <= local_start:
                continue

            layout_id = int(row["layout_id"])
            permutation_id = int(row["permutation_id"])
            repeat_index = int(row["repeat_index"])
            for color, ep in zip(("red", "blue", "yellow"), triplet_episodes, strict=True):
                episode_to_triplet_id[ep] = triplet_id
                episode_to_color_id[ep] = COLOR_TO_ID[color]
                episode_to_layout_id[ep] = layout_id
                episode_to_permutation_id[ep] = permutation_id
                episode_to_repeat_index[ep] = repeat_index

            for local_frame in range(local_start, local_end):
                group_indices: list[int] = []
                for ep in triplet_episodes:
                    ep_row = episode_meta[ep]
                    group_indices.append(int(ep_row["dataset_from_index"]) + local_frame)
                triplet_frame_indices.append(tuple(group_indices))  # type: ignore[arg-type]

            triplet_lengths.append(local_end - local_start)
            triplet_id += 1

    if triplet_id == 0:
        raise ValueError(f"No valid counterfactual triplets found from manifest {manifest}")

    summary = CounterfactualTripletSummary(
        manifest_path=str(manifest),
        triplet_count=triplet_id,
        triplet_frame_count=len(triplet_frame_indices),
        covered_episode_count=len(episode_to_triplet_id),
        min_triplet_frames=min(triplet_lengths),
        max_triplet_frames=max(triplet_lengths),
    )
    return CounterfactualTripletMetadata(
        manifest_path=manifest,
        summary=summary,
        episode_to_triplet_id=episode_to_triplet_id,
        episode_to_color_id=episode_to_color_id,
        episode_to_layout_id=episode_to_layout_id,
        episode_to_permutation_id=episode_to_permutation_id,
        episode_to_repeat_index=episode_to_repeat_index,
        triplet_frame_indices=triplet_frame_indices,
    )


class CounterfactualMetadataDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, metadata: CounterfactualTripletMetadata):
        self.dataset = dataset
        self.metadata = metadata

    def __getattr__(self, name: str) -> Any:
        return getattr(self.dataset, name)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = self.dataset[idx]
        episode_index = int(item["episode_index"].item() if hasattr(item["episode_index"], "item") else item["episode_index"])
        item["counterfactual_triplet_id"] = self.metadata.episode_to_triplet_id.get(episode_index, -1)
        item["counterfactual_color_id"] = self.metadata.episode_to_color_id.get(episode_index, -1)
        item["counterfactual_layout_id"] = self.metadata.episode_to_layout_id.get(episode_index, -1)
        item["counterfactual_permutation_id"] = self.metadata.episode_to_permutation_id.get(episode_index, -1)
        item["counterfactual_repeat_index"] = self.metadata.episode_to_repeat_index.get(episode_index, -1)
        item["layout_id"] = item["counterfactual_layout_id"]
        item["permutation_id"] = item["counterfactual_permutation_id"]
        item["repeat_index"] = item["counterfactual_repeat_index"]
        return item
