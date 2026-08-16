#!/usr/bin/env python3
"""Generate a reproducible fingerprint for a LeRobot dataset root."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    with path.open() as file:
        return json.load(file)


def to_python_scalar(value: Any) -> Any:
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return value
    return value


def summarize_episode_lengths(episodes: pd.DataFrame) -> dict[str, Any]:
    lengths = episodes["length"]
    return {
        "count": int(lengths.count()),
        "min": int(lengths.min()),
        "max": int(lengths.max()),
        "mean": float(lengths.mean()),
        "std": float(lengths.std()),
        "median": float(lengths.median()),
        "value_counts": {str(k): int(v) for k, v in lengths.value_counts().sort_index().items()},
        "per_episode": [
            {
                "episode_index": int(row["episode_index"]),
                "length": int(row["length"]),
                "dataset_from_index": int(row["dataset_from_index"]),
                "dataset_to_index": int(row["dataset_to_index"]),
                "tasks": [str(x) for x in row["tasks"]],
            }
            for _, row in episodes.iterrows()
        ],
    }


def build_fingerprint(dataset_root: Path) -> dict[str, Any]:
    info = load_json(dataset_root / "meta" / "info.json")
    stats = load_json(dataset_root / "meta" / "stats.json")
    tasks = pd.read_parquet(dataset_root / "meta" / "tasks.parquet")
    episodes = pd.read_parquet(dataset_root / "meta" / "episodes" / "chunk-000" / "file-000.parquet")

    files = sorted(path for path in dataset_root.rglob("*") if path.is_file())
    file_manifest = [
        {
            "relative_path": str(path.relative_to(dataset_root)),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in files
    ]

    features = info["features"]
    camera_keys = sorted(key for key in features if key.startswith("observation.images."))

    return {
        "dataset_root": str(dataset_root.resolve()),
        "codebase_version": info.get("codebase_version"),
        "robot_type": info.get("robot_type"),
        "fps": info.get("fps"),
        "total_episodes": int(info.get("total_episodes", 0)),
        "total_frames": int(info.get("total_frames", 0)),
        "total_tasks": int(info.get("total_tasks", 0)),
        "splits": info.get("splits"),
        "data_path": info.get("data_path"),
        "video_path": info.get("video_path"),
        "feature_keys": sorted(features.keys()),
        "camera_keys": camera_keys,
        "state_shape": features.get("observation.state", {}).get("shape"),
        "action_shape": features.get("action", {}).get("shape"),
        "feature_schema": features,
        "task_metadata": {
            "columns": tasks.columns.tolist(),
            "num_rows": len(tasks),
            "index_name": tasks.index.name,
            "rows": [
                {
                    "task": str(index),
                    **{column: to_python_scalar(row[column]) for column in tasks.columns},
                }
                for index, row in tasks.iterrows()
            ],
        },
        "episode_lengths": summarize_episode_lengths(episodes),
        "meta_info_json": info,
        "meta_stats_json": stats,
        "file_manifest": file_manifest,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    fingerprint = build_fingerprint(args.dataset_root.expanduser().resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as file:
        json.dump(fingerprint, file, indent=2)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
