#!/usr/bin/env python3
"""Compare two LeRobot dataset roots and report whether they are equivalent."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def load_json(path: Path) -> Any:
    with path.open() as file:
        return json.load(file)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compare_json(left: Any, right: Any) -> bool:
    return left == right


def compare_tasks(left_root: Path, right_root: Path) -> bool:
    left = pd.read_parquet(left_root / "meta" / "tasks.parquet")
    right = pd.read_parquet(right_root / "meta" / "tasks.parquet")
    return left.equals(right)


def compare_episodes(left_root: Path, right_root: Path) -> bool:
    left = pd.read_parquet(left_root / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
    right = pd.read_parquet(right_root / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
    return left.equals(right)


def compare_data_parquet(left_root: Path, right_root: Path, atol: float) -> dict[str, bool]:
    left = pd.read_parquet(left_root / "data" / "chunk-000" / "file-000.parquet")
    right = pd.read_parquet(right_root / "data" / "chunk-000" / "file-000.parquet")
    if left.columns.tolist() != right.columns.tolist() or len(left) != len(right):
        return {
            "schema_identical": False,
            "action_values_identical": False,
            "state_values_identical": False,
            "timestamps_identical": False,
        }

    action_equal = np.allclose(
        np.stack(left["action"].to_list()),
        np.stack(right["action"].to_list()),
        atol=atol,
        rtol=0.0,
    )
    state_equal = np.allclose(
        np.stack(left["observation.state"].to_list()),
        np.stack(right["observation.state"].to_list()),
        atol=atol,
        rtol=0.0,
    )
    timestamp_equal = np.allclose(
        left["timestamp"].to_numpy(),
        right["timestamp"].to_numpy(),
        atol=atol,
        rtol=0.0,
    )
    return {
        "schema_identical": True,
        "action_values_identical": bool(action_equal),
        "state_values_identical": bool(state_equal),
        "timestamps_identical": bool(timestamp_equal),
    }


def compare_file_manifest(left_root: Path, right_root: Path) -> dict[str, bool]:
    left_files = sorted(path.relative_to(left_root) for path in left_root.rglob("*") if path.is_file())
    right_files = sorted(path.relative_to(right_root) for path in right_root.rglob("*") if path.is_file())
    same_relpaths = left_files == right_files
    if not same_relpaths:
        return {"same_relpaths": False, "same_sizes": False}

    same_sizes = True
    same_sha256 = True
    for relpath in left_files:
        left_path = left_root / relpath
        right_path = right_root / relpath
        if left_path.stat().st_size != right_path.stat().st_size:
            same_sizes = False
        if sha256_file(left_path) != sha256_file(right_path):
            same_sha256 = False
    return {"same_relpaths": True, "same_sizes": same_sizes, "same_sha256": same_sha256}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left-root", type=Path, required=True)
    parser.add_argument("--right-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--atol", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    left_root = args.left_root.expanduser().resolve()
    right_root = args.right_root.expanduser().resolve()

    info_equal = compare_json(load_json(left_root / "meta" / "info.json"), load_json(right_root / "meta" / "info.json"))
    stats_equal = compare_json(load_json(left_root / "meta" / "stats.json"), load_json(right_root / "meta" / "stats.json"))
    tasks_equal = compare_tasks(left_root, right_root)
    episodes_equal = compare_episodes(left_root, right_root)
    parquet_checks = compare_data_parquet(left_root, right_root, args.atol)
    file_checks = compare_file_manifest(left_root, right_root)

    report = {
        "left_root": str(left_root),
        "right_root": str(right_root),
        "info_json_identical": info_equal,
        "stats_json_identical": stats_equal,
        "tasks_identical": tasks_equal,
        "episodes_identical": episodes_equal,
        "data_parquet_checks": parquet_checks,
        "file_manifest_checks": file_checks,
    }
    report["HUB_ROUNDTRIP_DATASET_MATCH"] = "PASS" if all(
        [
            info_equal,
            stats_equal,
            tasks_equal,
            episodes_equal,
            parquet_checks["schema_identical"],
            parquet_checks["action_values_identical"],
            parquet_checks["state_values_identical"],
            parquet_checks["timestamps_identical"],
            file_checks["same_relpaths"],
            file_checks["same_sizes"],
            file_checks["same_sha256"],
        ]
    ) else "FAIL"

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as file:
        json.dump(report, file, indent=2)
    print(args.output.resolve())
    print(report["HUB_ROUNDTRIP_DATASET_MATCH"])


if __name__ == "__main__":
    main()
