#!/usr/bin/env python3
"""Launch or validate SmolVLA fine-tuning on the real OpenArm 3-color dataset."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from collections import Counter
from pathlib import Path
from pprint import pformat

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_REPO_ID = "a126-kitech/openarm_pickcube_3colors_no_ep10_12"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs/train/smolvla_openarm_pickcube_3colors_pretrained"
EXPECTED_TASK_COLORS = ("red", "blue", "yellow")


def resolve_video_backend(requested_backend: str | None) -> str:
    if requested_backend is not None:
        return requested_backend

    try:
        import torchcodec  # noqa: F401

        return "torchcodec"
    except Exception:
        return "pyav"


def extract_task_counts(dataset_root: Path) -> Counter:
    episodes_dir = dataset_root / "meta" / "episodes"
    parquet_files = sorted(episodes_dir.glob("*/*.parquet"))
    counts: Counter = Counter()
    if not parquet_files:
        return counts

    frames = [pd.read_parquet(path, columns=["tasks"]) for path in parquet_files]
    episodes = pd.concat(frames, ignore_index=True)
    for tasks in episodes["tasks"]:
        if isinstance(tasks, list) and tasks:
            task = tasks[0]
            task_lower = task.lower()
            for color in EXPECTED_TASK_COLORS:
                if f"{color} cube" in task_lower:
                    counts[color] += 1
                    break
    return counts


def normalize_dataset_root(dataset_root: Path | None) -> Path | None:
    if dataset_root is None:
        return None

    resolved = dataset_root.expanduser()
    if str(resolved) == ".":
        raise ValueError(
            "--dataset-root resolved to the current directory. "
            "This usually means REAL_DATASET_ROOT was empty. "
            "Set REAL_DATASET_ROOT to the local Hugging Face dataset snapshot path."
        )

    resolved = resolved.resolve()
    info_path = resolved / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(
            f"--dataset-root does not point to a LeRobot dataset root: {resolved}\n"
            f"Missing: {info_path}\n"
            "Download the dataset locally first and pass the snapshot path."
        )
    return resolved


def validate_training_dataset_root(dataset_root: Path | None) -> None:
    if dataset_root is None:
        return

    data_dir = dataset_root / "data"
    video_dir = dataset_root / "videos"
    has_data = data_dir.is_dir() and any(data_dir.glob("*/*.parquet"))
    has_videos = video_dir.is_dir()
    if not has_data:
        raise FileNotFoundError(
            f"Training requires parquet data files under: {data_dir}\n"
            "The current dataset root appears to be a metadata-only snapshot. "
            "Download the full dataset snapshot before starting training."
        )
    if not has_videos:
        raise FileNotFoundError(
            f"Training requires videos under: {video_dir}\n"
            "Download the full dataset snapshot before starting training."
        )


def inspect_dataset(dataset_repo_id: str, dataset_root: Path | None) -> tuple[dict, dict]:
    import lerobot.policies  # noqa: F401
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
    from lerobot.datasets.utils import dataset_to_policy_features

    meta = LeRobotDatasetMetadata(dataset_repo_id, root=dataset_root)
    dataset_features = meta.info["features"]
    policy_features = dataset_to_policy_features(dataset_features)

    pretrained_cfg = PreTrainedConfig.from_pretrained(
        "lerobot/smolvla_base",
        cli_overrides=[
            "--input_features=null",
            "--output_features=null",
            "--load_vlm_weights=true",
        ],
    )

    final_output_features = {
        key: {"type": str(ft.type), "shape": ft.shape}
        for key, ft in policy_features.items()
        if str(ft.type).endswith("ACTION")
    }
    final_input_features = {
        key: {"type": str(ft.type), "shape": ft.shape}
        for key, ft in policy_features.items()
        if key not in final_output_features
    }

    tasks_path = meta.root / "meta" / "tasks.parquet"
    tasks = pd.read_parquet(tasks_path)
    task_counts = extract_task_counts(meta.root)

    report = {
        "dataset": {
            "repo_id": dataset_repo_id,
            "root": str(meta.root),
            "codebase_version": meta.info.get("codebase_version"),
            "robot_type": meta.info.get("robot_type"),
            "fps": meta.info.get("fps"),
            "total_episodes": meta.info.get("total_episodes"),
            "total_frames": meta.info.get("total_frames"),
            "total_tasks": meta.info.get("total_tasks"),
        },
        "dataset_features": dataset_features,
        "pretrained_policy": "lerobot/smolvla_base",
        "pretrained_source_features": {
            "input_features": {
                key: {"type": str(ft.type), "shape": ft.shape}
                for key, ft in (pretrained_cfg.input_features or {}).items()
            },
            "output_features": {
                key: {"type": str(ft.type), "shape": ft.shape}
                for key, ft in (pretrained_cfg.output_features or {}).items()
            },
        },
        "final_policy_input_features": final_input_features,
        "final_policy_output_features": final_output_features,
        "load_vlm_weights": getattr(pretrained_cfg, "load_vlm_weights", None),
        "tasks": {
            "strings": tasks.index.tolist(),
            "episode_counts": dict(task_counts),
        },
    }
    return report, {
        "meta_root": str(meta.root),
        "camera_keys": [k for k in dataset_features if k.startswith("observation.images.")],
        "state_shape": dataset_features["observation.state"]["shape"],
        "action_shape": dataset_features["action"]["shape"],
    }


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-repo-id", default=DEFAULT_DATASET_REPO_ID)
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--policy-path", default="lerobot/smolvla_base")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--job-name", default="smolvla_openarm_pickcube_3colors_pretrained")
    parser.add_argument("--steps", type=int, default=100_000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--save-freq", type=int, default=20_000)
    parser.add_argument("--log-freq", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--video-backend", choices=["torchcodec", "pyav"], default=None)
    parser.add_argument("--use-amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_known_args()


def main() -> None:
    args, extra = parse_args()
    args.dataset_root = normalize_dataset_root(args.dataset_root)
    report, summary = inspect_dataset(args.dataset_repo_id, args.dataset_root)
    print("[DATASET]", flush=True)
    print(pformat(report["dataset"]), flush=True)
    print("[DATASET FEATURES]", flush=True)
    print(pformat(report["dataset_features"]), flush=True)
    print(f"[PRETRAINED POLICY] {report['pretrained_policy']}", flush=True)
    print("[PRETRAINED / SOURCE FEATURES]", flush=True)
    print(pformat(report["pretrained_source_features"]), flush=True)
    print("[FINAL POLICY INPUT FEATURES]", flush=True)
    print(pformat(report["final_policy_input_features"]), flush=True)
    print("[FINAL POLICY OUTPUT FEATURES]", flush=True)
    print(pformat(report["final_policy_output_features"]), flush=True)
    print(f"[VLM WEIGHTS LOADED] {report['load_vlm_weights']}", flush=True)
    print("[TASKS]", flush=True)
    print(pformat(report["tasks"]), flush=True)
    print("[SUMMARY]", flush=True)
    print(
        pformat(
            {
                "dataset_repo_id": args.dataset_repo_id,
                "dataset_root": report["dataset"]["root"],
                "camera_keys": summary["camera_keys"],
                "state_shape": summary["state_shape"],
                "action_shape": summary["action_shape"],
                "batch_size": args.batch_size,
                "steps": args.steps,
                "seed": args.seed,
            }
        ),
        flush=True,
    )

    if args.validate_only:
        return

    validate_training_dataset_root(args.dataset_root)

    video_backend = resolve_video_backend(args.video_backend)
    train_exe = shutil.which("lerobot-train")
    if train_exe is None:
        candidate = Path(sys.executable).resolve().parent / "lerobot-train"
        train_exe = str(candidate) if candidate.is_file() else "lerobot-train"

    command = [
        train_exe,
        f"--policy.path={args.policy_path}",
        "--policy.input_features=null",
        "--policy.output_features=null",
        "--policy.load_vlm_weights=true",
        f"--dataset.repo_id={args.dataset_repo_id}",
        f"--dataset.video_backend={video_backend}",
        f"--batch_size={args.batch_size}",
        f"--num_workers={args.num_workers}",
        f"--steps={args.steps}",
        f"--save_freq={args.save_freq}",
        f"--log_freq={args.log_freq}",
        f"--seed={args.seed}",
        f"--output_dir={args.output_dir.resolve()}",
        f"--job_name={args.job_name}",
        f"--policy.device={args.device}",
        f"--policy.use_amp={str(args.use_amp).lower()}",
        "--policy.push_to_hub=false",
        f"--wandb.enable={str(args.wandb).lower()}",
        *extra,
    ]
    if args.dataset_root is not None:
        command.insert(6, f"--dataset.root={args.dataset_root.resolve()}")

    print("[TRAIN]", " ".join(command), flush=True)
    os.execvp(command[0], command)


if __name__ == "__main__":
    main()
