#!/usr/bin/env python3
"""Run paired counterfactual baseline sweeps and export CSV/JSON summaries."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EVAL_SCRIPT = PROJECT_ROOT / "scripts" / "eval" / "eval_smolvla_counterfactual_triplets.py"
DEFAULT_CHECKPOINT = (
    PROJECT_ROOT
    / "outputs/train/openarm_three_color_fixed_slots_perm_tilt50_r3_run3_early_targeted"
    / "checkpoints/020000/pretrained_model"
)
DEFAULT_DATASET_ROOT = (
    PROJECT_ROOT / "src/lerobot/datasets/openarm_three_color_fixed_slots_perm_tilt50_r3"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs/eval/counterfactual_baseline_020000"
DEFAULT_DATASET_REPO_ID = "local/openarm_three_color_fixed_slots_perm_tilt50_r3"
DEFAULT_FRAMES = (36, 50, 80, 100)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-script", type=Path, default=DEFAULT_EVAL_SCRIPT)
    parser.add_argument(
        "--python",
        default="/home/zxro/miniforge3/envs/lab-isaac5-py311/bin/python",
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--dataset-repo-id", default=DEFAULT_DATASET_REPO_ID)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--csv-name", default="baseline_paired_validation.csv")
    parser.add_argument("--json-name", default="baseline_paired_validation_summary.json")
    parser.add_argument("--num-groups", type=int, default=16)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--frames", nargs="+", type=int, default=list(DEFAULT_FRAMES))
    parser.add_argument("--hf-home", default="/tmp/hf_home")
    parser.add_argument("--hub-cache", default="/tmp/hf_home/hub")
    parser.add_argument("--datasets-cache", default="/tmp/hf_home/datasets")
    parser.add_argument("--pythonpath", default=str(PROJECT_ROOT / "src"))
    parser.add_argument("--offline", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def build_env(args: argparse.Namespace) -> dict[str, str]:
    env = os.environ.copy()
    env["HF_HOME"] = args.hf_home
    env["HUGGINGFACE_HUB_CACHE"] = args.hub_cache
    env["HF_DATASETS_CACHE"] = args.datasets_cache
    env["PYTHONPATH"] = args.pythonpath
    if args.offline:
        env["HF_HUB_OFFLINE"] = "1"
        env["TRANSFORMERS_OFFLINE"] = "1"
    return env


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / args.csv_name
    json_path = output_dir / args.json_name
    env = build_env(args)

    rows: list[dict[str, float | int]] = []
    frame_summaries: list[dict[str, object]] = []

    for frame in args.frames:
        output_json = output_dir / f"frame_{frame:03d}.json"
        command = [
            args.python,
            str(args.eval_script.expanduser().resolve()),
            "--checkpoint",
            str(args.checkpoint.expanduser().resolve()),
            "--dataset-root",
            str(args.dataset_root.expanduser().resolve()),
            "--dataset-repo-id",
            args.dataset_repo_id,
            "--num-groups",
            str(args.num_groups),
            "--local-frame",
            str(frame),
            "--device",
            args.device,
            "--output-json",
            str(output_json),
        ]

        print("[RUN]", " ".join(command), flush=True)
        try:
            subprocess.run(command, env=env, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as exc:
            if exc.stdout:
                print(exc.stdout, end="" if exc.stdout.endswith("\n") else "\n")
            if exc.stderr:
                print(exc.stderr, end="" if exc.stderr.endswith("\n") else "\n", file=sys.stderr)
            raise
        payload = json.loads(output_json.read_text())
        frame_summaries.append(payload)

        for group in payload.get("groups", []):
            gt_pairwise = group["gt_pairwise"]
            pred_pairwise = group["pred_pairwise"]
            rows.append(
                {
                    "local_frame": frame,
                    "triplet_id": group["triplet_id"],
                    "gt_red_blue": gt_pairwise["red_vs_blue"],
                    "gt_red_yellow": gt_pairwise["red_vs_yellow"],
                    "gt_blue_yellow": gt_pairwise["blue_vs_yellow"],
                    "pred_red_blue": pred_pairwise["red_vs_blue"],
                    "pred_red_yellow": pred_pairwise["red_vs_yellow"],
                    "pred_blue_yellow": pred_pairwise["blue_vs_yellow"],
                    "gt_mean": group["gt_separation"],
                    "pred_mean": group["prediction_separation"],
                    "pred_gt_ratio": group["ratio"],
                }
            )

    with csv_path.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "local_frame",
                "triplet_id",
                "gt_red_blue",
                "gt_red_yellow",
                "gt_blue_yellow",
                "pred_red_blue",
                "pred_red_yellow",
                "pred_blue_yellow",
                "gt_mean",
                "pred_mean",
                "pred_gt_ratio",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "checkpoint": str(args.checkpoint.expanduser().resolve()),
        "dataset_root": str(args.dataset_root.expanduser().resolve()),
        "dataset_repo_id": args.dataset_repo_id,
        "evaluated_steps": args.frames,
        "num_groups_per_step": args.num_groups,
        "num_rows": len(rows),
        "mean_gt_pairwise_distance": mean([float(row["gt_mean"]) for row in rows]),
        "mean_pred_pairwise_distance": mean([float(row["pred_mean"]) for row in rows]),
        "mean_pred_gt_ratio": mean([float(row["pred_gt_ratio"]) for row in rows]),
        "pair_means": {
            "red_blue": {
                "gt": mean([float(row["gt_red_blue"]) for row in rows]),
                "pred": mean([float(row["pred_red_blue"]) for row in rows]),
                "pred_gt": mean(
                    [
                        float(row["pred_red_blue"]) / float(row["gt_red_blue"])
                        for row in rows
                        if float(row["gt_red_blue"]) != 0.0
                    ]
                ),
            },
            "red_yellow": {
                "gt": mean([float(row["gt_red_yellow"]) for row in rows]),
                "pred": mean([float(row["pred_red_yellow"]) for row in rows]),
                "pred_gt": mean(
                    [
                        float(row["pred_red_yellow"]) / float(row["gt_red_yellow"])
                        for row in rows
                        if float(row["gt_red_yellow"]) != 0.0
                    ]
                ),
            },
            "blue_yellow": {
                "gt": mean([float(row["gt_blue_yellow"]) for row in rows]),
                "pred": mean([float(row["pred_blue_yellow"]) for row in rows]),
                "pred_gt": mean(
                    [
                        float(row["pred_blue_yellow"]) / float(row["gt_blue_yellow"])
                        for row in rows
                        if float(row["gt_blue_yellow"]) != 0.0
                    ]
                ),
            },
        },
        "per_frame": [
            {
                "local_frame": payload["local_frame"],
                "num_groups": payload["num_groups"],
                "gt_separation": payload["gt_separation"],
                "prediction_separation": payload["prediction_separation"],
                "prediction_gt_ratio": payload["prediction_gt_ratio"],
            }
            for payload in frame_summaries
        ],
        "source_jsons": [str(output_dir / f"frame_{frame:03d}.json") for frame in args.frames],
    }

    json_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"\nCSV saved to: {csv_path}")
    print(f"JSON saved to: {json_path}")


if __name__ == "__main__":
    main()
