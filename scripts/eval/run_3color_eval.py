#!/usr/bin/env python3
"""Run a 3-color evaluation sweep for red/blue/yellow instructions."""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EVAL_SCRIPT = PROJECT_ROOT / "scripts" / "eval" / "eval_smolvla_3color.py"
COLOR_NAMES = ("red", "blue", "yellow")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-script", type=Path, default=DEFAULT_EVAL_SCRIPT)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--policy-path", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument(
        "--dataset-repo-id",
        default="a126-kitech/openarm_pickcube_3colors_no_ep10_12",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episodes-per-color", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--use-amp", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--save-video", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--video-fps", type=int, default=30)
    parser.add_argument("--cube-layout", choices=["fixed_slots"], default="fixed_slots")
    parser.add_argument("--cube-jitter", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    combined_rows: list[dict[str, str]] = []

    for idx, color in enumerate(COLOR_NAMES):
        run_dir = args.output_dir / f"target_{color}"
        video_dir = run_dir / "videos"
        csv_path = run_dir / "results.csv"
        command = [
            args.python,
            str(args.eval_script),
            "--policy-path",
            str(args.policy_path),
            "--dataset-repo-id",
            args.dataset_repo_id,
            "--num-episodes",
            str(args.episodes_per_color),
            "--max-steps",
            str(args.max_steps),
            "--seed",
            str(args.seed + idx * args.episodes_per_color),
            "--device",
            args.device,
            "--video-fps",
            str(args.video_fps),
            "--target-color",
            color,
            "--cube-layout",
            args.cube_layout,
            "--cube-jitter",
            str(args.cube_jitter),
            "--output",
            str(csv_path),
        ]
        if args.dataset_root is not None:
            command.extend(["--dataset-root", str(args.dataset_root)])
        if args.use_amp:
            command.append("--use-amp")
        if args.save_video:
            command.extend(["--save-video", "--video-dir", str(video_dir)])

        print("[RUN]", " ".join(command), flush=True)
        subprocess.run(command, check=True)

        with csv_path.open(newline="") as file:
            combined_rows.extend(csv.DictReader(file))

    combined_path = args.output_dir / "combined_results.csv"
    if combined_rows:
        with combined_path.open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(combined_rows[0].keys()))
            writer.writeheader()
            writer.writerows(combined_rows)
        print(f"[RESULT] combined_csv={combined_path}", flush=True)


if __name__ == "__main__":
    main()
