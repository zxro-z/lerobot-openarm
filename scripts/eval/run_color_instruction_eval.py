#!/usr/bin/env python3
"""Thin wrapper around the three-color SmolVLA eval runner."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EVAL_SCRIPT = PROJECT_ROOT / "scripts" / "eval" / "eval_smolvla.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-script", type=Path, default=DEFAULT_EVAL_SCRIPT)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--policy-path", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--dataset-repo-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-episodes-per-color", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--use-amp", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--record-video",
        "--save-video",
        dest="record_video",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--video-dir", type=Path, default=Path("outputs/eval/openarm_three_color_smolvla/videos"))
    parser.add_argument("--video-camera", choices=["top", "wrist"], default="top")
    parser.add_argument("--video-fps", type=int, default=None)
    parser.add_argument("--success-video-tail-seconds", type=float, default=0.0)
    parser.add_argument("--instruction-order", choices=["grouped", "cycle"], default="grouped")
    parser.add_argument("--min-steps-before-success", type=int, default=50)
    parser.add_argument("--debug-success", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--reseed-per-episode", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        args.python,
        str(args.eval_script),
        "--policy-path",
        str(args.policy_path),
        "--dataset-root",
        str(args.dataset_root),
        "--dataset-repo-id",
        args.dataset_repo_id,
        "--output",
        str(args.output),
        "--num-episodes-per-color",
        str(args.num_episodes_per_color),
        "--max-steps",
        str(args.max_steps),
        "--seed",
        str(args.seed),
        "--device",
        args.device,
        "--instruction-order",
        args.instruction_order,
        "--min-steps-before-success",
        str(args.min_steps_before_success),
    ]
    if args.video_fps is not None:
        command.extend(["--video-fps", str(args.video_fps)])
    if args.success_video_tail_seconds != 0.0:
        command.extend(["--success-video-tail-seconds", str(args.success_video_tail_seconds)])
    if args.use_amp:
        command.append("--use-amp")
    if args.headless:
        command.append("--headless")
    if args.debug_success:
        command.append("--debug-success")
    if args.reseed_per_episode:
        command.append("--reseed-per-episode")
    if args.record_video:
        command.extend(
            ["--record-video", "--video-dir", str(args.video_dir), "--video-camera", args.video_camera]
        )
    print("[RUN]", " ".join(command), flush=True)
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
