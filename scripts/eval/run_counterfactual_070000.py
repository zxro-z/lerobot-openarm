#!/usr/bin/env python3
"""Run the fixed 70K counterfactual paired validation sweep."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PYTHON = Path("/home/zxro/miniforge3/envs/lab-isaac5-py311/bin/python")
WRAPPER = PROJECT_ROOT / "scripts" / "eval" / "run_counterfactual_baseline_020000.py"


def main() -> None:
    command = [
        str(PYTHON),
        str(WRAPPER),
        "--checkpoint",
        str(
            PROJECT_ROOT
            / "outputs/train/openarm_three_color_fixed_slots_cf_100k_run1b/checkpoints/070000/pretrained_model"
        ),
        "--output-dir",
        str(PROJECT_ROOT / "outputs/eval/counterfactual_070000"),
        "--csv-name",
        "paired_validation_070000.csv",
        "--json-name",
        "paired_validation_070000_summary.json",
        "--device",
        "cpu",
        "--hf-home",
        "/home/zxro/.cache/hf_lerobot",
        "--hub-cache",
        "/home/zxro/.cache/hf_lerobot/hub",
        "--datasets-cache",
        "/tmp/hf_lerobot/datasets",
        "--offline",
    ]
    subprocess.run(command, check=True)


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        raise SystemExit(exc.returncode) from exc
    except KeyboardInterrupt:
        raise SystemExit(130)
