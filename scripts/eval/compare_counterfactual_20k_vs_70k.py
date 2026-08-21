#!/usr/bin/env python3
"""Run fixed counterfactual validation for 20K and 70K, then compare row identity and predictions."""

from __future__ import annotations

import csv
import json
import math
import os
import statistics
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PYTHON = Path("/home/zxro/miniforge3/envs/lab-isaac5-py311/bin/python")
RUN_WRAPPER = PROJECT_ROOT / "scripts" / "eval" / "run_counterfactual_baseline_020000.py"
DATASET_ROOT = PROJECT_ROOT / "src/lerobot/datasets/openarm_three_color_fixed_slots_perm_tilt50_r3"
DATASET_REPO_ID = "local/openarm_three_color_fixed_slots_perm_tilt50_r3"
MANIFEST_PATH = DATASET_ROOT / "triplet_manifest.csv"
CHECKPOINT_20K = (
    PROJECT_ROOT
    / "outputs/train/openarm_three_color_fixed_slots_perm_tilt50_r3_run3_early_targeted/checkpoints/020000/pretrained_model"
)
CHECKPOINT_70K = (
    PROJECT_ROOT
    / "outputs/train/openarm_three_color_fixed_slots_cf_100k_run1b/checkpoints/070000/pretrained_model"
)
OUTPUT_20K = PROJECT_ROOT / "outputs/eval/counterfactual_compare_20k"
OUTPUT_70K = PROJECT_ROOT / "outputs/eval/counterfactual_compare_70k"
OUTPUT_COMPARE = PROJECT_ROOT / "outputs/eval/counterfactual_20k_vs_70k"
FRAMES = (36, 50, 80, 100)
NUM_GROUPS = 16
GT_NEAR_ZERO_THRESHOLD = 1e-3
GT_TOLERANCE = 1e-8
PAIR_COLUMNS = (
    ("red_blue", "gt_red_blue", "pred_red_blue"),
    ("red_yellow", "gt_red_yellow", "pred_red_yellow"),
    ("blue_yellow", "gt_blue_yellow", "pred_blue_yellow"),
)


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def to_float(row: dict[str, str], key: str) -> float:
    return float(row[key])


def load_triplet_metadata() -> dict[int, dict[str, int]]:
    triplets: dict[int, dict[str, int]] = {}
    with MANIFEST_PATH.open(newline="") as file:
        reader = csv.DictReader(file)
        for triplet_id, row in enumerate(reader):
            triplets[triplet_id] = {
                "layout_id": int(row["layout_id"]),
                "permutation_id": int(row["permutation_id"]),
                "repeat_index": int(row["repeat_index"]),
            }
    return triplets


def build_env() -> dict[str, str]:
    env = os.environ.copy()
    env["HF_HOME"] = "/home/zxro/.cache/hf_lerobot"
    env["HUGGINGFACE_HUB_CACHE"] = "/home/zxro/.cache/hf_lerobot/hub"
    env["HF_DATASETS_CACHE"] = "/tmp/hf_lerobot/datasets"
    env["PYTHONPATH"] = str(PROJECT_ROOT / "src")
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    return env


def run_validation(checkpoint: Path, output_dir: Path, csv_name: str, json_name: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        str(PYTHON),
        str(RUN_WRAPPER),
        "--checkpoint",
        str(checkpoint),
        "--dataset-root",
        str(DATASET_ROOT),
        "--dataset-repo-id",
        DATASET_REPO_ID,
        "--output-dir",
        str(output_dir),
        "--csv-name",
        csv_name,
        "--json-name",
        json_name,
        "--num-groups",
        str(NUM_GROUPS),
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
    print("[RUN]", " ".join(command), flush=True)
    subprocess.run(command, check=True, env=build_env())


def enrich_validation_csv(input_csv: Path, output_csv: Path, triplet_meta: dict[int, dict[str, int]]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with input_csv.open(newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            triplet_id = int(row["triplet_id"])
            meta = triplet_meta[triplet_id]
            enriched = dict(row)
            enriched["layout_id"] = str(meta["layout_id"])
            enriched["permutation_id"] = str(meta["permutation_id"])
            enriched["repeat_index"] = str(meta["repeat_index"])
            rows.append(enriched)

    rows.sort(
        key=lambda row: (
            int(row["local_frame"]),
            int(row["layout_id"]),
            int(row["permutation_id"]),
            int(row["repeat_index"]),
            int(row["triplet_id"]),
        )
    )
    with output_csv.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "local_frame",
                "triplet_id",
                "layout_id",
                "permutation_id",
                "repeat_index",
                "gt_red_blue",
                "gt_red_yellow",
                "gt_blue_yellow",
                "gt_mean",
                "pred_red_blue",
                "pred_red_yellow",
                "pred_blue_yellow",
                "pred_mean",
                "pred_gt_ratio",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    return rows


def make_key(row: dict[str, str]) -> tuple[int, int]:
    return int(row["local_frame"]), int(row["triplet_id"])


def compare_gt_rows(rows_20k: list[dict[str, str]], rows_70k: list[dict[str, str]]) -> tuple[list[dict[str, object]], dict[str, object]]:
    by_key_20k = {make_key(row): row for row in rows_20k}
    by_key_70k = {make_key(row): row for row in rows_70k}
    keys_20k = set(by_key_20k)
    keys_70k = set(by_key_70k)
    matched_keys = sorted(keys_20k & keys_70k)
    missing_in_20k = sorted(keys_70k - keys_20k)
    missing_in_70k = sorted(keys_20k - keys_70k)

    comparison_rows: list[dict[str, object]] = []
    gt_mismatch_rows = 0
    max_gt_abs_diff = 0.0
    for key in matched_keys:
        row_20k = by_key_20k[key]
        row_70k = by_key_70k[key]
        gt_diffs = {
            "gt_red_blue_abs_diff": abs(to_float(row_20k, "gt_red_blue") - to_float(row_70k, "gt_red_blue")),
            "gt_red_yellow_abs_diff": abs(to_float(row_20k, "gt_red_yellow") - to_float(row_70k, "gt_red_yellow")),
            "gt_blue_yellow_abs_diff": abs(to_float(row_20k, "gt_blue_yellow") - to_float(row_70k, "gt_blue_yellow")),
            "gt_mean_abs_diff": abs(to_float(row_20k, "gt_mean") - to_float(row_70k, "gt_mean")),
        }
        row_max_diff = max(gt_diffs.values())
        max_gt_abs_diff = max(max_gt_abs_diff, row_max_diff)
        same_meta = (
            row_20k["layout_id"] == row_70k["layout_id"]
            and row_20k["permutation_id"] == row_70k["permutation_id"]
            and row_20k["repeat_index"] == row_70k["repeat_index"]
        )
        gt_match = row_max_diff <= GT_TOLERANCE and same_meta
        if not gt_match:
            gt_mismatch_rows += 1
        comparison_rows.append(
            {
                "local_frame": key[0],
                "triplet_id": key[1],
                "layout_id_20k": row_20k["layout_id"],
                "layout_id_70k": row_70k["layout_id"],
                "permutation_id_20k": row_20k["permutation_id"],
                "permutation_id_70k": row_70k["permutation_id"],
                "repeat_index_20k": row_20k["repeat_index"],
                "repeat_index_70k": row_70k["repeat_index"],
                "gt_red_blue_20k": row_20k["gt_red_blue"],
                "gt_red_blue_70k": row_70k["gt_red_blue"],
                "gt_red_blue_abs_diff": gt_diffs["gt_red_blue_abs_diff"],
                "gt_red_yellow_20k": row_20k["gt_red_yellow"],
                "gt_red_yellow_70k": row_70k["gt_red_yellow"],
                "gt_red_yellow_abs_diff": gt_diffs["gt_red_yellow_abs_diff"],
                "gt_blue_yellow_20k": row_20k["gt_blue_yellow"],
                "gt_blue_yellow_70k": row_70k["gt_blue_yellow"],
                "gt_blue_yellow_abs_diff": gt_diffs["gt_blue_yellow_abs_diff"],
                "gt_mean_20k": row_20k["gt_mean"],
                "gt_mean_70k": row_70k["gt_mean"],
                "gt_mean_abs_diff": gt_diffs["gt_mean_abs_diff"],
                "gt_match": str(gt_match),
            }
        )

    summary = {
        "rows_20k": len(rows_20k),
        "rows_70k": len(rows_70k),
        "matched_rows": len(matched_keys),
        "missing_in_20k": len(missing_in_20k),
        "missing_in_70k": len(missing_in_70k),
        "gt_mismatch_rows": gt_mismatch_rows,
        "max_gt_abs_diff": max_gt_abs_diff,
        "missing_keys_in_20k": missing_in_20k,
        "missing_keys_in_70k": missing_in_70k,
    }
    return comparison_rows, summary


def ratio_for_row(pred_mean: float, gt_mean: float) -> float | None:
    if gt_mean <= GT_NEAR_ZERO_THRESHOLD:
        return None
    return pred_mean / gt_mean


def pair_ratio_for_row(pred: float, gt: float) -> float | None:
    if gt <= GT_NEAR_ZERO_THRESHOLD:
        return None
    return pred / gt


def build_prediction_comparison(rows_20k: list[dict[str, str]], rows_70k: list[dict[str, str]]) -> tuple[list[dict[str, object]], dict[str, object]]:
    by_key_20k = {make_key(row): row for row in rows_20k}
    by_key_70k = {make_key(row): row for row in rows_70k}
    matched_keys = sorted(set(by_key_20k) & set(by_key_70k))
    rows: list[dict[str, object]] = []
    for key in matched_keys:
        row_20k = by_key_20k[key]
        row_70k = by_key_70k[key]
        gt_mean = to_float(row_20k, "gt_mean")
        pred_mean_20k = to_float(row_20k, "pred_mean")
        pred_mean_70k = to_float(row_70k, "pred_mean")
        region = "GT_NEAR_ZERO" if gt_mean <= GT_NEAR_ZERO_THRESHOLD else "GT_TARGET_DEPENDENT"
        out: dict[str, object] = {
            "local_frame": key[0],
            "triplet_id": key[1],
            "layout_id": row_20k["layout_id"],
            "permutation_id": row_20k["permutation_id"],
            "repeat_index": row_20k["repeat_index"],
            "gt_mean": gt_mean,
            "pred_mean_20k": pred_mean_20k,
            "pred_mean_70k": pred_mean_70k,
            "delta_pred_mean": pred_mean_70k - pred_mean_20k,
            "region": region,
            "ratio_20k": ratio_for_row(pred_mean_20k, gt_mean),
            "ratio_70k": ratio_for_row(pred_mean_70k, gt_mean),
        }
        for pair_name, gt_key, pred_key in PAIR_COLUMNS:
            pred_20k = to_float(row_20k, pred_key)
            pred_70k = to_float(row_70k, pred_key)
            gt_val = to_float(row_20k, gt_key)
            out[f"gt_{pair_name}"] = gt_val
            out[f"pred_{pair_name}_20k"] = pred_20k
            out[f"pred_{pair_name}_70k"] = pred_70k
            out[f"delta_pred_{pair_name}"] = pred_70k - pred_20k
            out[f"ratio_{pair_name}_20k"] = pair_ratio_for_row(pred_20k, gt_val)
            out[f"ratio_{pair_name}_70k"] = pair_ratio_for_row(pred_70k, gt_val)
        rows.append(out)

    frame_summary: list[dict[str, object]] = []
    for frame in FRAMES:
        frame_rows = [row for row in rows if int(row["local_frame"]) == frame]
        target_rows = [row for row in frame_rows if row["region"] == "GT_TARGET_DEPENDENT"]
        frame_summary.append(
            {
                "frame": frame,
                "gt_mean": mean([float(row["gt_mean"]) for row in frame_rows]),
                "pred_20k": mean([float(row["pred_mean_20k"]) for row in frame_rows]),
                "pred_70k": mean([float(row["pred_mean_70k"]) for row in frame_rows]),
                "delta_pred": mean([float(row["delta_pred_mean"]) for row in frame_rows]),
                "ratio_20k": mean([float(row["ratio_20k"]) for row in target_rows if row["ratio_20k"] is not None]),
                "ratio_70k": mean([float(row["ratio_70k"]) for row in target_rows if row["ratio_70k"] is not None]),
                "target_row_count": len(target_rows),
            }
        )

    near_zero_rows = [row for row in rows if row["region"] == "GT_NEAR_ZERO"]
    target_rows = [row for row in rows if row["region"] == "GT_TARGET_DEPENDENT"]

    pairwise_summary: list[dict[str, object]] = []
    for pair_name, gt_key, _pred_key in PAIR_COLUMNS:
        pairwise_summary.append(
            {
                "pair": pair_name.replace("_", "-"),
                "gt": mean([float(row[f"gt_{pair_name}"]) for row in target_rows]),
                "pred_20k": mean([float(row[f"pred_{pair_name}_20k"]) for row in target_rows]),
                "pred_70k": mean([float(row[f"pred_{pair_name}_70k"]) for row in target_rows]),
                "pred_gt_20k": mean(
                    [float(row[f"ratio_{pair_name}_20k"]) for row in target_rows if row[f"ratio_{pair_name}_20k"] is not None]
                ),
                "pred_gt_70k": mean(
                    [float(row[f"ratio_{pair_name}_70k"]) for row in target_rows if row[f"ratio_{pair_name}_70k"] is not None]
                ),
            }
        )

    summary = {
        "gt_near_zero_threshold": GT_NEAR_ZERO_THRESHOLD,
        "frame_summary": frame_summary,
        "gt_near_zero_region": {
            "count": len(near_zero_rows),
            "pred_separation_20k_mean": mean([float(row["pred_mean_20k"]) for row in near_zero_rows]),
            "pred_separation_70k_mean": mean([float(row["pred_mean_70k"]) for row in near_zero_rows]),
            "pred_separation_20k_median": median([float(row["pred_mean_20k"]) for row in near_zero_rows]),
            "pred_separation_70k_median": median([float(row["pred_mean_70k"]) for row in near_zero_rows]),
        },
        "target_dependent_region": {
            "count": len(target_rows),
            "pred_20k_mean": mean([float(row["pred_mean_20k"]) for row in target_rows]),
            "pred_70k_mean": mean([float(row["pred_mean_70k"]) for row in target_rows]),
            "pred_20k_median": median([float(row["pred_mean_20k"]) for row in target_rows]),
            "pred_70k_median": median([float(row["pred_mean_70k"]) for row in target_rows]),
            "pred_gt_20k_mean": mean([float(row["ratio_20k"]) for row in target_rows if row["ratio_20k"] is not None]),
            "pred_gt_70k_mean": mean([float(row["ratio_70k"]) for row in target_rows if row["ratio_70k"] is not None]),
            "pred_gt_20k_median": median([float(row["ratio_20k"]) for row in target_rows if row["ratio_20k"] is not None]),
            "pred_gt_70k_median": median([float(row["ratio_70k"]) for row in target_rows if row["ratio_70k"] is not None]),
        },
        "pairwise_summary": pairwise_summary,
        "alignment": {
            "metric": None,
            "twenty_k": None,
            "seventy_k": None,
            "improved": None,
            "note": "Not computed: current validation outputs only pairwise distances, not action-vector directions.",
        },
    }
    return rows, summary


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def verdict_from_summary(prediction_summary: dict[str, object]) -> str:
    near_zero = prediction_summary["gt_near_zero_region"]
    target = prediction_summary["target_dependent_region"]
    pred_20k_near = near_zero["pred_separation_20k_mean"]
    pred_70k_near = near_zero["pred_separation_70k_mean"]
    pred_20k_target = target["pred_20k_mean"]
    pred_70k_target = target["pred_70k_mean"]
    ratio_20k = target["pred_gt_20k_mean"]
    ratio_70k = target["pred_gt_70k_mean"]
    if None in (pred_20k_near, pred_70k_near, pred_20k_target, pred_70k_target, ratio_20k, ratio_70k):
        return "NO MEANINGFUL CHANGE"
    near_zero_increase = float(pred_70k_near) - float(pred_20k_near)
    target_increase = float(pred_70k_target) - float(pred_20k_target)
    ratio_increase = float(ratio_70k) - float(ratio_20k)
    if near_zero_increase > 0.01 and target_increase <= 0.0:
        return "REGRESSED"
    if target_increase > 0.0 and ratio_increase > 0.0 and near_zero_increase <= 0.005:
        return "IMPROVED"
    if target_increase > 0.0 or ratio_increase > 0.0:
        return "PARTIALLY IMPROVED"
    if abs(target_increase) < 1e-4 and abs(ratio_increase) < 1e-4:
        return "NO MEANINGFUL CHANGE"
    return "REGRESSED"


def main() -> None:
    OUTPUT_COMPARE.mkdir(parents=True, exist_ok=True)
    Path("/tmp/hf_lerobot/datasets").mkdir(parents=True, exist_ok=True)

    run_validation(CHECKPOINT_20K, OUTPUT_20K, "baseline_paired_validation.csv", "baseline_paired_validation_summary.json")
    run_validation(CHECKPOINT_70K, OUTPUT_70K, "paired_validation_070000.csv", "paired_validation_070000_summary.json")

    triplet_meta = load_triplet_metadata()
    rows_20k = enrich_validation_csv(
        OUTPUT_20K / "baseline_paired_validation.csv",
        OUTPUT_COMPARE / "20k_current_validation.csv",
        triplet_meta,
    )
    rows_70k = enrich_validation_csv(
        OUTPUT_70K / "paired_validation_070000.csv",
        OUTPUT_COMPARE / "70k_current_validation.csv",
        triplet_meta,
    )

    gt_identity_rows, gt_identity_summary = compare_gt_rows(rows_20k, rows_70k)
    write_csv(OUTPUT_COMPARE / "gt_identity_check.csv", gt_identity_rows)

    if gt_identity_summary["gt_mismatch_rows"] != 0 or gt_identity_summary["missing_in_20k"] != 0 or gt_identity_summary["missing_in_70k"] != 0:
        summary = {
            "protocol": {
                "same_code": True,
                "same_dataset": True,
                "same_frames": list(FRAMES),
                "same_groups": NUM_GROUPS,
                "same_shared_noise": True,
                "checkpoint_only_difference": True,
            },
            "gt_identity_check": gt_identity_summary,
            "gt_verdict": "FAIL — GT mismatch, prediction comparison invalid",
            "verdict": "INVALID — GT MISMATCH",
        }
        (OUTPUT_COMPARE / "20k_vs_70k_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary, indent=2))
        return

    prediction_rows, prediction_summary = build_prediction_comparison(rows_20k, rows_70k)
    write_csv(OUTPUT_COMPARE / "20k_vs_70k_prediction_comparison.csv", prediction_rows)

    summary = {
        "protocol": {
            "same_code": True,
            "same_dataset": True,
            "same_frames": list(FRAMES),
            "same_groups": NUM_GROUPS,
            "same_shared_noise": True,
            "checkpoint_only_difference": True,
            "device": "cpu",
        },
        "gt_identity_check": gt_identity_summary,
        "gt_verdict": "PASS — GT rows are identical",
        "prediction_summary": prediction_summary,
        "verdict": verdict_from_summary(prediction_summary),
    }
    (OUTPUT_COMPARE / "20k_vs_70k_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        raise SystemExit(exc.returncode) from exc
    except KeyboardInterrupt:
        raise SystemExit(130)
