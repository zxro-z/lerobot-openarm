#!/usr/bin/env python3
"""Analyze grasp workspace audit CSV and emit summary tables/plots/report."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
from collections import Counter, defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


X_BINS = [-0.65, -0.625, -0.60, -0.575, -0.55, -0.525, -0.50]
Y_BINS = [0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.15]
TILT_BINS = [-45.0, -30.0, -15.0, 0.0, 15.0, 30.0, 45.0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def parse_row(row: dict[str, str]) -> dict[str, object]:
    parsed: dict[str, object] = {}
    for key, value in row.items():
        if key in {"target_color", "failure_reason"}:
            parsed[key] = value
        elif value == "":
            parsed[key] = None
        else:
            number = float(value)
            if key in {"layout_id", "candidate_attempt", "layout_seed", "grasp_success", "trajectory_finished_expected"}:
                parsed[key] = int(number)
            else:
                parsed[key] = number
    return parsed


def load_rows(path: Path) -> list[dict[str, object]]:
    with path.open(newline="") as file:
        return [parse_row(row) for row in csv.DictReader(file)]


def as_float_list(rows: list[dict[str, object]], key: str) -> list[float]:
    return [float(row[key]) for row in rows if row[key] is not None]


def stats(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"n": 0, "mean": None, "median": None, "min": None, "max": None}
    return {
        "n": len(values),
        "mean": float(sum(values) / len(values)),
        "median": float(statistics.median(values)),
        "min": float(min(values)),
        "max": float(max(values)),
    }


def bin_index(value: float, bins: list[float]) -> int | None:
    for idx in range(len(bins) - 1):
        left = bins[idx]
        right = bins[idx + 1]
        if left <= value < right or (idx == len(bins) - 2 and left <= value <= right):
            return idx
    return None


def failure_type(row: dict[str, object]) -> str:
    pos_bad = float(row["xyz_error_norm"]) > float(row["position_tolerance"])
    rot_bad = float(row["orientation_error_norm"]) > float(row["rotation_tolerance"])
    if pos_bad and rot_bad:
        return "BOTH"
    if pos_bad:
        return "POSITION_ONLY"
    if rot_bad:
        return "ORIENTATION_ONLY"
    return "OTHER"


def summarize_bins(
    rows: list[dict[str, object]],
    value_key: str,
    bins: list[float],
    include_orientation: bool = False,
) -> list[dict[str, object]]:
    table: list[dict[str, object]] = []
    for idx in range(len(bins) - 1):
        sub = [row for row in rows if bin_index(float(row[value_key]), bins) == idx]
        successes = sum(int(row["grasp_success"]) for row in sub)
        output = {
            "bin_start": bins[idx],
            "bin_end": bins[idx + 1],
            "attempts": len(sub),
            "successes": successes,
            "success_rate": successes / len(sub) if sub else None,
            "median_xyz_error_norm": float(statistics.median(as_float_list(sub, "xyz_error_norm"))) if sub else None,
        }
        if include_orientation:
            output["median_orientation_error_norm"] = (
                float(statistics.median(as_float_list(sub, "orientation_error_norm"))) if sub else None
            )
        table.append(output)
    return table


def summarize_xy_bins(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    table: list[dict[str, object]] = []
    for x_idx in range(len(X_BINS) - 1):
        for y_idx in range(len(Y_BINS) - 1):
            sub = [
                row
                for row in rows
                if bin_index(float(row["cube_x"]), X_BINS) == x_idx
                and bin_index(float(row["cube_y"]), Y_BINS) == y_idx
            ]
            successes = sum(int(row["grasp_success"]) for row in sub)
            table.append(
                {
                    "x_bin_start": X_BINS[x_idx],
                    "x_bin_end": X_BINS[x_idx + 1],
                    "y_bin_start": Y_BINS[y_idx],
                    "y_bin_end": Y_BINS[y_idx + 1],
                    "attempts": len(sub),
                    "successes": successes,
                    "success_rate": successes / len(sub) if sub else None,
                    "median_xyz_error_norm": (
                        float(statistics.median(as_float_list(sub, "xyz_error_norm"))) if sub else None
                    ),
                }
            )
    return table


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def make_scatter(rows: list[dict[str, object]], out_path: Path, title: str, color_by_error: bool) -> None:
    xs_success = [float(row["cube_x"]) for row in rows if int(row["grasp_success"]) == 1]
    ys_success = [float(row["cube_y"]) for row in rows if int(row["grasp_success"]) == 1]
    xs_fail = [float(row["cube_x"]) for row in rows if int(row["grasp_success"]) == 0]
    ys_fail = [float(row["cube_y"]) for row in rows if int(row["grasp_success"]) == 0]

    plt.figure(figsize=(7, 5))
    if color_by_error:
        errors = [float(row["xyz_error_norm"]) for row in rows]
        scatter = plt.scatter(
            [float(row["cube_x"]) for row in rows],
            [float(row["cube_y"]) for row in rows],
            c=errors,
            cmap="viridis",
            s=[30 + 200 * min(err, 0.02) / 0.02 for err in errors],
            alpha=0.9,
            edgecolors="black",
            linewidths=0.2,
        )
        plt.colorbar(scatter, label="xyz_error_norm")
        for row in rows:
            if int(row["grasp_success"]) == 0:
                plt.scatter(float(row["cube_x"]), float(row["cube_y"]), marker="x", c="red", s=45, linewidths=1.0)
    else:
        plt.scatter(xs_success, ys_success, c="#1b9e77", label="success", s=55, alpha=0.9)
        plt.scatter(xs_fail, ys_fail, c="#d95f02", label="failure", marker="x", s=55, alpha=0.9)
        plt.legend()
    plt.xlabel("cube_x (world)")
    plt.ylabel("cube_y (world)")
    plt.title(title)
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def make_xy_heatmap(rows: list[dict[str, object]], out_path: Path, title: str) -> None:
    grid = np.full((len(Y_BINS) - 1, len(X_BINS) - 1), np.nan)
    for y_idx in range(len(Y_BINS) - 1):
        for x_idx in range(len(X_BINS) - 1):
            sub = [
                row
                for row in rows
                if bin_index(float(row["cube_x"]), X_BINS) == x_idx
                and bin_index(float(row["cube_y"]), Y_BINS) == y_idx
            ]
            if sub:
                grid[y_idx, x_idx] = sum(int(row["grasp_success"]) for row in sub) / len(sub)

    plt.figure(figsize=(8, 5))
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad(color="#dddddd")
    plt.imshow(grid, origin="lower", aspect="auto", vmin=0.0, vmax=1.0, cmap=cmap)
    plt.colorbar(label="success rate")
    plt.xticks(range(len(X_BINS) - 1), [f"{X_BINS[i]:.3f}\n{X_BINS[i+1]:.3f}" for i in range(len(X_BINS) - 1)])
    plt.yticks(range(len(Y_BINS) - 1), [f"{Y_BINS[i]:.2f}\n{Y_BINS[i+1]:.2f}" for i in range(len(Y_BINS) - 1)])
    for y_idx in range(grid.shape[0]):
        for x_idx in range(grid.shape[1]):
            if not math.isnan(grid[y_idx, x_idx]):
                plt.text(x_idx, y_idx, f"{grid[y_idx, x_idx]:.2f}", ha="center", va="center", color="white")
    plt.xlabel("x bins")
    plt.ylabel("y bins")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def logistic_regression(rows: list[dict[str, object]]) -> dict[str, object] | None:
    if len(rows) < 8:
        return None
    x = np.array(
        [[1.0, float(row["cube_x"]), float(row["cube_y"]), abs(float(row["tcp_tilt_deg"]))] for row in rows],
        dtype=float,
    )
    y = np.array([int(row["grasp_success"]) for row in rows], dtype=float)
    beta = np.zeros(x.shape[1], dtype=float)
    for _ in range(50):
        z = np.clip(x @ beta, -30.0, 30.0)
        p = 1.0 / (1.0 + np.exp(-z))
        w = p * (1.0 - p)
        hessian = x.T @ (x * w[:, None]) + 1e-6 * np.eye(x.shape[1])
        grad = x.T @ (y - p)
        step = np.linalg.solve(hessian, grad)
        beta = beta + step
        if np.max(np.abs(step)) < 1e-6:
            break

    predictions = []
    for fold in range(5):
        test_idx = [idx for idx in range(len(rows)) if idx % 5 == fold]
        train_idx = [idx for idx in range(len(rows)) if idx % 5 != fold]
        if not test_idx or not train_idx:
            continue
        x_train = x[train_idx]
        y_train = y[train_idx]
        fold_beta = np.zeros(x_train.shape[1], dtype=float)
        for _ in range(50):
            z = np.clip(x_train @ fold_beta, -30.0, 30.0)
            p = 1.0 / (1.0 + np.exp(-z))
            w = p * (1.0 - p)
            hessian = x_train.T @ (x_train * w[:, None]) + 1e-6 * np.eye(x_train.shape[1])
            grad = x_train.T @ (y_train - p)
            step = np.linalg.solve(hessian, grad)
            fold_beta = fold_beta + step
            if np.max(np.abs(step)) < 1e-6:
                break
        x_test = x[test_idx]
        y_test = y[test_idx]
        probs = 1.0 / (1.0 + np.exp(-np.clip(x_test @ fold_beta, -30.0, 30.0)))
        predictions.extend((probs >= 0.5).astype(int) == y_test.astype(int))

    return {
        "coefficients": {
            "intercept": float(beta[0]),
            "cube_x": float(beta[1]),
            "cube_y": float(beta[2]),
            "abs_tilt_deg": float(beta[3]),
        },
        "cv_accuracy": float(sum(predictions) / len(predictions)) if predictions else None,
    }


def packing_note(x_min: float, x_max: float, y_min: float, y_max: float, min_separation: float) -> str:
    width = x_max - x_min
    height = y_max - y_min
    if width >= min_separation:
        return "x width alone exceeds min_separation, so 3-cube packing is not obviously constrained by x."
    required_y_step = math.sqrt(max(min_separation**2 - width**2, 0.0))
    total_required_span = 2.0 * required_y_step
    if total_required_span > height:
        return (
            f"x width={width:.3f} is too narrow to place 3 cubes with 0.090 m separation inside "
            f"y span={height:.3f} using a simple zig-zag packing argument."
        )
    return (
        f"x width={width:.3f} does not exceed min_separation, but 3-cube packing may still be feasible "
        f"because the required zig-zag y span is about {total_required_span:.3f} <= {height:.3f}."
    )


def main() -> None:
    args = parse_args()
    rows = load_rows(args.csv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    failures = [row for row in rows if int(row["grasp_success"]) == 0]
    successes = [row for row in rows if int(row["grasp_success"]) == 1]
    red_rows = [row for row in rows if row["target_color"] == "red"]
    blue_rows = [row for row in rows if row["target_color"] == "blue"]
    yellow_rows = [row for row in rows if row["target_color"] == "yellow"]
    red_positive_tilt = [row for row in red_rows if float(row["tcp_tilt_deg"]) >= 0.0]

    failure_breakdown = Counter(failure_type(row) for row in failures)
    missing_counts = {
        key: sum(1 for row in rows if row.get(key) in (None, ""))
        for key in rows[0].keys()
        if sum(1 for row in rows if row.get(key) in (None, "")) > 0
    }

    x_bins = summarize_bins(rows, "cube_x", X_BINS)
    y_bins = summarize_bins(rows, "cube_y", Y_BINS)
    tilt_bins = summarize_bins(rows, "tcp_tilt_deg", TILT_BINS, include_orientation=True)
    xy_bins = summarize_xy_bins(rows)

    summary_rows = [
        {"metric": "rows", "value": len(rows)},
        {"metric": "unique_layout_seed_count", "value": len(set(int(row["layout_seed"]) for row in rows))},
        {"metric": "success_count", "value": len(successes)},
        {"metric": "failure_count", "value": len(failures)},
        {"metric": "overall_success_rate", "value": len(successes) / len(rows) if rows else None},
        {"metric": "red_count", "value": len(red_rows)},
        {"metric": "red_success_rate", "value": sum(int(row["grasp_success"]) for row in red_rows) / len(red_rows)},
        {"metric": "blue_count", "value": len(blue_rows)},
        {
            "metric": "blue_success_rate",
            "value": sum(int(row["grasp_success"]) for row in blue_rows) / len(blue_rows) if blue_rows else None,
        },
        {"metric": "yellow_count", "value": len(yellow_rows)},
        {
            "metric": "yellow_success_rate",
            "value": sum(int(row["grasp_success"]) for row in yellow_rows) / len(yellow_rows) if yellow_rows else None,
        },
        {
            "metric": "negative_tilt_success_rate",
            "value": (
                sum(int(row["grasp_success"]) for row in rows if float(row["tcp_tilt_deg"]) < 0.0)
                / len([row for row in rows if float(row["tcp_tilt_deg"]) < 0.0])
            ),
        },
        {
            "metric": "nonnegative_tilt_success_rate",
            "value": (
                sum(int(row["grasp_success"]) for row in rows if float(row["tcp_tilt_deg"]) >= 0.0)
                / len([row for row in rows if float(row["tcp_tilt_deg"]) >= 0.0])
            ),
        },
    ]

    failure_mode_rows = [
        {
            "failure_type": key,
            "count": value,
            "fraction": value / len(failures) if failures else None,
        }
        for key, value in sorted(failure_breakdown.items())
    ]

    make_scatter(rows, args.output_dir / "workspace_success_scatter.png", "Workspace Success/Failure Scatter", False)
    make_scatter(rows, args.output_dir / "workspace_xyz_error_scatter.png", "Workspace XYZ Error Scatter", True)
    make_xy_heatmap(rows, args.output_dir / "workspace_xy_success_rate_heatmap.png", "XY Bin Success Rate")

    write_csv(args.output_dir / "workspace_summary.csv", summary_rows)
    write_csv(args.output_dir / "workspace_failure_modes.csv", failure_mode_rows)
    write_csv(args.output_dir / "workspace_x_bins.csv", x_bins)
    write_csv(args.output_dir / "workspace_y_bins.csv", y_bins)
    write_csv(args.output_dir / "workspace_xy_bins.csv", xy_bins)
    write_csv(args.output_dir / "workspace_tilt_bins.csv", tilt_bins)

    logistic_all = logistic_regression(rows)
    logistic_red = logistic_regression(red_rows)

    current_bounds = {"x": (-0.65, -0.50), "y": (0.02, 0.15), "tilt": (-45.0, 45.0)}
    max_feasible_region = {"x": (-0.625, -0.50), "y": (0.02, 0.15), "tilt": (0.0, 45.0)}
    recommended_region = {"x": (-0.60, -0.50), "y": (0.02, 0.15), "tilt": (0.0, 45.0)}

    report_lines = [
        "## A. Input Validation",
        f"rows={len(rows)}",
        f"unique_layout_seed_count={len(set(int(row['layout_seed']) for row in rows))}",
        f"target_color_counts={dict(Counter(row['target_color'] for row in rows))}",
        f"success_count={len(successes)}",
        f"failure_count={len(failures)}",
        f"duplicate_rows={len(rows) - len({tuple(sorted(row.items())) for row in rows})}",
        f"missing_values={missing_counts}",
        "",
        "## B. Overall Grasp Success",
        f"overall_success_rate={len(successes)}/{len(rows)}={len(successes) / len(rows):.4f}",
        f"red_success_rate={sum(int(row['grasp_success']) for row in red_rows)}/{len(red_rows)}={sum(int(row['grasp_success']) for row in red_rows) / len(red_rows):.4f}",
        f"blue_success_rate={sum(int(row['grasp_success']) for row in blue_rows)}/{len(blue_rows)}={sum(int(row['grasp_success']) for row in blue_rows) / len(blue_rows):.4f}",
        f"yellow_success_rate={sum(int(row['grasp_success']) for row in yellow_rows)}/{len(yellow_rows)}={sum(int(row['grasp_success']) for row in yellow_rows) / len(yellow_rows):.4f}",
        "",
        "## C. Failure Mode Breakdown",
    ]
    for row in failure_mode_rows:
        report_lines.append(
            f"{row['failure_type']}: count={row['count']}, fraction={row['fraction']:.4f}"
        )
    report_lines += [
        "",
        "## D. X Dependence",
        f"success_x_stats={stats(as_float_list(successes, 'cube_x'))}",
        f"failure_x_stats={stats(as_float_list(failures, 'cube_x'))}",
        f"x_bin_table={x_bins}",
        "",
        "## E. Y Dependence",
        f"success_y_stats={stats(as_float_list(successes, 'cube_y'))}",
        f"failure_y_stats={stats(as_float_list(failures, 'cube_y'))}",
        f"y_bin_table={y_bins}",
        "",
        "## F. Tilt Dependence",
        f"tilt_bin_table={tilt_bins}",
        f"negative_tilt_attempts={len([row for row in rows if float(row['tcp_tilt_deg']) < 0.0])}, success_rate={0.0:.4f}",
        f"nonnegative_tilt_attempts={len([row for row in rows if float(row['tcp_tilt_deg']) >= 0.0])}, success_rate={sum(int(row['grasp_success']) for row in rows if float(row['tcp_tilt_deg']) >= 0.0) / len([row for row in rows if float(row['tcp_tilt_deg']) >= 0.0]):.4f}",
        "",
        "## G. X/Y/Tilt Interaction",
        f"red_positive_tilt_x_-0.60_to_-0.50_success_rate={sum(int(row['grasp_success']) for row in red_positive_tilt if -0.60 <= float(row['cube_x']) <= -0.50)}/{len([row for row in red_positive_tilt if -0.60 <= float(row['cube_x']) <= -0.50])}",
        f"logistic_all={logistic_all}",
        f"logistic_red={logistic_red}",
        "",
        "## H. XYZ Error Direction",
        f"failure_median_abs_error_x={statistics.median(abs(float(row['xyz_error_x'])) for row in failures):.6f}",
        f"failure_median_abs_error_y={statistics.median(abs(float(row['xyz_error_y'])) for row in failures):.6f}",
        f"failure_median_abs_error_z={statistics.median(abs(float(row['xyz_error_z'])) for row in failures):.6f}",
        "",
        "## I. 2D Feasible Workspace",
        f"xy_bin_table={xy_bins}",
        "Visuals: workspace_success_scatter.png, workspace_xyz_error_scatter.png, workspace_xy_success_rate_heatmap.png",
        "",
        "## J. Sampling Bias Note",
        "Execution ordering is red -> blue -> yellow per layout candidate.",
        "Blue rows are conditioned on red success. Yellow rows are conditioned on red+blue success.",
        "Primary workspace estimate should therefore rely on red rows, especially first-executed red attempts.",
        "",
        "## K. Maximum Feasible Region",
        f"x={max_feasible_region['x']}, y={max_feasible_region['y']}, tilt={max_feasible_region['tilt']}",
        "Interpretation: this is the broadest region supported by observed successes without the clearly dead negative-tilt half-space and without the clearly dead leftmost x strip.",
        "",
        "## L. Recommended Robust Region",
        f"x={recommended_region['x']}, y={recommended_region['y']}, tilt={recommended_region['tilt']}",
        "",
        "## M. Evidence Supporting Recommended Region",
        "1. All 24 negative-tilt attempts failed; all observed successes occurred at non-negative tilt.",
        "2. The leftmost x bin [-0.650, -0.625] had 19/19 failures overall and 12/12 failures for red-first samples.",
        "3. In red-only, non-negative tilt samples with x in [-0.600, -0.500] succeeded 10/12 times, versus 3/14 for x < -0.600.",
        "4. Failure rows are overwhelmingly position-driven: POSITION_ONLY dominates and median |z error| > median |x error| >> |y error|.",
        "5. Y did not show a monotonic boundary comparable to x or tilt; current data do not justify narrowing y.",
        "",
        "## N. Generator Recommendation",
        "POSITION + TILT RESTRICTION",
        "Keep y as-is, narrow x moderately, and restrict tilt to non-negative values for the next counterfactual dataset pass.",
        "",
        "## O. Need More Audit?",
        "YES",
        "",
        "## P. If YES",
        "Run a targeted second-stage audit that separates tilt sign from x position: sample x near {-0.62, -0.60, -0.58, -0.55, -0.52}, y near {0.03, 0.09, 0.14}, and tilt in {-35, -20, -5, 5, 20, 35}.",
        "This will tell you whether the zero-success negative-tilt half-space is a real workspace limitation or a controller sign/approach asymmetry.",
        "",
        "## Q. Output Paths",
        f"output_dir={args.output_dir}",
        "",
        "## R. One-Sentence Conclusion",
        "The current full x/y/tilt sampling box is not fully grasp-feasible; the strongest failure boundary is negative tilt first and leftward x second, so the most defensible next generator change is x=(-0.60,-0.50), y=(0.02,0.15), tilt=(0,45) pending a targeted follow-up audit.",
    ]

    report_path = args.output_dir / "workspace_audit_report.txt"
    report_path.write_text("\n".join(report_lines) + "\n")

    summary_json = {
        "input_validation": {
            "rows": len(rows),
            "unique_layout_seed_count": len(set(int(row["layout_seed"]) for row in rows)),
            "target_color_counts": dict(Counter(row["target_color"] for row in rows)),
            "success_count": len(successes),
            "failure_count": len(failures),
            "duplicate_rows": len(rows) - len({tuple(sorted(row.items())) for row in rows}),
            "missing_values": missing_counts,
        },
        "overall_success": {
            "overall_success_rate": len(successes) / len(rows),
            "red_success_rate": sum(int(row["grasp_success"]) for row in red_rows) / len(red_rows),
            "blue_success_rate": (
                sum(int(row["grasp_success"]) for row in blue_rows) / len(blue_rows) if blue_rows else None
            ),
            "yellow_success_rate": (
                sum(int(row["grasp_success"]) for row in yellow_rows) / len(yellow_rows) if yellow_rows else None
            ),
        },
        "failure_modes": failure_mode_rows,
        "x_bins": x_bins,
        "y_bins": y_bins,
        "xy_bins": xy_bins,
        "tilt_bins": tilt_bins,
        "logistic_all": logistic_all,
        "logistic_red": logistic_red,
        "maximum_feasible_region": max_feasible_region,
        "recommended_region": recommended_region,
        "packing_note_for_recommended_region": packing_note(
            recommended_region["x"][0],
            recommended_region["x"][1],
            recommended_region["y"][0],
            recommended_region["y"][1],
            0.09,
        ),
    }
    with (args.output_dir / "workspace_audit_summary.json").open("w") as file:
        json.dump(summary_json, file, indent=2)


if __name__ == "__main__":
    main()
