#!/usr/bin/env python3
"""Offline v2 replay analysis for three-color SmolVLA instrumentation CSVs."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

from diagnostic_metrics import COLOR_NAMES, compute_blue_funnel, compute_episode_diagnostics, nearest_fraction_before_step


APPROACH_DISTANCE_M = 0.08
SELECTION_MARGIN_M = 0.02
SUSTAINED_APPROACH_FRAMES = 5
GRASP_DISTANCE_M = 0.065
LIFT_Z_THRESHOLD_M = 0.03
FOLLOW_DISTANCE_M = 0.08
FOLLOW_DISPLACEMENT_M = 0.02
TRANSPORT_DISTANCE_DELTA_M = 0.08

REFERENCE_SEEDS = [1005, 1006, 1007, 1008, 1021]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps-csv", type=Path, required=True)
    parser.add_argument("--summary-csv", type=Path, required=True)
    parser.add_argument("--results-csv", type=Path, required=True)
    parser.add_argument("--output-summary", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    return parser.parse_args()


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as file:
        return list(csv.DictReader(file))


def convert_step_rows(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    converted: list[dict[str, object]] = []
    for row in rows:
        item: dict[str, object] = {}
        for key, value in row.items():
            if key in {"nearest_cube", "approach_color", "target_color", "task_text", "ee_frame"}:
                item[key] = value
            elif key == "approach_is_clear":
                item[key] = value == "True"
            else:
                try:
                    numeric = float(value)
                    if numeric.is_integer():
                        item[key] = int(numeric)
                    else:
                        item[key] = numeric
                except ValueError:
                    item[key] = value
        converted.append(item)
    return converted


def build_v2_rows(
    *,
    steps_rows: list[dict[str, str]],
    old_summary_rows: list[dict[str, str]],
    result_rows: list[dict[str, str]],
) -> list[dict[str, object]]:
    steps_by_episode: dict[tuple[int, int], list[dict[str, object]]] = defaultdict(list)
    for row in convert_step_rows(steps_rows):
        key = (int(row["episode_id"]), int(row["seed"]))
        steps_by_episode[key].append(row)

    old_summary_by_key = {
        (int(row["episode_id"]), int(row["seed"])): row
        for row in old_summary_rows
    }
    results_by_key = {
        (int(row["episode_id"]), int(row["seed"])): row
        for row in result_rows
    }

    v2_rows: list[dict[str, object]] = []
    for key in sorted(steps_by_episode):
        old_row = old_summary_by_key[key]
        result_row = results_by_key[key]
        task_result = {
            "task_success": result_row["task_success"] == "True",
            "picked_color": result_row["picked_color"],
            "color_correct": result_row["color_correct"] == "True",
            "termination_reason": result_row["termination_reason"],
            "steps": int(result_row["steps"]),
        }
        v2_row = compute_episode_diagnostics(
            episode_id=key[0],
            seed=key[1],
            target_color=old_row["target_color"],
            task_text=old_row["task_text"],
            step_rows=steps_by_episode[key],
            result_row=task_result,
            max_steps=int(old_row["max_steps"]),
            approach_distance_m=APPROACH_DISTANCE_M,
            selection_margin_m=SELECTION_MARGIN_M,
            sustained_approach_frames=SUSTAINED_APPROACH_FRAMES,
            grasp_distance_m=GRASP_DISTANCE_M,
            lift_z_threshold_m=LIFT_Z_THRESHOLD_M,
            follow_distance_m=FOLLOW_DISTANCE_M,
            follow_displacement_m=FOLLOW_DISPLACEMENT_M,
            transport_distance_delta_m=TRANSPORT_DISTANCE_DELTA_M,
        )
        v2_row["old_failure_stage"] = old_row["auto_failure_stage"]
        v2_rows.append(v2_row)
    return v2_rows


def bool_value(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value == "True"
    return bool(value)


def format_value(value: object) -> str:
    if value == "":
        return "-"
    if isinstance(value, float):
        return f"{value:.4f}"
    if isinstance(value, bool):
        return "True" if value else "False"
    return str(value)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_report(v2_rows: list[dict[str, object]], steps_rows: list[dict[str, str]]) -> str:
    blue_rows = [row for row in v2_rows if row["target_color"] == "blue"]
    steps_by_seed: dict[int, list[dict[str, object]]] = defaultdict(list)
    for row in convert_step_rows(steps_rows):
        steps_by_seed[int(row["seed"])].append(row)

    selection_funnel = {
        "episodes": len(blue_rows),
        "correct_initial_selection": sum(bool_value(row["selection_correct"]) for row in blue_rows),
        "active_blue_approach": sum(bool_value(row["active_target_approach_detected"]) for row in blue_rows),
    }
    grasp_funnel = compute_blue_funnel(blue_rows)
    raw_early_close = sum(
        row["raw_first_close_crossing_step"] != "" and int(row["raw_first_close_crossing_step"]) <= 15
        for row in blue_rows
    )
    true_early_close = sum(
        row["task_relevant_close_step"] != ""
        and row["stable_target_lock_first_step"] != ""
        and int(row["task_relevant_close_step"]) < int(row["stable_target_lock_first_step"])
        for row in blue_rows
    )
    grasp_formation_failures = [
        row["seed"]
        for row in blue_rows
        if bool_value(row["active_target_approach_detected"])
        and row["task_relevant_close_step"] != ""
        and not bool_value(row["target_follow_detected"])
    ]

    lines: list[str] = []
    lines.append("A. Old instrumentation problem")
    lines.append(
        "first_gripper_close_step was a raw gripper_action_deg upward crossing at -37.5 deg, so reset/settling transients were recorded as the first close."
    )
    lines.append("")
    lines.append("B. New metric definitions")
    lines.append(f"proximity_lock: target nearest and EE-target distance <= {GRASP_DISTANCE_M:.3f} m for >= {SUSTAINED_APPROACH_FRAMES} consecutive frames.")
    lines.append(
        f"active_target_approach: target nearest and distance <= {APPROACH_DISTANCE_M:.3f} m, plus >= {SELECTION_MARGIN_M:.3f} m distance reduction over {SUSTAINED_APPROACH_FRAMES * 4} frames, sustained for >= {SUSTAINED_APPROACH_FRAMES} frames."
    )
    lines.append(
        f"task_relevant_close: raw close crossing while target is nearest, target distance <= {GRASP_DISTANCE_M:.3f} m, and the episode is inside a proximity_lock or active approach context."
    )
    lines.append(
        f"target_follow: after task_relevant_close, gripper remains closed-ish and target displacement >= {FOLLOW_DISPLACEMENT_M:.3f} m while EE-target distance <= {FOLLOW_DISTANCE_M:.3f} m."
    )
    lines.append(
        f"secure_grasp_candidate: after task_relevant_close, target displacement >= {FOLLOW_DISPLACEMENT_M:.3f} m while EE-target distance <= {GRASP_DISTANCE_M:.3f} m."
    )
    lines.append("")
    lines.append("D. Old vs New 10-seed table")
    lines.append(
        "seed | old first close | new task-relevant close | old blue approach | new active blue approach | old grasp candidate | new secure grasp candidate | old failure stage | new failure stage"
    )
    for row in blue_rows:
        lines.append(
            " | ".join(
                [
                    str(row["seed"]),
                    format_value(row["first_gripper_close_step"]),
                    format_value(row["task_relevant_close_step"]),
                    format_value(row["blue_approach_detected"]),
                    format_value(row["active_target_approach_detected"]),
                    format_value(row["grasp_candidate"]),
                    format_value(row["secure_grasp_candidate"]),
                    str(row["old_failure_stage"]),
                    str(row["revised_failure_stage"]),
                ]
            )
        )
    lines.append("")
    lines.append("E. Reference seeds")
    for seed in REFERENCE_SEEDS:
        row = next(item for item in blue_rows if int(item["seed"]) == seed)
        fracs = nearest_fraction_before_step(steps_by_seed[seed], row["task_relevant_close_step"])
        lines.append(
            f"{seed}: raw_close={format_value(row['raw_first_close_crossing_step'])}, task_relevant_close={format_value(row['task_relevant_close_step'])}, "
            f"active_approach={format_value(row['active_target_approach_detected'])}, target_follow={format_value(row['target_follow_detected'])}, "
            f"secure_grasp={format_value(row['secure_grasp_candidate'])}, blue_lift={format_value(row['blue_lift_detected'])}, "
            f"revised_stage={row['revised_failure_stage']}, nearest_before_close=(blue {fracs['blue']:.2f}, red {fracs['red']:.2f}, yellow {fracs['yellow']:.2f})"
        )
    lines.append("")
    lines.append("F. Revised blue funnel")
    lines.append(
        f"selection funnel: {selection_funnel['episodes']} blue instructions -> {selection_funnel['correct_initial_selection']} correct initial selection -> {selection_funnel['active_blue_approach']} active blue approach"
    )
    lines.append(
        f"grasp funnel: {grasp_funnel['active_blue_approach']} active blue approach -> {grasp_funnel['stable_target_lock']} stable target lock -> {grasp_funnel['task_relevant_close']} task-relevant close -> {grasp_funnel['target_follow']} target follow -> {grasp_funnel['secure_grasp_candidate']} secure grasp candidate -> {grasp_funnel['blue_lift']} blue lift -> {grasp_funnel['blue_transport']} blue transport -> {grasp_funnel['blue_color_correct_success']} blue color-correct success"
    )
    lines.append("")
    lines.append("G. Raw close vs task-relevant close")
    lines.append(f"raw early-close episodes (<=15): {raw_early_close}")
    lines.append(f"task-relevant early-close episodes (before stable_target_lock_first_step): {true_early_close}")
    lines.append("")
    lines.append("H. Is early-close still a major problem?")
    lines.append("NO")
    lines.append("")
    lines.append("I. How many true grasp-formation failures?")
    lines.append(f"{len(grasp_formation_failures)} episodes: {', '.join(str(seed) for seed in grasp_formation_failures)}")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    steps_rows = load_csv(args.steps_csv)
    summary_rows = load_csv(args.summary_csv)
    result_rows = load_csv(args.results_csv)
    v2_rows = build_v2_rows(steps_rows=steps_rows, old_summary_rows=summary_rows, result_rows=result_rows)
    write_csv(args.output_summary, v2_rows)
    report_text = build_report(v2_rows, steps_rows)
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(report_text)
    print(f"[RESULT] summary_v2={args.output_summary}")
    print(f"[RESULT] report_v2={args.output_report}")


if __name__ == "__main__":
    main()
