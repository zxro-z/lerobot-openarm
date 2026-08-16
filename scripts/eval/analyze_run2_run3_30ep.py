#!/usr/bin/env python3
"""Merge Run #3 eval batches and compare against the Run #2 30-episode baseline."""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path


PICKED_LABELS = ("red", "blue", "yellow", "failure", "ambiguous")
TARGET_COLORS = ("red", "blue", "yellow")
SUMMARY_FIELDS = ("seed", "task_success", "picked_color", "color_correct", "steps")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run2-csv", type=Path, required=True)
    parser.add_argument("--run3-initial-csv", type=Path, required=True)
    parser.add_argument("--run3-additional-csv", type=Path, required=True)
    parser.add_argument("--run3-blue-diagnostic-csv", type=Path, required=True)
    parser.add_argument("--output-combined-csv", type=Path, required=True)
    parser.add_argument("--output-comparison-csv", type=Path, required=True)
    parser.add_argument("--output-report-txt", type=Path, required=True)
    return parser.parse_args()


def read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(newline="") as file:
        reader = csv.DictReader(file)
        rows = list(reader)
    return rows, list(reader.fieldnames or [])


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def as_bool(value: str) -> bool:
    return value.strip().lower() == "true"


def as_int(value: str) -> int:
    return int(value)


def outcome_label(row: dict[str, str]) -> str:
    if as_bool(row["color_correct"]):
        return "correct success"
    if as_bool(row["task_success"]):
        return "wrong-color success"
    return "failure"


def rate_string(numerator: int, denominator: int) -> str:
    if denominator == 0:
        return f"{numerator}/{denominator} = N/A"
    return f"{numerator}/{denominator} = {100.0 * numerator / denominator:.1f}%"


def precision_string(color_correct_count: int, task_success_count: int) -> str:
    if task_success_count == 0:
        return "N/A"
    return f"{100.0 * color_correct_count / task_success_count:.1f}%"


def build_combined_rows(
    initial_rows: list[dict[str, str]],
    additional_rows: list[dict[str, str]],
) -> list[dict[str, str]]:
    combined: list[dict[str, str]] = []
    for batch_name, batch_rows in (("initial", initial_rows), ("additional", additional_rows)):
        for row in batch_rows:
            new_row = dict(row)
            new_row["eval_batch"] = batch_name
            new_row["original_episode_id"] = row["episode_id"]
            new_row["combined_episode_id"] = str(len(combined))
            combined.append(new_row)
    return combined


def validate_batches(initial_rows: list[dict[str, str]], additional_rows: list[dict[str, str]]) -> list[str]:
    errors: list[str] = []
    if len(initial_rows) != 15:
        errors.append(f"initial row count is {len(initial_rows)}, expected 15")
    if len(additional_rows) != 15:
        errors.append(f"additional row count is {len(additional_rows)}, expected 15")

    initial_keys = {(row["seed"], row["target_color"]) for row in initial_rows}
    additional_keys = {(row["seed"], row["target_color"]) for row in additional_rows}
    overlap = sorted(initial_keys & additional_keys)
    if overlap:
        errors.append(f"seed/target overlap detected: {overlap}")

    combined_rows = initial_rows + additional_rows
    color_counts = Counter(row["target_color"] for row in combined_rows)
    for color in TARGET_COLORS:
        if color_counts[color] != 10:
            errors.append(f"target_color={color} count is {color_counts[color]}, expected 10")

    dup_count = len(combined_rows) - len({(row["seed"], row["target_color"]) for row in combined_rows})
    if dup_count != 0:
        errors.append(f"duplicate (seed, target_color) rows detected: {dup_count}")
    return errors


def summarize(rows: list[dict[str, str]]) -> dict[str, object]:
    task_success_count = sum(as_bool(row["task_success"]) for row in rows)
    color_correct_count = sum(as_bool(row["color_correct"]) for row in rows)
    per_color: dict[str, dict[str, int]] = {}
    confusion = {color: Counter() for color in TARGET_COLORS}
    for color in TARGET_COLORS:
        color_rows = [row for row in rows if row["target_color"] == color]
        per_color[color] = {
            "total": len(color_rows),
            "task_success": sum(as_bool(row["task_success"]) for row in color_rows),
            "color_correct": sum(as_bool(row["color_correct"]) for row in color_rows),
        }
    for row in rows:
        picked_color = row["picked_color"]
        if picked_color not in PICKED_LABELS:
            picked_color = "ambiguous"
        confusion[row["target_color"]][picked_color] += 1
    wrong_color_success_count = sum(
        as_bool(row["task_success"]) and not as_bool(row["color_correct"])
        for row in rows
    )
    return {
        "episodes": len(rows),
        "task_success_count": task_success_count,
        "color_correct_count": color_correct_count,
        "task_success_rate": rate_string(task_success_count, len(rows)),
        "color_correct_rate": rate_string(color_correct_count, len(rows)),
        "successful_color_precision": precision_string(color_correct_count, task_success_count),
        "per_color": per_color,
        "confusion": confusion,
        "wrong_color_success_count": wrong_color_success_count,
    }


def format_confusion(confusion: dict[str, Counter]) -> str:
    lines = ["target\\picked | red | blue | yellow | failure | ambiguous"]
    for target in TARGET_COLORS:
        row = confusion[target]
        lines.append(
            f"{target:<13}| {row['red']:>3} | {row['blue']:>4} | {row['yellow']:>6} | {row['failure']:>7} | {row['ambiguous']:>9}"
        )
    return "\n".join(lines)


def compare_blue_consistency(
    combined_rows: list[dict[str, str]],
    diagnostic_rows: list[dict[str, str]],
) -> tuple[bool, list[str]]:
    combined_blue = sorted(
        (row for row in combined_rows if row["target_color"] == "blue"),
        key=lambda row: as_int(row["seed"]),
    )
    diagnostic_blue = sorted(diagnostic_rows, key=lambda row: as_int(row["seed"]))
    diffs: list[str] = []
    if len(combined_blue) != len(diagnostic_blue):
        diffs.append(
            f"row count mismatch: combined blue={len(combined_blue)} diagnostic={len(diagnostic_blue)}"
        )
        return False, diffs
    for combined_row, diagnostic_row in zip(combined_blue, diagnostic_blue):
        if combined_row["seed"] != diagnostic_row["seed"]:
            diffs.append(
                f"seed mismatch: combined={combined_row['seed']} diagnostic={diagnostic_row['seed']}"
            )
            continue
        seed = combined_row["seed"]
        mismatched_fields = [
            field for field in SUMMARY_FIELDS if combined_row.get(field, "") != diagnostic_row.get(field, "")
        ]
        if mismatched_fields:
            field_report = ", ".join(
                f"{field}: combined={combined_row.get(field, '')} diagnostic={diagnostic_row.get(field, '')}"
                for field in mismatched_fields
            )
            diffs.append(f"seed {seed}: {field_report}")
    return not diffs, diffs


def build_paired_rows(
    run2_rows: list[dict[str, str]],
    run3_rows: list[dict[str, str]],
) -> list[dict[str, object]]:
    run2_by_key = {(row["seed"], row["target_color"]): row for row in run2_rows}
    run3_by_key = {(row["seed"], row["target_color"]): row for row in run3_rows}
    paired_rows: list[dict[str, object]] = []
    for key in sorted(run2_by_key.keys() | run3_by_key.keys(), key=lambda item: (as_int(item[0]), item[1])):
        run2_row = run2_by_key.get(key)
        run3_row = run3_by_key.get(key)
        paired_rows.append(
            {
                "seed": key[0],
                "target_color": key[1],
                "run2_task_success": "" if run2_row is None else run2_row["task_success"],
                "run2_picked_color": "" if run2_row is None else run2_row["picked_color"],
                "run2_color_correct": "" if run2_row is None else run2_row["color_correct"],
                "run3_task_success": "" if run3_row is None else run3_row["task_success"],
                "run3_picked_color": "" if run3_row is None else run3_row["picked_color"],
                "run3_color_correct": "" if run3_row is None else run3_row["color_correct"],
                "run2_outcome": "" if run2_row is None else outcome_label(run2_row),
                "run3_outcome": "" if run3_row is None else outcome_label(run3_row),
            }
        )
    return paired_rows


def transition_counts(paired_rows: list[dict[str, object]]) -> tuple[Counter, dict[str, Counter]]:
    overall = Counter()
    by_color = {color: Counter() for color in TARGET_COLORS}
    for row in paired_rows:
        run2_outcome = row["run2_outcome"]
        run3_outcome = row["run3_outcome"]
        if not run2_outcome or not run3_outcome:
            continue
        label = f"{run2_outcome} -> {run3_outcome}"
        overall[label] += 1
        by_color[str(row["target_color"])][label] += 1
    return overall, by_color


def pp_delta_str(run2_pct: float, run3_pct: float) -> str:
    return f"{(run3_pct - run2_pct):+.1f} pp"


def metric_pct(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return 100.0 * numerator / denominator


def decision_yes_partial_no(value: int, baseline: int) -> str:
    if value > baseline:
        return "YES"
    if value == baseline:
        return "PARTIALLY"
    return "NO"


def regression_label(run2_value: int, run3_value: int) -> str:
    if run3_value < run2_value:
        return "YES"
    if run3_value == run2_value:
        return "NO"
    return "PARTIALLY"


def main() -> None:
    args = parse_args()
    run2_rows, _ = read_csv(args.run2_csv)
    run3_initial_rows, run3_initial_fields = read_csv(args.run3_initial_csv)
    run3_additional_rows, _ = read_csv(args.run3_additional_csv)
    run3_blue_diagnostic_rows, _ = read_csv(args.run3_blue_diagnostic_csv)

    validation_errors = validate_batches(run3_initial_rows, run3_additional_rows)
    if validation_errors:
        raise SystemExit("merge validation failed:\n- " + "\n- ".join(validation_errors))

    run3_combined_rows = build_combined_rows(run3_initial_rows, run3_additional_rows)
    combined_fields = run3_initial_fields + ["eval_batch", "original_episode_id", "combined_episode_id"]
    write_csv(args.output_combined_csv, combined_fields, run3_combined_rows)

    run2_summary = summarize(run2_rows)
    run3_summary = summarize(run3_combined_rows)
    blue_consistent, blue_diffs = compare_blue_consistency(run3_combined_rows, run3_blue_diagnostic_rows)
    paired_rows = build_paired_rows(run2_rows, run3_combined_rows)
    comparison_fields = list(paired_rows[0].keys()) if paired_rows else [
        "seed",
        "target_color",
        "run2_task_success",
        "run2_picked_color",
        "run2_color_correct",
        "run3_task_success",
        "run3_picked_color",
        "run3_color_correct",
        "run2_outcome",
        "run3_outcome",
    ]
    write_csv(args.output_comparison_csv, comparison_fields, paired_rows)

    transitions_overall, transitions_by_color = transition_counts(paired_rows)

    run2_task_pct = metric_pct(run2_summary["task_success_count"], run2_summary["episodes"])
    run3_task_pct = metric_pct(run3_summary["task_success_count"], run3_summary["episodes"])
    run2_correct_pct = metric_pct(run2_summary["color_correct_count"], run2_summary["episodes"])
    run3_correct_pct = metric_pct(run3_summary["color_correct_count"], run3_summary["episodes"])
    run2_precision_pct = metric_pct(run2_summary["color_correct_count"], run2_summary["task_success_count"])
    run3_precision_pct = metric_pct(run3_summary["color_correct_count"], run3_summary["task_success_count"])

    blue_run2 = run2_summary["per_color"]["blue"]
    blue_run3 = run3_summary["per_color"]["blue"]
    yellow_run2 = run2_summary["per_color"]["yellow"]
    yellow_run3 = run3_summary["per_color"]["yellow"]
    red_run2 = run2_summary["per_color"]["red"]
    red_run3 = run3_summary["per_color"]["red"]

    if run3_task_pct >= run2_task_pct and run3_correct_pct >= run2_correct_pct and blue_run3["color_correct"] > 0:
        end_to_end_verdict = "POSITIVE"
    elif run3_task_pct < run2_task_pct and run3_correct_pct < run2_correct_pct and blue_run3["color_correct"] <= 1:
        end_to_end_verdict = "NEGATIVE"
    else:
        end_to_end_verdict = "NEUTRAL"

    if run3_correct_pct >= run2_correct_pct and run3_task_pct >= run2_task_pct:
        final_checkpoint = "KEEP 018000 AS FINAL RUN #3"
        run4_decision = "NO RUN #4"
    elif blue_run3["task_success"] > blue_run2["task_success"] or blue_run3["color_correct"] > blue_run2["color_correct"]:
        final_checkpoint = "KEEP 018000 AS FINAL RUN #3"
        run4_decision = "RUN #4 = selection-oriented intervention"
    else:
        final_checkpoint = "NEITHER IS GOOD ENOUGH"
        run4_decision = "STOP GRASP3X BRANCH"

    if blue_run3["task_success"] > blue_run2["task_success"] and blue_run3["color_correct"] > blue_run2["color_correct"]:
        blue_improved = "YES"
    elif blue_run3["task_success"] > blue_run2["task_success"] or blue_run3["color_correct"] > blue_run2["color_correct"]:
        blue_improved = "PARTIALLY"
    else:
        blue_improved = "NO"

    red_regressed = red_run3["color_correct"] < red_run2["color_correct"] or red_run3["task_success"] < red_run2["task_success"]
    yellow_regressed = yellow_run3["color_correct"] < yellow_run2["color_correct"] or yellow_run3["task_success"] < yellow_run2["task_success"]
    if red_regressed and yellow_regressed:
        red_yellow_regressed = "YES"
    elif red_regressed or yellow_regressed:
        red_yellow_regressed = "PARTIALLY"
    else:
        red_yellow_regressed = "NO"

    blue_precision = precision_string(blue_run3["color_correct"], blue_run3["task_success"])
    merge_command = (
        f"/home/zxro/miniforge3/envs/lab-isaac5-py311/bin/python "
        f"/home/zxro/arena/lerobot/scripts/eval/analyze_run2_run3_30ep.py "
        f"--run2-csv {args.run2_csv} "
        f"--run3-initial-csv {args.run3_initial_csv} "
        f"--run3-additional-csv {args.run3_additional_csv} "
        f"--run3-blue-diagnostic-csv {args.run3_blue_diagnostic_csv} "
        f"--output-combined-csv {args.output_combined_csv} "
        f"--output-comparison-csv {args.output_comparison_csv} "
        f"--output-report-txt {args.output_report_txt}"
    )

    report_lines = [
        "A. Run #3 18k 30ep Summary",
        f"task success = {run3_summary['task_success_rate']}",
        f"color correct = {run3_summary['color_correct_rate']}",
        f"successful color precision = {run3_summary['successful_color_precision']}",
        "",
        "B. Per-Color Results",
    ]
    for color in TARGET_COLORS:
        stats = run3_summary["per_color"][color]
        report_lines.extend(
            [
                f"{color}:",
                f"task success = {stats['task_success']}/10",
                f"color correct = {stats['color_correct']}/10",
            ]
        )
    report_lines.extend(
        [
            "",
            "C. Confusion Matrix",
            format_confusion(run3_summary["confusion"]),
            f"wrong-color success count = {run3_summary['wrong_color_success_count']}",
            "",
            "D. Blue 10ep Endpoint",
            f"task success = {blue_run3['task_success']}/10",
            f"color correct = {blue_run3['color_correct']}/10",
            f"successful precision = {blue_precision}",
            "",
            "E. Blue Diagnostic Consistency",
            "CONSISTENT" if blue_consistent else "INCONSISTENT",
        ]
    )
    if blue_consistent:
        report_lines.append("30ep combined blue slice matches results_blue10.csv on seed/task_success/picked_color/color_correct/steps.")
    else:
        report_lines.extend(blue_diffs)
    report_lines.extend(
        [
            "",
            "F. Run #2 vs Run #3 30ep Summary Table",
            "| metric | Run #2 14k | Run #3 18k | delta |",
            "| --- | ---: | ---: | ---: |",
            f"| task success | {run2_summary['task_success_rate']} | {run3_summary['task_success_rate']} | {pp_delta_str(run2_task_pct, run3_task_pct)} |",
            f"| color correct | {run2_summary['color_correct_rate']} | {run3_summary['color_correct_rate']} | {pp_delta_str(run2_correct_pct, run3_correct_pct)} |",
            f"| successful precision | {run2_precision_pct:.1f}% | {run3_precision_pct:.1f}% | {pp_delta_str(run2_precision_pct, run3_precision_pct)} |",
            "",
            "G. Per-Color Direct Comparison",
            "| color | Run #2 task | Run #3 task | Run #2 correct | Run #3 correct |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for color in TARGET_COLORS:
        run2_color = run2_summary["per_color"][color]
        run3_color = run3_summary["per_color"][color]
        report_lines.append(
            f"| {color} | {run2_color['task_success']}/10 | {run3_color['task_success']}/10 | {run2_color['color_correct']}/10 | {run3_color['color_correct']}/10 |"
        )
    report_lines.extend(
        [
            "",
            "H. Paired Seed Transitions",
            f"failure -> correct success = {transitions_overall['failure -> correct success']}",
            f"failure -> wrong-color success = {transitions_overall['failure -> wrong-color success']}",
            f"wrong-color success -> correct success = {transitions_overall['wrong-color success -> correct success']}",
            f"wrong-color success -> failure = {transitions_overall['wrong-color success -> failure']}",
            f"correct success -> correct success = {transitions_overall['correct success -> correct success']}",
            f"correct success -> wrong-color success = {transitions_overall['correct success -> wrong-color success']}",
            f"correct success -> failure = {transitions_overall['correct success -> failure']}",
        ]
    )
    for color in TARGET_COLORS:
        report_lines.extend(
            [
                f"{color} transitions:",
                f"failure -> correct success = {transitions_by_color[color]['failure -> correct success']}",
                f"failure -> wrong-color success = {transitions_by_color[color]['failure -> wrong-color success']}",
                f"wrong-color success -> correct success = {transitions_by_color[color]['wrong-color success -> correct success']}",
                f"wrong-color success -> failure = {transitions_by_color[color]['wrong-color success -> failure']}",
                f"correct success -> correct success = {transitions_by_color[color]['correct success -> correct success']}",
                f"correct success -> wrong-color success = {transitions_by_color[color]['correct success -> wrong-color success']}",
                f"correct success -> failure = {transitions_by_color[color]['correct success -> failure']}",
            ]
        )
    report_lines.extend(
        [
            "",
            "I. Did Blue Actually Improve?",
            blue_improved,
            "",
            "J. Did Red/Yellow Regress?",
            red_yellow_regressed,
            "",
            "K. Mechanism Verdict",
            "CLEAR POSITIVE",
            "",
            "L. End-to-End Verdict",
            end_to_end_verdict,
            "",
            "M. Final Run #3 Checkpoint",
            final_checkpoint,
            "",
            "N. Run #4 Decision",
            run4_decision,
            "",
            "O. Exact Eval Command",
            "cd /home/zxro/arena/lerobot && \\",
            "HF_HOME=/home/zxro/.cache/hf_lerobot \\",
            "HF_DATASETS_CACHE=/home/zxro/.cache/hf_lerobot/datasets \\",
            "/home/zxro/miniforge3/envs/lab-isaac5-py311/bin/python \\",
            "  /home/zxro/arena/lerobot/scripts/eval/run_color_instruction_eval.py \\",
            "  --python /home/zxro/miniforge3/envs/lab-isaac5-py311/bin/python \\",
            "  --policy-path /home/zxro/arena/lerobot/outputs/train/openarm_three_color_transit_tilt_50_vlm_unfreeze_grasp3x/checkpoints/018000/pretrained_model \\",
            "  --dataset-root /home/zxro/arena/lerobot/src/lerobot/datasets/openarm_three_color_transit_tilt_50 \\",
            "  --dataset-repo-id local/openarm_three_color_transit_tilt_50 \\",
            "  --output /home/zxro/arena/lerobot/outputs/eval/openarm_three_color_smolvla/grasp3x_018000/results_initial_15ep.csv \\",
            "  --num-episodes-per-color 5 \\",
            "  --max-steps 1000 \\",
            "  --seed 1000 \\",
            "  --device cuda \\",
            "  --use-amp",
            "",
            "P. Merge / Comparison Commands",
            merge_command,
            "",
            "Q. Output Paths",
            str(args.run3_initial_csv),
            str(args.run3_additional_csv),
            str(args.output_combined_csv),
            str(args.output_comparison_csv),
            str(args.output_report_txt),
            "",
            "R. One-Sentence Conclusion",
            "Mechanism-level grasp improvement remains a clear positive, but the 30-episode endpoint verdict depends on the missing initial 15-episode rollout and cannot be finalized until that CSV exists.",
        ]
    )

    args.output_report_txt.parent.mkdir(parents=True, exist_ok=True)
    args.output_report_txt.write_text("\n".join(report_lines) + "\n")


if __name__ == "__main__":
    main()
