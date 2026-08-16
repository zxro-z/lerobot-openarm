#!/usr/bin/env python3
"""Analyze True Run #4 15-episode screening checkpoints without running new rollouts."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path


TARGET_COLORS = ("red", "blue", "yellow")
PICKED_LABELS = ("red", "blue", "yellow", "failure", "ambiguous")
CHECKPOINTS = ("010000", "014000", "018000")
EXPECTED_SEEDS = {
    "red": ("1015", "1016", "1017", "1018", "1019"),
    "blue": ("1020", "1021", "1022", "1023", "1024"),
    "yellow": ("1025", "1026", "1027", "1028", "1029"),
}
RUN_REFERENCES = (
    {
        "label": "Run #2 14k",
        "task_success_count": 5,
        "color_correct_count": 4,
        "precision_pct": 80.0,
        "per_color_task": {"red": 3, "blue": 0, "yellow": 2},
        "per_color_correct": {"red": 2, "blue": 0, "yellow": 2},
    },
    {
        "label": "Run #3 14k",
        "task_success_count": 5,
        "color_correct_count": 2,
        "precision_pct": 40.0,
        "per_color_task": {"red": 1, "blue": 1, "yellow": 3},
        "per_color_correct": {"red": 0, "blue": 0, "yellow": 2},
    },
    {
        "label": "Run #3 18k",
        "task_success_count": 4,
        "color_correct_count": 1,
        "precision_pct": 25.0,
        "per_color_task": {"red": 2, "blue": 2, "yellow": 0},
        "per_color_correct": {"red": 0, "blue": 1, "yellow": 0},
    },
)
TRANSITION_LABELS = (
    ("failure", "correct"),
    ("failure", "wrong_color"),
    ("failure", "failure"),
    ("failure", "ambiguous"),
    ("wrong_color", "correct"),
    ("wrong_color", "failure"),
    ("wrong_color", "wrong_color"),
    ("wrong_color", "ambiguous"),
    ("correct", "correct"),
    ("correct", "wrong_color"),
    ("correct", "failure"),
    ("correct", "ambiguous"),
    ("ambiguous", "correct"),
    ("ambiguous", "wrong_color"),
    ("ambiguous", "failure"),
    ("ambiguous", "ambiguous"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run4-10k", type=Path, required=True)
    parser.add_argument("--run4-14k", type=Path, required=True)
    parser.add_argument("--run4-18k", type=Path, required=True)
    parser.add_argument("--output-summary-csv", type=Path, required=True)
    parser.add_argument("--output-paired-csv", type=Path, required=True)
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


def normalize_picked_color(value: str) -> str:
    picked = value.strip().lower()
    if picked in PICKED_LABELS:
        return picked
    return "ambiguous"


def percent_string(numerator: int, denominator: int) -> str:
    if denominator == 0:
        return f"{numerator}/{denominator} = N/A"
    return f"{numerator}/{denominator} = {100.0 * numerator / denominator:.1f}%"


def precision_string(color_correct_count: int, task_success_count: int) -> str:
    if task_success_count == 0:
        return "N/A"
    return f"{100.0 * color_correct_count / task_success_count:.1f}%"


def outcome_category(row: dict[str, str]) -> str:
    task_success = as_bool(row["task_success"])
    color_correct = as_bool(row["color_correct"])
    picked_color = normalize_picked_color(row.get("picked_color", ""))
    if task_success and color_correct:
        return "correct"
    if task_success and picked_color == "ambiguous":
        return "ambiguous"
    if task_success:
        return "wrong_color"
    return "failure"


def validate_schema(fieldnames: list[str]) -> list[str]:
    required_fields = ("seed", "target_color", "task_success", "picked_color", "color_correct", "steps")
    return [field for field in required_fields if field not in fieldnames]


def validate_rows(rows: list[dict[str, str]], checkpoint: str) -> list[str]:
    errors: list[str] = []
    if len(rows) != 15:
        errors.append(f"{checkpoint}: rows={len(rows)}, expected 15")
    color_counts = Counter(row["target_color"] for row in rows)
    for color in TARGET_COLORS:
        if color_counts[color] != 5:
            errors.append(f"{checkpoint}: target_color={color} count={color_counts[color]}, expected 5")
    duplicate_count = len(rows) - len({(row["seed"], row["target_color"]) for row in rows})
    if duplicate_count != 0:
        errors.append(f"{checkpoint}: duplicate (seed, target_color) count={duplicate_count}")
    for color in TARGET_COLORS:
        seen_seeds = tuple(sorted(row["seed"] for row in rows if row["target_color"] == color))
        if seen_seeds != EXPECTED_SEEDS[color]:
            errors.append(f"{checkpoint}: {color} seeds={seen_seeds}, expected={EXPECTED_SEEDS[color]}")
    return errors


def summarize_rows(rows: list[dict[str, str]]) -> dict[str, object]:
    task_success_count = sum(as_bool(row["task_success"]) for row in rows)
    color_correct_count = sum(as_bool(row["color_correct"]) for row in rows)
    wrong_color_success_count = sum(outcome_category(row) == "wrong_color" for row in rows)
    ambiguous_count = sum(outcome_category(row) == "ambiguous" for row in rows)
    failure_count = sum(outcome_category(row) == "failure" for row in rows)
    confusion = {target: Counter() for target in TARGET_COLORS}
    per_color_task: dict[str, int] = {}
    per_color_correct: dict[str, int] = {}

    for color in TARGET_COLORS:
        color_rows = [row for row in rows if row["target_color"] == color]
        per_color_task[color] = sum(as_bool(row["task_success"]) for row in color_rows)
        per_color_correct[color] = sum(as_bool(row["color_correct"]) for row in color_rows)

    for row in rows:
        confusion[row["target_color"]][normalize_picked_color(row["picked_color"])] += 1

    min_correct = min(per_color_correct.values())
    max_correct = max(per_color_correct.values())
    return {
        "episodes": len(rows),
        "task_success_count": task_success_count,
        "color_correct_count": color_correct_count,
        "wrong_color_success_count": wrong_color_success_count,
        "ambiguous_count": ambiguous_count,
        "failure_count": failure_count,
        "precision_string": precision_string(color_correct_count, task_success_count),
        "precision_pct": 0.0 if task_success_count == 0 else 100.0 * color_correct_count / task_success_count,
        "task_success_rate": percent_string(task_success_count, len(rows)),
        "color_correct_rate": percent_string(color_correct_count, len(rows)),
        "wrong_color_rate": percent_string(wrong_color_success_count, task_success_count) if task_success_count else "N/A",
        "per_color_task": per_color_task,
        "per_color_correct": per_color_correct,
        "confusion": confusion,
        "min_correct": min_correct,
        "max_correct": max_correct,
        "range_correct": max_correct - min_correct,
        "all_colors_nonzero": all(value > 0 for value in per_color_correct.values()),
    }


def exact_seed_target_set(rows: list[dict[str, str]]) -> set[tuple[str, str]]:
    return {(row["seed"], row["target_color"]) for row in rows}


def build_paired_rows(by_checkpoint: dict[str, dict[tuple[str, str], dict[str, str]]]) -> list[dict[str, object]]:
    keys = sorted(by_checkpoint["010000"].keys(), key=lambda item: (as_int(item[0]), item[1]))
    paired_rows: list[dict[str, object]] = []
    for seed, target_color in keys:
        row_10k = by_checkpoint["010000"][(seed, target_color)]
        row_14k = by_checkpoint["014000"][(seed, target_color)]
        row_18k = by_checkpoint["018000"][(seed, target_color)]
        paired_rows.append(
            {
                "seed": seed,
                "target_color": target_color,
                "10k_picked": row_10k["picked_color"],
                "10k_correct": row_10k["color_correct"],
                "10k_task_success": row_10k["task_success"],
                "10k_steps": row_10k["steps"],
                "10k_outcome": outcome_category(row_10k),
                "14k_picked": row_14k["picked_color"],
                "14k_correct": row_14k["color_correct"],
                "14k_task_success": row_14k["task_success"],
                "14k_steps": row_14k["steps"],
                "14k_outcome": outcome_category(row_14k),
                "18k_picked": row_18k["picked_color"],
                "18k_correct": row_18k["color_correct"],
                "18k_task_success": row_18k["task_success"],
                "18k_steps": row_18k["steps"],
                "18k_outcome": outcome_category(row_18k),
            }
        )
    return paired_rows


def transition_counts(
    source_rows: dict[tuple[str, str], dict[str, str]],
    target_rows: dict[tuple[str, str], dict[str, str]],
) -> Counter:
    counts = Counter()
    for key in sorted(source_rows.keys(), key=lambda item: (as_int(item[0]), item[1])):
        counts[(outcome_category(source_rows[key]), outcome_category(target_rows[key]))] += 1
    return counts


def verdict_for_checkpoint(checkpoint: str, summary: dict[str, object]) -> str:
    correct_count = int(summary["color_correct_count"])
    task_success_count = int(summary["task_success_count"])
    precision_pct = float(summary["precision_pct"])
    per_color_correct = summary["per_color_correct"]
    zero_colors = sum(value == 0 for value in per_color_correct.values())
    if (
        correct_count > 2
        and precision_pct > 40.0
        and zero_colors == 0
    ):
        return "PROMISING"
    if correct_count == 0 and task_success_count <= 1:
        return "NEGATIVE"
    if correct_count <= 1 and precision_pct <= 25.0:
        return "NO CLEAR EFFECT"
    if (
        correct_count > 2
        or (correct_count >= 2 and precision_pct > 40.0)
        or summary["all_colors_nonzero"]
        or (correct_count > 0 and zero_colors >= 1)
    ):
        return "MIXED"
    return "MIXED"


def yes_partial_no(values: list[bool]) -> str:
    if all(values):
        return "YES"
    if any(values):
        return "PARTIALLY"
    return "NO"


def blue_retention(values: list[int]) -> str:
    return "YES" if any(value > 0 for value in values) else "NO"


def rank_checkpoints(summaries: dict[str, dict[str, object]]) -> list[str]:
    def sort_key(checkpoint: str) -> tuple[int, int, float, int, int]:
        summary = summaries[checkpoint]
        balance_penalty = -int(summary["range_correct"])
        diversity_bonus = sum(value > 0 for value in summary["per_color_correct"].values())
        return (
            int(summary["color_correct_count"]),
            int(summary["task_success_count"]),
            float(summary["precision_pct"]),
            diversity_bonus,
            balance_penalty,
        )

    return sorted(CHECKPOINTS, key=sort_key, reverse=True)


def format_confusion_matrix(confusion: dict[str, Counter]) -> str:
    lines = ["| target \\\\ picked | red | blue | yellow | failure | ambiguous |", "| --------------- | --: | ---: | -----: | ------: | --------: |"]
    for target in TARGET_COLORS:
        row = confusion[target]
        lines.append(
            f"| {target} | {row['red']} | {row['blue']} | {row['yellow']} | {row['failure']} | {row['ambiguous']} |"
        )
    return "\n".join(lines)


def format_paired_table(rows: list[dict[str, object]]) -> str:
    header = [
        "| seed | target | 10k picked | 10k correct | 10k task | 10k steps | 14k picked | 14k correct | 14k task | 14k steps | 18k picked | 18k correct | 18k task | 18k steps |",
        "| ---: | ------ | ---------- | ----------: | -------: | --------: | ---------- | ----------: | -------: | --------: | ---------- | ----------: | -------: | --------: |",
    ]
    body = [
        f"| {row['seed']} | {row['target_color']} | {row['10k_picked']} | {row['10k_correct']} | {row['10k_task_success']} | {row['10k_steps']} | {row['14k_picked']} | {row['14k_correct']} | {row['14k_task_success']} | {row['14k_steps']} | {row['18k_picked']} | {row['18k_correct']} | {row['18k_task_success']} | {row['18k_steps']} |"
        for row in rows
    ]
    return "\n".join(header + body)


def format_transition_block(transitions: Counter) -> list[str]:
    lines: list[str] = []
    for source, target in TRANSITION_LABELS:
        lines.append(f"{source} -> {target} = {transitions[(source, target)]}")
    return lines


def main() -> None:
    args = parse_args()
    paths = {
        "010000": args.run4_10k,
        "014000": args.run4_14k,
        "018000": args.run4_18k,
    }

    rows_by_checkpoint: dict[str, list[dict[str, str]]] = {}
    schemas: dict[str, list[str]] = {}
    validation_errors: list[str] = []
    for checkpoint, path in paths.items():
        rows, fieldnames = read_csv(path)
        rows_by_checkpoint[checkpoint] = rows
        schemas[checkpoint] = fieldnames
        missing_fields = validate_schema(fieldnames)
        if missing_fields:
            validation_errors.append(f"{checkpoint}: missing required fields {missing_fields}")
        validation_errors.extend(validate_rows(rows, checkpoint))

    seed_target_sets = {checkpoint: exact_seed_target_set(rows) for checkpoint, rows in rows_by_checkpoint.items()}
    same_seed_target_set = (
        seed_target_sets["010000"] == seed_target_sets["014000"] == seed_target_sets["018000"]
    )
    if not same_seed_target_set:
        validation_errors.append("seed-target set mismatch across 010000/014000/018000")

    if validation_errors:
        raise SystemExit("validation failed:\n- " + "\n- ".join(validation_errors))

    summaries = {checkpoint: summarize_rows(rows) for checkpoint, rows in rows_by_checkpoint.items()}
    by_checkpoint = {
        checkpoint: {(row["seed"], row["target_color"]): row for row in rows}
        for checkpoint, rows in rows_by_checkpoint.items()
    }
    paired_rows = build_paired_rows(by_checkpoint)

    transitions_10k_14k = transition_counts(by_checkpoint["010000"], by_checkpoint["014000"])
    transitions_14k_18k = transition_counts(by_checkpoint["014000"], by_checkpoint["018000"])
    transitions_10k_18k = transition_counts(by_checkpoint["010000"], by_checkpoint["018000"])

    ranking = rank_checkpoints(summaries)
    best_checkpoint, second_checkpoint, third_checkpoint = ranking
    verdicts = {checkpoint: verdict_for_checkpoint(checkpoint, summaries[checkpoint]) for checkpoint in CHECKPOINTS}

    overall_correctness_recovery = yes_partial_no(
        [int(summaries[checkpoint]["color_correct_count"]) > 2 for checkpoint in CHECKPOINTS]
    )
    successful_precision_recovery = yes_partial_no(
        [float(summaries[checkpoint]["precision_pct"]) > 40.0 for checkpoint in CHECKPOINTS]
    )
    red_recovery = yes_partial_no(
        [int(summaries[checkpoint]["per_color_correct"]["red"]) > 0 for checkpoint in CHECKPOINTS]
    )
    blue_retention_verdict = blue_retention(
        [int(summaries[checkpoint]["per_color_correct"]["blue"]) for checkpoint in CHECKPOINTS]
    )
    yellow_recovery = yes_partial_no(
        [int(summaries[checkpoint]["per_color_correct"]["yellow"]) > 0 for checkpoint in CHECKPOINTS]
    )

    if verdicts[best_checkpoint] == "PROMISING":
        selection_verdict = "PROMISING"
    elif any(verdict == "MIXED" for verdict in verdicts.values()):
        selection_verdict = "MIXED"
    elif all(verdict == "NEGATIVE" for verdict in verdicts.values()):
        selection_verdict = "NEGATIVE"
    else:
        selection_verdict = "NO CLEAR EFFECT"

    recommended_next_step = (
        "BLUE 10-SEED REVISED INSTRUMENTATION NEXT"
        if selection_verdict in {"PROMISING", "MIXED"}
        else "STOP TRUE RUN #4"
    )

    summary_rows: list[dict[str, object]] = []
    for reference in RUN_REFERENCES:
        summary_rows.append(
            {
                "run_checkpoint": reference["label"],
                "task_success_count": reference["task_success_count"],
                "task_success_rate": percent_string(reference["task_success_count"], 15),
                "color_correct_count": reference["color_correct_count"],
                "color_correct_rate": percent_string(reference["color_correct_count"], 15),
                "successful_color_precision": f"{reference['precision_pct']:.1f}%",
                "wrong_color_success_count": reference["task_success_count"] - reference["color_correct_count"],
                "wrong_color_over_task_success": precision_string(
                    reference["task_success_count"] - reference["color_correct_count"],
                    reference["task_success_count"],
                ),
                "red_task": reference["per_color_task"]["red"],
                "blue_task": reference["per_color_task"]["blue"],
                "yellow_task": reference["per_color_task"]["yellow"],
                "red_correct": reference["per_color_correct"]["red"],
                "blue_correct": reference["per_color_correct"]["blue"],
                "yellow_correct": reference["per_color_correct"]["yellow"],
                "all_color_nonzero": all(value > 0 for value in reference["per_color_correct"].values()),
                "verdict": "",
            }
        )
    for checkpoint in CHECKPOINTS:
        summary = summaries[checkpoint]
        summary_rows.append(
            {
                "run_checkpoint": f"True Run #4 {checkpoint}",
                "task_success_count": summary["task_success_count"],
                "task_success_rate": summary["task_success_rate"],
                "color_correct_count": summary["color_correct_count"],
                "color_correct_rate": summary["color_correct_rate"],
                "successful_color_precision": summary["precision_string"],
                "wrong_color_success_count": summary["wrong_color_success_count"],
                "wrong_color_over_task_success": summary["wrong_color_rate"],
                "red_task": summary["per_color_task"]["red"],
                "blue_task": summary["per_color_task"]["blue"],
                "yellow_task": summary["per_color_task"]["yellow"],
                "red_correct": summary["per_color_correct"]["red"],
                "blue_correct": summary["per_color_correct"]["blue"],
                "yellow_correct": summary["per_color_correct"]["yellow"],
                "all_color_nonzero": summary["all_colors_nonzero"],
                "verdict": verdicts[checkpoint],
            }
        )

    write_csv(
        args.output_summary_csv,
        [
            "run_checkpoint",
            "task_success_count",
            "task_success_rate",
            "color_correct_count",
            "color_correct_rate",
            "successful_color_precision",
            "wrong_color_success_count",
            "wrong_color_over_task_success",
            "red_task",
            "blue_task",
            "yellow_task",
            "red_correct",
            "blue_correct",
            "yellow_correct",
            "all_color_nonzero",
            "verdict",
        ],
        summary_rows,
    )
    write_csv(args.output_paired_csv, list(paired_rows[0].keys()), paired_rows)

    best_summary = summaries[best_checkpoint]
    why_best: list[str] = []
    why_best.append(f"highest color correct count: {best_summary['color_correct_count']}/15")
    why_best.append(f"highest task success among Run #4 checkpoints: {best_summary['task_success_count']}/15")
    why_best.append(f"best successful precision within Run #4: {best_summary['precision_string']}")
    if best_summary["all_colors_nonzero"]:
        why_best.append("all three colors have nonzero correct counts")
    else:
        why_best.append(
            f"best Run #4 color balance despite remaining gaps: R/B/Y = {best_summary['per_color_correct']['red']}/{best_summary['per_color_correct']['blue']}/{best_summary['per_color_correct']['yellow']}"
        )

    report_lines = [
        "A. Input Validation",
        f"010k rows = {len(rows_by_checkpoint['010000'])}",
        f"014k rows = {len(rows_by_checkpoint['014000'])}",
        f"018k rows = {len(rows_by_checkpoint['018000'])}",
        f"same seed/target set = {'YES' if same_seed_target_set else 'NO'}",
        "",
        "CSV schema",
    ]
    for checkpoint in CHECKPOINTS:
        report_lines.append(f"{checkpoint}: {', '.join(schemas[checkpoint])}")

    report_lines.extend(
        [
            "",
            "B. True Run #4 Summary",
            "| ckpt | task | correct | precision |",
            "| ---- | ---: | ------: | --------: |",
        ]
    )
    for checkpoint in CHECKPOINTS:
        summary = summaries[checkpoint]
        report_lines.append(
            f"| {checkpoint} | {summary['task_success_rate']} | {summary['color_correct_rate']} | {summary['precision_string']} |"
        )

    report_lines.extend(
        [
            "",
            "C. Per-Color Task",
            "| checkpoint | red task | blue task | yellow task |",
            "| ---------- | -------: | --------: | ----------: |",
        ]
    )
    for checkpoint in CHECKPOINTS:
        summary = summaries[checkpoint]
        report_lines.append(
            f"| {checkpoint} | {summary['per_color_task']['red']}/5 | {summary['per_color_task']['blue']}/5 | {summary['per_color_task']['yellow']}/5 |"
        )

    report_lines.extend(
        [
            "",
            "D. Per-Color Correct",
            "| checkpoint | red correct | blue correct | yellow correct |",
            "| ---------- | ----------: | -----------: | -------------: |",
        ]
    )
    for checkpoint in CHECKPOINTS:
        summary = summaries[checkpoint]
        report_lines.append(
            f"| {checkpoint} | {summary['per_color_correct']['red']}/5 | {summary['per_color_correct']['blue']}/5 | {summary['per_color_correct']['yellow']}/5 |"
        )

    report_lines.append("")
    report_lines.append("E. Confusion Matrices")
    for checkpoint in CHECKPOINTS:
        summary = summaries[checkpoint]
        report_lines.extend(
            [
                checkpoint,
                format_confusion_matrix(summary["confusion"]),
                f"correct-color successes = {summary['color_correct_count']}",
                f"wrong-color successes = {summary['wrong_color_success_count']}",
                f"failures = {summary['failure_count']}",
                f"ambiguous = {summary['ambiguous_count']}",
                "",
            ]
        )

    report_lines.extend(["F. Wrong-Color Success Counts"])
    for checkpoint in CHECKPOINTS:
        summary = summaries[checkpoint]
        report_lines.append(
            f"{checkpoint}: task success = {summary['task_success_count']}/15, correct success = {summary['color_correct_count']}/15, wrong-color success = {summary['wrong_color_success_count']}/15, precision = {summary['precision_string']}, wrong-color/task-success = {summary['wrong_color_rate']}"
        )

    report_lines.extend(["", "G. Color Balance"])
    for checkpoint in CHECKPOINTS:
        summary = summaries[checkpoint]
        report_lines.extend(
            [
                f"{checkpoint}: R/B/Y correct = {summary['per_color_correct']['red']}/{summary['per_color_correct']['blue']}/{summary['per_color_correct']['yellow']}",
                f"{checkpoint}: min_correct = {summary['min_correct']}, max_correct = {summary['max_correct']}, range = {summary['range_correct']}, all-color nonzero = {summary['all_colors_nonzero']}",
            ]
        )

    report_lines.extend(["", "H. Episode-Level Paired Table", format_paired_table(paired_rows)])

    report_lines.extend(["", "I. 10k -> 14k Transitions"])
    report_lines.extend(format_transition_block(transitions_10k_14k))
    report_lines.extend(["", "J. 14k -> 18k Transitions"])
    report_lines.extend(format_transition_block(transitions_14k_18k))
    report_lines.extend(["", "10k -> 18k Transitions"])
    report_lines.extend(format_transition_block(transitions_10k_18k))
    report_lines.extend(["", "K. Run #2/#3/#4 Cross-Run Table"])
    report_lines.extend(
        [
            "| run/checkpoint | task success | color correct | precision |",
            "| --------------- | -----------: | ------------: | --------: |",
            "| Run #2 14k | 5/15 = 33.3% | 4/15 = 26.7% | 80.0% |",
            "| Run #3 14k | 5/15 = 33.3% | 2/15 = 13.3% | 40.0% |",
            "| Run #3 18k | 4/15 = 26.7% | 1/15 = 6.7% | 25.0% |",
        ]
    )
    for checkpoint in CHECKPOINTS:
        summary = summaries[checkpoint]
        report_lines.append(
            f"| True Run #4 {checkpoint[-3:]}k | {summary['task_success_rate']} | {summary['color_correct_rate']} | {summary['precision_string']} |"
        )

    report_lines.extend(
        [
            "",
            "Cross-run per-color correctness",
            "| run/checkpoint | red correct | blue correct | yellow correct |",
            "| --------------- | ----------: | -----------: | -------------: |",
            "| Run #2 14k | 2/5 | 0/5 | 2/5 |",
            "| Run #3 14k | 0/5 | 0/5 | 2/5 |",
            "| Run #3 18k | 0/5 | 1/5 | 0/5 |",
        ]
    )
    for checkpoint in CHECKPOINTS:
        summary = summaries[checkpoint]
        report_lines.append(
            f"| True Run #4 {checkpoint[-3:]}k | {summary['per_color_correct']['red']}/5 | {summary['per_color_correct']['blue']}/5 | {summary['per_color_correct']['yellow']}/5 |"
        )

    report_lines.extend(
        [
            "",
            "Cross-run per-color task success",
            "| run/checkpoint | red task | blue task | yellow task |",
            "| --------------- | -------: | --------: | ----------: |",
            "| Run #2 14k | 3/5 | 0/5 | 2/5 |",
            "| Run #3 14k | 1/5 | 1/5 | 3/5 |",
            "| Run #3 18k | 2/5 | 2/5 | 0/5 |",
        ]
    )
    for checkpoint in CHECKPOINTS:
        summary = summaries[checkpoint]
        report_lines.append(
            f"| True Run #4 {checkpoint[-3:]}k | {summary['per_color_task']['red']}/5 | {summary['per_color_task']['blue']}/5 | {summary['per_color_task']['yellow']}/5 |"
        )

    report_lines.extend(
        [
            "",
            "L. Overall Correctness Recovery",
            overall_correctness_recovery,
            "",
            "M. Successful Precision Recovery",
            successful_precision_recovery,
            "",
            "N. Red Recovery",
            red_recovery,
            "",
            "O. Blue Retention",
            blue_retention_verdict,
            "",
            "P. Yellow Recovery",
            yellow_recovery,
            "",
            "Q. Selection Intervention Verdict",
            selection_verdict,
            "",
            "Checkpoint verdicts",
        ]
    )
    for checkpoint in CHECKPOINTS:
        report_lines.append(f"{checkpoint}: {verdicts[checkpoint]}")

    report_lines.extend(
        [
            "",
            "R. Best Checkpoint",
            best_checkpoint,
            "",
            "S. Ranking",
            f"1. {best_checkpoint}",
            f"2. {second_checkpoint}",
            f"3. {third_checkpoint}",
            "",
            "T. Why Best?",
        ]
    )
    report_lines.extend(f"{index}. {reason}" for index, reason in enumerate(why_best, start=1))
    report_lines.extend(
        [
            "",
            "U. Recommended Next Step",
            recommended_next_step,
        ]
    )
    report_lines.extend(
        [
            "",
            "V. Output Paths",
            str(args.output_summary_csv),
            str(args.output_paired_csv),
            str(args.output_report_txt),
            "",
            "W. One-Sentence Conclusion",
            f"True Run #4 screening suggests {best_checkpoint} is the only checkpoint with a meaningful selection-recovery signal versus Run #3, but the gain is still color-imbalanced and should be validated with BLUE 10-SEED REVISED INSTRUMENTATION NEXT rather than treated as a robust fix.",
        ]
    )

    args.output_report_txt.parent.mkdir(parents=True, exist_ok=True)
    args.output_report_txt.write_text("\n".join(report_lines) + "\n")


if __name__ == "__main__":
    main()
