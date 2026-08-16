#!/usr/bin/env python3
"""Summarize three-color eval CSV into an instruction-vs-picked confusion matrix."""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path


PICKED_LABELS = ("red", "blue", "yellow", "failure", "ambiguous")
INSTRUCTION_LABELS = ("red", "blue", "yellow")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", type=Path)
    return parser.parse_args()


def format_table(matrix: dict[str, Counter]) -> str:
    widths = {
        "instruction": max(len("Instruction"), *(len(label) for label in INSTRUCTION_LABELS)),
        **{label: max(len(label), 11) for label in PICKED_LABELS},
    }
    header = (
        f"{'Instruction':<{widths['instruction']}} "
        + " ".join(f"{label:>{widths[label]}}" for label in PICKED_LABELS)
    )
    rows = [header]
    for instruction in INSTRUCTION_LABELS:
        row = matrix[instruction]
        rows.append(
            f"{instruction:<{widths['instruction']}} "
            + " ".join(f"{row[label]:>{widths[label]}}" for label in PICKED_LABELS)
        )
    return "\n".join(rows)


def main() -> None:
    args = parse_args()
    with args.csv_path.open(newline="") as file:
        rows = list(csv.DictReader(file))

    matrix: dict[str, Counter] = defaultdict(Counter)
    task_successes = 0
    color_correct_successes = 0
    per_instruction_total = Counter()
    per_instruction_correct = Counter()

    for row in rows:
        instruction = row.get("target_color") or "unknown"
        picked_color = row.get("picked_color") or "failure"
        task_success = row.get("task_success", "").lower() == "true"
        color_correct = row.get("color_correct", "").lower() == "true"

        if instruction in INSTRUCTION_LABELS and picked_color in PICKED_LABELS:
            matrix[instruction][picked_color] += 1
            per_instruction_total[instruction] += 1
            per_instruction_correct[instruction] += int(color_correct)

        task_successes += int(task_success)
        color_correct_successes += int(color_correct)

    print(format_table(matrix))
    if rows:
        overall_color_accuracy = color_correct_successes / len(rows)
        task_success_rate = task_successes / len(rows)
        color_correct_success_rate = color_correct_successes / len(rows)
        print()
        print(f"overall_color_accuracy: {overall_color_accuracy:.3f}")
        print(f"task_success_rate: {task_success_rate:.3f}")
        print(f"color_correct_success_rate: {color_correct_success_rate:.3f}")
        for instruction in INSTRUCTION_LABELS:
            total = per_instruction_total[instruction]
            acc = (per_instruction_correct[instruction] / total) if total else 0.0
            print(f"{instruction}_instruction_accuracy: {acc:.3f}")


if __name__ == "__main__":
    main()
