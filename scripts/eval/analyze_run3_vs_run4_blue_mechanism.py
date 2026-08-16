#!/usr/bin/env python3
"""Compare Run #3 18k vs True Run #4 14k blue diagnostic outputs."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

from diagnostic_metrics import compute_blue_funnel


PICKED_LABELS = ("red", "blue", "yellow", "failure", "ambiguous")
STAGE_ORDER = (
    "selection_failure",
    "initial_selection",
    "active_approach",
    "target_lock",
    "relevant_close",
    "target_follow",
    "secure_grasp",
    "blue_lift",
    "blue_transport",
    "correct_success",
)
KEY_SEEDS = ("1005", "1006", "1021", "1024")
CONSISTENCY_FIELDS = ("task_success", "picked_color", "color_correct", "steps")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run3-results-csv", type=Path, required=True)
    parser.add_argument("--run3-summary-v2-csv", type=Path, required=True)
    parser.add_argument("--run4-results-csv", type=Path, required=True)
    parser.add_argument("--run4-summary-v2-csv", type=Path, required=True)
    parser.add_argument("--run4-screening-blue-csv", type=Path, required=True)
    parser.add_argument("--output-comparison-csv", type=Path, required=True)
    parser.add_argument("--output-paired-csv", type=Path, required=True)
    parser.add_argument("--output-report-txt", type=Path, required=True)
    return parser.parse_args()


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as file:
        return list(csv.DictReader(file))


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() == "true"


def normalize_picked(value: object) -> str:
    picked = str(value).strip().lower()
    return picked if picked in PICKED_LABELS else "ambiguous"


def format_rate(numerator: int, denominator: int) -> str:
    if denominator == 0:
        return f"{numerator}/{denominator} = N/A"
    return f"{numerator}/{denominator} = {100.0 * numerator / denominator:.1f}%"


def precision_string(correct: int, task: int) -> str:
    if task == 0:
        return "N/A"
    return f"{100.0 * correct / task:.1f}%"


def ratio_string(numerator: int, denominator: int) -> str:
    if denominator == 0:
        return "N/A"
    return f"{numerator}/{denominator} = {100.0 * numerator / denominator:.1f}%"


def signed_delta(current: int, baseline: int) -> str:
    delta = current - baseline
    return f"{delta:+d}"


def parse_int(value: object) -> int:
    return int(str(value))


def stage_label(row: dict[str, str]) -> str:
    if as_bool(row["color_correct"]):
        return "correct_success"
    if as_bool(row["blue_transport_detected"]):
        return "blue_transport"
    if as_bool(row["blue_lift_detected"]):
        return "blue_lift"
    if as_bool(row["secure_grasp_candidate"]):
        return "secure_grasp"
    if as_bool(row["target_follow_detected"]):
        return "target_follow"
    if row["task_relevant_close_step"] != "":
        return "relevant_close"
    if as_bool(row["stable_target_lock_detected"]):
        if as_bool(row["active_target_approach_detected"]):
            return "active_approach"
        if as_bool(row["selection_correct"]):
            return "target_lock"
    if as_bool(row["active_target_approach_detected"]):
        return "active_approach"
    if as_bool(row["selection_correct"]):
        return "initial_selection"
    return "selection_failure"


def verify_blue_seed_set(rows: list[dict[str, str]], expected: list[str], label: str) -> None:
    blue_rows = [row for row in rows if row["target_color"] == "blue"]
    actual = sorted(row["seed"] for row in blue_rows)
    if actual != sorted(expected):
        raise SystemExit(f"{label} blue seed set mismatch: actual={actual} expected={sorted(expected)}")


def summarize_endpoint(results_rows: list[dict[str, str]]) -> dict[str, object]:
    blue_rows = [row for row in results_rows if row["target_color"] == "blue"]
    picked = Counter(normalize_picked(row["picked_color"]) for row in blue_rows)
    task_success = sum(as_bool(row["task_success"]) for row in blue_rows)
    color_correct = sum(as_bool(row["color_correct"]) for row in blue_rows)
    wrong_color_success = sum(as_bool(row["task_success"]) and not as_bool(row["color_correct"]) for row in blue_rows)
    return {
        "episodes": len(blue_rows),
        "task_success": task_success,
        "color_correct": color_correct,
        "precision": precision_string(color_correct, task_success),
        "picked": picked,
        "wrong_color_success": wrong_color_success,
    }


def summarize_mechanism(summary_rows: list[dict[str, str]]) -> dict[str, object]:
    blue_rows = [row for row in summary_rows if row["target_color"] == "blue"]
    funnel = compute_blue_funnel(blue_rows)
    true_grasp_failures = [
        row["seed"]
        for row in blue_rows
        if as_bool(row["active_target_approach_detected"])
        and row["task_relevant_close_step"] != ""
        and not as_bool(row["target_follow_detected"])
    ]
    selection_failure_family = {
        "no_correct_initial_selection": sum(not as_bool(row["selection_correct"]) for row in blue_rows),
        "active_approach_missing": sum(
            as_bool(row["selection_correct"]) and not as_bool(row["active_target_approach_detected"])
            for row in blue_rows
        ),
        "approach_but_no_relevant_close": sum(
            as_bool(row["active_target_approach_detected"]) and row["task_relevant_close_step"] == ""
            for row in blue_rows
        ),
    }
    return {
        "funnel": funnel,
        "true_grasp_failures": true_grasp_failures,
        "selection_failure_family": selection_failure_family,
    }


def conditional_metrics(funnel: dict[str, int]) -> dict[str, str]:
    return {
        "relevant_close_over_active_approach": ratio_string(funnel["task_relevant_close"], funnel["active_blue_approach"]),
        "target_follow_over_relevant_close": ratio_string(funnel["target_follow"], funnel["task_relevant_close"]),
        "secure_grasp_over_target_follow": ratio_string(funnel["secure_grasp_candidate"], funnel["target_follow"]),
        "blue_lift_over_secure_grasp": ratio_string(funnel["blue_lift"], funnel["secure_grasp_candidate"]),
        "blue_transport_over_blue_lift": ratio_string(funnel["blue_transport"], funnel["blue_lift"]),
    }


def keyed_by_seed(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    return {row["seed"]: row for row in rows if row["target_color"] == "blue"}


def stage_change(run3_row: dict[str, str], run4_row: dict[str, str]) -> str:
    return f"{stage_label(run3_row)} -> {stage_label(run4_row)}"


def mechanism_selection_verdict(run3: dict[str, object], run4: dict[str, object]) -> str:
    run3_funnel = run3["funnel"]
    run4_funnel = run4["funnel"]
    initial_delta = run4_funnel["correct_initial_selection"] - run3_funnel["correct_initial_selection"]
    active_delta = run4_funnel["active_blue_approach"] - run3_funnel["active_blue_approach"]
    lock_delta = run4_funnel["stable_target_lock"] - run3_funnel["stable_target_lock"]
    if initial_delta > 0 and active_delta > 0:
        return "CLEAR POSITIVE"
    if initial_delta > 0 or active_delta > 0 or lock_delta > 0:
        return "PARTIAL POSITIVE"
    if initial_delta == 0 and active_delta == 0:
        return "NEUTRAL"
    return "NEGATIVE"


def grasp_preservation_verdict(run3: dict[str, object], run4: dict[str, object]) -> str:
    run3_funnel = run3["funnel"]
    run4_funnel = run4["funnel"]
    run3_follow_close = run3_funnel["target_follow"] == run3_funnel["task_relevant_close"]
    run3_secure_follow = run3_funnel["secure_grasp_candidate"] == run3_funnel["target_follow"]
    run4_follow_close = run4_funnel["target_follow"] == run4_funnel["task_relevant_close"]
    run4_secure_follow = run4_funnel["secure_grasp_candidate"] == run4_funnel["target_follow"]
    if run4_funnel["task_relevant_close"] == 0:
        return "INSUFFICIENT"
    if run4_follow_close and run4_secure_follow and (run3_follow_close or run3_secure_follow):
        return "PRESERVED"
    if run4_funnel["target_follow"] > 0 and run4_funnel["secure_grasp_candidate"] > 0:
        return "PARTIALLY PRESERVED"
    return "REGRESSED"


def tradeoff_verdict(selection_verdict: str, grasp_verdict: str) -> str:
    if selection_verdict in {"CLEAR POSITIVE", "PARTIAL POSITIVE"} and grasp_verdict == "PRESERVED":
        return "RESOLVED TRADEOFF"
    if selection_verdict in {"CLEAR POSITIVE", "PARTIAL POSITIVE"} and grasp_verdict in {"PARTIALLY PRESERVED", "REGRESSED"}:
        return "SELECTION GAIN / GRASP REGRESSION"
    if selection_verdict in {"NEUTRAL", "NEGATIVE"} and grasp_verdict in {"PRESERVED", "PARTIALLY PRESERVED"}:
        return "GRASP PRESERVED / NO SELECTION GAIN"
    if selection_verdict == "NEGATIVE" and grasp_verdict == "REGRESSED":
        return "BOTH REGRESS"
    return "MIXED / INSUFFICIENT"


def combined_verdict(selection_verdict: str, grasp_verdict: str, run4_endpoint: dict[str, object], run3_endpoint: dict[str, object]) -> str:
    if selection_verdict in {"CLEAR POSITIVE", "PARTIAL POSITIVE"} and grasp_verdict == "PRESERVED":
        return "PROMISING MECHANISM COMBINATION"
    if selection_verdict in {"CLEAR POSITIVE", "PARTIAL POSITIVE"} and grasp_verdict in {"PARTIALLY PRESERVED", "REGRESSED"}:
        return "MIXED TRADEOFF"
    if selection_verdict == "NEGATIVE" and grasp_verdict == "REGRESSED":
        return "NEGATIVE"
    if run4_endpoint["color_correct"] <= run3_endpoint["color_correct"] and selection_verdict == "NEUTRAL":
        return "NO CLEAR BENEFIT"
    return "MIXED TRADEOFF"


def expansion_decision(selection_verdict: str, grasp_verdict: str, combined: str) -> str:
    if selection_verdict == "CLEAR POSITIVE" and grasp_verdict == "PRESERVED":
        return "EXPAND TRUE RUN #4 014000 TO 30EP"
    if combined == "NEGATIVE":
        return "STOP TRUE RUN #4"
    return "DO NOT EXPAND YET"


def consistency_diffs(run4_results: list[dict[str, str]], screening_rows: list[dict[str, str]]) -> list[str]:
    run4_by_seed = {row["seed"]: row for row in run4_results if row["target_color"] == "blue"}
    screening_by_seed = {row["seed"]: row for row in screening_rows if row["target_color"] == "blue"}
    diffs: list[str] = []
    for seed in ("1020", "1021", "1022", "1023", "1024"):
        run4_row = run4_by_seed.get(seed)
        screening_row = screening_by_seed.get(seed)
        if run4_row is None or screening_row is None:
            diffs.append(f"{seed}: missing row in run4 diagnostic or screening blue slice")
            continue
        changed = [field for field in CONSISTENCY_FIELDS if run4_row[field] != screening_row[field]]
        if changed:
            details = ", ".join(
                f"{field}: diagnostic={run4_row[field]} screening={screening_row[field]}"
                for field in changed
            )
            diffs.append(f"{seed}: {details}")
    return diffs


def build_paired_rows(run3_rows: list[dict[str, str]], run4_rows: list[dict[str, str]]) -> list[dict[str, object]]:
    run3_by_seed = keyed_by_seed(run3_rows)
    run4_by_seed = keyed_by_seed(run4_rows)
    seeds = sorted(run3_by_seed.keys() | run4_by_seed.keys(), key=int)
    paired_rows: list[dict[str, object]] = []
    for seed in seeds:
        run3_row = run3_by_seed[seed]
        run4_row = run4_by_seed[seed]
        paired_rows.append(
            {
                "seed": seed,
                "run3_stage": stage_label(run3_row),
                "run4_stage": stage_label(run4_row),
                "change": stage_change(run3_row, run4_row),
                "run3_selection_correct": run3_row["selection_correct"],
                "run4_selection_correct": run4_row["selection_correct"],
                "run3_active_target_approach": run3_row["active_target_approach_detected"],
                "run4_active_target_approach": run4_row["active_target_approach_detected"],
                "run3_task_relevant_close_step": run3_row["task_relevant_close_step"],
                "run4_task_relevant_close_step": run4_row["task_relevant_close_step"],
                "run3_target_follow": run3_row["target_follow_detected"],
                "run4_target_follow": run4_row["target_follow_detected"],
                "run3_secure_grasp": run3_row["secure_grasp_candidate"],
                "run4_secure_grasp": run4_row["secure_grasp_candidate"],
                "run3_blue_lift": run3_row["blue_lift_detected"],
                "run4_blue_lift": run4_row["blue_lift_detected"],
                "run3_blue_transport": run3_row["blue_transport_detected"],
                "run4_blue_transport": run4_row["blue_transport_detected"],
                "run3_task_success": run3_row["task_success"],
                "run4_task_success": run4_row["task_success"],
                "run3_picked_color": run3_row["picked_color"],
                "run4_picked_color": run4_row["picked_color"],
                "run3_color_correct": run3_row["color_correct"],
                "run4_color_correct": run4_row["color_correct"],
                "run3_steps": run3_row["steps"],
                "run4_steps": run4_row["steps"],
            }
        )
    return paired_rows


def main() -> None:
    args = parse_args()

    run3_results = load_csv(args.run3_results_csv)
    run3_summary = load_csv(args.run3_summary_v2_csv)
    run4_results = load_csv(args.run4_results_csv)
    run4_summary = load_csv(args.run4_summary_v2_csv)
    run4_screening = load_csv(args.run4_screening_blue_csv)

    expected_seeds = ["1005", "1006", "1007", "1008", "1009", "1020", "1021", "1022", "1023", "1024"]
    verify_blue_seed_set(run3_results, expected_seeds, "run3 results")
    verify_blue_seed_set(run3_summary, expected_seeds, "run3 summary v2")
    verify_blue_seed_set(run4_results, expected_seeds, "run4 results")
    verify_blue_seed_set(run4_summary, expected_seeds, "run4 summary v2")

    run3_endpoint = summarize_endpoint(run3_results)
    run4_endpoint = summarize_endpoint(run4_results)
    run3_mech = summarize_mechanism(run3_summary)
    run4_mech = summarize_mechanism(run4_summary)
    run3_cond = conditional_metrics(run3_mech["funnel"])
    run4_cond = conditional_metrics(run4_mech["funnel"])

    selection_verdict = mechanism_selection_verdict(run3_mech, run4_mech)
    grasp_verdict = grasp_preservation_verdict(run3_mech, run4_mech)
    tradeoff = tradeoff_verdict(selection_verdict, grasp_verdict)
    combined = combined_verdict(selection_verdict, grasp_verdict, run4_endpoint, run3_endpoint)
    decision = expansion_decision(selection_verdict, grasp_verdict, combined)
    paired_rows = build_paired_rows(run3_summary, run4_summary)
    diffs = consistency_diffs(run4_results, run4_screening)

    comparison_rows = [
        {
            "metric": "correct initial selection",
            "run3_18k": f"{run3_mech['funnel']['correct_initial_selection']}/10",
            "run4_14k": f"{run4_mech['funnel']['correct_initial_selection']}/10",
            "delta": signed_delta(run4_mech["funnel"]["correct_initial_selection"], run3_mech["funnel"]["correct_initial_selection"]),
        },
        {
            "metric": "active blue approach",
            "run3_18k": f"{run3_mech['funnel']['active_blue_approach']}/10",
            "run4_14k": f"{run4_mech['funnel']['active_blue_approach']}/10",
            "delta": signed_delta(run4_mech["funnel"]["active_blue_approach"], run3_mech["funnel"]["active_blue_approach"]),
        },
        {
            "metric": "stable target lock",
            "run3_18k": f"{run3_mech['funnel']['stable_target_lock']}/10",
            "run4_14k": f"{run4_mech['funnel']['stable_target_lock']}/10",
            "delta": signed_delta(run4_mech["funnel"]["stable_target_lock"], run3_mech["funnel"]["stable_target_lock"]),
        },
        {
            "metric": "relevant close / active approach",
            "run3_18k": f"{run3_mech['funnel']['task_relevant_close']}/{run3_mech['funnel']['active_blue_approach']}",
            "run4_14k": f"{run4_mech['funnel']['task_relevant_close']}/{run4_mech['funnel']['active_blue_approach']}",
            "delta": "",
        },
        {
            "metric": "target follow / relevant close",
            "run3_18k": run3_cond["target_follow_over_relevant_close"],
            "run4_14k": run4_cond["target_follow_over_relevant_close"],
            "delta": "",
        },
        {
            "metric": "secure grasp / target follow",
            "run3_18k": run3_cond["secure_grasp_over_target_follow"],
            "run4_14k": run4_cond["secure_grasp_over_target_follow"],
            "delta": "",
        },
        {
            "metric": "blue lift / secure grasp",
            "run3_18k": run3_cond["blue_lift_over_secure_grasp"],
            "run4_14k": run4_cond["blue_lift_over_secure_grasp"],
            "delta": "",
        },
        {
            "metric": "blue transport / blue lift",
            "run3_18k": run3_cond["blue_transport_over_blue_lift"],
            "run4_14k": run4_cond["blue_transport_over_blue_lift"],
            "delta": "",
        },
        {
            "metric": "task success",
            "run3_18k": f"{run3_endpoint['task_success']}/10",
            "run4_14k": f"{run4_endpoint['task_success']}/10",
            "delta": signed_delta(run4_endpoint["task_success"], run3_endpoint["task_success"]),
        },
        {
            "metric": "color correct",
            "run3_18k": f"{run3_endpoint['color_correct']}/10",
            "run4_14k": f"{run4_endpoint['color_correct']}/10",
            "delta": signed_delta(run4_endpoint["color_correct"], run3_endpoint["color_correct"]),
        },
        {
            "metric": "wrong-color success",
            "run3_18k": f"{run3_endpoint['wrong_color_success']}/10",
            "run4_14k": f"{run4_endpoint['wrong_color_success']}/10",
            "delta": signed_delta(run4_endpoint["wrong_color_success"], run3_endpoint["wrong_color_success"]),
        },
    ]
    write_csv(args.output_comparison_csv, list(comparison_rows[0].keys()), comparison_rows)
    write_csv(args.output_paired_csv, list(paired_rows[0].keys()), paired_rows)

    report_lines = [
        "A. True Run #4 14k Blue Endpoint",
        f"task success = {run4_endpoint['task_success']}/10 = {100.0 * run4_endpoint['task_success'] / 10:.1f}%",
        f"color correct = {run4_endpoint['color_correct']}/10 = {100.0 * run4_endpoint['color_correct'] / 10:.1f}%",
        f"precision = {run4_endpoint['precision']}",
        f"picked-color distribution = red {run4_endpoint['picked']['red']}, blue {run4_endpoint['picked']['blue']}, yellow {run4_endpoint['picked']['yellow']}, failure {run4_endpoint['picked']['failure']}, ambiguous {run4_endpoint['picked']['ambiguous']}",
        f"wrong-color success = {run4_endpoint['wrong_color_success']}/10",
        "",
        "B. True Run #4 14k Funnel",
    ]
    for key, label in (
        ("episodes", "blue episodes"),
        ("correct_initial_selection", "correct initial selection"),
        ("active_blue_approach", "active blue approach"),
        ("stable_target_lock", "stable target lock"),
        ("task_relevant_close", "task-relevant close"),
        ("target_follow", "target follow"),
        ("secure_grasp_candidate", "secure grasp candidate"),
        ("blue_lift", "blue lift"),
        ("blue_transport", "blue transport"),
        ("blue_color_correct_success", "blue color-correct success"),
    ):
        denom = 10 if key != "episodes" else 10
        report_lines.append(f"{label} = {run4_mech['funnel'][key]}/10 = {100.0 * run4_mech['funnel'][key] / denom:.1f}%")

    report_lines.extend(
        [
            "",
            "C. Run #3 vs Run #4 Funnel Table",
            "| stage | Run #3 18k | True Run #4 14k |",
            "| ----- | ---------: | --------------: |",
        ]
    )
    for key, label in (
        ("episodes", "blue episodes"),
        ("correct_initial_selection", "initial selection"),
        ("active_blue_approach", "active approach"),
        ("stable_target_lock", "target lock"),
        ("task_relevant_close", "relevant close"),
        ("target_follow", "follow"),
        ("secure_grasp_candidate", "secure grasp"),
        ("blue_lift", "lift"),
        ("blue_transport", "transport"),
        ("blue_color_correct_success", "correct success"),
    ):
        report_lines.append(
            f"| {label} | {run3_mech['funnel'][key]} | {run4_mech['funnel'][key]} |"
        )

    report_lines.extend(
        [
            "",
            "D. Selection Comparison",
            "| metric | Run #3 18k | True Run #4 14k | delta |",
            "| ----- | ---------: | --------------: | ----: |",
            f"| correct initial selection | {run3_mech['funnel']['correct_initial_selection']}/10 | {run4_mech['funnel']['correct_initial_selection']}/10 | {signed_delta(run4_mech['funnel']['correct_initial_selection'], run3_mech['funnel']['correct_initial_selection'])} |",
            f"| active blue approach | {run3_mech['funnel']['active_blue_approach']}/10 | {run4_mech['funnel']['active_blue_approach']}/10 | {signed_delta(run4_mech['funnel']['active_blue_approach'], run3_mech['funnel']['active_blue_approach'])} |",
            f"| stable target lock | {run3_mech['funnel']['stable_target_lock']}/10 | {run4_mech['funnel']['stable_target_lock']}/10 | {signed_delta(run4_mech['funnel']['stable_target_lock'], run3_mech['funnel']['stable_target_lock'])} |",
            f"| relevant close / active approach | {run3_mech['funnel']['task_relevant_close']}/{run3_mech['funnel']['active_blue_approach']} | {run4_mech['funnel']['task_relevant_close']}/{run4_mech['funnel']['active_blue_approach']} |  |",
            "",
            "E. Grasp Preservation Comparison",
            f"Run #3: relevant close = {run3_mech['funnel']['task_relevant_close']}, follow = {run3_mech['funnel']['target_follow']}, secure grasp = {run3_mech['funnel']['secure_grasp_candidate']}, lift = {run3_mech['funnel']['blue_lift']}, transport = {run3_mech['funnel']['blue_transport']}",
            f"Run #4: relevant close = {run4_mech['funnel']['task_relevant_close']}, follow = {run4_mech['funnel']['target_follow']}, secure grasp = {run4_mech['funnel']['secure_grasp_candidate']}, lift = {run4_mech['funnel']['blue_lift']}, transport = {run4_mech['funnel']['blue_transport']}",
            "",
            "F. Primary Conditional Metrics",
            f"Run #3 relevant_close / active_approach = {run3_cond['relevant_close_over_active_approach']}",
            f"Run #4 relevant_close / active_approach = {run4_cond['relevant_close_over_active_approach']}",
            f"Run #3 target_follow / task_relevant_close = {run3_cond['target_follow_over_relevant_close']}",
            f"Run #4 target_follow / task_relevant_close = {run4_cond['target_follow_over_relevant_close']}",
            f"Run #3 secure_grasp / target_follow = {run3_cond['secure_grasp_over_target_follow']}",
            f"Run #4 secure_grasp / target_follow = {run4_cond['secure_grasp_over_target_follow']}",
            f"Run #3 blue_lift / secure_grasp = {run3_cond['blue_lift_over_secure_grasp']}",
            f"Run #4 blue_lift / secure_grasp = {run4_cond['blue_lift_over_secure_grasp']}",
            f"Run #3 blue_transport / blue_lift = {run3_cond['blue_transport_over_blue_lift']}",
            f"Run #4 blue_transport / blue_lift = {run4_cond['blue_transport_over_blue_lift']}",
            "",
            "G. True Grasp-Formation Failures",
            f"Run #3 = {len(run3_mech['true_grasp_failures'])} ({', '.join(run3_mech['true_grasp_failures']) or '-'})",
            f"Run #4 = {len(run4_mech['true_grasp_failures'])} ({', '.join(run4_mech['true_grasp_failures']) or '-'})",
            "",
            "H. Selection Failure Family",
            f"Run #4 no correct initial selection = {run4_mech['selection_failure_family']['no_correct_initial_selection']}",
            f"Run #4 active approach missing = {run4_mech['selection_failure_family']['active_approach_missing']}",
            f"Run #4 approach but no relevant close = {run4_mech['selection_failure_family']['approach_but_no_relevant_close']}",
            "",
            "I. Paired Seed Table",
            "| seed | Run #3 18k stage | Run #4 14k stage | change |",
            "| ---: | ---------------- | ---------------- | ------ |",
        ]
    )
    for row in paired_rows:
        report_lines.append(
            f"| {row['seed']} | {row['run3_stage']} | {row['run4_stage']} | {row['change']} |"
        )

    report_lines.extend(["", "J. Key Seed Changes"])
    paired_by_seed = {row["seed"]: row for row in paired_rows}
    for seed in KEY_SEEDS:
        row = paired_by_seed[seed]
        report_lines.append(
            f"{seed}: Run #3 = {row['run3_stage']}, Run #4 = {row['run4_stage']}, change = {row['change']}"
        )

    report_lines.extend(
        [
            "",
            "K. Wrong-Color Behavior",
            f"Run #3 picked-color distribution = red {run3_endpoint['picked']['red']}, blue {run3_endpoint['picked']['blue']}, yellow {run3_endpoint['picked']['yellow']}, failure {run3_endpoint['picked']['failure']}, ambiguous {run3_endpoint['picked']['ambiguous']}",
            f"Run #4 picked-color distribution = red {run4_endpoint['picked']['red']}, blue {run4_endpoint['picked']['blue']}, yellow {run4_endpoint['picked']['yellow']}, failure {run4_endpoint['picked']['failure']}, ambiguous {run4_endpoint['picked']['ambiguous']}",
            "",
            "L. Screening-vs-Diagnostic Consistency",
        ]
    )
    if diffs:
        report_lines.extend(diffs)
    else:
        report_lines.append("1020..1024 match the 15ep screening blue slice on task_success, picked_color, color_correct, and steps.")

    report_lines.extend(
        [
            "",
            "M. Selection Mechanism Verdict",
            selection_verdict,
            "",
            "N. Grasp Preservation Verdict",
            grasp_verdict,
            "",
            "O. Tradeoff Verdict",
            tradeoff,
            "",
            "P. Combined Run #4 Verdict",
            combined,
            "",
            "Q. 30ep Decision",
            decision,
            "",
            "R. Exact Commands",
            "1. Run blue diagnostic eval for True Run #4 14k.",
            "2. Run offline v2 analyzer.",
            "3. Run this comparison script.",
            "",
            "S. Output Paths",
            str(args.output_comparison_csv),
            str(args.output_paired_csv),
            str(args.output_report_txt),
            "",
            "T. One-Sentence Conclusion",
            f"True Run #4 14k {combined.lower()} versus Run #3 18k, so the current 10-seed mechanism evidence {('supports 30ep expansion' if decision == 'EXPAND TRUE RUN #4 014000 TO 30EP' else 'does not yet support 30ep expansion')} within this sample.",
        ]
    )

    args.output_report_txt.parent.mkdir(parents=True, exist_ok=True)
    args.output_report_txt.write_text("\n".join(report_lines) + "\n")


if __name__ == "__main__":
    main()
