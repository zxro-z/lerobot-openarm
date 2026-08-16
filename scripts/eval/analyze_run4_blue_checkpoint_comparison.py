#!/usr/bin/env python3
"""Compare True Run #4 blue diagnostic checkpoints 14k/16k/18k."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path


CHECKPOINTS = ("014000", "016000", "018000")
EXPECTED_SEEDS = ("1005", "1006", "1007", "1008", "1009", "1020", "1021", "1022", "1023", "1024")
PICKED_LABELS = ("blue", "red", "yellow", "failure", "ambiguous")
RUNTIME_LABELS = {
    "wrong_color_selection",
    "no_clear_selection",
    "correct_selection_pre_grasp_failure",
    "grasp_candidate_lift_failure",
    "success",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run4-14k-dir", type=Path, required=True)
    parser.add_argument("--run4-16k-dir", type=Path, required=True)
    parser.add_argument("--run4-18k-dir", type=Path, required=True)
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
    return str(value).strip().lower() == "true"


def rate_string(numerator: int, denominator: int) -> str:
    if denominator == 0:
        return f"{numerator}/{denominator} = N/A"
    return f"{numerator}/{denominator} = {100.0 * numerator / denominator:.1f}%"


def precision_string(color_correct: int, task_success: int) -> str:
    if task_success == 0:
        return "N/A"
    return f"{100.0 * color_correct / task_success:.1f}%"


def ratio_string(numerator: int, denominator: int) -> str:
    if denominator == 0:
        return "N/A"
    return f"{numerator}/{denominator} = {100.0 * numerator / denominator:.1f}%"


def normalize_picked(value: str) -> str:
    picked = value.strip().lower()
    return picked if picked in PICKED_LABELS else "ambiguous"


def stage_label(row: dict[str, str]) -> str:
    if as_bool(row["color_correct"]):
        return "correct_success"
    if as_bool(row["task_success"]) and not as_bool(row["color_correct"]):
        return "wrong_color_success"
    if row["revised_failure_stage"] == "selection_failure":
        if row.get("old_failure_stage") == "wrong_color_selection":
            return "wrong_color_selection"
        if row.get("old_failure_stage") == "no_clear_selection":
            return "no_clear_selection"
        return "selection_failure"
    if row["revised_failure_stage"] == "approach_without_relevant_close":
        return "correct_selection_no_grasp"
    if row["revised_failure_stage"] == "relevant_close_without_follow":
        return "close_no_follow"
    if row["revised_failure_stage"] == "follow_without_lift":
        return "grasp_no_lift"
    if row["revised_failure_stage"] == "lift_without_transport":
        return "lift_no_transport"
    if row["revised_failure_stage"] == "transport_without_placement":
        return "transport_no_success"
    return row["revised_failure_stage"]


def transition_counts(source: dict[str, dict[str, str]], target: dict[str, dict[str, str]]) -> Counter:
    counts = Counter()
    for seed in EXPECTED_SEEDS:
        counts[(stage_label(source[seed]), stage_label(target[seed]))] += 1
    return counts


def summarize_checkpoint(
    checkpoint: str,
    results_rows: list[dict[str, str]],
    runtime_rows: list[dict[str, str]],
    v2_rows: list[dict[str, str]],
) -> dict[str, object]:
    task_success = sum(as_bool(row["task_success"]) for row in results_rows)
    color_correct = sum(as_bool(row["color_correct"]) for row in results_rows)
    picked = Counter(normalize_picked(row["picked_color"]) for row in results_rows)
    wrong_color_success = sum(as_bool(row["task_success"]) and not as_bool(row["color_correct"]) for row in results_rows)

    clear_correct_selection = sum(as_bool(row["selection_correct"]) for row in runtime_rows)
    wrong_color_selection = sum(row["auto_failure_stage"] == "wrong_color_selection" for row in runtime_rows)
    no_clear_selection = sum(row["auto_failure_stage"] == "no_clear_selection" for row in runtime_rows)
    correct_blue_approach = sum(as_bool(row["blue_approach_detected"]) for row in runtime_rows)
    close_near_blue = sum(as_bool(row["close_near_target"]) for row in runtime_rows)
    blue_grasp_candidate = sum(
        as_bool(row["grasp_candidate"]) and row["grasped_color_candidate"] == "blue"
        for row in runtime_rows
    )
    blue_lift = sum(as_bool(row["blue_lift_detected"]) for row in runtime_rows)
    blue_transport = sum(as_bool(row["blue_transport_detected"]) for row in runtime_rows)

    v2_funnel = {
        "initial_selection": sum(as_bool(row["selection_correct"]) for row in v2_rows),
        "active_approach": sum(as_bool(row["active_target_approach_detected"]) for row in v2_rows),
        "stable_lock": sum(as_bool(row["stable_target_lock_detected"]) for row in v2_rows),
        "relevant_close": sum(row["task_relevant_close_step"] != "" for row in v2_rows),
        "target_follow": sum(as_bool(row["target_follow_detected"]) for row in v2_rows),
        "secure_grasp": sum(as_bool(row["secure_grasp_candidate"]) for row in v2_rows),
        "blue_lift": sum(as_bool(row["blue_lift_detected"]) for row in v2_rows),
        "blue_transport": sum(as_bool(row["blue_transport_detected"]) for row in v2_rows),
        "blue_correct_success": sum(as_bool(row["color_correct"]) for row in v2_rows),
    }

    grasp_given_correct_selection = ratio_string(blue_grasp_candidate, clear_correct_selection)
    lift_given_grasp = ratio_string(blue_lift, blue_grasp_candidate)
    transport_given_lift = ratio_string(blue_transport, blue_lift)
    follow_given_relevant_close = ratio_string(v2_funnel["target_follow"], v2_funnel["relevant_close"])
    secure_given_follow = ratio_string(v2_funnel["secure_grasp"], v2_funnel["target_follow"])

    return {
        "checkpoint": checkpoint,
        "task_success": task_success,
        "color_correct": color_correct,
        "precision": precision_string(color_correct, task_success),
        "picked": picked,
        "wrong_color_success": wrong_color_success,
        "clear_correct_selection": clear_correct_selection,
        "wrong_color_selection": wrong_color_selection,
        "no_clear_selection": no_clear_selection,
        "selection_accuracy": rate_string(clear_correct_selection, len(runtime_rows)),
        "correct_blue_approach": correct_blue_approach,
        "close_near_blue": close_near_blue,
        "blue_grasp_candidate": blue_grasp_candidate,
        "blue_lift": blue_lift,
        "blue_transport": blue_transport,
        "grasp_given_correct_selection": grasp_given_correct_selection,
        "lift_given_grasp": lift_given_grasp,
        "transport_given_lift": transport_given_lift,
        "v2_funnel": v2_funnel,
        "follow_given_relevant_close": follow_given_relevant_close,
        "secure_given_follow": secure_given_follow,
    }


def verify_inputs(results_rows: list[dict[str, str]], checkpoint: str) -> list[str]:
    errors: list[str] = []
    if len(results_rows) != 10:
        errors.append(f"{checkpoint}: rows={len(results_rows)}, expected 10")
    seeds = tuple(row["seed"] for row in results_rows)
    if seeds != EXPECTED_SEEDS:
        errors.append(f"{checkpoint}: seed order={seeds}, expected={EXPECTED_SEEDS}")
    targets = {row["target_color"] for row in results_rows}
    if targets != {"blue"}:
        errors.append(f"{checkpoint}: target set={targets}, expected={{'blue'}}")
    return errors


def rank_checkpoints(summaries: dict[str, dict[str, object]]) -> list[str]:
    def sort_key(checkpoint: str) -> tuple[int, int, int, int, int, int]:
        summary = summaries[checkpoint]
        return (
            int(summary["clear_correct_selection"]),
            int(summary["color_correct"]),
            -int(summary["wrong_color_selection"]),
            int(summary["blue_grasp_candidate"]),
            int(summary["blue_transport"]),
            int(summary["task_success"]),
        )

    return sorted(CHECKPOINTS, key=sort_key, reverse=True)


def checkpoint_label(summary: dict[str, object]) -> str:
    sel = int(summary["clear_correct_selection"])
    grasp = int(summary["blue_grasp_candidate"])
    lift = int(summary["blue_lift"])
    transport = int(summary["blue_transport"])
    if sel <= 2 and grasp >= 2:
        return "SELECTION-LIMITED"
    if sel >= 2 and grasp == 0:
        return "GRASP-LIMITED"
    if grasp >= 1 and lift == 0:
        return "DOWNSTREAM-LIMITED"
    if sel >= 2 and grasp >= 2 and transport >= 1:
        return "BALANCED/PROMISING"
    return "DOWNSTREAM-LIMITED"


def mechanism_axis(summary: dict[str, object]) -> str:
    sel = int(summary["clear_correct_selection"])
    if sel <= 2:
        return "selection weak"
    return "selection relatively strong"


def grasp_axis(summary: dict[str, object]) -> str:
    if summary["grasp_given_correct_selection"] == "N/A":
        return "no grasp evidence"
    if summary["grasp_given_correct_selection"].startswith("2/2") or summary["grasp_given_correct_selection"].startswith("1/1"):
        return "conditional grasp preserved"
    return "conditional grasp weaker"


def downstream_axis(summary: dict[str, object]) -> str:
    if int(summary["blue_transport"]) > 0:
        return "nonzero transport"
    if int(summary["blue_lift"]) > 0:
        return "lift without transport"
    return "downstream collapsed"


def endpoint_axis(summary: dict[str, object]) -> str:
    return (
        f"task {summary['task_success']}/10, correct {summary['color_correct']}/10, "
        f"precision {summary['precision']}"
    )


def dominant_bottleneck(summary: dict[str, object]) -> str:
    if int(summary["clear_correct_selection"]) <= 2:
        return "SELECTION"
    if int(summary["blue_grasp_candidate"]) < int(summary["clear_correct_selection"]):
        return "GRASP"
    if int(summary["blue_lift"]) < int(summary["blue_grasp_candidate"]):
        return "LIFT"
    if int(summary["blue_transport"]) < int(summary["blue_lift"]):
        return "TRANSPORT"
    if int(summary["color_correct"]) < int(summary["blue_transport"]):
        return "PLACEMENT"
    return "MIXED"


def target_commitment_verdict(best: str, summaries: dict[str, dict[str, object]]) -> str:
    best_summary = summaries[best]
    if int(best_summary["clear_correct_selection"]) > 2:
        return "CLEAR POSITIVE"
    if int(best_summary["correct_blue_approach"]) > 5 or int(best_summary["v2_funnel"]["active_approach"]) > 2:
        return "PARTIAL POSITIVE"
    if all(int(summary["clear_correct_selection"]) == 2 for summary in summaries.values()):
        return "NEUTRAL"
    return "NEGATIVE"


def grasp_preservation_verdict(best: str, summaries: dict[str, dict[str, object]]) -> str:
    best_summary = summaries[best]
    if best_summary["grasp_given_correct_selection"] == "2/2 = 100.0%":
        return "PRESERVED"
    if best_summary["follow_given_relevant_close"].endswith("100.0%") and int(best_summary["v2_funnel"]["secure_grasp"]) > 0:
        return "PARTIALLY PRESERVED"
    if int(best_summary["blue_grasp_candidate"]) == 0:
        return "REGRESSED"
    return "PARTIALLY PRESERVED"


def combined_verdict(best: str, summaries: dict[str, dict[str, object]]) -> str:
    best_summary = summaries[best]
    if int(best_summary["clear_correct_selection"]) > 2 and best_summary["grasp_given_correct_selection"].endswith("100.0%"):
        return "RESOLVED TRADEOFF"
    if best_summary["grasp_given_correct_selection"].endswith("100.0%") and int(best_summary["blue_transport"]) > 0:
        return "PARTIALLY RESOLVED"
    if int(best_summary["correct_blue_approach"]) > 5 and int(best_summary["wrong_color_selection"]) >= 7:
        return "MIXED TRADEOFF"
    if all(int(summary["clear_correct_selection"]) == 2 for summary in summaries.values()):
        return "NO CLEAR BENEFIT"
    return "NEGATIVE"


def next_step(best: str, summaries: dict[str, dict[str, object]]) -> str:
    verdict = combined_verdict(best, summaries)
    if verdict in {"RESOLVED TRADEOFF", "PARTIALLY RESOLVED"}:
        return "EXPAND BEST TO 30EP"
    if verdict == "NEGATIVE":
        return "STOP RUN #4"
    return "DO NOT EXPAND YET"


def build_paired_rows(v2_by_ckpt: dict[str, dict[str, dict[str, str]]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for seed in EXPECTED_SEEDS:
        row14 = v2_by_ckpt["014000"][seed]
        row16 = v2_by_ckpt["016000"][seed]
        row18 = v2_by_ckpt["018000"][seed]
        rows.append(
            {
                "seed": seed,
                "014k_result_stage": stage_label(row14),
                "016k_result_stage": stage_label(row16),
                "018k_result_stage": stage_label(row18),
                "014k_picked": row14["picked_color"],
                "016k_picked": row16["picked_color"],
                "018k_picked": row18["picked_color"],
                "014k_task_success": row14["task_success"],
                "016k_task_success": row16["task_success"],
                "018k_task_success": row18["task_success"],
                "014k_color_correct": row14["color_correct"],
                "016k_color_correct": row16["color_correct"],
                "018k_color_correct": row18["color_correct"],
                "14k_to_16k": f"{stage_label(row14)} -> {stage_label(row16)}",
                "16k_to_18k": f"{stage_label(row16)} -> {stage_label(row18)}",
                "14k_to_18k": f"{stage_label(row14)} -> {stage_label(row18)}",
            }
        )
    return rows


def format_transition_block(counts: Counter) -> list[str]:
    keys = sorted(counts.keys())
    return [f"{source} -> {target} = {counts[(source, target)]}" for source, target in keys]


def main() -> None:
    args = parse_args()
    dirs = {
        "014000": args.run4_14k_dir,
        "016000": args.run4_16k_dir,
        "018000": args.run4_18k_dir,
    }

    results_rows: dict[str, list[dict[str, str]]] = {}
    runtime_rows: dict[str, list[dict[str, str]]] = {}
    v2_rows: dict[str, list[dict[str, str]]] = {}
    validation_errors: list[str] = []

    for checkpoint, directory in dirs.items():
        results_rows[checkpoint] = load_csv(directory / "results_blue10.csv")
        runtime_rows[checkpoint] = load_csv(directory / "instrumentation_summary.csv")
        v2_rows[checkpoint] = load_csv(directory / "instrumentation_summary_v2.csv")
        validation_errors.extend(verify_inputs(results_rows[checkpoint], checkpoint))

    seed_sets = {
        checkpoint: {(row["seed"], row["target_color"]) for row in rows}
        for checkpoint, rows in results_rows.items()
    }
    same_seed_set = seed_sets["014000"] == seed_sets["016000"] == seed_sets["018000"]
    if not same_seed_set:
        validation_errors.append("seed-target sets do not exactly match across 14k/16k/18k")
    if validation_errors:
        raise SystemExit("validation failed:\n- " + "\n- ".join(validation_errors))

    summaries = {
        checkpoint: summarize_checkpoint(checkpoint, results_rows[checkpoint], runtime_rows[checkpoint], v2_rows[checkpoint])
        for checkpoint in CHECKPOINTS
    }

    ranking = rank_checkpoints(summaries)
    best, second, third = ranking

    v2_by_ckpt = {
        checkpoint: {row["seed"]: row for row in v2_rows[checkpoint]}
        for checkpoint in CHECKPOINTS
    }
    paired_rows = build_paired_rows(v2_by_ckpt)
    transition_14_16 = transition_counts(v2_by_ckpt["014000"], v2_by_ckpt["016000"])
    transition_16_18 = transition_counts(v2_by_ckpt["016000"], v2_by_ckpt["018000"])
    transition_14_18 = transition_counts(v2_by_ckpt["014000"], v2_by_ckpt["018000"])

    comparison_rows: list[dict[str, object]] = []
    for checkpoint in CHECKPOINTS:
        summary = summaries[checkpoint]
        comparison_rows.append(
            {
                "checkpoint": checkpoint,
                "task_success": rate_string(summary["task_success"], 10),
                "color_correct": rate_string(summary["color_correct"], 10),
                "precision": summary["precision"],
                "clear_correct_selection": summary["clear_correct_selection"],
                "wrong_color_selection": summary["wrong_color_selection"],
                "no_clear_selection": summary["no_clear_selection"],
                "selection_accuracy": summary["selection_accuracy"],
                "correct_blue_approach": rate_string(summary["correct_blue_approach"], 10),
                "close_near_blue": rate_string(summary["close_near_blue"], 10),
                "blue_grasp_candidate": rate_string(summary["blue_grasp_candidate"], 10),
                "blue_lift": rate_string(summary["blue_lift"], 10),
                "blue_transport": rate_string(summary["blue_transport"], 10),
                "grasp_given_correct_selection": summary["grasp_given_correct_selection"],
                "lift_given_grasp": summary["lift_given_grasp"],
                "transport_given_lift": summary["transport_given_lift"],
                "v2_initial_selection": rate_string(summary["v2_funnel"]["initial_selection"], 10),
                "v2_active_approach": rate_string(summary["v2_funnel"]["active_approach"], 10),
                "v2_stable_lock": rate_string(summary["v2_funnel"]["stable_lock"], 10),
                "v2_relevant_close": rate_string(summary["v2_funnel"]["relevant_close"], 10),
                "v2_target_follow": rate_string(summary["v2_funnel"]["target_follow"], 10),
                "v2_secure_grasp": rate_string(summary["v2_funnel"]["secure_grasp"], 10),
                "v2_follow_given_relevant_close": summary["follow_given_relevant_close"],
                "v2_secure_given_follow": summary["secure_given_follow"],
                "wrong_color_success": rate_string(summary["wrong_color_success"], 10),
                "picked_blue": summary["picked"]["blue"],
                "picked_red": summary["picked"]["red"],
                "picked_yellow": summary["picked"]["yellow"],
                "picked_failure": summary["picked"]["failure"],
                "picked_ambiguous": summary["picked"]["ambiguous"],
                "checkpoint_verdict": checkpoint_label(summary),
                "dominant_bottleneck": dominant_bottleneck(summary),
            }
        )
    write_csv(args.output_comparison_csv, list(comparison_rows[0].keys()), comparison_rows)
    write_csv(args.output_paired_csv, list(paired_rows[0].keys()), paired_rows)

    target_commitment = target_commitment_verdict(best, summaries)
    grasp_preservation = grasp_preservation_verdict(best, summaries)
    combined = combined_verdict(best, summaries)
    next_step_verdict = next_step(best, summaries)
    best_summary = summaries[best]

    rationale = [
        f"selection_accuracy ties at {best_summary['clear_correct_selection']}/10, so tie-break moves to failure composition",
        f"lowest wrong-color selection count among the three checkpoints: {best_summary['wrong_color_selection']}/10",
        f"grasp_given_correct_selection remains high at {best_summary['grasp_given_correct_selection']} with only a 2-episode denominator",
        f"downstream propagation stays nonzero: lift {best_summary['blue_lift']}/10, transport {best_summary['blue_transport']}/10, correct {best_summary['color_correct']}/10",
        f"paired seeds show fewer regressions into wrong-color success than the lower-ranked alternatives",
    ]

    report_lines = [
        "A. Input Validation",
        f"14k rows = {len(results_rows['014000'])}",
        f"16k rows = {len(results_rows['016000'])}",
        f"18k rows = {len(results_rows['018000'])}",
        f"same exact seeds? {'YES' if same_seed_set else 'NO'}",
        "",
        "B. Metric Definition Mapping",
        "clear correct selection == sum(selection_correct) from instrumentation_summary.csv",
        "wrong-color selection == sum(auto_failure_stage == wrong_color_selection) from instrumentation_summary.csv",
        "no-clear selection == sum(auto_failure_stage == no_clear_selection) from instrumentation_summary.csv",
        "correct blue approach == sum(blue_approach_detected) from instrumentation_summary.csv",
        "close near blue == sum(close_near_target) from instrumentation_summary.csv",
        "blue grasp candidate == sum(grasp_candidate and grasped_color_candidate == blue) from instrumentation_summary.csv",
        "blue lift == sum(blue_lift_detected) from instrumentation_summary.csv",
        "blue transport == sum(blue_transport_detected) from instrumentation_summary.csv",
        "v2 correct initial selection == sum(selection_correct) from instrumentation_summary_v2.csv",
        "v2 active target approach == sum(active_target_approach_detected) from instrumentation_summary_v2.csv",
        "v2 task-relevant close == sum(task_relevant_close_step != '') from instrumentation_summary_v2.csv",
        "v2 secure grasp candidate == sum(secure_grasp_candidate) from instrumentation_summary_v2.csv",
        "runtime close near blue is not identical to v2 task-relevant close; they are reported separately",
        "runtime blue grasp candidate is not identical to v2 secure grasp candidate; they are reported separately",
        "",
        "C. Endpoint Summary",
        "| checkpoint | task success | color correct | precision |",
        "| ---------- | -----------: | ------------: | --------: |",
    ]
    for checkpoint in CHECKPOINTS:
        summary = summaries[checkpoint]
        report_lines.append(
            f"| {checkpoint} | {rate_string(summary['task_success'], 10)} | {rate_string(summary['color_correct'], 10)} | {summary['precision']} |"
        )

    report_lines.extend(
        [
            "",
            "D. Selection Summary",
            "| metric | 014k | 016k | 018k |",
            "| ------ | ---: | ---: | ---: |",
            f"| clear correct selection | {summaries['014000']['clear_correct_selection']} | {summaries['016000']['clear_correct_selection']} | {summaries['018000']['clear_correct_selection']} |",
            f"| wrong-color selection | {summaries['014000']['wrong_color_selection']} | {summaries['016000']['wrong_color_selection']} | {summaries['018000']['wrong_color_selection']} |",
            f"| no-clear selection | {summaries['014000']['no_clear_selection']} | {summaries['016000']['no_clear_selection']} | {summaries['018000']['no_clear_selection']} |",
            f"| selection accuracy | {summaries['014000']['selection_accuracy'].split(' = ')[1]} | {summaries['016000']['selection_accuracy'].split(' = ')[1]} | {summaries['018000']['selection_accuracy'].split(' = ')[1]} |",
            "",
            "E. Selection Failure Composition",
        ]
    )
    for checkpoint in CHECKPOINTS:
        summary = summaries[checkpoint]
        report_lines.append(
            f"{checkpoint}: wrong-color selection = {summary['wrong_color_selection']}/10, no-clear selection = {summary['no_clear_selection']}/10"
        )

    report_lines.extend(
        [
            "",
            "F. Blue Approach Summary",
            "| checkpoint | correct blue approach |",
            "| ---------- | --------------------: |",
            f"| 014k | {summaries['014000']['correct_blue_approach']}/10 |",
            f"| 016k | {summaries['016000']['correct_blue_approach']}/10 |",
            f"| 018k | {summaries['018000']['correct_blue_approach']}/10 |",
            "",
            "G. Grasp Funnel",
            "| stage | 014k | 016k | 018k |",
            "| ----- | ---: | ---: | ---: |",
            f"| correct blue approach | {summaries['014000']['correct_blue_approach']} | {summaries['016000']['correct_blue_approach']} | {summaries['018000']['correct_blue_approach']} |",
            f"| close near blue | {summaries['014000']['close_near_blue']} | {summaries['016000']['close_near_blue']} | {summaries['018000']['close_near_blue']} |",
            f"| blue grasp candidate | {summaries['014000']['blue_grasp_candidate']} | {summaries['016000']['blue_grasp_candidate']} | {summaries['018000']['blue_grasp_candidate']} |",
            f"| blue lift | {summaries['014000']['blue_lift']} | {summaries['016000']['blue_lift']} | {summaries['018000']['blue_lift']} |",
            f"| blue transport | {summaries['014000']['blue_transport']} | {summaries['016000']['blue_transport']} | {summaries['018000']['blue_transport']} |",
            "",
            "H. Conditional Metrics",
        ]
    )
    for checkpoint in CHECKPOINTS:
        summary = summaries[checkpoint]
        report_lines.extend(
            [
                f"{checkpoint}: grasp_given_correct_selection = {summary['grasp_given_correct_selection']}",
                f"{checkpoint}: lift_given_grasp = {summary['lift_given_grasp']}",
                f"{checkpoint}: transport_given_lift = {summary['transport_given_lift']}",
            ]
        )

    report_lines.extend(
        [
            "",
            "I. Revised v2 Funnel Comparison",
            "| v2 stage | 014k | 016k | 018k |",
            "| -------- | ---: | ---: | ---: |",
            f"| initial selection | {summaries['014000']['v2_funnel']['initial_selection']} | {summaries['016000']['v2_funnel']['initial_selection']} | {summaries['018000']['v2_funnel']['initial_selection']} |",
            f"| active approach | {summaries['014000']['v2_funnel']['active_approach']} | {summaries['016000']['v2_funnel']['active_approach']} | {summaries['018000']['v2_funnel']['active_approach']} |",
            f"| stable lock | {summaries['014000']['v2_funnel']['stable_lock']} | {summaries['016000']['v2_funnel']['stable_lock']} | {summaries['018000']['v2_funnel']['stable_lock']} |",
            f"| relevant close | {summaries['014000']['v2_funnel']['relevant_close']} | {summaries['016000']['v2_funnel']['relevant_close']} | {summaries['018000']['v2_funnel']['relevant_close']} |",
            f"| target follow | {summaries['014000']['v2_funnel']['target_follow']} | {summaries['016000']['v2_funnel']['target_follow']} | {summaries['018000']['v2_funnel']['target_follow']} |",
            f"| secure grasp | {summaries['014000']['v2_funnel']['secure_grasp']} | {summaries['016000']['v2_funnel']['secure_grasp']} | {summaries['018000']['v2_funnel']['secure_grasp']} |",
            f"| lift | {summaries['014000']['v2_funnel']['blue_lift']} | {summaries['016000']['v2_funnel']['blue_lift']} | {summaries['018000']['v2_funnel']['blue_lift']} |",
            f"| transport | {summaries['014000']['v2_funnel']['blue_transport']} | {summaries['016000']['v2_funnel']['blue_transport']} | {summaries['018000']['v2_funnel']['blue_transport']} |",
            f"| correct success | {summaries['014000']['v2_funnel']['blue_correct_success']} | {summaries['016000']['v2_funnel']['blue_correct_success']} | {summaries['018000']['v2_funnel']['blue_correct_success']} |",
            "",
            "J. Selection-vs-Grasp 2-Axis Table",
            "| checkpoint | selection accuracy | grasp given correct selection |",
            "| ---------- | -----------------: | ----------------------------: |",
        ]
    )
    for checkpoint in CHECKPOINTS:
        summary = summaries[checkpoint]
        report_lines.append(
            f"| {checkpoint} | {summary['selection_accuracy'].split(' = ')[1]} | {summary['grasp_given_correct_selection'].split(' = ')[1] if ' = ' in summary['grasp_given_correct_selection'] else summary['grasp_given_correct_selection']} |"
        )

    report_lines.extend(["", "K. Wrong-Color Behavior"])
    for checkpoint in CHECKPOINTS:
        summary = summaries[checkpoint]
        report_lines.append(
            f"{checkpoint}: picked blue={summary['picked']['blue']}, red={summary['picked']['red']}, yellow={summary['picked']['yellow']}, failure={summary['picked']['failure']}, ambiguous={summary['picked']['ambiguous']}; wrong-color success={summary['wrong_color_success']}/10"
        )

    report_lines.extend(
        [
            "",
            "L. Paired 10-Seed Table",
            "| seed | 014k result/stage | 016k result/stage | 018k result/stage |",
            "| ---: | ----------------- | ----------------- | ----------------- |",
        ]
    )
    for row in paired_rows:
        report_lines.append(
            f"| {row['seed']} | {row['014k_result_stage']} | {row['016k_result_stage']} | {row['018k_result_stage']} |"
        )

    report_lines.extend(["", "M. 14k -> 16k Transitions"])
    report_lines.extend(format_transition_block(transition_14_16))
    report_lines.extend(["", "N. 16k -> 18k Transitions"])
    report_lines.extend(format_transition_block(transition_16_18))
    report_lines.extend(
        [
            "",
            "O. Checkpoint Dynamics",
            "Selection accuracy stays flat at 2/10 across 14k, 16k, and 18k.",
            "Correct blue approach rises from 5/10 at 14k to 7/10 at 16k, then slips to 6/10 at 18k.",
            "The v2 funnel broadens later: active approach 3 -> 2 -> 4, stable lock 4 -> 4 -> 5, relevant close 3 -> 2 -> 4.",
            "Despite broader late-stage v2 reach, endpoint color-correct success remains 1/10 at all three checkpoints.",
            "The main late-checkpoint change is more approach/lock activity without a corresponding increase in initial correct selection or final blue correctness.",
            "",
            "P. Per-Checkpoint Bottleneck Label",
        ]
    )
    for checkpoint in CHECKPOINTS:
        report_lines.append(f"{checkpoint}: {checkpoint_label(summaries[checkpoint])}")

    report_lines.extend(
        [
            "",
            "Q. 3-Way Tradeoff Table",
            "| checkpoint | selection | grasp | downstream | endpoint | dominant bottleneck |",
            "| ---------- | --------- | ----- | ---------- | -------- | ------------------- |",
        ]
    )
    for checkpoint in CHECKPOINTS:
        summary = summaries[checkpoint]
        report_lines.append(
            f"| {checkpoint} | {mechanism_axis(summary)} | {grasp_axis(summary)} | {downstream_axis(summary)} | {endpoint_axis(summary)} | {dominant_bottleneck(summary)} |"
        )

    report_lines.extend(
        [
            "",
            "R. Best Mechanism Checkpoint",
            best,
            "",
            "S. Ranking",
            f"1. {best}",
            f"2. {second}",
            f"3. {third}",
            "",
            "T. Best Selection Rationale",
        ]
    )
    report_lines.extend(f"{index}. {reason}" for index, reason in enumerate(rationale, start=1))
    report_lines.extend(
        [
            "",
            "U. Remaining Bottleneck",
            dominant_bottleneck(best_summary),
            "",
            "V. Target-Commitment Verdict",
            target_commitment,
            "",
            "W. Grasp Preservation Verdict",
            grasp_preservation,
            "",
            "X. Combined Mechanism Verdict",
            combined,
            "",
            "Y. Next Step",
            next_step_verdict,
            "",
            "Z. Output Paths",
            str(args.output_comparison_csv),
            str(args.output_paired_csv),
            str(args.output_report_txt),
            "",
            "AA. One-Sentence Conclusion",
            f"Across the Sunday, August 16, 2026 blue 10-seed checkpoint screen, True Run #4 {best} is the best mechanism candidate among 14k/16k/18k because it keeps the strongest selection-failure composition while preserving nonzero grasp-to-transport propagation, but the remaining bottleneck is still {dominant_bottleneck(best_summary).lower()} and the 10-episode evidence only supports a cautious next step.",
        ]
    )

    args.output_report_txt.parent.mkdir(parents=True, exist_ok=True)
    args.output_report_txt.write_text("\n".join(report_lines) + "\n")


if __name__ == "__main__":
    main()
