#!/usr/bin/env python3
"""Shared diagnostic metric logic for three-color SmolVLA eval instrumentation."""

from __future__ import annotations

from collections import Counter


COLOR_NAMES = ("red", "blue", "yellow")
RAW_CLOSE_ACTION_THRESHOLD_DEG = -37.5


def _first_sustained_step(mask: list[bool], min_frames: int) -> int | str:
    streak = 0
    for idx, value in enumerate(mask):
        if value:
            streak += 1
        else:
            streak = 0
        if streak >= min_frames:
            return idx - min_frames + 1
    return ""


def _build_target_lock_mask(
    step_rows: list[dict[str, object]],
    target_color: str,
    lock_distance_m: float,
) -> list[bool]:
    return [
        str(row["nearest_cube"]) == target_color and float(row[f"dist_ee_{target_color}"]) <= lock_distance_m
        for row in step_rows
    ]


def _build_active_approach_mask(
    step_rows: list[dict[str, object]],
    target_color: str,
    approach_distance_m: float,
    sustained_frames: int,
    motion_window_frames: int,
    motion_delta_m: float,
) -> list[bool]:
    target_lock_mask = _build_target_lock_mask(step_rows, target_color, approach_distance_m)
    active_mask: list[bool] = []
    streak = 0
    for idx, row in enumerate(step_rows):
        if target_lock_mask[idx]:
            streak += 1
        else:
            streak = 0
        if idx < motion_window_frames or streak < sustained_frames:
            active_mask.append(False)
            continue
        current_distance = float(row[f"dist_ee_{target_color}"])
        prior_window = step_rows[idx - motion_window_frames:idx]
        prior_max_distance = max(float(prev[f"dist_ee_{target_color}"]) for prev in prior_window)
        active_mask.append((prior_max_distance - current_distance) >= motion_delta_m)
    return active_mask


def _find_first_raw_close_crossing(step_rows: list[dict[str, object]]) -> tuple[int | str, str, bool, dict[str, float | str]]:
    dist_at_close: dict[str, float | str] = {color: "" for color in COLOR_NAMES}
    for idx in range(1, len(step_rows)):
        prev = step_rows[idx - 1]
        row = step_rows[idx]
        if float(prev["gripper_action_deg"]) <= RAW_CLOSE_ACTION_THRESHOLD_DEG < float(row["gripper_action_deg"]):
            for color in COLOR_NAMES:
                dist_at_close[color] = float(row[f"dist_ee_{color}"])
            nearest_cube = str(row["nearest_cube"])
            return int(row["step"]), nearest_cube, False, dist_at_close
    return "", "none", False, dist_at_close


def _find_task_relevant_close(
    step_rows: list[dict[str, object]],
    target_color: str,
    grasp_distance_m: float,
    lock_mask: list[bool],
    active_mask: list[bool],
    active_window_after_frames: int,
    active_first_idx: int | str,
) -> dict[str, object]:
    if active_first_idx == "":
        return {
            "task_relevant_close_step": "",
            "task_relevant_close_distance": "",
            "task_relevant_close_nearest_cube": "none",
            "task_relevant_close_during_active_approach": False,
        }
    active_indices = [idx for idx, flag in enumerate(active_mask) if flag]
    last_active_idx = active_indices[-1] if active_indices else None
    for idx in range(1, len(step_rows)):
        if idx < int(active_first_idx):
            continue
        prev = step_rows[idx - 1]
        row = step_rows[idx]
        if not (float(prev["gripper_action_deg"]) <= RAW_CLOSE_ACTION_THRESHOLD_DEG < float(row["gripper_action_deg"])):
            continue
        if str(row["nearest_cube"]) != target_color:
            continue
        if float(row[f"dist_ee_{target_color}"]) > grasp_distance_m:
            continue
        active_now = active_mask[idx]
        active_recent = last_active_idx is not None and idx <= last_active_idx + active_window_after_frames
        if not (lock_mask[idx] or active_now or active_recent):
            continue
        return {
            "task_relevant_close_step": int(row["step"]),
            "task_relevant_close_distance": float(row[f"dist_ee_{target_color}"]),
            "task_relevant_close_nearest_cube": str(row["nearest_cube"]),
            "task_relevant_close_during_active_approach": bool(active_now or active_recent),
        }
    return {
        "task_relevant_close_step": "",
        "task_relevant_close_distance": "",
        "task_relevant_close_nearest_cube": "none",
        "task_relevant_close_during_active_approach": False,
    }


def _find_target_follow(
    step_rows: list[dict[str, object]],
    target_color: str,
    start_step: int | str,
    follow_distance_m: float,
    follow_displacement_m: float,
) -> tuple[bool, int | str]:
    if start_step == "":
        return False, ""
    for row in step_rows:
        if int(row["step"]) < int(start_step):
            continue
        if (
            float(row["gripper_action_deg"]) > RAW_CLOSE_ACTION_THRESHOLD_DEG
            and float(row[f"dist_ee_{target_color}"]) <= follow_distance_m
            and float(row[f"disp_{target_color}"]) >= follow_displacement_m
        ):
            return True, int(row["step"])
    return False, ""


def _find_secure_grasp_candidate(
    step_rows: list[dict[str, object]],
    target_color: str,
    start_step: int | str,
    grasp_distance_m: float,
    follow_displacement_m: float,
) -> tuple[bool, int | str]:
    if start_step == "":
        return False, ""
    for row in step_rows:
        if int(row["step"]) < int(start_step):
            continue
        if (
            float(row["gripper_action_deg"]) > RAW_CLOSE_ACTION_THRESHOLD_DEG
            and float(row[f"dist_ee_{target_color}"]) <= grasp_distance_m
            and float(row[f"disp_{target_color}"]) >= follow_displacement_m
        ):
            return True, int(row["step"])
    return False, ""


def infer_revised_failure_stage(
    *,
    result_row: dict[str, object],
    target_color: str,
    active_target_approach_detected: bool,
    selection_correct: bool,
    task_relevant_close_step: int | str,
    target_follow_detected: bool,
    lift_detected: dict[str, bool],
    transport_detected: dict[str, bool],
) -> str:
    if bool(result_row["task_success"]) and bool(result_row["color_correct"]):
        return "success"
    if bool(result_row["task_success"]) and not bool(result_row["color_correct"]):
        return "wrong_color_success"
    if not active_target_approach_detected or not selection_correct:
        return "selection_failure"
    if task_relevant_close_step == "":
        return "approach_without_relevant_close"
    if not target_follow_detected:
        return "relevant_close_without_follow"
    if not lift_detected[target_color]:
        return "follow_without_lift"
    if not transport_detected[target_color]:
        return "lift_without_transport"
    return "transport_without_placement"


def compute_episode_diagnostics(
    *,
    episode_id: int,
    seed: int,
    target_color: str,
    task_text: str,
    step_rows: list[dict[str, object]],
    result_row: dict[str, object],
    max_steps: int,
    approach_distance_m: float,
    selection_margin_m: float,
    sustained_approach_frames: int,
    grasp_distance_m: float,
    lift_z_threshold_m: float,
    follow_distance_m: float,
    follow_displacement_m: float,
    transport_distance_delta_m: float,
) -> dict[str, object]:
    if not step_rows:
        raise ValueError(f"No diagnostic rows captured for episode {episode_id}")

    def min_for(color: str) -> float:
        return min(float(row[f"dist_ee_{color}"]) for row in step_rows)

    def max_for(color: str) -> float:
        return max(float(row[f"disp_{color}"]) for row in step_rows)

    min_dists = {color: min_for(color) for color in COLOR_NAMES}
    max_disps = {color: max_for(color) for color in COLOR_NAMES}
    max_delta_z = {
        color: max(float(row[f"{color}_delta_z"]) for row in step_rows)
        for color in COLOR_NAMES
    }
    min_box_distance = {
        color: min(float(row[f"{color}_box_distance"]) for row in step_rows)
        for color in COLOR_NAMES
    }

    first_approached_color = "none"
    first_approach_step: int | str = ""
    sustained_selected_color = "none"
    sustained_selection_step: int | str = ""
    streak_color: str | None = None
    streak_len = 0
    for row in step_rows:
        approached_color = str(row["approach_color"])
        if first_approached_color == "none" and approached_color != "none":
            first_approached_color = approached_color
            first_approach_step = int(row["step"])
        if approached_color != "none" and approached_color == streak_color:
            streak_len += 1
        elif approached_color != "none":
            streak_color = approached_color
            streak_len = 1
        else:
            streak_color = None
            streak_len = 0
        if sustained_selected_color == "none" and streak_color is not None and streak_len >= sustained_approach_frames:
            sustained_selected_color = streak_color
            sustained_selection_step = int(row["step"]) - sustained_approach_frames + 1

    selection_correct_first = first_approached_color == target_color
    selection_correct_sustained = sustained_selected_color == target_color
    selection_correct = selection_correct_sustained or (
        sustained_selected_color == "none" and selection_correct_first
    )

    raw_close_step, nearest_cube_at_close, _, dist_at_close = _find_first_raw_close_crossing(step_rows)
    close_near_target = nearest_cube_at_close == target_color

    grasp_candidate = False
    grasped_color_candidate = "none"
    grasp_candidate_step: int | str = ""
    follow_detected = False
    followed_color = "none"
    for color in COLOR_NAMES:
        for row in step_rows:
            if (
                float(row["gripper_action_deg"]) > RAW_CLOSE_ACTION_THRESHOLD_DEG
                and float(row[f"dist_ee_{color}"]) <= grasp_distance_m
                and float(row[f"disp_{color}"]) >= follow_displacement_m
            ):
                grasp_candidate = True
                grasped_color_candidate = color
                grasp_candidate_step = int(row["step"])
                break
        if grasp_candidate:
            break
    for color in COLOR_NAMES:
        for row in step_rows:
            if (
                float(row["gripper_action_deg"]) > RAW_CLOSE_ACTION_THRESHOLD_DEG
                and float(row[f"dist_ee_{color}"]) <= follow_distance_m
                and float(row[f"disp_{color}"]) >= follow_displacement_m
            ):
                follow_detected = True
                followed_color = color
                break
        if follow_detected:
            break

    lift_detected = {
        color: max_delta_z[color] >= lift_z_threshold_m
        for color in COLOR_NAMES
    }
    first_lift_step: dict[str, int | str] = {}
    for color in COLOR_NAMES:
        first_lift_step[color] = ""
        for row in step_rows:
            if float(row[f"{color}_delta_z"]) >= lift_z_threshold_m:
                first_lift_step[color] = int(row["step"])
                break

    box_initial_distance = {
        color: float(step_rows[0][f"{color}_box_distance"])
        for color in COLOR_NAMES
    }
    transport_detected = {
        color: lift_detected[color]
        and (box_initial_distance[color] - min_box_distance[color]) >= transport_distance_delta_m
        for color in COLOR_NAMES
    }

    if bool(result_row["task_success"]):
        auto_failure_stage = "success"
    elif not selection_correct:
        auto_failure_stage = (
            "wrong_color_selection"
            if first_approached_color not in ("none", target_color)
            or sustained_selected_color not in ("none", target_color)
            else "no_clear_selection"
        )
    elif not grasp_candidate:
        auto_failure_stage = "correct_selection_pre_grasp_failure"
    elif not lift_detected[target_color]:
        auto_failure_stage = "grasp_candidate_lift_failure"
    elif not transport_detected[target_color]:
        auto_failure_stage = "lift_transport_failure"
    else:
        auto_failure_stage = "transport_placement_failure"

    stable_lock_distance_m = grasp_distance_m
    active_motion_window_frames = sustained_approach_frames * 4
    active_motion_delta_m = selection_margin_m
    close_active_window_after_frames = sustained_approach_frames

    lock_mask = _build_target_lock_mask(step_rows, target_color, stable_lock_distance_m)
    stable_target_lock_first_idx = _first_sustained_step(lock_mask, sustained_approach_frames)
    stable_target_lock_detected = stable_target_lock_first_idx != ""
    stable_target_lock_first_step: int | str = (
        int(step_rows[int(stable_target_lock_first_idx)]["step"]) if stable_target_lock_detected else ""
    )

    active_mask = _build_active_approach_mask(
        step_rows,
        target_color,
        approach_distance_m=approach_distance_m,
        sustained_frames=sustained_approach_frames,
        motion_window_frames=active_motion_window_frames,
        motion_delta_m=active_motion_delta_m,
    )
    active_target_approach_first_idx = _first_sustained_step(active_mask, sustained_approach_frames)
    active_target_approach_detected = active_target_approach_first_idx != ""
    active_target_approach_first_step: int | str = (
        int(step_rows[int(active_target_approach_first_idx)]["step"]) if active_target_approach_detected else ""
    )

    close_info = _find_task_relevant_close(
        step_rows,
        target_color=target_color,
        grasp_distance_m=grasp_distance_m,
        lock_mask=lock_mask,
        active_mask=active_mask,
        active_window_after_frames=close_active_window_after_frames,
        active_first_idx=active_target_approach_first_idx,
    )
    target_follow_detected, target_follow_start_step = _find_target_follow(
        step_rows,
        target_color=target_color,
        start_step=close_info["task_relevant_close_step"],
        follow_distance_m=follow_distance_m,
        follow_displacement_m=follow_displacement_m,
    )
    secure_grasp_candidate, secure_grasp_candidate_step = _find_secure_grasp_candidate(
        step_rows,
        target_color=target_color,
        start_step=close_info["task_relevant_close_step"],
        grasp_distance_m=grasp_distance_m,
        follow_displacement_m=follow_displacement_m,
    )

    revised_failure_stage = infer_revised_failure_stage(
        result_row=result_row,
        target_color=target_color,
        active_target_approach_detected=active_target_approach_detected,
        selection_correct=selection_correct,
        task_relevant_close_step=close_info["task_relevant_close_step"],
        target_follow_detected=target_follow_detected,
        lift_detected=lift_detected,
        transport_detected=transport_detected,
    )

    summary = {
        "episode_id": episode_id,
        "seed": seed,
        "target_color": target_color,
        "task_text": task_text,
        "max_steps": max_steps,
        "first_approached_color": first_approached_color,
        "first_approach_step": first_approach_step,
        "sustained_selected_color": sustained_selected_color,
        "sustained_selection_step": sustained_selection_step,
        "selection_correct_first": selection_correct_first,
        "selection_correct_sustained": selection_correct_sustained,
        "selection_correct": selection_correct,
        "min_dist_ee_red": min_dists["red"],
        "min_dist_ee_blue": min_dists["blue"],
        "min_dist_ee_yellow": min_dists["yellow"],
        "blue_approach_detected": min_dists["blue"] <= approach_distance_m,
        "blue_approach_first_step": first_approach_step if first_approached_color == "blue" else "",
        "blue_min_ee_distance": min_dists["blue"],
        "raw_first_close_crossing_step": raw_close_step,
        "first_gripper_close_step": raw_close_step,
        "first_gripper_close_step_note": (
            "RAW action threshold crossing. May include reset/settling transients. "
            "Not necessarily a task-level grasp attempt."
        ),
        "nearest_cube_at_close": nearest_cube_at_close,
        "dist_ee_red_at_close": dist_at_close["red"],
        "dist_ee_blue_at_close": dist_at_close["blue"],
        "dist_ee_yellow_at_close": dist_at_close["yellow"],
        "close_near_target": close_near_target,
        "grasp_candidate": grasp_candidate,
        "grasped_color_candidate": grasped_color_candidate,
        "grasp_candidate_step": grasp_candidate_step,
        "max_disp_red": max_disps["red"],
        "max_disp_blue": max_disps["blue"],
        "max_disp_yellow": max_disps["yellow"],
        "most_moved_cube": max(max_disps, key=max_disps.get),
        "red_lift_detected": lift_detected["red"],
        "blue_lift_detected": lift_detected["blue"],
        "yellow_lift_detected": lift_detected["yellow"],
        "red_max_delta_z": max_delta_z["red"],
        "blue_max_delta_z": max_delta_z["blue"],
        "yellow_max_delta_z": max_delta_z["yellow"],
        "red_first_lift_step": first_lift_step["red"],
        "blue_first_lift_step": first_lift_step["blue"],
        "yellow_first_lift_step": first_lift_step["yellow"],
        "follow_detected": follow_detected,
        "followed_color": followed_color,
        "red_box_initial_distance": box_initial_distance["red"],
        "blue_box_initial_distance": box_initial_distance["blue"],
        "yellow_box_initial_distance": box_initial_distance["yellow"],
        "red_box_min_distance": min_box_distance["red"],
        "blue_box_min_distance": min_box_distance["blue"],
        "yellow_box_min_distance": min_box_distance["yellow"],
        "red_transport_detected": transport_detected["red"],
        "blue_transport_detected": transport_detected["blue"],
        "yellow_transport_detected": transport_detected["yellow"],
        "transport_detected": transport_detected[target_color],
        "task_success": result_row["task_success"],
        "picked_color": result_row["picked_color"],
        "color_correct": result_row["color_correct"],
        "termination_reason": result_row["termination_reason"],
        "steps": result_row["steps"],
        "auto_failure_stage": auto_failure_stage,
        "proximity_lock_distance_m": stable_lock_distance_m,
        "active_approach_motion_window_frames": active_motion_window_frames,
        "active_approach_motion_delta_m": active_motion_delta_m,
        "stable_target_lock_detected": stable_target_lock_detected,
        "stable_target_lock_first_step": stable_target_lock_first_step,
        "active_target_approach_detected": active_target_approach_detected,
        "active_target_approach_first_step": active_target_approach_first_step,
        **close_info,
        "blue_task_relevant_close": target_color == "blue" and close_info["task_relevant_close_step"] != "",
        "target_follow_detected": target_follow_detected,
        "target_follow_start_step": target_follow_start_step,
        "secure_grasp_candidate": secure_grasp_candidate,
        "secure_grasp_candidate_step": secure_grasp_candidate_step,
        "revised_failure_stage": revised_failure_stage,
    }
    return summary


def compute_blue_funnel(summary_rows: list[dict[str, object]]) -> dict[str, int]:
    blue_rows = [row for row in summary_rows if row["target_color"] == "blue"]
    return {
        "episodes": len(blue_rows),
        "correct_initial_selection": sum(bool(row["selection_correct"]) for row in blue_rows),
        "active_blue_approach": sum(bool(row["active_target_approach_detected"]) for row in blue_rows),
        "stable_target_lock": sum(bool(row["stable_target_lock_detected"]) for row in blue_rows),
        "task_relevant_close": sum(row["task_relevant_close_step"] != "" for row in blue_rows),
        "target_follow": sum(bool(row["target_follow_detected"]) for row in blue_rows),
        "secure_grasp_candidate": sum(bool(row["secure_grasp_candidate"]) for row in blue_rows),
        "blue_lift": sum(bool(row["blue_lift_detected"]) for row in blue_rows),
        "blue_transport": sum(bool(row["blue_transport_detected"]) for row in blue_rows),
        "blue_color_correct_success": sum(bool(row["color_correct"]) for row in blue_rows),
    }


def nearest_fraction_before_step(
    step_rows: list[dict[str, object]],
    step_value: int | str,
    window_frames: int = 30,
) -> dict[str, float]:
    if step_value == "":
        return {color: 0.0 for color in COLOR_NAMES}
    idx = next((i for i, row in enumerate(step_rows) if int(row["step"]) == int(step_value)), None)
    if idx is None:
        return {color: 0.0 for color in COLOR_NAMES}
    start = max(0, idx - window_frames)
    window = step_rows[start:idx]
    if not window:
        return {color: 0.0 for color in COLOR_NAMES}
    counts = Counter(str(row["nearest_cube"]) for row in window)
    return {color: counts[color] / len(window) for color in COLOR_NAMES}
