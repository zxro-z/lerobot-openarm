"""Generate a random-cube/random-tilt dataset with real-gripper calibration.

The base simulation represents ``openarm_left_finger_joint1`` from 0.0 rad
(closed) to 0.044 rad (open).  Before writing each dataset frame, this launcher
linearly maps only the gripper element of observation.state and action to the
measured real-motor range:

    closed: -15 degrees = -0.261799... rad
    open:   -60 degrees = -1.047197... rad

All seven arm joints remain unchanged and are recorded in radians.
"""

from __future__ import annotations

import argparse
import math
import random
import sys


def _parse_launcher_args() -> tuple[argparse.Namespace, list[str]]:
    """Consume launcher-only arguments and leave base-script arguments."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--tilt_deg_range",
        type=float,
        nargs=2,
        metavar=("MIN", "MAX"),
        default=(-45.0, 45.0),
        help="Sample the robot-base-y TCP tilt uniformly from this range for each attempt.",
    )
    parser.add_argument(
        "--tilt_random_seed",
        type=int,
        default=None,
        help="Seed for reproducible tilt sampling.",
    )
    return parser.parse_known_args()


launcher_args, base_argv = _parse_launcher_args()
tilt_min, tilt_max = launcher_args.tilt_deg_range
if tilt_min > tilt_max:
    raise SystemExit("--tilt_deg_range MIN must be less than or equal to MAX")

# The imported base module owns the remaining command-line arguments.
sys.argv = [sys.argv[0], *base_argv]

import openarm_table_dual_realsense_ik_pick_place_make_dataset_random_cube as base  # noqa: E402


SIM_GRIPPER_CLOSED_RAD = 0.0
SIM_GRIPPER_OPEN_RAD = 0.044
REAL_GRIPPER_CLOSED_RAD = math.radians(-15.0)
REAL_GRIPPER_OPEN_RAD = math.radians(-60.0)
GRIPPER_INDEX = 7


def map_sim_gripper_to_real_rad(value: float) -> float:
    """Map and clamp a simulated gripper angle to the measured real range."""
    alpha = (float(value) - SIM_GRIPPER_CLOSED_RAD) / (
        SIM_GRIPPER_OPEN_RAD - SIM_GRIPPER_CLOSED_RAD
    )
    alpha = min(max(alpha, 0.0), 1.0)
    return REAL_GRIPPER_CLOSED_RAD + alpha * (
        REAL_GRIPPER_OPEN_RAD - REAL_GRIPPER_CLOSED_RAD
    )


_tilt_rng = random.Random(launcher_args.tilt_random_seed)
_original_controller_init = base.PickPlaceController.__init__
_original_get_record_state_action = base.PickPlaceController.get_record_state_action


def _mapped_controller_init(self, robot, scene) -> None:
    """Sample one TCP tilt before constructing each episode controller."""
    sampled_tilt = _tilt_rng.uniform(tilt_min, tilt_max)
    base.args_cli.tilt_deg = sampled_tilt
    print(
        f"[EPISODE] Sampled TCP tilt={sampled_tilt:.2f} deg "
        f"from [{tilt_min:.2f}, {tilt_max:.2f}]"
    )
    _original_controller_init(self, robot, scene)


def _get_mapped_record_state_action(self):
    """Return arm radians plus a real-motor-calibrated gripper value."""
    state, action = _original_get_record_state_action(self)
    state[GRIPPER_INDEX] = map_sim_gripper_to_real_rad(state[GRIPPER_INDEX])
    action[GRIPPER_INDEX] = map_sim_gripper_to_real_rad(action[GRIPPER_INDEX])
    return state, action


base.PickPlaceController.__init__ = _mapped_controller_init
base.PickPlaceController.get_record_state_action = _get_mapped_record_state_action


if __name__ == "__main__":
    print(
        f"[INFO] TCP tilt randomization range=[{tilt_min:.2f}, {tilt_max:.2f}] deg, "
        f"seed={launcher_args.tilt_random_seed}"
    )
    print(
        "[INFO] Dataset gripper mapping: "
        f"sim [{SIM_GRIPPER_CLOSED_RAD:.3f}, {SIM_GRIPPER_OPEN_RAD:.3f}] rad -> "
        f"real [{REAL_GRIPPER_CLOSED_RAD:.4f}, {REAL_GRIPPER_OPEN_RAD:.4f}] rad "
        "(closed -> open)"
    )
    base.main()
    base.simulation_app.close()
