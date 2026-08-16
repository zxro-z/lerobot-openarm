"""Degree-output variant of the random-cube/tilt mapped dataset generator.

The simulator and IK controller continue to operate in radians.  Only the
eight values returned to the LeRobot recorder are converted to degrees:

    [joint_1, ..., joint_7, gripper]

This keeps Isaac Lab's native unit contract intact while producing a dataset
whose observation.state and action features are entirely degree based.
"""

from __future__ import annotations

import numpy as np

import openarm_table_dual_realsense_ik_pick_place_make_dataset_random_cube_random_tilt_gripper_mapped as mapped


base = mapped.base
_get_record_state_action_rad = base.PickPlaceController.get_record_state_action


def _get_record_state_action_degree(self):
    """Return mapped arm and gripper positions in degrees."""
    state_rad, action_rad = _get_record_state_action_rad(self)
    state_deg = np.rad2deg(state_rad).astype(np.float32)
    action_deg = np.rad2deg(action_rad).astype(np.float32)
    return state_deg, action_deg


base.PickPlaceController.get_record_state_action = _get_record_state_action_degree


if __name__ == "__main__":
    print(
        "[INFO] Dataset units: observation.state/action "
        "[joint_1, ..., joint_7, gripper] = degrees"
    )
    base.main()
    base.simulation_app.close()
