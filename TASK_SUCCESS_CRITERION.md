# OpenArm Pick-and-Place Task Success Criterion

## 1. Evaluation Path

Current reference-style evaluation uses:

- `scripts/eval/eval_smolvla.py`
- `scripts/eval/openarm_smolvla_env.py`

Execution path:

1. `scripts/eval/eval_smolvla.py:43-59`
   - parses CLI
   - imports `OpenArmEnv`, `TASK`, `ROBOT_TYPE`, `simulation_app` from `openarm_smolvla_env.py`
2. `scripts/eval/eval_smolvla.py:74-86`
   - builds policy and processors
   - creates `env = OpenArmEnv(max_episode_steps=args.max_steps)`
3. `scripts/eval/eval_smolvla.py:94-152`
   - evaluation loop
   - calls `obs, _, terminated, truncated, info = env.step(action_np)`
   - reads success from `info["is_success"]`
   - sets per-episode result from that success flag
4. `scripts/eval/eval_smolvla.py:153-172`
   - writes per-episode CSV
   - prints `[RESULT] successes=X/Y`

The actual task success condition is computed inside:

- `scripts/eval/openarm_smolvla_env.py`
- `OpenArmEnv.step()`
- lines `507-575`

Current native `lerobot-eval` path:

- `src/lerobot/scripts/lerobot_eval.py:191-220`

That script does not define the OpenArm success condition itself. It only reads `info["final_info"]["is_success"]` from the environment and aggregates it. So for OpenArm, the success criterion is still environment-defined, not `lerobot_eval.py`-defined.

## 2. Success Boolean Definition

Current success boolean is defined in:

- `scripts/eval/openarm_smolvla_env.py:549-560`

Code-level definition:

```python
cube_pos_w = self.cube.data.root_pos_w[0].detach().cpu().numpy()
(success_x_min, success_x_max), (success_y_min, success_y_max), (success_z_min, success_z_max) = (
    get_success_box_bounds()
)
in_box_x = success_x_min < cube_pos_w[0] < success_x_max
in_box_y = success_y_min < cube_pos_w[1] < success_y_max
in_box_z = success_z_min < cube_pos_w[2] < success_z_max

cube_vel_w = self.cube.data.root_vel_w[0].detach().cpu().numpy()
is_stationary = np.linalg.norm(cube_vel_w[:3]) < 0.05

is_success = bool(in_box_x and in_box_y and in_box_z and is_stationary)
```

Readable form:

`is_success = cube_inside_success_box_x AND cube_inside_success_box_y AND cube_inside_success_box_z AND cube_linear_speed_below_threshold`

There are no additional success requirements in the current code for:

- angular velocity
- gripper openness
- end-effector position
- robot joint state
- minimum step count
- consecutive stationary frames
- object release detection

## 3. Position Conditions

Success position bounds are produced by:

- `scripts/eval/openarm_smolvla_env.py:128-134`
- function `get_success_box_bounds()`

That function uses:

- `STORAGE_BOX_SIZE = (0.55, 0.75, 0.15)` at `scripts/eval/openarm_smolvla_env.py:92`
- `STORAGE_BOX_POS = (-0.53584, -0.15, 0.04664)` at `scripts/eval/openarm_smolvla_env.py:93`
- `SUCCESS_BOX_INNER_FRACTION_X = 0.20` at `scripts/eval/openarm_smolvla_env.py:99`
- `SUCCESS_BOX_INNER_FRACTION_Y = 0.40` at `scripts/eval/openarm_smolvla_env.py:100`

Expanded bounds:

- `half_x = 0.55 * 0.20 / 2 = 0.055`
- `half_y = 0.75 * 0.40 / 2 = 0.15`

So:

- `x` must satisfy `-0.59084 < cube_x < -0.48084`
- `y` must satisfy `-0.30 < cube_y < 0.00`
- `z` must satisfy `0.03664 < cube_z < 0.12164`

Important distinction:

- The storage box asset is placed at `STORAGE_BOX_POS`
- Success does not use the full outer box footprint
- Success uses a reduced interior region computed from the box center/size

## 4. Velocity / Stationary Conditions

Current stationary check is defined in:

- `scripts/eval/openarm_smolvla_env.py:557-558`

Code:

```python
cube_vel_w = self.cube.data.root_vel_w[0].detach().cpu().numpy()
is_stationary = np.linalg.norm(cube_vel_w[:3]) < 0.05
```

This means:

- only the linear velocity components `vx, vy, vz` are used
- angular velocity is ignored
- no multi-frame dwell time is required
- success can happen as soon as one step satisfies both position and linear-speed conditions

## 5. Coordinate Frames

The success computation compares positions in the world frame.

Evidence:

- cube position comes from `self.cube.data.root_pos_w`
  - `scripts/eval/openarm_smolvla_env.py:549`
  - suffix `_w` indicates world-frame tensor
- cube velocity comes from `self.cube.data.root_vel_w`
  - `scripts/eval/openarm_smolvla_env.py:557`
- storage box placement is defined directly in scene init state:
  - `scripts/eval/openarm_smolvla_env.py:274-285`
  - `init_state=AssetBaseCfg.InitialStateCfg(pos=STORAGE_BOX_POS)`

No transform into robot base frame or box-local frame is applied in the success check. The code directly compares world-frame cube coordinates against world-frame numeric box bounds.

## 6. Reward Definition

Reward is defined in:

- `scripts/eval/openarm_smolvla_env.py:560-561`

Code:

```python
reward = 1.0 if is_success else 0.0
```

So the relationship is exact:

- if `is_success == True`, then `reward == 1.0`
- if `is_success == False`, then `reward == 0.0`

There is no other reward shaping term in the current eval environment.

## 7. Termination / Truncation

Defined in:

- `scripts/eval/openarm_smolvla_env.py:563-564`

Code:

```python
terminated = is_success
truncated = self.current_step >= self.max_episode_steps
```

Implications:

- success immediately terminates the episode
- timeout truncates the episode when `current_step >= max_episode_steps`
- if success is false and max steps are not reached, the episode continues

Wrapper behavior:

- `scripts/eval/eval_smolvla.py:137-152`
- the loop breaks on `terminated or truncated`

Special current wrapper behavior for video saving:

- `scripts/eval/eval_smolvla.py:138-146`
- if success occurs and `--save-video` is enabled, the wrapper performs extra `env.step(action_np)` calls for tail recording before writing the MP4
- this does not change the original episode result already computed from the first success-triggering step

## 8. Numerical Thresholds

parameter: `STORAGE_BOX_POS.x`
value: `-0.53584`
unit: world-frame position
source file: `scripts/eval/openarm_smolvla_env.py`
source variable: `STORAGE_BOX_POS[0]`

parameter: `STORAGE_BOX_POS.y`
value: `-0.15`
unit: world-frame position
source file: `scripts/eval/openarm_smolvla_env.py`
source variable: `STORAGE_BOX_POS[1]`

parameter: `STORAGE_BOX_POS.z`
value: `0.04664`
unit: world-frame position
source file: `scripts/eval/openarm_smolvla_env.py`
source variable: `STORAGE_BOX_POS[2]`

parameter: `STORAGE_BOX_SIZE.x`
value: `0.55`
unit: world-frame length
source file: `scripts/eval/openarm_smolvla_env.py`
source variable: `STORAGE_BOX_SIZE[0]`

parameter: `STORAGE_BOX_SIZE.y`
value: `0.75`
unit: world-frame length
source file: `scripts/eval/openarm_smolvla_env.py`
source variable: `STORAGE_BOX_SIZE[1]`

parameter: `STORAGE_BOX_SIZE.z`
value: `0.15`
unit: world-frame length
source file: `scripts/eval/openarm_smolvla_env.py`
source variable: `STORAGE_BOX_SIZE[2]`

parameter: `SUCCESS_BOX_INNER_FRACTION_X`
value: `0.20`
unit: fraction
source file: `scripts/eval/openarm_smolvla_env.py`
source variable: `SUCCESS_BOX_INNER_FRACTION_X`

parameter: `SUCCESS_BOX_INNER_FRACTION_Y`
value: `0.40`
unit: fraction
source file: `scripts/eval/openarm_smolvla_env.py`
source variable: `SUCCESS_BOX_INNER_FRACTION_Y`

parameter: `success_x_range`
value: `(-0.59084, -0.48084)`
unit: world-frame position
source file: `scripts/eval/openarm_smolvla_env.py`
source variable: `get_success_box_bounds()`

parameter: `success_y_range`
value: `(-0.30, 0.00)`
unit: world-frame position
source file: `scripts/eval/openarm_smolvla_env.py`
source variable: `get_success_box_bounds()`

parameter: `success_z_range`
value: `(0.03664, 0.12164)`
unit: world-frame position
source file: `scripts/eval/openarm_smolvla_env.py`
source variable: `get_success_box_bounds()`

parameter: `linear_velocity_threshold`
value: `0.05`
unit: world-frame speed magnitude
source file: `scripts/eval/openarm_smolvla_env.py`
source variable: `np.linalg.norm(cube_vel_w[:3]) < 0.05`

parameter: `required_consecutive_steps`
value: `none`
unit: n/a
source file: `scripts/eval/openarm_smolvla_env.py`
source variable: no such variable exists

parameter: `angular_velocity_threshold`
value: `none`
unit: n/a
source file: `scripts/eval/openarm_smolvla_env.py`
source variable: no such variable exists

parameter: `min_step_before_success_check`
value: `none`
unit: n/a
source file: `scripts/eval/openarm_smolvla_env.py`
source variable: no such variable exists

## 9. Example Success and Failure Cases

CASE A — SUCCESS

- cube position: `(-0.53, -0.12, 0.08)`
- cube linear velocity: `(0.01, 0.00, -0.01)`
- speed norm: about `0.014`

Why it succeeds:

- `x = -0.53` is inside `(-0.59084, -0.48084)`
- `y = -0.12` is inside `(-0.30, 0.00)`
- `z = 0.08` is inside `(0.03664, 0.12164)`
- linear speed `0.014 < 0.05`

CASE B — POSITION FAIL

- cube position: `(-0.53, 0.05, 0.08)`
- cube linear velocity: `(0.00, 0.00, 0.00)`

Why it fails:

- `x` is valid
- `z` is valid
- speed is stationary
- but `y = 0.05` is outside success range because the allowed range is `(-0.30, 0.00)`

CASE C — STATIONARY FAIL

- cube position: `(-0.53, -0.10, 0.08)`
- cube linear velocity: `(0.04, 0.04, 0.00)`
- speed norm: about `0.0566`

Why it fails:

- position passes all three axes
- but `sqrt(0.04^2 + 0.04^2) = 0.0566 > 0.05`
- so `is_stationary` is false

## 10. Successful Rollout Evidence

Current `ab_local_verified` artifacts found:

- `outputs/eval/ab_local_verified/videos/episode_000_success_1.mp4`
- `outputs/eval/ab_local_verified/videos/episode_001_success_1.mp4`
- `outputs/eval/ab_local_verified/videos/episode_002_success_1.mp4`
- `outputs/eval/ab_local_verified/videos/episode_003_success_1.mp4`
- `outputs/eval/ab_local_verified/videos/episode_004_success_1.mp4`
- `outputs/eval/ab_local_verified/videos/episode_005_success_0.mp4`
- `outputs/eval/ab_local_verified/videos/episode_006_success_1.mp4`

However, the current run does not have a saved `results.csv` or step-by-step diagnostic JSON under `outputs/eval/ab_local_verified/`.

Therefore:

- success/failure at the file level is observable from video filenames
- per-step values for the current successful run are not available from saved artifacts
- success-triggering state for the current `ab_local_verified` rollout is `UNKNOWN`

There are older diagnostic JSON files in other output folders, but those correspond to earlier diagnostic environments and should not be treated as direct evidence for the current eval wrapper unless explicitly cross-validated.

## 11. Potential Edge Cases

cube가 box 위쪽에 떠 있어도 성공 가능한지

- Yes, if `cube_z` is within `(0.03664, 0.12164)` and linear speed norm is `< 0.05`.
- The current code does not check contact with the box floor.
- It only checks world-frame z interval and linear speed.

cube가 box 경계에 걸쳐 있어도 성공 가능한지

- Only the cube root position is checked.
- The code does not use cube extent, corners, or overlap volume.
- So a cube can be partially hanging out and still succeed if its root position is inside the bounds.

cube가 빠르게 움직이는 동안 성공 가능한지

- No, not if linear speed norm is `>= 0.05`.
- Angular speed does not matter because it is not checked.

gripper가 cube를 잡고 있는 상태에서도 success 가능한지

- Yes.
- There is no condition requiring the gripper to open or release.
- There is no EE/gripper contact condition in the success logic.

success가 한 frame만 true여도 종료되는지

- Yes.
- `terminated = is_success`
- No consecutive-frame confirmation exists.
- One step satisfying the boolean is enough to end the episode.

## SUCCESS_CRITERION_SUMMARY

The current OpenArm eval environment declares success when the cube root position is inside a reduced storage-box region in world coordinates and the cube linear speed norm is below `0.05`.

Numerically, success requires:

- `-0.59084 < cube_x < -0.48084`
- `-0.30 < cube_y < 0.00`
- `0.03664 < cube_z < 0.12164`
- `sqrt(vx^2 + vy^2 + vz^2) < 0.05`

If that boolean is true for a single step, reward becomes `1.0`, `terminated=True`, and the episode ends.
