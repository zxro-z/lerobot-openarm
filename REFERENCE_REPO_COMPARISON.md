# SmolVLA Reference Repository Comparison

## 1. Executive Summary

현재 repository와 성공한 동료 repository의 공통 upstream 코드는 거의 동일하다. 제외 규칙을 적용한 구조 비교 결과는 다음과 같다.

- current files: 667
- reference files: 629
- common files: 629
- current only: 38
- reference only: 0
- same: 622
- different: 7

핵심 결론은 다음과 같다.

- 현재 실패/성공 차이를 설명하는 가장 큰 코드 차이는 upstream SmolVLA 자체가 아니라, 현재 repo에만 추가된 OpenArm 통합 코드와 reference wrapper path 차이다.
- reference wrapper의 train 경로는 `reference_success/train_smolvla.py`가 `lerobot-train --policy.path=lerobot/smolvla_base ...`를 실행하는 형태다. 즉 base checkpoint를 명시적으로 로드한다.
- 현재 repo의 `src/lerobot/configs/train.py` auto-load patch는 reference reproduction의 source of truth가 아니다. reproduction 목표는 이 patch에 의존하지 않고도 reference wrapper command가 그대로 실행되게 만드는 것이다.
- reference eval은 `lerobot-eval`을 쓰지 않는다. `reference_success/eval_smolvla.py`가 custom `OpenArmEnv`와 `predict_action()`을 직접 사용한다.
- 현재 실험에서 이미 확인된 `GRIPPER_PRIMARY_FAILURE = CONFIRMED`는 reference code diff와도 정합적이다. current OpenArm env에는 reference에 없는 `dataset_schedule` 그리퍼 오버라이드가 있고, 이 오버라이드가 success를 회복했다. 이는 기존 checkpoint 기준으로 policy가 그리퍼 timing/control을 충분히 학습하지 못했거나, eval path에서 그리퍼 제어가 behaviorally mismatch였음을 강하게 시사한다. 다만 reference wrapper path로 다시 학습하면 이 문제가 해결된다고 아직 단정할 수는 없다.

## 2. Repositories Compared

Current repository root:

- `/home/zxro/arena/lerobot`

Reference repository root:

- `/home/zxro/arena/lerobot/reference_success/lerobot_SG/lerobot`

Auxiliary reference wrappers:

- `/home/zxro/arena/lerobot/reference_success/train_smolvla.py`
- `/home/zxro/arena/lerobot/reference_success/eval_smolvla.py`
- `/home/zxro/arena/lerobot/reference_success/openarm_smolvla_env.py`
- `/home/zxro/arena/lerobot/reference_success/openarm_table_dual_realsense_ik_pick_place_make_dataset_random_cube_random_tilt_gripper_mapped_degree.py`

Dataset path to use in current environment:

- `/home/zxro/arena/lerobot/outputs/lerobot_datasets/random_cube_tilt_30_gripper_mapped_box_blue_50_degree`

Git/version evidence:

- current branch: `main`
- current HEAD: `d324ffe8`
- current describe: `v0.4.4-14-gd324ffe8-dirty`
- reference HEAD: `d324ffe8`
- reference describe: `v0.4.4-14-gd324ffe8-dirty`
- `pyproject.toml` version in both repos: `0.4.5`

Reference repo git state:

- detached HEAD
- one local diff in `src/lerobot/policies/groot/groot_n1.py`
- no SmolVLA-specific source diff inside reference repo itself

## 3. Experimental Background

Experiment goals must be kept distinct.

REPRODUCTION GOAL:

- reproduce the successful colleague train/eval execution path in the current repository

ROOT-CAUSE GOAL:

- check whether a newly trained checkpoint from the same reference path still exhibits gripper timing failure

Already verified facts from prior experiments:

- `DEGREE_DATASET_CONTRACT = PASS`
- `GRIPPER_DATASET_CONTRACT = PASS`
- `DEGREE_CHECKPOINT_CONTRACT = PASS`
- `DEGREE_EVAL_CONTRACT = PASS`
- `GRIPPER_FAILURE_DIAGNOSIS = PASS`

Fixed dataset gripper schedule ablation:

- `NORMAL`: success=False, reward_sum=0, min_EE_cube_distance=0.01579, cube_displacement=0.0420, cube_height_change=0.0231
- `DATASET_SCHEDULE`: success=True, terminated_step=449, reward_sum=1, min_EE_cube_distance=0.01719, cube_displacement=0.13795, cube_height_change=0.11146

Interpretation:

- degree/radian mismatch is not the primary remaining issue
- dataset/checkpoint contract is already consistent
- gripper timing/control remains the dominant behavioral failure axis
- this conclusion is confirmed for the existing checkpoint and eval path already tested
- it does not yet prove that retraining with the reference wrapper path will fix the problem

## 4. Train Architecture Comparison

### A. Current `lerobot-train`

CLI entry:

- `pyproject.toml` -> `lerobot-train="lerobot.scripts.lerobot_train:main"`
- implementation: `src/lerobot/scripts/lerobot_train.py`

Path:

1. CLI parse via `@parser.wrap()` in `src/lerobot/scripts/lerobot_train.py`
2. config validate in `TrainPipelineConfig.validate()` at `src/lerobot/configs/train.py`
3. dataset metadata via `LeRobotDatasetMetadata(...)` in `src/lerobot/datasets/factory.py:87-91`
4. dataset via `LeRobotDataset(...)` in `src/lerobot/datasets/factory.py:93-102`
5. feature inference via `make_policy(..., ds_meta=dataset.meta)` in `src/lerobot/scripts/lerobot_train.py`
6. policy config finalize in `src/lerobot/policies/factory.py:460-477`
7. pretrained loading in `src/lerobot/policies/factory.py:488-492`
8. pre/post processor creation in `src/lerobot/scripts/lerobot_train.py` and `src/lerobot/policies/factory.py`
9. normalization via `NormalizerProcessorStep` in `src/lerobot/policies/smolvla/processor_smolvla.py:80-88`
10. dataloader creation in `src/lerobot/scripts/lerobot_train.py`
11. forward/loss in `SmolVLAPolicy.forward()` at `src/lerobot/policies/smolvla/modeling_smolvla.py:356-401`
12. optimizer/scheduler presets from `SmolVLAConfig` at `src/lerobot/policies/smolvla/configuration_smolvla.py:130-144`
13. checkpoint save via `save_checkpoint` and `update_last_checkpoint` in `src/lerobot/scripts/lerobot_train.py`

### B. Reference repo `lerobot-train`

Same as upstream current except one critical absence:

- `reference_success/lerobot_SG/lerobot/src/lerobot/configs/train.py` does not auto-initialize SmolVLA from `lerobot/smolvla_base` when only `policy.type=smolvla` is provided

Otherwise train path is the same as current upstream.

### C. Reference wrapper `reference_success/train_smolvla.py`

Entry:

- `reference_success/train_smolvla.py`

Actual command emitted:

- `lerobot-train --policy.path=lerobot/smolvla_base --dataset.repo_id=... --dataset.root=... --dataset.video_backend=torchcodec --batch_size=4 --num_workers=4 --steps=20000 --save_freq=2000 --log_freq=100 --output_dir=... --job_name=random_cube_tilt_30_blue_box_smolvla --policy.device=cuda --policy.use_amp=true --policy.push_to_hub=false --wandb.enable=false`

Wrapper behavior:

- validates dataset metadata before launch
- forces pretrained SmolVLA base load through `--policy.path`
- is the actual reference training source of truth
- does not rely on current repo's patched auto-load branch in `TrainPipelineConfig.validate()`

### Train Differences

[TR-001]

file:

- current: `src/lerobot/configs/train.py`
- reference: `reference_success/lerobot_SG/lerobot/src/lerobot/configs/train.py`
- reference wrapper: `reference_success/train_smolvla.py`

function/class:

- `TrainPipelineConfig.validate`

CURRENT:

- if `policy_path` is absent and `policy.type == "smolvla"`, current auto-loads `lerobot/smolvla_base` and forces `--input_features=null`, `--output_features=null`

REFERENCE:

- no such branch exists

REFERENCE WRAPPER:

- always passes `--policy.path=lerobot/smolvla_base`

실제 training path에 들어감:

- YES

checkpoint에 영향을 줄 수 있음:

- YES

training quality에 영향을 줄 가능성:

- MEDIUM

confidence:

- CONFIRMED

reason:

- the patch changes current fallback behavior, but reference reproduction does not require it as long as the wrapper keeps passing `--policy.path=lerobot/smolvla_base`
- without pretrained path, `make_policy()` only loads weights when `cfg.pretrained_path` is set (`src/lerobot/policies/factory.py:488-492`)
- `SmolVLAConfig.load_vlm_weights` default is `False` (`src/lerobot/policies/smolvla/configuration_smolvla.py:88`)

[TR-002]

file:

- current: `src/lerobot/policies/factory.py`
- reference: same file in reference repo
- reference wrapper: `reference_success/train_smolvla.py`

function/class:

- `make_policy`

CURRENT:

- infers `cfg.output_features` from dataset/env features
- if `cfg.input_features` is empty, infers them from dataset/env features
- passes `dataset.meta.stats` into policy/processor when metadata exists

REFERENCE:

- same

REFERENCE WRAPPER:

- dataset contract ensures action/state shape `(8,)` and cameras `top`, `wrist` before invoking same logic

실제 training path에 들어감:

- YES

checkpoint에 영향을 줄 수 있음:

- YES

training quality에 영향을 줄 가능성:

- MEDIUM

confidence:

- CONFIRMED

reason:

- wrapper-side metadata validation prevents silent mismatch before the common train path starts

[TR-003]

file:

- current: `src/lerobot/policies/smolvla/processor_smolvla.py`
- reference: same
- reference wrapper: `reference_success/train_smolvla.py`

function/class:

- `make_smolvla_pre_post_processors`

CURRENT:

- normalizes `STATE` and `ACTION` using dataset stats
- leaves `VISUAL` identity
- tokenizes `task` with newline canonicalization

REFERENCE:

- same

REFERENCE WRAPPER:

- uses same processor path indirectly through `lerobot-train`

실제 training path에 들어감:

- YES

checkpoint에 영향을 줄 수 있음:

- YES

training quality에 영향을 줄 가능성:

- LOW

confidence:

- CONFIRMED

reason:

- no code difference here; this area is not a repo diff root cause

[TR-004]

file:

- current: `src/lerobot/datasets/factory.py`
- reference: same
- reference wrapper: `reference_success/train_smolvla.py`

function/class:

- `make_dataset`

CURRENT:

- resolves action chunk timestamps from `cfg.action_delta_indices`
- loads dataset meta from local root/repo id
- passes `video_backend` from CLI

REFERENCE:

- same

REFERENCE WRAPPER:

- forces `--dataset.video_backend=torchcodec`

실제 training path에 들어감:

- YES

checkpoint에 영향을 줄 수 있음:

- POSSIBLY

training quality에 영향을 줄 가능성:

- LOW

confidence:

- LIKELY

reason:

- backend affects decoding/runtime stability, not feature semantics; still part of exact reproducibility

## 5. Eval Architecture Comparison

### A. Current `lerobot-eval` + current OpenArm env

Path:

1. `lerobot-eval` -> `src/lerobot/scripts/lerobot_eval.py`
2. config parse via `EvalPipelineConfig`
3. env creation via `make_env()` in `src/lerobot/envs/factory.py`
4. raw observation conversion via `preprocess_observation()` in `src/lerobot/envs/utils.py`
5. task injection via `add_envs_task()` in `src/lerobot/envs/utils.py`
6. policy pre/post processors via `make_pre_post_processors()`
7. `policy.select_action()` in rollout loop
8. action chunk queue handled inside `SmolVLAPolicy.select_action()`
9. env.step
10. reward/success metrics inside `rollout()`

### B. Reference repo eval/OpenArm implementation

Inside reference repo there is no built-in OpenArm env integration in `src/lerobot/envs/*`.

- reference `src/lerobot/envs/configs.py` has no `OpenArmEnv`
- reference `src/lerobot/envs/factory.py` has no `openarm` case
- reference `src/lerobot/envs/utils.py` has no explicit support for `observation.images.*` keys

This means OpenArm eval in the successful setup is not a native `lerobot-eval` path.

### C. Reference wrapper `eval_smolvla.py` + `openarm_smolvla_env.py`

Path:

1. parse args in `reference_success/eval_smolvla.py`
2. import `openarm_smolvla_env.py`, which starts Isaac Lab
3. load dataset metadata via `LeRobotDatasetMetadata`
4. load `policy_cfg = PreTrainedConfig.from_pretrained(policy_path)`
5. `make_policy(policy_cfg, ds_meta=metadata)`
6. `make_pre_post_processors(policy_cfg, pretrained_path=policy_path)`
7. loop:
8. `obs, _ = env.reset(seed=...)`
9. `policy.reset(); preprocessor.reset(); postprocessor.reset()`
10. `predict_action(observation=obs, policy=policy, preprocessor=preprocessor, postprocessor=postprocessor, task=TASK, robot_type=ROBOT_TYPE)`
11. `env.step(action_np)`
12. success from `info["is_success"]`

This bypasses:

- `lerobot-eval`
- vector env wrappers
- `preprocess_observation()`
- `add_envs_task()`
- env pre/post processors in `src/lerobot/envs/factory.py`

### Eval Differences

[EV-001]

file:

- current: `src/lerobot/scripts/lerobot_eval.py`
- reference: `reference_success/lerobot_SG/lerobot/src/lerobot/scripts/lerobot_eval.py`
- reference wrapper: `reference_success/eval_smolvla.py`

function/class:

- `rollout`

CURRENT:

- vectorized env rollout, uses `preprocess_observation()`, `add_envs_task()`, `policy.select_action()`
- current local patch adds fallback hardcoded task prompt

REFERENCE:

- same upstream code

REFERENCE WRAPPER:

- does not call `rollout()` at all
- calls `predict_action()` directly on a non-vector custom env

실제 training path에 들어감:

- NO

checkpoint에 영향을 줄 수 있음:

- NO

training quality에 영향을 줄 가능성:

- LOW

confidence:

- CONFIRMED

reason:

- difference matters for eval behavior, not train checkpoint generation

[EV-002]

file:

- current: `src/lerobot/envs/configs.py`, `src/lerobot/envs/factory.py`, `src/lerobot/envs/utils.py`
- reference: corresponding files in reference repo
- reference wrapper: `reference_success/openarm_smolvla_env.py`

function/class:

- `OpenArmEnv`, `make_env`, `preprocess_observation`, `add_envs_task`

CURRENT:

- adds native `openarm` env config and factory path
- adds explicit preprocessing for `observation.images.top`, `observation.images.wrist`, `observation.state`
- adds task injection fallbacks and vector wrapper attribute injection

REFERENCE:

- none of these OpenArm integration changes exist

REFERENCE WRAPPER:

- custom env already emits the exact observation dict expected by `prepare_observation_for_inference()`
- task string is passed explicitly to `predict_action()`

실제 training path에 들어감:

- NO

checkpoint에 영향을 줄 수 있음:

- NO

training quality에 영향을 줄 가능성:

- LOW

confidence:

- CONFIRMED

reason:

- these changes are compatibility plumbing for native `lerobot-eval`; the successful reference path bypasses them

[EV-003]

file:

- current: `gym_openarm/openarm_env.py`
- reference: `reference_success/openarm_smolvla_env.py`
- reference wrapper: `reference_success/eval_smolvla.py`

function/class:

- `OpenArmEnv.step`, `OpenArmEnv._get_obs`

CURRENT:

- accepts degree action `(8,)`
- converts to radians
- inverse-maps real gripper range `[-15 deg, -60 deg]` to sim range `[0.0, 0.044]`
- includes optional `gripper_override_mode="dataset_schedule"`

REFERENCE:

- same degree-based action contract
- same gripper inverse mapping
- no dataset schedule override path

REFERENCE WRAPPER:

- directly uses this custom env contract

실제 training path에 들어감:

- NO

checkpoint에 영향을 줄 수 있음:

- NO

training quality에 영향을 줄 가능성:

- HIGH for closed-loop success

confidence:

- CONFIRMED

reason:

- same unit contract, but current-only schedule override materially changes success

[EV-004]

file:

- current: `gym_openarm/openarm_env.py`
- reference: `reference_success/openarm_smolvla_env.py`
- reference wrapper: `reference_success/eval_smolvla.py`

function/class:

- `OpenArmEnv.reset`

CURRENT:

- resets `cube_rng` with fixed seed 15 set at init
- does not reseed RNG from `reset(seed=...)`

REFERENCE:

- on reset, if seed provided, rebuilds `cube_rng = np.random.default_rng(seed)`

REFERENCE WRAPPER:

- calls `env.reset(seed=args.seed + episode)`

실제 training path에 들어감:

- NO

checkpoint에 영향을 줄 수 있음:

- NO

training quality에 영향을 줄 가능성:

- LOW

confidence:

- CONFIRMED

reason:

- affects episode reproducibility during eval, not model weights

## 6. Dataset / Feature Contract Comparison

Confirmed same contract across successful reference wrapper and current validated experiments:

- observation key: `observation.state`
- image keys: `observation.images.top`, `observation.images.wrist`
- state dim: 8
- action dim: 8
- task prompt: `Pick up the red cube and place it in the storage box.`
- dataset unit contract: all 8 state/action values are stored in degrees in the successful degree dataset wrapper

Evidence:

- `reference_success/train_smolvla.py` validates shape `[8]` and camera set
- `reference_success/openarm_table_dual_realsense_ik_pick_place_make_dataset_random_cube_random_tilt_gripper_mapped_degree.py` converts both state and action from radians to degrees before recording
- `gym_openarm/openarm_env.py` and `reference_success/openarm_smolvla_env.py` both return observation.state in degrees during eval

Implication:

- state/action dimensionality and degree/radian representation are not the remaining primary mismatch

## 7. SmolVLA Configuration Differences

SmolVLA source files inside current and reference repo are identical:

- `src/lerobot/policies/smolvla/configuration_smolvla.py`
- `src/lerobot/policies/smolvla/processor_smolvla.py`
- `src/lerobot/policies/smolvla/modeling_smolvla.py`

Important defaults:

- `chunk_size = 50`
- `n_action_steps = 50`
- `normalization_mapping = {VISUAL: IDENTITY, STATE: MEAN_STD, ACTION: MEAN_STD}`
- `load_vlm_weights = False`

Crucial interaction:

- `load_vlm_weights=False` is safe only if policy weights are loaded from a pretrained checkpoint path
- reference wrapper guarantees that via `--policy.path=lerobot/smolvla_base`
- unpatched upstream `lerobot-train` does not

## 8. Normalization / Processor Differences

No SmolVLA processor diff exists between current and reference repos.

Shared behavior:

- preprocessor adds batch dim
- appends newline to task prompt
- tokenizes language
- normalizes state/action using dataset stats
- postprocessor unnormalizes action

Potentially relevant shared assumption:

- dataset stats must match the degree-based dataset used for both training and eval metadata

Assessment:

- not a repo diff root cause
- still part of exact migration because reference eval loads `LeRobotDatasetMetadata` and then calls `make_policy(..., ds_meta=metadata)`

## 9. OpenArm Environment Differences

Current-only files:

- `gym_openarm/__init__.py`
- `gym_openarm/openarm_env.py`
- `gym_openarm/openarm_table_dual_realsense_ik_pick_place_make_dataset_random_cube_random_tilt_gripper_mapped.py`
- `src/lerobot/assets/openarm_use/*`

Reference-only at wrapper level:

- `reference_success/openarm_smolvla_env.py`
- `reference_success/openarm_table_dual_realsense_ik_pick_place_make_dataset_random_cube_random_tilt_gripper_mapped_degree.py`

Important differences:

- current env launches Isaac Lab headless by default
- reference env launches with GUI (`headless: False`)
- current env adds `gripper_override_mode`
- current env logs extra unit-debug info
- current env reset seed handling differs from reference

Important same behaviors:

- degree input action contract
- gripper inverse mapping from real degree scale to sim range
- observation.state emitted in degrees
- dual camera observation keys
- 4 physics substeps per control step
- same success condition geometry and stationarity test

## 10. Gripper Pipeline Differences

This section is the strongest connection between repo diff and prior experimental evidence.

[GR-001]

current file:

- `gym_openarm/openarm_env.py:395-407`

reference file:

- no equivalent logic in `reference_success/openarm_smolvla_env.py`

current behavior:

- can override policy gripper output with a hardcoded dataset-derived schedule:
  - steps 0-14: `-15`
  - steps 15-253: `-60`
  - steps 254-443: `-15`
  - 444+: `-60`

reference behavior:

- always uses policy-predicted gripper output

impact:

- directly changes closed-loop success

confidence:

- CONFIRMED

This exactly matches the prior ablation where replacing policy gripper timing with dataset-like timing flips failure to success.

[GR-002]

current file:

- `gym_openarm/openarm_env.py:476-503`

reference file:

- `reference_success/openarm_smolvla_env.py:469-489`

current behavior:

- converts degree gripper target to radians and maps from real range to sim finger joints

reference behavior:

- same

impact:

- proves the remaining issue is not the basic gripper unit/inverse-mapping formula

confidence:

- CONFIRMED

[GR-003]

current file:

- `reference_success/train_smolvla.py`
- `src/lerobot/configs/train.py`

reference file:

- same wrapper

current behavior:

- patched current `TrainPipelineConfig.validate()` now emulates wrapper behavior by auto-using `smolvla_base`

reference behavior:

- wrapper explicitly fine-tunes from base

impact:

- likely improves gripper behavior because training starts from pretrained SmolVLA policy rather than scratch/random expert state

confidence:

- LIKELY

Why this matters for gripper:

- gripper timing is often the hardest low-margin output in pick-place tasks
- if earlier runs accidentally trained from scratch or from a different initialization path, the gripper dimension can degrade first while arm approach still looks plausible

## 11. Dependency / Version Differences

Repository metadata:

- `pyproject.toml` identical in current and reference repo
- package version identical: `0.4.5`
- git commit identical: `d324ffe8`

Known repo-local differences outside SmolVLA:

- current repo has 7 modified tracked files
- reference repo has 1 modified tracked file (`groot_n1.py`)

Dependency conclusion:

- there is no repository-level dependency/version divergence explaining SmolVLA behavior difference

## 12. Confirmed Root Causes

1. Pretrained SmolVLA base initialization via the wrapper command is required to match the successful wrapper behavior.

- current file: `src/lerobot/configs/train.py`
- reference file: `reference_success/train_smolvla.py`
- current behavior: a local patch exists, but reproduction should not depend on it
- reference behavior: wrapper explicitly passes `--policy.path=lerobot/smolvla_base`
- impact: if older/current runs omitted this, resulting checkpoints are materially different
- confidence: CONFIRMED

2. The successful eval path is not `lerobot-eval`; it is `predict_action()` plus a custom OpenArm env.

- current file: `src/lerobot/scripts/lerobot_eval.py`
- reference file: `reference_success/eval_smolvla.py`
- current behavior: vector env rollout path with env preprocess plumbing
- reference behavior: direct single-env loop
- impact: exact successful command form must follow wrapper path, not native `lerobot-eval`
- confidence: CONFIRMED

3. Gripper timing/control is the primary residual behavioral failure axis for the existing checkpoint already tested.

- current file: `gym_openarm/openarm_env.py`
- reference file: `reference_success/openarm_smolvla_env.py`
- current behavior: only current diagnostic env has schedule override proving policy-gripper weakness
- reference behavior: policy must succeed without such override
- impact: failure is not arm reach or unit contract; it is gripper learning/control quality
- confidence: CONFIRMED

## 13. Likely Contributing Causes

1. Earlier current training may have diverged because SmolVLA was not initialized from `lerobot/smolvla_base`.

- impact: HIGH
- confidence: LIKELY

2. Wrapper-side dataset validation may have prevented subtle train-time contract mistakes that native `lerobot-train` alone would not catch early.

- impact: MEDIUM
- confidence: LIKELY

3. Reproducing eval via `lerobot-eval` instead of the wrapper loop may have introduced extra plumbing variables without matching the successful setup.

- impact: MEDIUM
- confidence: LIKELY

## 14. Differences Proven Irrelevant

1. SmolVLA internal code under `src/lerobot/policies/smolvla/` is identical between current and reference repos.

- impact: no repo diff root cause here
- confidence: CONFIRMED

2. `pyproject.toml` and package version metadata are identical.

- impact: no dependency metadata root cause here
- confidence: CONFIRMED

3. Degree/radian dataset and checkpoint contract has already been experimentally validated.

- impact: not the remaining primary issue
- confidence: CONFIRMED

4. Gripper inverse mapping formula in eval env is the same between current and reference custom envs.

- impact: not the remaining primary issue
- confidence: CONFIRMED

## 15. Current Repository Custom Modifications

Tracked diffs relative to reference repo:

- `src/lerobot/configs/train.py`
- `src/lerobot/datasets/v30/augment_dataset_quantile_stats.py`
- `src/lerobot/envs/configs.py`
- `src/lerobot/envs/factory.py`
- `src/lerobot/envs/utils.py`
- `src/lerobot/policies/groot/groot_n1.py`
- `src/lerobot/scripts/lerobot_eval.py`

Current-only untracked OpenArm-related additions:

- `gym_openarm/*`
- `src/lerobot/assets/openarm_use/*`
- `scripts/openarm_dataset_scale_diagnostic.py`
- `scripts/openarm_rollout_diagnostic.py`

Assessment:

- only `src/lerobot/configs/train.py` clearly aligns current behavior closer to reference wrapper behavior
- most `src/lerobot/envs/*` and `src/lerobot/scripts/lerobot_eval.py` changes are convenience plumbing for native `lerobot-eval`, not the successful reference path
- `gym_openarm/openarm_env.py` is a diagnostic/custom env, not source-of-truth reference behavior

## 16. Changes Required to Match Reference

### P0

1. P0-1: port `reference_success/train_smolvla.py` to `scripts/train/train_smolvla.py`.

- current file: no current equivalent wrapper
- reference file: `reference_success/train_smolvla.py`
- 적용할 변경: preserve reference wrapper CLI and `lerobot-train --policy.path=lerobot/smolvla_base` launch behavior
- 그대로 copy 가능한지: mostly
- 현재 repo 구조에 맞춰 port해야 하는지: yes, for current dataset path defaults
- regression risk: low
- 검증 방법: emitted train command must include `--policy.path=lerobot/smolvla_base`

2. P0-2: port `reference_success/eval_smolvla.py` to `scripts/eval/eval_smolvla.py`.

- current file: no current equivalent wrapper
- reference file: `reference_success/eval_smolvla.py`
- 적용할 변경: preserve reference wrapper CLI and direct `predict_action()` eval loop
- 그대로 copy 가능한지: mostly
- 현재 repo 구조에 맞춰 port해야 하는지: yes, for local import path and current dataset path defaults
- regression risk: low
- 검증 방법: wrapper help and smoke eval should run without touching `lerobot-eval`

3. P0-3: port `reference_success/openarm_smolvla_env.py` so reference behavior runs as-is.

- current file: no exact equivalent under target wrapper path
- reference file: `reference_success/openarm_smolvla_env.py`
- 적용할 변경: preserve env behavior; only resolve asset path for current repo layout
- 그대로 copy 가능한지: mostly
- 현재 repo 구조에 맞춰 port해야 하는지: yes
- regression risk: medium
- 검증 방법: wrapper import, reset, observation keys, and 20-step eval smoke

4. P0-4: make the colleague-compatible command interface actually run in the current environment.

- current file: wrapper path not yet present
- reference file: wrapper trio above
- 적용할 변경: keep successful colleague CLI form and update only local paths
- 그대로 copy 가능한지: no, requires integration glue
- 현재 repo 구조에 맞춰 port해야 하는지: yes
- regression risk: medium
- 검증 방법: `--help`, dataset validation, train dry validation, 10-step train smoke, 20-step eval smoke

### P1

1. Add dataset validation / seed / portability improvements around the wrapper path.

- current file: `scripts/train/train_smolvla.py`, `scripts/eval/eval_smolvla.py`, `scripts/eval/openarm_smolvla_env.py`
- reference file: `reference_success/train_smolvla.py`, `reference_success/eval_smolvla.py`, `reference_success/openarm_smolvla_env.py`
- 적용할 변경: keep the port stable across machines and preserve seeded behavior
- 그대로 copy 가능한지: mostly
- 현재 repo 구조에 맞춰 port해야 하는지: yes
- regression risk: medium
- 검증 방법: seeded resets should reproduce reference cube positions

2. Add wrapper-side dataset validation to reduce silent contract drift.

- current file: no equivalent wrapper
- reference file: `reference_success/train_smolvla.py`
- 적용할 변경: validate feature shapes and camera keys before training
- 그대로 copy 가능한지: yes
- 현재 repo 구조에 맞춰 port해야 하는지: minimal
- regression risk: low
- 검증 방법: invalid dataset should fail before `lerobot-train`

### P2

1. Decide whether native `lerobot-eval` OpenArm support should remain.

- current file: `src/lerobot/envs/configs.py`, `src/lerobot/envs/factory.py`, `src/lerobot/envs/utils.py`, `src/lerobot/scripts/lerobot_eval.py`
- reference file: no equivalent
- 적용할 변경: keep only if still needed for convenience; it is not part of the successful source-of-truth path
- 그대로 copy 가능한지: not applicable
- 현재 repo 구조에 맞춰 port해야 하는지: not applicable
- regression risk: medium because current experiments may depend on it
- 검증 방법: only after wrapper path is working

2. Clean up unrelated local diffs such as quantile stat push/tag suppression if they are not part of SmolVLA migration.

- current file: `src/lerobot/datasets/v30/augment_dataset_quantile_stats.py`
- reference file: reference counterpart
- 적용할 변경: optional cleanup
- 그대로 copy 가능한지: yes
- 현재 repo 구조에 맞춰 port해야 하는지: no
- regression risk: low
- 검증 방법: run that specific dataset tool if needed

## 17. Migration Plan

1. Port `reference_success/train_smolvla.py` to `scripts/train/train_smolvla.py`.
2. Port `reference_success/eval_smolvla.py` to `scripts/eval/eval_smolvla.py`.
3. Port `reference_success/openarm_smolvla_env.py` with only asset path resolution changes.
4. Validate that the colleague-compatible CLI works without depending on core upstream source edits.
5. Keep current experimental facts as regression tests, especially the gripper ablation evidence.
6. Only after wrapper-equivalent commands work, decide whether native `lerobot-eval` OpenArm support should be retained.

## 18. Validation Plan

1. Train command validation.

- confirm emitted command includes `--policy.path=lerobot/smolvla_base`
- confirm dataset root/repo id/video backend match reference wrapper
- confirm saved config/checkpoint contains expected `policy.pretrained_path`

2. Eval command validation.

- confirm eval path uses wrapper-style direct `predict_action()` loop
- confirm observation keys are exactly `observation.state`, `observation.images.top`, `observation.images.wrist`
- confirm degree-based action contract remains intact

3. Behavioral validation.

- run identical checkpoint through reference-style eval path
- verify gripper behavior without schedule override
- compare success rate and per-episode steps with reference

4. Diagnostic regression validation.

- rerun the current gripper schedule ablation
- if a wrapper-equivalent retrained checkpoint improves the issue, `NORMAL` should improve toward `DATASET_SCHEDULE`

## 19. Target Train Command

Reference source-of-truth form:

```bash
python scripts/train/train_smolvla.py \
  --dataset-root /home/zxro/arena/lerobot/outputs/lerobot_datasets/random_cube_tilt_30_gripper_mapped_box_blue_50_degree \
  --dataset-repo-id local/random_cube_tilt_30_gripper_mapped_box_blue_50_degree \
  --policy-path lerobot/smolvla_base \
  --output-dir /home/zxro/arena/lerobot/outputs/train/random_cube_tilt_30_blue_box_smolvla \
  --steps 20000 \
  --batch-size 4 \
  --num-workers 4 \
  --save-freq 2000 \
  --log-freq 100 \
  --device cuda \
  --use-amp \
  --no-wandb
```

Equivalent underlying command:

```bash
lerobot-train \
  --policy.path=lerobot/smolvla_base \
  --dataset.repo_id=local/random_cube_tilt_30_gripper_mapped_box_blue_50_degree \
  --dataset.root=/home/zxro/arena/lerobot/outputs/lerobot_datasets/random_cube_tilt_30_gripper_mapped_box_blue_50_degree \
  --dataset.video_backend=torchcodec \
  --batch_size=4 \
  --num_workers=4 \
  --steps=20000 \
  --save_freq=2000 \
  --log_freq=100 \
  --output_dir=/home/zxro/arena/lerobot/outputs/train/random_cube_tilt_30_blue_box_smolvla \
  --job_name=random_cube_tilt_30_blue_box_smolvla \
  --policy.device=cuda \
  --policy.use_amp=true \
  --policy.push_to_hub=false \
  --wandb.enable=false
```

## 20. Target Eval Command

Reference source-of-truth form:

```bash
python scripts/eval/eval_smolvla.py \
  --policy-path /ABS/PATH/TO/CHECKPOINT/pretrained_model \
  --dataset-root /home/zxro/arena/lerobot/outputs/lerobot_datasets/random_cube_tilt_30_gripper_mapped_box_blue_50_degree \
  --dataset-repo-id local/random_cube_tilt_30_gripper_mapped_box_blue_50_degree \
  --num-episodes 10 \
  --max-steps 1000 \
  --seed 1000 \
  --device cuda \
  --use-amp \
  --output outputs/eval/openarm_smolvla/results.csv
```

Important note:

- the target eval command to reproduce the successful setup is wrapper-based Python execution, not `lerobot-eval`

## Appendix A. Structural Comparison

### A. Current에만 있는 파일

- `HANDOFF.md`
- `gym_openarm.zip`
- `gym_openarm/__init__.py`
- `gym_openarm/openarm_env.py`
- `gym_openarm/openarm_table_dual_realsense_ik_pick_place_make_dataset_random_cube_random_tilt_gripper_mapped.py`
- `scripts/openarm_dataset_scale_diagnostic.py`
- `scripts/openarm_rollout_diagnostic.py`
- `src/lerobot/assets.zip`
- `src/lerobot/assets/openarm_use/...`
- `src/lerobot/processor/outputs/...`

### B. Reference에만 있는 파일

- none under the filtered compare set

### C. 양쪽에 있지만 내용이 다른 파일

- `src/lerobot/configs/train.py`
- `src/lerobot/datasets/v30/augment_dataset_quantile_stats.py`
- `src/lerobot/envs/configs.py`
- `src/lerobot/envs/factory.py`
- `src/lerobot/envs/utils.py`
- `src/lerobot/policies/groot/groot_n1.py`
- `src/lerobot/scripts/lerobot_eval.py`

### D. 동일한 파일

- 622 files identical after filtering

## Appendix B. Current State Preservation

Current `git status --short --branch` summary at audit time:

- branch: `main`
- behind origin by 235 commits
- modified tracked files: 7
- untracked includes `reference_success/`, `gym_openarm/`, `models/`, `scripts/`, `src/lerobot/assets/`

Current `git diff --stat` summary:

- 7 files changed
- 211 insertions
- 34 deletions

Current recent log at audit time:

- `d324ffe8 (HEAD -> main) fix(ci): test only multi-gpu tests in multi-gpu runner (#3092)`
- `1a24f770 Feat/slurm compute rabc script (#3041)`
- `92fba372 fix(num_frames): fixing redundant frames count in conversion script (#3091)`
- `3e451202 fix(ci): log in HF for gated repo in nightly workflows (#3089)`
- `f0d2b37b chore(dependencies): bump transformers v5 (#2964)`

Most important preservation note:

- no source files were modified during this audit
- only this report file was added
