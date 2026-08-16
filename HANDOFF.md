# SmolVLA Training Debug Handoff

## 1. Goal
현재 `~/arena/lerobot` 레포에서 `lerobot-train` 기반 SmolVLA 학습 경로를, 전달받은 `reference_success/train_smolvla.py`의 confirmed behavior와 최대한 동일하게 맞춘 뒤, 짧은 검증을 거쳐 최종 20,000-step training을 실행하는 것이 목표다.

핵심 목표:
- `policy.type=smolvla` 경로에서도 pretrained SmolVLA base (`lerobot/smolvla_base`)를 사용
- `input_features` / `output_features`가 dataset metadata 기반 inference로 이어지도록 보장
- 이후 reference 조건에 최대한 맞춘 `lerobot-train` CLI 확정

## 2. Repository / Environment
- Repository path: `/home/zxro/arena/lerobot`
- Current working branch/status: branch name는 미확인, dirty worktree 존재
- Python / conda environment used for validation: `/home/zxro/miniforge3/envs/lab-isaac5-py311/bin/python`
- Default system `python3`: `/usr/bin/python3`였고, `draccus` 미설치라 validation에 사용 불가
- Reference file path: `/home/zxro/arena/lerobot/reference_success/train_smolvla.py`
- Current dataset repo_id used by user: `a126-kitech/openarm_dual_realsense_pick_place_random_cube_tilt_30_box_blue`
- Device intended by user: `cuda`
- Actual validation device used during smoke/config checks: `cpu` fallback occurred in the conda env because `cuda` device was not available in that validation context
- Additional local test dataset used for lightweight validation:
  `/home/zxro/arena/lerobot/outputs/lerobot_datasets/openarm_pick_lift_test_20260709_191801`
- `HF_HUB_DISABLE_XET=1` was part of the successful reference command provided by the user
- UNKNOWN:
  - actual GPU availability in the final intended training shell
  - actual branch name
  - exact Hugging Face auth/token state for the next session

## 3. Original Training Command
User’s original command:

```bash
lerobot-train \
  --dataset.repo_id=a126-kitech/openarm_dual_realsense_pick_place_random_cube_tilt_30_box_blue \
  --dataset.video_backend=pyav \
  --policy.type=smolvla \
  --output_dir=outputs/train/smolvla_openarm_v10 \
  --job_name=smolvla_openarm_v9 \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --batch_size=8 \
  --steps=20000 \
  --seed=1000
```

## 4. Successful Reference Command
Successful command provided by the user:

```bash
HF_HUB_DISABLE_XET=1 \
python /home/zxro/smolVLA-isaacLab/scripts/train/train_smolvla.py \
  --dataset-root "/home/zxro/outputs/lerobot_datasets/random_cube_tilt_30_gripper_mapped_box_blue_50_degree" \
  --dataset-repo-id "local/random_cube_tilt_30_gripper_mapped_box_blue_50_degree" \
  --output-dir "/home/zxro/smolVLA-isaacLab/outputs/train/random_cube_tilt_30_blue_box_smolvla_20k" \
  --steps 20000 \
  --save-freq 2000 \
  --log-freq 100 \
  --batch-size 4 \
  --num-workers 4 \
  --device cuda \
  --use-amp \
  --policy.input_features=null \
  --policy.output_features=null
```

## 5. Reference Script
`reference_success/train_smolvla.py` is **not** the full successful project. It is a **single reference wrapper script** that was provided for comparison.

What it does:
- validates local dataset metadata
- builds a `lerobot-train` command
- forwards unknown CLI args
- `os.execvp(...)` into `lerobot-train`

It is **not** a separate training implementation with its own dataloader / optimizer / training loop.

Relevant file:
- [`reference_success/train_smolvla.py`](./reference_success/train_smolvla.py)

## 6. Changes Made So Far
### 6.1 Actual source code change
- File: [`src/lerobot/configs/train.py`](./src/lerobot/configs/train.py)
- Change:
  - Added a SmolVLA-specific fallback inside `TrainPipelineConfig.validate()`
  - When `--policy.path` is absent and `policy.type == "smolvla"`, it now:
    - loads config from `lerobot/smolvla_base`
    - injects `--input_features=null` if not already provided
    - injects `--output_features=null` if not already provided
    - sets `self.policy.pretrained_path = Path("lerobot/smolvla_base")`
- Reason:
  - To match the confirmed reference behavior without requiring wrapper-script changes or forcing the user to pass `--policy.path`, `--policy.input_features=null`, `--policy.output_features=null` manually.
- Git diff 핵심:
  - new `elif self.policy is not None and self.policy.type == "smolvla": ...`
  - forced pretrained base path and null feature overrides

Exact diff seen during session:

```diff
diff --git a/src/lerobot/configs/train.py b/src/lerobot/configs/train.py
index 7a5eee77..0e1ceb9f 100644
--- a/src/lerobot/configs/train.py
+++ b/src/lerobot/configs/train.py
@@ -86,6 +86,18 @@ class TrainPipelineConfig(HubMixin):
             cli_overrides = parser.get_cli_overrides("policy")
             self.policy = PreTrainedConfig.from_pretrained(policy_path, cli_overrides=cli_overrides)
             self.policy.pretrained_path = Path(policy_path)
+        elif self.policy is not None and self.policy.type == "smolvla":
+            # Align the default SmolVLA training path with the reference wrapper:
+            # initialize from the published base checkpoint, while forcing feature
+            # inference from dataset metadata instead of reusing saved feature specs.
+            policy_path = "lerobot/smolvla_base"
+            cli_overrides = list(parser.get_cli_overrides("policy") or [])
+            if not any(arg.startswith("--input_features=") for arg in cli_overrides):
+                cli_overrides.append("--input_features=null")
+            if not any(arg.startswith("--output_features=") for arg in cli_overrides):
+                cli_overrides.append("--output_features=null")
+            self.policy = PreTrainedConfig.from_pretrained(policy_path, cli_overrides=cli_overrides)
+            self.policy.pretrained_path = Path(policy_path)
         elif self.resume:
```

### 6.2 No other source code changes were intentionally made in this session
- No other files were edited by this session.
- The repo already had unrelated dirty changes before this work.

## 7. Confirmed Findings
Only direct code/runtime confirmations are listed here.

### 7.1 `policy.type=smolvla` now sets pretrained base during `cfg.validate()`
- Status: `CONFIRMED`
- File / function:
  - [`src/lerobot/configs/train.py:81-100`](./src/lerobot/configs/train.py#L81)
  - `TrainPipelineConfig.validate`
- Confirmed behavior:
  - if `policy.path` is absent and `policy.type == "smolvla"`, `policy_path = "lerobot/smolvla_base"` is used
  - config is loaded via `PreTrainedConfig.from_pretrained(...)`

### 7.2 `input_features` / `output_features` are forced to `None` before policy creation in the new SmolVLA fallback
- Status: `CONFIRMED`
- File / function:
  - [`src/lerobot/configs/train.py:94-99`](./src/lerobot/configs/train.py#L94)
  - `TrainPipelineConfig.validate`
- Confirmed behavior:
  - `--input_features=null` and `--output_features=null` are injected unless already present

### 7.3 Dataset metadata feature inference path in `make_policy()`
- Status: `CONFIRMED`
- Files / functions:
  - [`src/lerobot/policies/factory.py:457-472`](./src/lerobot/policies/factory.py#L457)
  - [`src/lerobot/datasets/utils.py:698-741`](./src/lerobot/datasets/utils.py#L698)
  - `make_policy`
  - `dataset_to_policy_features`
- Confirmed behavior:
  - `cfg.output_features` is always rebuilt from dataset features for ACTION-type entries
  - `cfg.input_features` is inferred from the remaining dataset features if `not cfg.input_features`

### 7.4 Reference script is a wrapper around current `lerobot-train`, not a separate trainer
- Status: `CONFIRMED`
- File / function:
  - [`reference_success/train_smolvla.py:74-98`](./reference_success/train_smolvla.py#L74)
  - `main`
- Confirmed behavior:
  - validates dataset
  - builds `lerobot-train` command
  - calls `os.execvp(command[0], command)`

### 7.5 Current `lerobot-train` entry point
- Status: `CONFIRMED`
- File / function:
  - [`src/lerobot/scripts/lerobot_train.py:153-188`](./src/lerobot/scripts/lerobot_train.py#L153)
  - `train`
- Confirmed behavior:
  - `cfg.validate()` is called before dataset/policy creation

### 7.6 AMP config field exists, but trainer wiring is incomplete/unclear
- Confirmed part:
  - `policy.use_amp` exists in config
  - trainer uses `with accelerator.autocast():`
  - `Accelerator(...)` is created without explicit `mixed_precision=...`
- Files:
  - [`src/lerobot/configs/policies.py:62-65`](./src/lerobot/configs/policies.py#L62)
  - [`src/lerobot/scripts/lerobot_train.py:176-187`](./src/lerobot/scripts/lerobot_train.py#L176)
  - [`src/lerobot/scripts/lerobot_train.py:101-102`](./src/lerobot/scripts/lerobot_train.py#L101)
- Status:
  - existence of field and current trainer code shape: `CONFIRMED`
  - exact runtime effect on mixed precision: not confirmed; see section 9

### 7.7 Previously saved run configs exist and were inspected
- `smolvla_openarm_v9` saved config showed pre-change fresh path characteristics:
  - `pretrained_path: null`
  - `load_vlm_weights: false`
- File:
  - [`outputs/train/smolvla_openarm_v9/checkpoints/020000/pretrained_model/config.json`](./outputs/train/smolvla_openarm_v9/checkpoints/020000/pretrained_model/config.json)
- Status: `CONFIRMED`

### 7.8 Previously saved run config `smolvla_openarm_v10` showed pretrained-base-like path
- File:
  - [`outputs/train/smolvla_openarm_v10/checkpoints/020000/pretrained_model/train_config.json`](./outputs/train/smolvla_openarm_v10/checkpoints/020000/pretrained_model/train_config.json)
- Confirmed values:
  - `pretrained_path = "lerobot/smolvla_base"`
  - `load_vlm_weights = true`
  - `prefix_length = 0`
  - `pad_language_to = "max_length"`
- Status: `CONFIRMED`

### 7.9 Local lightweight validation dataset metadata
- File:
  - [`outputs/lerobot_datasets/openarm_pick_lift_test_20260709_191801/meta/info.json`](./outputs/lerobot_datasets/openarm_pick_lift_test_20260709_191801/meta/info.json)
- Confirmed keys:
  - `observation.state`
  - `action`
  - `observation.images.front`
  - `timestamp`
  - `frame_index`
  - `episode_index`
  - `index`
  - `task_index`
- Confirmed camera keys in this local test dataset:
  - `observation.images.front`
- Confirmed shapes in this local test dataset:
  - `observation.state`: `[9]`
  - `action`: `[9]`
- Status: `CONFIRMED`
- Important:
  - this is **not** the user’s target dataset
  - it was only used to validate the feature inference code path

## 8. Runtime Validation Results
### 8.1 Config validation before/after SmolVLA fallback
Executed with:
- conda python: `/home/zxro/miniforge3/envs/lab-isaac5-py311/bin/python`
- local lightweight dataset root
- `sys.argv = ['smoke', '--policy.type=smolvla', '--policy.device=cpu', '--policy.push_to_hub=false']`

Observed output:

```text
before None {} {}
after_pretrained_path lerobot/smolvla_base
after_input_features None
after_output_features None
```

Interpretation:
- `cfg.validate()` now changes `pretrained_path` to `lerobot/smolvla_base`
- `input_features` and `output_features` remain `None` after validation
- this matches the intended pre-inference state

### 8.2 Dataset metadata load on local lightweight validation dataset
Observed output:

```text
dataset_features ['action', 'episode_index', 'frame_index', 'index', 'observation.images.front', 'observation.state', 'task_index', 'timestamp']
dataset_camera_keys ['observation.images.front']
dataset_state_shape (9,)
dataset_action_shape (9,)
```

Interpretation:
- dataset loading succeeded
- metadata was accessible
- feature inference path had valid metadata to work from

### 8.3 First smoke attempt failed at HF config download due to network restriction
Observed traceback start:

```text
'[Errno -3] Temporary failure in name resolution' thrown while requesting HEAD https://huggingface.co/lerobot/smolvla_base/resolve/main/config.json
...
File "/home/zxro/arena/lerobot/src/lerobot/configs/train.py", line 99, in validate
```

Interpretation:
- failure was environmental/network-related
- code path did reach the new SmolVLA pretrained fallback

### 8.4 Escalated network validation entered pretrained loading path
Observed output included:

```text
after_pretrained_path lerobot/smolvla_base
after_input_features None
after_output_features None
...
Loading  HuggingFaceTB/SmolVLM2-500M-Video-Instruct weights ...
```

Interpretation:
- pretrained config load path was reached
- model creation advanced into backbone loading

### 8.5 Full 1-step smoke was NOT confirmed complete
- `forward`, `loss`, `backward`, `optimizer.step` completion were **not** captured as successful end-state in this session
- the run progressed into pretrained model loading, but no final “optimizer_step_ok” output was obtained

## 9. Incomplete / Unverified Items
All items below are **not confirmed**.

- `forward` completion on the real SmolVLA pretrained path: `UNKNOWN`
- `loss` calculation completion: `UNKNOWN`
- `backward` completion: `UNKNOWN`
- `optimizer.step()` completion: `UNKNOWN`
- full 1-step smoke success end-to-end: `UNKNOWN`
- whether AMP is truly active in current trainer when `--policy.use_amp=true`: `UNKNOWN`
  - likely not fully wired, but not runtime-confirmed
- whether `HF_HUB_DISABLE_XET=1` is required in practice for the next run: `UNKNOWN`
- whether the user’s target dataset differs in a way that materially impacts training quality/success: `UNKNOWN`
- exact feature keys / camera keys / state/action shapes of the user’s target dataset repo:
  `a126-kitech/openarm_dual_realsense_pick_place_random_cube_tilt_30_box_blue`
  - not directly opened from local metadata in this session
  - inferred from previous saved training artifacts only

## 10. Current Understanding
### Already resolved / confirmed
- `policy.type=smolvla` no longer implies the old fresh-init path during `cfg.validate()`
- the code now forces SmolVLA config loading from `lerobot/smolvla_base`
- `input_features` / `output_features` are forced to `None` before `make_policy()`
- the feature inference code path from dataset metadata is confirmed in code
- the reference script is only a wrapper around `lerobot-train`

### Not yet resolved
- full runtime success of pretrained SmolVLA creation + one optimization step
- exact AMP behavior in the current trainer
- actual next-run behavior on the user’s target dataset repo

## 11. Next Training Task
Recommended order for the next session:

1. Read this `HANDOFF.md` first and do **not** re-investigate already confirmed items.
2. Check current `git status` and keep the existing source change in `src/lerobot/configs/train.py`.
3. Re-run a short validation on the **actual target dataset** if local metadata is available:
   - confirm feature keys
   - confirm camera keys
   - confirm state/action shapes
4. Run a **true short smoke training** on the real target dataset:
   - `batch_size=1`
   - `steps=1`
   - aim to confirm `forward`, `loss`, `backward`, `optimizer.step`
5. If that succeeds, finalize the full 20,000-step training CLI.
6. Run the full training.
7. Only after training is stable, compare eval path with reference eval conditions.

Reason for this order:
- the code-path alignment work is already done
- the remaining risk is runtime, not static config plumbing

## 12. CLI Requirements for Next Run
Required constraints for the next intended real training run:

- `dataset.repo_id = a126-kitech/openarm_dual_realsense_pick_place_random_cube_tilt_30_box_blue`
- `dataset.video_backend = pyav`
- `steps = 20000`
- `batch_size = 4`
- `num_workers = 4`
- save frequency should map to current config field `save_freq = 2000`
- log frequency should map to current config field `log_freq = 100`
- `seed = 1000`
- `policy.device = cuda`
- `policy.push_to_hub = false`
- pretrained SmolVLA base should be used
- dataset metadata feature inference should remain enabled
- if practical, include `HF_HUB_DISABLE_XET=1`
- AMP usage should only be claimed after verifying current trainer behavior

Current valid CLI field names confirmed from code:
- `--steps`
- `--save_freq`
- `--log_freq`
- `--batch_size`
- `--num_workers`
- `--seed`
- `--policy.device`
- `--policy.push_to_hub`
- `--policy.use_amp`
- `--dataset.video_backend`

TODO:
- confirm at runtime whether `--policy.use_amp=true` actually enables training mixed precision in current trainer

## 13. Commands Already Run
Meaningful commands from this session:

```bash
git status --short
git diff -- src/lerobot/configs/train.py
sed -n '1,220p' reference_success/train_smolvla.py
sed -n '1,240p' outputs/train/smolvla_openarm_v9/checkpoints/020000/pretrained_model/config.json
sed -n '1,260p' outputs/train/smolvla_openarm_v10/checkpoints/020000/pretrained_model/train_config.json
sed -n '1,220p' outputs/lerobot_datasets/openarm_pick_lift_test_20260709_191801/meta/info.json
```

Config validation / lightweight runtime checks:

```bash
/home/zxro/miniforge3/envs/lab-isaac5-py311/bin/python -c "import draccus, torch, accelerate, huggingface_hub; print('deps-ok')"
```

Python validation snippets were run inline to:
- instantiate `TrainPipelineConfig`
- call `cfg.validate()`
- print `pretrained_path`, `input_features`, `output_features`
- load local lightweight dataset metadata
- enter pretrained SmolVLA creation path

Escalated network validation was used to allow Hugging Face downloads during runtime checks.

## 14. Important Files
Files the next session should read first:

- [`HANDOFF.md`](./HANDOFF.md)
  - session summary and confirmed/unverified state

- [`reference_success/train_smolvla.py`](./reference_success/train_smolvla.py)
  - reference wrapper script behavior

- [`src/lerobot/configs/train.py`](./src/lerobot/configs/train.py)
  - SmolVLA fallback change in `TrainPipelineConfig.validate()`

- [`src/lerobot/scripts/lerobot_train.py`](./src/lerobot/scripts/lerobot_train.py)
  - main training entry point and training loop

- [`src/lerobot/policies/factory.py`](./src/lerobot/policies/factory.py)
  - dataset metadata → `input_features` / `output_features` inference

- [`src/lerobot/datasets/factory.py`](./src/lerobot/datasets/factory.py)
  - dataset construction path

- [`src/lerobot/datasets/utils.py`](./src/lerobot/datasets/utils.py)
  - `dataset_to_policy_features`

- [`src/lerobot/policies/smolvla/configuration_smolvla.py`](./src/lerobot/policies/smolvla/configuration_smolvla.py)
  - SmolVLA config defaults and optimizer/scheduler presets

- [`src/lerobot/policies/smolvla/modeling_smolvla.py`](./src/lerobot/policies/smolvla/modeling_smolvla.py)
  - SmolVLA policy and forward path

- [`src/lerobot/policies/smolvla/smolvlm_with_expert.py`](./src/lerobot/policies/smolvla/smolvlm_with_expert.py)
  - backbone/expert model loading path

- [`outputs/train/smolvla_openarm_v9/checkpoints/020000/pretrained_model/config.json`](./outputs/train/smolvla_openarm_v9/checkpoints/020000/pretrained_model/config.json)
  - pre-change fresh-init run artifact

- [`outputs/train/smolvla_openarm_v10/checkpoints/020000/pretrained_model/train_config.json`](./outputs/train/smolvla_openarm_v10/checkpoints/020000/pretrained_model/train_config.json)
  - pretrained-base-like run artifact used for comparison

- [`outputs/lerobot_datasets/openarm_pick_lift_test_20260709_191801/meta/info.json`](./outputs/lerobot_datasets/openarm_pick_lift_test_20260709_191801/meta/info.json)
  - local lightweight validation dataset metadata

## 15. Do Not Assume
- Do not assume `policy.type=smolvla` still means fresh initialization.
- Do not assume `--policy.path=lerobot/smolvla_base` must be passed manually; current code now auto-applies that fallback for SmolVLA.
- Do not assume `input_features` / `output_features` must be manually set to `null`; current SmolVLA fallback now injects that behavior.
- Do not assume AMP is definitely active just because `--policy.use_amp=true` exists.
- Do not assume `reference_success/` contains the entire successful project.
- Do not assume the local lightweight validation dataset represents the user’s actual target dataset.
- Do not skip a short runtime smoke validation before the next full 20,000-step run.

## 16. Next Session Starter Prompt
Use this as the starter prompt for the next Codex session:

```text
Read /home/zxro/arena/lerobot/HANDOFF.md first and do not re-investigate items already marked CONFIRMED there. Then check the current repository state with git status and git diff. The immediate goal is to finalize the lerobot-train CLI for SmolVLA and run a short runtime validation on the actual target dataset path/repo if possible. Do not modify code yet. First confirm runtime behavior: pretrained SmolVLA base loading, dataset metadata feature inference, and whether a 1-step smoke run can complete forward/loss/backward/optimizer.step.
```
