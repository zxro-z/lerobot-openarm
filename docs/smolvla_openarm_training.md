# SmolVLA OpenArm Training Contract

This document records the current reproducible training and evaluation contract for the OpenArm SmolVLA path in this repository.

## Required Dataset Feature Contract

The OpenArm SmolVLA path expects dataset metadata to resolve to the following policy features:

```text
input
- observation.state               : shape [8]
- observation.images.top          : shape [3, 480, 640]
- observation.images.wrist        : shape [3, 480, 640]

output
- action                          : shape [8]
```

The local dataset metadata currently stores:

```text
observation.state
action
observation.images.top
observation.images.wrist
```

`dataset_to_policy_features()` converts image/video features from `(H, W, C)` to `(C, H, W)` for policy use.

## Standard Training Command

Use the successful OpenArm training contract as the canonical baseline:

```bash
DATASET_PATH=<DATASET_PATH>
OUTPUT_DIR=<OUTPUT_DIR>

PATH=/home/zxro/miniforge3/envs/lab-isaac5-py311/bin:$PATH \
HF_HOME=/home/zxro/.cache/hf_lerobot \
HF_DATASETS_CACHE=/home/zxro/.cache/hf_lerobot/datasets \
/home/zxro/miniforge3/envs/lab-isaac5-py311/bin/python3.11 /home/zxro/arena/lerobot/scripts/train/train_smolvla.py \
  --dataset-source local \
  --dataset-root "${DATASET_PATH}" \
  --dataset-repo-id local/random_cube_tilt_30_gripper_mapped_box_blue_50_degree \
  --policy-path lerobot/smolvla_base \
  --output-dir "${OUTPUT_DIR}" \
  --steps 20000 \
  --batch-size 4 \
  --num-workers 4 \
  --save-freq 2000 \
  --log-freq 100 \
  --device cuda \
  --video-backend pyav \
  --use-amp \
  --no-wandb
```

## Meaning of `input_features=null` / `output_features=null`

The training wrapper always forwards:

```text
--policy.input_features=null
--policy.output_features=null
```

This prevents the published pretrained SmolVLA feature schema from being reused unchanged. Instead:

1. `PreTrainedConfig.from_pretrained("lerobot/smolvla_base", cli_overrides=...)` loads the base config.
2. Draccus applies the CLI overrides and sets `input_features=None`, `output_features=None`.
3. `make_policy(..., ds_meta=dataset.meta)` resolves dataset features via `dataset_to_policy_features()`.
4. `cfg.output_features` is always set from dataset action features.
5. `cfg.input_features` is set from the remaining dataset features only when it is empty/null.

This is required because the published base config does not match the OpenArm dataset contract.

## VLM Weight Loading Setting

The successful run saved:

```text
policy.pretrained_path = lerobot/smolvla_base
policy.load_vlm_weights = true
```

`load_vlm_weights` controls whether `SmolVLMWithExpertModel` initializes the VLM backbone with pretrained Hugging Face weights:

- `true`: `AutoModelForImageTextToText.from_pretrained(...)`
- `false`: `AutoConfig.from_pretrained(...)` + fresh `SmolVLMForConditionalGeneration(config)`

For reproducibility, the standard command should explicitly preserve the successful behavior:

```text
--policy.path=lerobot/smolvla_base
```

The saved successful checkpoint already confirms `load_vlm_weights=true`.

## Pre-Training Sanity Check

Before starting a long run, confirm:

1. dataset task metadata contains:

```text
Pick up the red cube and place it in the storage box.
```

2. dataset features resolve to:

```text
observation.state         -> STATE [8]
observation.images.top    -> VISUAL [3, 480, 640]
observation.images.wrist  -> VISUAL [3, 480, 640]
action                    -> ACTION [8]
```

3. training startup logs print:

```text
[PRETRAINED POLICY FEATURES]
[DATASET FEATURES]
[FINAL POLICY INPUT FEATURES]
[FINAL POLICY OUTPUT FEATURES]
[VLM WEIGHTS LOADED]
```

## Expected Saved Checkpoint Feature Config

The saved checkpoint config should contain:

```json
"input_features": {
  "observation.state": {"type": "STATE", "shape": [8]},
  "observation.images.top": {"type": "VISUAL", "shape": [3, 480, 640]},
  "observation.images.wrist": {"type": "VISUAL", "shape": [3, 480, 640]}
},
"output_features": {
  "action": {"type": "ACTION", "shape": [8]}
}
```

## Simulation Color Evaluation Protocol

Current OpenArm eval environment in this repository is a single-cube environment with color override, not a three-cube environment.

Supported protocol today:

- red instruction, red cube
- red instruction, blue cube
- red instruction, yellow cube
- blue instruction, red/blue/yellow single-cube variants
- yellow instruction, red/blue/yellow single-cube variants

The result CSV records:

```text
episode_id / seed / instruction / instruction_color / target_color
picked_color / task_success / color_correct
cube_initial_* / termination_reason
```

For the current single-cube environment:

- `picked_color = object_color` if the cube reaches the success region and stabilizes
- `picked_color = failure` otherwise

This is not yet a multi-object language-selection benchmark.

## Confusion Matrix Generation

```bash
/home/zxro/miniforge3/envs/lab-isaac5-py311/bin/python3.11 \
  /home/zxro/arena/lerobot/scripts/eval/analyze_color_eval.py \
  <RESULTS_CSV>
```

## Real-World Caveat

Simulation evaluation validates whether the trained policy exhibits language-conditioned color selection in the simulation environment.

It must not be interpreted as evidence of real-world OpenArm performance.
