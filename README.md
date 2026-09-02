# SmolVLA Simulation Final Archive

This archive records only the final simulation experiments. It intentionally
excludes real-robot experiments, staging data, smoke tests, failed runs, and
intermediate analyses. Dataset, checkpoint, and video binaries are external
artifacts and must not be committed to Git.

## 1. Overview

Two final simulation results are retained:

1. Single-cube pick-and-place baseline: **9/10 success (90.0%)**.
2. Three-color, language-conditioned pick-and-place: the final checkpoint is
   **018000**, selected from the 10k/14k/18k/20k comparison.

## 2. Final Experiments

### 2.1 Single-cube Pick-and-Place

Task: pick one cube and place it at the target/storage location.

| Metric | Result |
| --- | ---: |
| Successes | 9 / 10 |
| Task Success | 90.0% |

TODO: verify the exact final single-cube checkpoint, evaluation command, and
video archive path. No matching final-result artifact was found during this
archive pass, so no intermediate single-cube run is named as canonical.

### 2.2 Three-color Language-conditioned Pick-and-Place

Three cubes (red, blue, yellow) are present. The instruction selects the cube
to move into the storage box. The final result evaluates both manipulation and
instruction-conditioned color selection.

## 3. Dataset

### Dataset path

External archive path (not tracked by Git):

```text
/home/zxro/arena/lerobot/src/lerobot/datasets/openarm_three_color_triplet_tilt50_matte/openarm_three_color_triplet_tilt50_matte_dataset
```

LeRobot repo ID: `local/openarm_three_color_triplet_tilt50_matte`.

### Dataset composition

| Item | Verified value |
| --- | --- |
| Episodes | 150 |
| Frames | 77,160 |
| Tasks | 3 |
| Per color | 50 red / 50 blue / 50 yellow |
| FPS | 30 |
| Cube world X range observed in position log | -0.589884 to -0.470105 m |
| Cube world Y range observed in position log | 0.010078 to 0.139903 m |
| Cube material | saturated matte, non-emissive, roughness 1.0 |
| Gripper angle | 50 degrees fixed (experiment contract) |

### Observation / action features

| Feature | Value |
| --- | --- |
| `observation.state` | 8-D: joint_1..joint_7, gripper |
| `action` | 8-D: joint_1..joint_7, gripper |
| Cameras | `observation.images.top`, `observation.images.wrist` |
| Camera format | 640×480 RGB video, 30 FPS |

### Task strings

```text
Pick up the red cube and place it in the storage box.
Pick up the blue cube and place it in the storage box.
Pick up the yellow cube and place it in the storage box.
```

### Dataset generation rule

The generator samples one three-cube layout, restores the robot and all cubes
to that same layout before each target episode, and records the sequence:

```text
Red → Blue → Yellow → Red → Blue → Yellow → ...
```

50 accepted triplets produce 150 episodes. A triplet is accepted atomically:
if any color episode fails collection quality checks, none of its three episodes
is merged into the final dataset.

Generation sources:

```text
src/lerobot/scripts/openarm_three_color_triplet_atomic_dataset_with_positions.py
src/lerobot/scripts/openarm_three_color_triplet_atomic_dataset_with_positions_matte.py
```

## 4. Training

### Training settings

The values below are from the saved `018000/pretrained_model/train_config.json`
and policy `config.json`.

| Setting | Final value |
| --- | --- |
| Base policy | `lerobot/smolvla_base` |
| Input/output features | inferred from dataset (resolved to 8-D state/action, top+wrist) |
| `load_vlm_weights` | true |
| `train_expert_only` | false |
| `freeze_vision_encoder` | true |
| `attention_mode` | `cross_attn` |
| `train_state_proj` | true |
| Batch size | 4 |
| Workers | 4 |
| Steps | 20,000 |
| Save frequency | 2,000 |
| Seed | 1,000 |
| Grasp sampler | off (`grasp_positive_manifest: null`) |
| Commitment sampler | off (`target_commitment_manifest: null`) |

`train_expert_only=false` permits training beyond the action expert subject to
the policy's trainable-parameter rules; `freeze_vision_encoder=true` keeps the
vision encoder frozen. These are saved resolved policy values, not inferred
from the result table.

### Full training command

The original shell history is not stored. The following is a **reconstructed,
executable command from the saved resolved configuration**; option ordering is
not claimed to be historical.

```bash
lerobot-train \
  --policy.path=lerobot/smolvla_base \
  --policy.input_features=null \
  --policy.output_features=null \
  --policy.load_vlm_weights=true \
  --policy.train_expert_only=false \
  --policy.freeze_vision_encoder=true \
  --policy.attention_mode=cross_attn \
  --policy.train_state_proj=true \
  --policy.device=cuda \
  --policy.use_amp=true \
  --policy.push_to_hub=false \
  --dataset.repo_id=local/openarm_three_color_triplet_tilt50_matte \
  --dataset.root=/home/zxro/arena/lerobot/src/lerobot/datasets/openarm_three_color_triplet_tilt50_matte/openarm_three_color_triplet_tilt50_matte_dataset \
  --dataset.video_backend=pyav \
  --output_dir=/home/zxro/arena/lerobot/outputs/train/openarm_three_color_triplet_tilt50_matte_run2_contract \
  --job_name=openarm_three_color_transit_tilt_50_smolvla \
  --batch_size=4 \
  --num_workers=4 \
  --steps=20000 \
  --save_freq=2000 \
  --seed=1000 \
  --wandb.enable=false
```

TODO: verify whether the original invocation explicitly supplied `--eval_freq`.
The saved train configuration alone does not establish the original CLI spelling.

## 5. Checkpoints

Training output directory (external; Git-ignored):

```text
/home/zxro/arena/lerobot/outputs/train/openarm_three_color_triplet_tilt50_matte_run2_contract
```

Checkpoint naming rule: zero-padded step directories, for example
`checkpoints/018000/pretrained_model`.

**Final / best checkpoint:**

```text
/home/zxro/arena/lerobot/outputs/train/openarm_three_color_triplet_tilt50_matte_run2_contract/checkpoints/018000/pretrained_model
```

| Checkpoint | Task Success | Color Accuracy | Wrong-color Success | Precision among Successes |
| --- | ---: | ---: | ---: | ---: |
| 10k | 16.7% | 10.0% | 6.7% | 60.0% |
| 14k | 20.0% | 3.3% | 16.7% | 16.7% |
| **18k (final)** | **40.0%** | **20.0%** | **20.0%** | **50.0%** |
| 20k | 20.0% | 6.7% | 13.3% | 33.3% |

Performance did not monotonically improve with training steps. The reason for
the 20k degradation was not conclusively identified.

## 6. Evaluation

### Protocol

Each checkpoint is evaluated for 30 episodes:

| Instruction | Episodes |
| --- | ---: |
| Red | 10 |
| Blue | 10 |
| Yellow | 10 |
| Total | 30 |

The protocol uses a fixed seed schedule beginning at 1000. The evaluator uses
one deterministic environment reset seed per episode.

Metrics:

- `picked_color`
- `task_success`
- `color_correct`
- instruction-color × picked-color confusion matrix

Interpretation:

- diagonal: selected the instructed color and succeeded;
- off-diagonal success: manipulation succeeded but selected the wrong color;
- failure: no task success within the episode limit.

### Full evaluation command

The final results were produced by the top+wrist three-color evaluator:
`scripts/eval/eval_smolvla.py`. The following command reproduces the stored
30-episode protocol and writes a new external result CSV.

```bash
python scripts/eval/eval_smolvla.py \
  --policy-path /home/zxro/arena/lerobot/outputs/train/openarm_three_color_triplet_tilt50_matte_run2_contract/checkpoints/018000/pretrained_model \
  --dataset-root /home/zxro/arena/lerobot/src/lerobot/datasets/openarm_three_color_triplet_tilt50_matte/openarm_three_color_triplet_tilt50_matte_dataset \
  --dataset-repo-id local/openarm_three_color_triplet_tilt50_matte \
  --num-episodes-per-color 10 \
  --max-steps 1000 \
  --seed 1000 \
  --device cuda \
  --instruction-order grouped \
  --output /home/zxro/arena/lerobot/outputs/eval/openarm_three_color_triplet_tilt50_matte_run2_contract/reproduced_018000_results.csv
```

Summarize the resulting CSV:

```bash
python scripts/eval/analyze_color_eval.py \
  /home/zxro/arena/lerobot/outputs/eval/openarm_three_color_triplet_tilt50_matte_run2_contract/reproduced_018000_results.csv
```

TODO: verify whether the historical final invocation used `--use-amp`; the
stored result CSV does not preserve its CLI arguments.

## 7. Final Results

### Single-cube result

```text
successes = 9/10
Task Success = 90.0%
```

### Three-color result: 018000 (final)

| Instruction | Picked Red | Picked Blue | Picked Yellow | Failure |
| --- | ---: | ---: | ---: | ---: |
| Red | 2 | 1 | 1 | 6 |
| Blue | 1 | 2 | 2 | 5 |
| Yellow | 0 | 1 | 2 | 7 |

| Metric | Result |
| --- | --- |
| Task Success | 12 / 30 = 40.0% |
| Color Accuracy | 6 / 30 = 20.0% |
| Wrong-color Success | 6 / 30 = 20.0% |
| Precision among task successes | 6 / 12 = 50.0% |
| Red instruction accuracy | 2 / 10 = 20.0% |
| Blue instruction accuracy | 2 / 10 = 20.0% |
| Yellow instruction accuracy | 2 / 10 = 20.0% |

## 8. Interpretation

The dominant bottleneck was manipulation robustness: 18/30 episodes (60%)
failed before a successful placement. Among the 12 successful manipulation
episodes, six selected the correct color and six placed the wrong color.

The model showed partial language-conditioned color selection, but the result
was not sufficient to conclude robust language grounding. The dominant
bottleneck was manipulation failure, followed by language/color-selection
failure.

## 9. Reproduction

1. Restore the external dataset to the path in section 3.
2. Run the training command in section 4 to create the output directory.
3. Evaluate `018000/pretrained_model` with section 6.
4. Run `analyze_color_eval.py` on the emitted CSV and compare with section 7.

Expected final 18k output: 12/30 task success, 6/30 correct-color success,
and the confusion matrix in section 7.

## 10. External Archive Paths

| Artifact | Path | Git status |
| --- | --- | --- |
| Final dataset | `/home/zxro/arena/lerobot/src/lerobot/datasets/openarm_three_color_triplet_tilt50_matte/openarm_three_color_triplet_tilt50_matte_dataset` | excluded |
| Position log / generation provenance | `/home/zxro/arena/lerobot/src/lerobot/datasets/openarm_three_color_triplet_tilt50_matte/openarm_three_color_triplet_tilt50_matte_cube_positions` | excluded |
| Training output | `/home/zxro/arena/lerobot/outputs/train/openarm_three_color_triplet_tilt50_matte_run2_contract` | excluded |
| Final evaluation CSV/matrices | `/home/zxro/arena/lerobot/outputs/eval/openarm_three_color_triplet_tilt50_matte_run2_contract` | excluded |
| Final demo video | TODO: verify; no video was found under the final evaluation output directory |

The generation source scripts and evaluation source scripts remain in Git; only
their large generated artifacts are excluded.
