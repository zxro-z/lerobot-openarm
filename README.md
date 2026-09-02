# SmolVLA Simulation Final Archive

This archive records the final simulation experiments: a single-cube
pick-and-place baseline and a three-color language-conditioned task. Dataset,
checkpoint, and video binaries remain outside Git.

## 1. Overview

| Experiment | Final result |
| --- | --- |
| Single-cube pick-and-place | 9 / 10 task success (90.0%) |
| Three-color language-conditioned pick-and-place | Checkpoint 18k selected from the 10k / 14k / 18k / 20k comparison |

## 2. Final Experiments

### 2.1 Single-cube Pick-and-Place

Task: pick one cube and place it at the target/storage location.

| Metric | Result |
| --- | ---: |
| Successes | 9 / 10 |
| Task Success | 90.0% |

#### Dataset

- Repo ID: `local/random_cube_tilt_30_gripper_mapped_box_blue_50_degree`
- Root: `/home/zxro/arena/lerobot/outputs/lerobot_datasets/random_cube_tilt_30_gripper_mapped_box_blue_50_degree`
- 50 episodes, 25,378 frames, one task, 30 FPS.
- Cameras: `observation.images.top`, `observation.images.wrist`.
- `observation.state`: 8-D. `action`: 8-D.

#### Training

Final training output:

```text
/home/zxro/arena/lerobot/outputs/train/ab_local_verified
```

Final checkpoint:

```text
/home/zxro/arena/lerobot/outputs/train/ab_local_verified/checkpoints/020000/pretrained_model
```

### Final single-cube training command

```bash
PATH=/home/zxro/miniforge3/envs/lab-isaac5-py311/bin:$PATH \
HF_HOME=/home/zxro/.cache/hf_lerobot \
HF_DATASETS_CACHE=/home/zxro/.cache/hf_lerobot/datasets \
/home/zxro/miniforge3/envs/lab-isaac5-py311/bin/python3.11 \
  /home/zxro/arena/lerobot/scripts/train/train_smolvla.py \
  --dataset-source local \
  --dataset-root /home/zxro/arena/lerobot/outputs/lerobot_datasets/random_cube_tilt_30_gripper_mapped_box_blue_50_degree \
  --dataset-repo-id local/random_cube_tilt_30_gripper_mapped_box_blue_50_degree \
  --output-dir /home/zxro/arena/lerobot/outputs/train/ab_local_verified \
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

#### Evaluation

- Result CSV: `/home/zxro/arena/lerobot/outputs/eval/ab_local_verified/results.csv`
- Video directory: `/home/zxro/arena/lerobot/outputs/eval/ab_local_verified/videos`
- Representative success video: [`episode_000_success_1.mp4`](https://github.com/user-attachments/assets/8657caf1-2273-4117-87f1-e1e0ee9582ec)

### Final single-cube evaluation command

```bash
PATH=/home/zxro/miniforge3/envs/lab-isaac5-py311/bin:$PATH \
/home/zxro/miniforge3/envs/lab-isaac5-py311/bin/python3.11 \
  /home/zxro/arena/lerobot/scripts/eval/eval_smolvla.py \
  --policy-path /home/zxro/arena/lerobot/outputs/train/ab_local_verified/checkpoints/020000/pretrained_model \
  --dataset-root /home/zxro/arena/lerobot/outputs/lerobot_datasets/random_cube_tilt_30_gripper_mapped_box_blue_50_degree \
  --dataset-repo-id local/random_cube_tilt_30_gripper_mapped_box_blue_50_degree \
  --num-episodes 10 \
  --max-steps 1000 \
  --seed 1000 \
  --device cuda \
  --use-amp \
  --save-video \
  --video-dir /home/zxro/arena/lerobot/outputs/eval/ab_local_verified/videos \
  --output /home/zxro/arena/lerobot/outputs/eval/ab_local_verified/results.csv
```

### 2.2 Three-color Language-conditioned Pick-and-Place

Three cubes (red, blue, yellow) are present. The instruction selects the cube
to move into the storage box. The evaluation separates manipulation success
from instruction-conditioned color selection.

## 3. Dataset

### Three-color dataset

- Repo ID: `local/openarm_three_color_triplet_tilt50_matte`
- Root: `/home/zxro/arena/lerobot/src/lerobot/datasets/openarm_three_color_triplet_tilt50_matte/openarm_three_color_triplet_tilt50_matte_dataset`
- 150 episodes: 50 red, 50 blue, and 50 yellow.
- 77,160 frames at 30 FPS.
- Cameras: `observation.images.top`, `observation.images.wrist`.
- `observation.state`: 8-D. `action`: 8-D.

Task strings:

```text
Pick up the red cube and place it in the storage box.
Pick up the blue cube and place it in the storage box.
Pick up the yellow cube and place it in the storage box.
```

For each cube layout, the generator records the three instructions in the
following sequence before sampling the next layout:

```text
Red → Blue → Yellow → Red → Blue → Yellow → ...
```

## 4. Training

### Training settings

| Setting | Final value |
| --- | --- |
| Base policy | `lerobot/smolvla_base` |
| Input/output features | Inferred from the dataset: 8-D state/action, top+wrist |
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
| AMP | true |
| Grasp sampler | off |
| Commitment sampler | off |

Training output:

```text
/home/zxro/arena/lerobot/outputs/train/openarm_three_color_triplet_tilt50_matte_run2_contract
```

### Final three-color training command

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

## 5. Checkpoints

Checkpoint directories use zero-padded step names under the training output.

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

Performance did not monotonically improve with training steps, and the best
result was obtained at checkpoint 18k.

## 6. Evaluation

### Protocol

Each checkpoint is evaluated with a fixed seed schedule beginning at 1000.

| Instruction | Episodes |
| --- | ---: |
| Red | 10 |
| Blue | 10 |
| Yellow | 10 |
| Total | 30 |

- `max_steps`: 1000
- Device: CUDA
- AMP: enabled
- Metrics: `picked_color`, `task_success`, `color_correct`, and the
  instruction-color × picked-color confusion matrix.

Diagonal entries are instruction-consistent successes. Off-diagonal entries
are wrong-color successes. The failure column represents unsuccessful task
completion.

### Final three-color evaluation command

```bash
python scripts/eval/eval_smolvla.py \
  --policy-path /home/zxro/arena/lerobot/outputs/train/openarm_three_color_triplet_tilt50_matte_run2_contract/checkpoints/018000/pretrained_model \
  --dataset-root /home/zxro/arena/lerobot/src/lerobot/datasets/openarm_three_color_triplet_tilt50_matte/openarm_three_color_triplet_tilt50_matte_dataset \
  --dataset-repo-id local/openarm_three_color_triplet_tilt50_matte \
  --num-episodes-per-color 10 \
  --max-steps 1000 \
  --seed 1000 \
  --device cuda \
  --use-amp \
  --instruction-order grouped \
  --output /home/zxro/arena/lerobot/outputs/eval/openarm_three_color_triplet_tilt50_matte_run2_contract/reproduced_018000_results.csv
```

Summarize the result CSV:

```bash
python scripts/eval/analyze_color_eval.py \
  /home/zxro/arena/lerobot/outputs/eval/openarm_three_color_triplet_tilt50_matte_run2_contract/reproduced_018000_results.csv
```

## 7. Final Results

### Single-cube result

```text
Successes = 9 / 10
Task Success = 90.0%
```

### Three-color result: checkpoint 018000

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

The dominant bottleneck was manipulation robustness: 18 of 30 episodes failed
before successful placement. Among the 12 successful manipulation episodes,
six selected the instructed color and six selected the wrong color.

The model demonstrated partial language-conditioned color selection. Language
grounding remains partial, and manipulation failure is the dominant bottleneck.

## 9. Reproduction

### Single-cube

1. Use the local dataset in section 2.1.
2. Run the final single-cube training command.
3. Evaluate checkpoint `020000/pretrained_model` for 10 episodes with seed 1000.
4. Expected result: 9 / 10 task success.

### Three-color

1. Use the dataset in section 3.
2. Run the final three-color training command.
3. Evaluate checkpoint `018000/pretrained_model` with the command in section 6.
4. Expected result: 12 / 30 task success, 6 / 30 correct-color success, and
   the confusion matrix in section 7.
