# SmolVLA OpenArm Training And Evaluation Report

## A. Current training contract

```text
Training entrypoint:
- Wrapper: scripts/train/train_smolvla.py
- Actual trainer: src/lerobot/scripts/lerobot_train.py via lerobot-train

Dataset:
- repo_id: local/random_cube_tilt_30_gripper_mapped_box_blue_50_degree
- root: /home/zxro/arena/lerobot/outputs/lerobot_datasets/random_cube_tilt_30_gripper_mapped_box_blue_50_degree

Policy:
- policy.path / pretrained_path: lerobot/smolvla_base
- policy type: smolvla

Input feature handling:
- wrapper passes --policy.input_features=null
- final resolved input features:
  - observation.state: [8]
  - observation.images.top: [3, 480, 640]
  - observation.images.wrist: [3, 480, 640]

Output feature handling:
- wrapper passes --policy.output_features=null
- final resolved output features:
  - action: [8]

VLM loading:
- config field: load_vlm_weights
- successful saved run value: true

Batch size:
- 4

Training steps:
- 20000

Learning rate:
- 1e-4

Optimizer:
- AdamW
- betas: (0.9, 0.95)
- eps: 1e-8
- weight_decay: 1e-10
- grad_clip_norm: 10.0

Scheduler:
- cosine_decay_with_warmup
- warmup_steps: 1000
- decay_steps: 30000
- decay_lr: 2.5e-6

Save frequency:
- 2000

Logging frequency:
- 100

AMP:
- true

Seed:
- 1000

Device:
- cuda

Dataloader workers:
- 4

Video backend:
- pyav

WandB:
- disabled

Other saved successful config values:
- eval_freq: 20000
- use_policy_training_preset: true
- freeze_vision_encoder: true
- train_expert_only: true
- train_state_proj: true
- attention_mode: cross_attn
- prefix_length: 0
- pad_language_to: max_length
- num_expert_layers: 0
- num_vlm_layers: 16
- self_attn_every_n_layers: 2
- expert_width_multiplier: 0.75
```

근거:
- wrapper: [train_smolvla.py](/home/zxro/arena/lerobot/scripts/train/train_smolvla.py)
- 실제 성공 run 저장 config: [train_config.json](/home/zxro/arena/lerobot/outputs/train/ab_local_verified/checkpoints/020000/pretrained_model/train_config.json)
- saved policy config: [config.json](/home/zxro/arena/lerobot/outputs/train/ab_local_verified/checkpoints/020000/pretrained_model/config.json)

## B. Feature resolution flow

```text
pretrained config
    ↓
CLI override
    ↓
dataset metadata
    ↓
final policy config
    ↓
saved checkpoint
```

상세:

1. pretrained config
- `lerobot/smolvla_base`의 pretrained config는 원래 OpenArm contract와 다를 수 있음
- user가 이미 확인한 old base feature:
  - observation.state [6]
  - observation.image
  - observation.image2
  - observation.image3
  - action [6]

2. CLI override
- wrapper가 항상 추가:
  - `--policy.input_features=null`
  - `--policy.output_features=null`
- 위치: [train_smolvla.py](/home/zxro/arena/lerobot/scripts/train/train_smolvla.py#L102)
- `TrainPipelineConfig.validate()`도 default smolvla path일 때 같은 보정 로직을 가짐:
  - [train.py](/home/zxro/arena/lerobot/src/lerobot/configs/train.py#L90)

3. dataset metadata
- dataset metadata 실제 key:
  - `observation.state`
  - `action`
  - `observation.images.top`
  - `observation.images.wrist`
- 직접 확인 결과:
  - `observation.state`: float32, shape `(8,)`
  - `action`: float32, shape `(8,)`
  - `observation.images.top`: video, shape `(480, 640, 3)`
  - `observation.images.wrist`: video, shape `(480, 640, 3)`

4. final policy config
- `make_policy()`가 dataset metadata에서 feature를 만듦:
  - [factory.py](/home/zxro/arena/lerobot/src/lerobot/policies/factory.py#L465)
- 변환 함수:
  - [dataset_to_policy_features()](/home/zxro/arena/lerobot/src/lerobot/datasets/utils.py#L698)
- 최종 resolved features:
  - `observation.state`: STATE `(8,)`
  - `observation.images.top`: VISUAL `(3, 480, 640)`
  - `observation.images.wrist`: VISUAL `(3, 480, 640)`
  - `action`: ACTION `(8,)`

5. saved checkpoint
- successful checkpoint config confirms exactly those final features:
  - [config.json](/home/zxro/arena/lerobot/outputs/train/ab_local_verified/checkpoints/020000/pretrained_model/config.json)

## C. VLM loading verification

- 관련 config 이름:
  - `load_vlm_weights`
  - 정의: [configuration_smolvla.py](/home/zxro/arena/lerobot/src/lerobot/policies/smolvla/configuration_smolvla.py#L88)

- 현재 값:
  - successful saved run: `true`
  - 확인 위치: [train_config.json](/home/zxro/arena/lerobot/outputs/train/ab_local_verified/checkpoints/020000/pretrained_model/train_config.json)

- 코드상 실제 동작:
  - [modeling_smolvla.py](/home/zxro/arena/lerobot/src/lerobot/policies/smolvla/modeling_smolvla.py#L563)
  - `SmolVLMWithExpertModel(... load_vlm_weights=self.config.load_vlm_weights, ...)`
  - 실제 분기:
    - `true`:
      - `AutoModelForImageTextToText.from_pretrained(model_id, ...)`
      - 위치: [smolvlm_with_expert.py](/home/zxro/arena/lerobot/src/lerobot/policies/smolvla/smolvlm_with_expert.py#L76)
    - `false`:
      - `AutoConfig.from_pretrained(model_id)`
      - fresh `SmolVLMForConditionalGeneration(config)`
      - 위치: [smolvlm_with_expert.py](/home/zxro/arena/lerobot/src/lerobot/policies/smolvla/smolvlm_with_expert.py#L83)

- `policy.path=lerobot/smolvla_base`와 관계:
  - `policy.path`는 base SmolVLA checkpoint config/weights source
  - `load_vlm_weights=true`는 내부 VLM backbone도 pretrained VLM weights로 초기화함
  - 둘 다 있어야 “published base에서 이어 fine-tune” 계약이 명확함

- standard command에 명시했는지:
  - 문서상 재현성 때문에 명시 권장
  - 이유: wrapper default인 `SmolVLAConfig.load_vlm_weights`는 코드상 `false`지만, successful run 저장값은 `true`
  - 따라서 long-term reproducibility를 위해 standard command에 `--policy.load_vlm_weights=true`를 넣는 편이 안전함

## D. Evaluation implementation

- 기존 success criterion 위치:

## Three-Color Eval Success Timing Note

The three-color dataset generator uses the storage-box success check only as a
post-controller quality gate after the scripted trajectory is done. The eval
environment checks success every step, so it must use both:

- a tighter box-interior success volume that does not overlap the reset spawn workspace
- a `min_steps_before_success` guard to prevent reset-time false positives
  - [openarm_smolvla_env.py](/home/zxro/arena/lerobot/scripts/eval/openarm_smolvla_env.py#L754)
- 현재 로직:
  - cube position in success box region
  - and cube linear velocity norm < 0.05
  - then `reward = 1.0`
  - `terminated = is_success`

- picked_color 판정 방법:
  - 현재 env는 single-cube 구조
  - 따라서 현재 구현은:
    - success면 `picked_color = object_color`
    - 실패면 `picked_color = failure`
  - 위치: [eval_smolvla.py](/home/zxro/arena/lerobot/scripts/eval/eval_smolvla.py)
  - 주석으로 명시함

- CSV 생성 위치:
  - [eval_smolvla.py](/home/zxro/arena/lerobot/scripts/eval/eval_smolvla.py)
  - 기록 항목:
    - `episode_id`
    - `seed`
    - `instruction`
    - `instruction_color`
    - `target_color`
    - `picked_color`
    - `task_success`
    - `color_correct`
    - cube init pose 일부
    - `termination_reason`

- confusion matrix 생성 방법:
  - script 추가: [analyze_color_eval.py](/home/zxro/arena/lerobot/scripts/eval/analyze_color_eval.py)
  - 입력: eval CSV
  - 출력:
    - instruction x picked_color matrix
    - overall color accuracy
    - per-color accuracy
    - task success rate
    - color-correct task success rate

중요:
- 현재 repository의 OpenArm eval scene은 multi-cube가 아니라 single-cube입니다.
- 따라서 “instruction color와 실제 어떤 색 cube를 골랐는가”라는 진짜 selection confusion matrix는 아직 아닙니다.
- 현재 가능한 것은 “single-cube color-conditioned success/failure logging”입니다.

## E. Position randomization

- 현재 구현:
  - single cube only
  - random reset mode:
    - `x ~ uniform(-0.60, -0.46)`
    - `y ~ uniform(0.03, 0.17)`
    - z fixed `0.1`
    - 위치: [openarm_smolvla_env.py](/home/zxro/arena/lerobot/scripts/eval/openarm_smolvla_env.py#L608)
  - dataset replay mode:
    - deterministic poses reconstructed from dataset seed/ranges
    - 위치: [openarm_smolvla_env.py](/home/zxro/arena/lerobot/scripts/eval/openarm_smolvla_env.py#L151), [openarm_smolvla_env.py](/home/zxro/arena/lerobot/scripts/eval/openarm_smolvla_env.py#L591)

- seed 적용 위치:
  - reset 시 `self.cube_rng = np.random.default_rng(seed)`
  - 위치: [openarm_smolvla_env.py](/home/zxro/arena/lerobot/scripts/eval/openarm_smolvla_env.py#L621)

- shortcut 위험:
  - 현재 single-cube env라서 color-position correlation shortcut 문제 자체는 “색 중 하나를 고르는” 문제로 아직 나타나지 않음
  - 하지만 future 3-cube eval로 가면 color-specific fixed positions가 생기면 바로 shortcut risk가 생김

- 필요한 최소 수정:
  - 지금은 큰 구조 변경 없이 문서화만 했음
  - full color-selection confusion matrix를 진짜로 만들려면:
    - same scene에 red/blue/yellow cube 3개를 동시에 spawn
    - per-episode position permutation/randomization 필요
    - success region 안에 들어간 cube identity를 판정해야 함

## F. Modified files

[path: scripts/eval/eval_smolvla.py](/home/zxro/arena/lerobot/scripts/eval/eval_smolvla.py)  
변경 내용:
- `--instruction-color` 추가
- instruction color 추론 로직 추가
- eval CSV에 `instruction_color`, `picked_color`, `task_success`, `color_correct`, `termination_reason` 등 추가
- debug/invariant logging 강화
변경 이유:
- color-conditioned eval logging과 confusion-matrix용 CSV 생성

[path: scripts/eval/openarm_smolvla_env.py](/home/zxro/arena/lerobot/scripts/eval/openarm_smolvla_env.py)  
변경 내용:
- step info에 final cube position / velocity 추가
- reset info와 env config에 color/task/debug 정보 유지
변경 이유:
- eval result interpretation과 picked_color/task_success logging 보강

[path: scripts/eval/analyze_color_eval.py](/home/zxro/arena/lerobot/scripts/eval/analyze_color_eval.py)  
변경 내용:
- eval CSV를 confusion matrix로 집계하는 분석 스크립트 추가
변경 이유:
- instruction x picked_color 집계 자동화

[path: scripts/eval/run_color_instruction_eval.py](/home/zxro/arena/lerobot/scripts/eval/run_color_instruction_eval.py)  
변경 내용:
- red/blue/yellow instruction sweep runner 추가
- combined CSV 생성
변경 이유:
- 3색 평가 프로토콜 자동 실행

[path: src/lerobot/scripts/lerobot_train.py](/home/zxro/arena/lerobot/src/lerobot/scripts/lerobot_train.py)  
변경 내용:
- training startup 시 아래 로그 추가:
  - `[PRETRAINED POLICY]`
  - `[PRETRAINED POLICY FEATURES]`
  - `[PRETRAINED VLM WEIGHTS LOADED]`
  - `[DATASET FEATURES]`
  - `[FINAL POLICY INPUT FEATURES]`
  - `[FINAL POLICY OUTPUT FEATURES]`
  - `[VLM WEIGHTS LOADED]`
변경 이유:
- reproducible training contract 확인용

[path: docs/smolvla_openarm_training.md](/home/zxro/arena/lerobot/docs/smolvla_openarm_training.md)  
변경 내용:
- OpenArm SmolVLA training contract 문서 추가
- feature resolution, VLM loading, standard command, eval protocol 문서화
변경 이유:
- future sim/real training reproducibility 확보

## G. Commands

1. dataset / feature sanity check

```bash
HF_HOME=/tmp/hf_lerobot \
HF_DATASETS_CACHE=/tmp/hf_lerobot/datasets \
/home/zxro/miniforge3/envs/lab-isaac5-py311/bin/python3.11 - <<'PY'
from pathlib import Path
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.datasets.utils import dataset_to_policy_features

root = Path("/home/zxro/arena/lerobot/outputs/lerobot_datasets/random_cube_tilt_30_gripper_mapped_box_blue_50_degree")
meta = LeRobotDatasetMetadata("local/random_cube_tilt_30_gripper_mapped_box_blue_50_degree", root=root)

print("[DATASET FEATURES]")
for k, v in meta.info["features"].items():
    print(k, v)

print("\n[POLICY FEATURES]")
for k, v in dataset_to_policy_features(meta.info["features"]).items():
    print(k, {"type": str(v.type), "shape": v.shape})
PY
```

2. standard SmolVLA training

```bash
DATASET_PATH=/home/zxro/arena/lerobot/outputs/lerobot_datasets/random_cube_tilt_30_gripper_mapped_box_blue_50_degree
OUTPUT_DIR=/home/zxro/arena/lerobot/outputs/train/<RUN_NAME>

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
  --no-wandb \
  --policy.load_vlm_weights=true
```

3. 3-color simulation evaluation

현재 가능한 것은 single-cube actual-color sweep입니다. 예: actual cube = blue, instruction ∈ {red, blue, yellow}

```bash
PATH=/home/zxro/miniforge3/envs/lab-isaac5-py311/bin:$PATH \
/home/zxro/miniforge3/envs/lab-isaac5-py311/bin/python3.11 \
/home/zxro/arena/lerobot/scripts/eval/run_color_instruction_eval.py \
  --policy-path /home/zxro/arena/lerobot/outputs/train/ab_local_verified/checkpoints/020000/pretrained_model \
  --dataset-root /home/zxro/arena/lerobot/outputs/lerobot_datasets/random_cube_tilt_30_gripper_mapped_box_blue_50_degree \
  --dataset-repo-id local/random_cube_tilt_30_gripper_mapped_box_blue_50_degree \
  --output-dir /home/zxro/arena/lerobot/outputs/eval/color_instruction_blue_single_cube \
  --actual-cube-color blue \
  --num-episodes 10 \
  --max-steps 1000 \
  --seed 1000 \
  --device cuda \
  --use-amp \
  --cube-reset-mode dataset_replay \
  --dataset-pose-start-episode 0
```

baseline red instruction / red actual cube 단일 실행:

```bash
PATH=/home/zxro/miniforge3/envs/lab-isaac5-py311/bin:$PATH \
/home/zxro/miniforge3/envs/lab-isaac5-py311/bin/python3.11 /home/zxro/arena/lerobot/scripts/eval/eval_smolvla.py \
  --policy-path /home/zxro/arena/lerobot/outputs/train/ab_local_verified/checkpoints/020000/pretrained_model \
  --dataset-root /home/zxro/arena/lerobot/outputs/lerobot_datasets/random_cube_tilt_30_gripper_mapped_box_blue_50_degree \
  --dataset-repo-id local/random_cube_tilt_30_gripper_mapped_box_blue_50_degree \
  --num-episodes 10 \
  --max-steps 1000 \
  --seed 1000 \
  --device cuda \
  --use-amp \
  --cube-reset-mode dataset_replay \
  --dataset-pose-start-episode 0 \
  --object-color red \
  --instruction-color red \
  --output /home/zxro/arena/lerobot/outputs/eval/red_red_single_cube/results.csv
```

4. confusion matrix generation

```bash
/home/zxro/miniforge3/envs/lab-isaac5-py311/bin/python3.11 \
  /home/zxro/arena/lerobot/scripts/eval/analyze_color_eval.py \
  /home/zxro/arena/lerobot/outputs/eval/color_instruction_blue_single_cube/combined_results.csv
```

## H. Remaining issues

- 현재 OpenArm eval environment는 single-cube 구조입니다.
  - 따라서 요청한 “Instruction × Red picked / Blue picked / Yellow picked / Failure” confusion matrix를 엄밀히 구현하려면 multi-cube scene이 필요합니다.
  - 이번 작업에서는 current repo behavior를 보존하면서 single-cube color-conditioned logging/analysis infrastructure까지만 넣었습니다.

- 실제 Isaac Sim rollout smoke test는 이 세션에서 shared semaphore permission error 때문에 실행 검증을 완료하지 못했습니다.
  - 문법 검증은 통과했습니다.
  - 실제 render/video/CSV 생성은 Isaac Sim이 정상 실행되는 환경에서 확인해야 합니다.

- pretrained base config의 원래 feature schema는 user가 제시한 값과 current saved successful checkpoint 결과를 함께 문서화했습니다.
  - base repo를 다시 직접 조회하는 네트워크 검증은 이 세션에서 수행하지 않았습니다.

- future true 3-color selection eval을 하려면 최소 추가 작업이 필요합니다.
  - 3개 cube 동시 spawn
  - per-episode spatial permutation/randomization
  - success region에 들어간 cube identity 판정
  - 그때부터 requested confusion matrix가 실질적으로 의미를 가집니다.
