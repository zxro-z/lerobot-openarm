# SmolVLA OpenArm Training Root Cause Report

## 1. 문제 요약

처음 확인된 문제는 다음과 같았다.

- training 자체는 실행되고 checkpoint도 생성됐다.
- 하지만 실제 OpenArm rollout에서는 기대한 pick-and-place behavior가 안정적으로 나오지 않았다.
- dataset degree/gripper contract, checkpoint degree contract, eval degree contract는 별도 검증에서 통과했다.
- 이후 성공한 동료 repository와 실제 train/eval 경로를 대조하면서, 최종적으로 training feature contract mismatch가 핵심 원인으로 확인됐다.

## 2. 최종 Root Cause

이번 문제의 가장 중요한 원인은 `lerobot/smolvla_base` pretrained base checkpoint가 이미 가지고 있는 기본 feature schema와, 현재 OpenArm training dataset의 실제 feature schema가 서로 달랐다는 점이다.

현재 OpenArm dataset의 실제 contract는 checkpoint와 dataset metadata에서 다음과 같이 확인됐다.

- `observation.images.top`
- `observation.images.wrist`
- `observation.state`: shape `[8]`
- `action`: shape `[8]`

근거:

- dataset validation wrapper: [`scripts/train/train_smolvla.py`](/home/zxro/arena/lerobot/scripts/train/train_smolvla.py)
- 저장된 checkpoint config: [`config.json`](/home/zxro/arena/lerobot/outputs/train/random_cube_tilt_30_blue_box_smolvla/checkpoints/020000/pretrained_model/config.json)
- 저장된 train config: [`train_config.json`](/home/zxro/arena/lerobot/outputs/train/random_cube_tilt_30_blue_box_smolvla/checkpoints/020000/pretrained_model/train_config.json)

반면, 수정 전 실제 training error와 config/log에서는 pretrained base 쪽 feature schema가 현재 dataset과 다른 이름과 차원을 사용하고 있었다.

확인된 로그:

```text
ValueError: Feature mismatch between dataset/environment and policy config.
- Missing features: ['observation.images.camera1', 'observation.images.camera2', 'observation.images.camera3']
- Extra features: ['observation.images.top', 'observation.images.wrist']
```

그리고 같은 조사 과정에서 수정 전 policy config/log에 다음 값이 관찰됐다.

- `input_features`: `observation.images.camera1`, `observation.images.camera2`, `observation.images.camera3`, `observation.state` shape `[6]`
- `output_features`: `action` shape `[6]`

즉, pretrained base의 feature schema를 그대로 유지하면 현재 OpenArm dataset metadata와 일치하지 않는 policy input/output contract가 형성될 수 있었다. 이것이 이번 문제의 핵심 root cause다.

Confidence: `CONFIRMED`

## 3. 왜 문제가 발생했는가

실제 training path는 다음 순서로 이어진다.

1. pretrained checkpoint load
2. pretrained config load
3. 기존 `input_features` / `output_features` 유지 또는 override
4. OpenArm dataset load
5. dataset metadata의 `top` / `wrist` / `state(8)` / `action(8)` schema와 비교

관련 코드 역할은 아래와 같다.

- [`src/lerobot/configs/train.py#L81`](/home/zxro/arena/lerobot/src/lerobot/configs/train.py#L81)
  `TrainPipelineConfig.validate()`는 CLI에서 `policy.path`가 들어오면 `PreTrainedConfig.from_pretrained(...)`로 policy config를 로드한다.
- [`src/lerobot/configs/policies.py#L58`](/home/zxro/arena/lerobot/src/lerobot/configs/policies.py#L58)
  `PreTrainedConfig`는 `input_features`와 `output_features`를 가진다. 주석에 명시돼 있듯 `input_features`는 `None/null`이면 dataset에서 추론할 수 있다.
- [`src/lerobot/policies/factory.py#L406`](/home/zxro/arena/lerobot/src/lerobot/policies/factory.py#L406)
  `make_policy()`는 `ds_meta`가 있으면 dataset metadata를 policy feature로 변환한다.
- [`src/lerobot/policies/factory.py#L470`](/home/zxro/arena/lerobot/src/lerobot/policies/factory.py#L470)
  `cfg.output_features`는 dataset/env feature에서 action 항목으로 다시 채워진다.
- [`src/lerobot/policies/factory.py#L471`](/home/zxro/arena/lerobot/src/lerobot/policies/factory.py#L471)
  `cfg.input_features`는 비어 있거나 `None`일 때만 dataset/env feature 기준으로 채워진다.
- [`src/lerobot/policies/factory.py#L488`](/home/zxro/arena/lerobot/src/lerobot/policies/factory.py#L488)
  `cfg.pretrained_path`가 있으면 pretrained policy weights를 로드한다.
- [`src/lerobot/policies/factory.py#L526`](/home/zxro/arena/lerobot/src/lerobot/policies/factory.py#L526)
  마지막에 dataset/env feature와 policy config의 visual feature 이름이 일치하는지 검증한다.

중요한 분기는 이것이다.

- `input_features` / `output_features`가 이미 채워져 있으면:
  pretrained config에 저장된 feature schema가 그대로 남을 수 있다.
- `input_features=null` / `output_features=null`로 넘기면:
  dataset metadata 기준으로 현재 dataset contract를 다시 구성하게 된다.

이번 문제는 pretrained weights 자체가 아니라, pretrained config 안에 저장돼 있던 feature schema가 현재 dataset contract와 맞지 않았던 데서 발생했다.

## 4. 실제로 확인된 Feature Mismatch

| 항목 | 수정 전 | 수정 후 |
|---|---|---|
| policy base | `lerobot/smolvla_base` | `lerobot/smolvla_base` |
| input_features | pretrained base schema 유지 | dataset metadata에서 재추론 |
| output_features | pretrained base schema 유지 | dataset metadata에서 재추론 |
| cameras | `observation.images.camera1`, `observation.images.camera2`, `observation.images.camera3` | `observation.images.top`, `observation.images.wrist` |
| state | `observation.state` shape `[6]` | `observation.state` shape `[8]` |
| action | `action` shape `[6]` | `action` shape `[8]` |

수정 전 근거:

- 실제 training error log의 missing/extra visual feature 목록
- 조사 과정에서 확인한 pretrained policy config/log의 state/action 차원

수정 후 근거:

- [`config.json`](/home/zxro/arena/lerobot/outputs/train/random_cube_tilt_30_blue_box_smolvla/checkpoints/020000/pretrained_model/config.json)
- [`train_config.json`](/home/zxro/arena/lerobot/outputs/train/random_cube_tilt_30_blue_box_smolvla/checkpoints/020000/pretrained_model/train_config.json)

## 5. 해결 방법

최종적으로 해결된 핵심은 pretrained base를 버리는 것이 아니었다. 해결된 방식은 다음 세 가지를 동시에 만족시키는 것이었다.

1. pretrained SmolVLA base weights/config는 계속 사용한다.
2. `input_features`와 `output_features`만 `null`로 override한다.
3. 현재 dataset metadata를 기준으로 feature schema를 다시 inference하게 만든다.

핵심 argument는 다음과 같다.

```text
--policy.path=lerobot/smolvla_base
--policy.input_features=null
--policy.output_features=null
```

각 argument의 의미:

- `--policy.path=lerobot/smolvla_base`
  pretrained model weights와 base config를 로드한다.
- `--policy.input_features=null`
  pretrained config에 저장된 기존 input feature schema를 그대로 쓰지 않고, dataset metadata에서 다시 추론하도록 만든다.
- `--policy.output_features=null`
  pretrained config에 저장된 기존 output feature schema를 그대로 쓰지 않고, dataset metadata에서 다시 추론하도록 만든다.

실제 wrapper 구현도 이 방식으로 고정돼 있다.

- [`scripts/train/train_smolvla.py#L91`](/home/zxro/arena/lerobot/scripts/train/train_smolvla.py#L91)
  `--policy.path=...`
- [`scripts/train/train_smolvla.py#L95`](/home/zxro/arena/lerobot/scripts/train/train_smolvla.py#L95)
  `--policy.input_features=null`
- [`scripts/train/train_smolvla.py#L96`](/home/zxro/arena/lerobot/scripts/train/train_smolvla.py#L96)
  `--policy.output_features=null`

## 6. 수정 후 검증

생성된 성공 checkpoint에서 다음을 직접 확인했다.

- `input_features`에 `observation.images.top` 존재
- `input_features`에 `observation.images.wrist` 존재
- `observation.state` shape = `[8]`
- `output action` shape = `[8]`
- dataset normalization stats 관련 processor 파일이 checkpoint에 저장됨
- pretrained base가 실제 사용됨

확인 근거:

- [`config.json`](/home/zxro/arena/lerobot/outputs/train/random_cube_tilt_30_blue_box_smolvla/checkpoints/020000/pretrained_model/config.json)
  - `pretrained_path = "lerobot/smolvla_base"`
  - `load_vlm_weights = true`
  - `input_features` contains `observation.state: shape [8]`, `observation.images.top: shape [3, 480, 640]`, `observation.images.wrist: shape [3, 480, 640]`
  - `output_features` contains `action: shape [8]`
- [`train_config.json`](/home/zxro/arena/lerobot/outputs/train/random_cube_tilt_30_blue_box_smolvla/checkpoints/020000/pretrained_model/train_config.json)
  - `dataset.root = "/home/zxro/arena/lerobot/outputs/lerobot_datasets/random_cube_tilt_30_gripper_mapped_box_blue_50_degree"`
  - `dataset.repo_id = "local/random_cube_tilt_30_gripper_mapped_box_blue_50_degree"`
  - `dataset.video_backend = "pyav"`
- checkpoint directory
  - [`policy_preprocessor.json`](/home/zxro/arena/lerobot/outputs/train/random_cube_tilt_30_blue_box_smolvla/checkpoints/020000/pretrained_model/policy_preprocessor.json)
  - [`policy_postprocessor.json`](/home/zxro/arena/lerobot/outputs/train/random_cube_tilt_30_blue_box_smolvla/checkpoints/020000/pretrained_model/policy_postprocessor.json)
  - [`policy_preprocessor_step_5_normalizer_processor.safetensors`](/home/zxro/arena/lerobot/outputs/train/random_cube_tilt_30_blue_box_smolvla/checkpoints/020000/pretrained_model/policy_preprocessor_step_5_normalizer_processor.safetensors)
  - [`policy_postprocessor_step_0_unnormalizer_processor.safetensors`](/home/zxro/arena/lerobot/outputs/train/random_cube_tilt_30_blue_box_smolvla/checkpoints/020000/pretrained_model/policy_postprocessor_step_0_unnormalizer_processor.safetensors)

## 7. 잘못된 접근과 차이

### 잘못될 수 있는 방식

다음 방식은 현재 OpenArm dataset contract와 충돌할 수 있다.

- `--policy.type=smolvla`만 사용해서 scratch/default policy config를 쓰는 방식
- pretrained path는 사용하지만, pretrained checkpoint에 저장된 기존 `input_features` / `output_features`를 그대로 유지하는 방식

이 경우 camera key, state dimension, action dimension이 현재 dataset과 다를 수 있다. 실제로 이번 조사에서는 visual feature mismatch와 state/action dimension mismatch가 모두 확인됐다.

### 성공한 방식

성공한 방식은 pretrained base를 명시적으로 사용하면서 아래 두 override를 함께 넣는 것이었다.

```text
--policy.path=lerobot/smolvla_base
--policy.input_features=null
--policy.output_features=null
```

이 방식이 OpenArm dataset contract와 일치하는 이유는, pretrained weights는 유지하되 feature contract만 현재 dataset metadata 기준으로 다시 맞추기 때문이다. 즉 weights와 feature schema를 분리해서 다룬 것이다.

## 8. 최종 Training Command

현재 repository의 실제 wrapper 구현 기준으로 다시 확인한 최종 training command는 다음과 같다.

```bash
PATH=/home/zxro/miniforge3/envs/lab-isaac5-py311/bin:$PATH \
HF_HOME=/tmp/hf_lerobot \
HF_DATASETS_CACHE=/tmp/hf_lerobot/datasets \
/home/zxro/miniforge3/envs/lab-isaac5-py311/bin/python3.11 scripts/train/train_smolvla.py \
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
  --video-backend pyav \
  --use-amp \
  --no-wandb
```

주의:

- wrapper CLI는 `--policy-path`를 받는다.
- wrapper 내부에서 실제 `lerobot-train` 호출 시 `--policy.input_features=null`과 `--policy.output_features=null`를 자동으로 추가한다.

## 9. 최종 Evaluation Command

현재 repository의 실제 reference-style eval wrapper 기준으로 다시 확인한 최종 evaluation command는 다음과 같다.

```bash
PATH=/home/zxro/miniforge3/envs/lab-isaac5-py311/bin:$PATH \
/home/zxro/miniforge3/envs/lab-isaac5-py311/bin/python3.11 scripts/eval/eval_smolvla.py \
  --policy-path /home/zxro/arena/lerobot/outputs/train/random_cube_tilt_30_blue_box_smolvla/checkpoints/020000/pretrained_model \
  --dataset-root /home/zxro/arena/lerobot/outputs/lerobot_datasets/random_cube_tilt_30_gripper_mapped_box_blue_50_degree \
  --dataset-repo-id local/random_cube_tilt_30_gripper_mapped_box_blue_50_degree \
  --num-episodes 10 \
  --max-steps 1000 \
  --seed 1000 \
  --device cuda \
  --use-amp \
  --output outputs/eval/openarm_smolvla/results.csv
```

이 문서에서 eval wrapper는 재현에 사용된 실행 경로로만 기록한다. 최종 root cause는 eval wrapper 차이가 아니라 training feature contract mismatch다.

## 10. 핵심 교훈

- pretrained model을 사용한다고 해서 pretrained feature schema까지 현재 dataset에 맞는 것은 아니다.
- robotics policy에서는 camera key, state dimension, action dimension이 dataset과 정확히 일치해야 한다.
- pretrained weights와 dataset-specific feature contract는 별개로 다뤄야 한다.
- training이 crash 없이 진행되고 loss가 나온다고 해서 feature contract가 올바른 것은 아니다.
- checkpoint의 `config.json`과 `train_config.json`을 반드시 확인해야 한다.

## 11. Quick Checklist

- [ ] dataset camera keys 확인
- [ ] state dimension 확인
- [ ] action dimension 확인
- [ ] pretrained base 사용 여부 확인
- [ ] `input_features` / `output_features`가 dataset 기준으로 inference되는지 확인
- [ ] checkpoint `config.json` 확인
- [ ] checkpoint `train_config.json` 확인
- [ ] normalization stats processor 파일 확인
- [ ] short train smoke 확인
- [ ] short eval 확인
- [ ] full 20k train 확인
