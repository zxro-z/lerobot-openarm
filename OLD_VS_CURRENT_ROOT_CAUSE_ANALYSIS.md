# OLD vs CURRENT Root Cause Analysis

## 1. 원래 실패 상태

이번 forensic analysis에서 "원래 실패 상태"는 두 층으로 나눠서 봐야 했다.

1. 역사적으로 가장 이른 실패 train command
2. 현재 성공 wrapper들이 추가되기 직전의 repository 상태

이 둘은 동일한 시점이 아니다.

확인된 가장 이른 실패 train command는 사용자 제공 명령과 저장된 checkpoint config 기준으로 다음이다.

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

근거:

- 사용자 제공 command
- [`outputs/train/smolvla_openarm_v9/checkpoints/020000/pretrained_model/config.json`](/home/zxro/arena/lerobot/outputs/train/smolvla_openarm_v9/checkpoints/020000/pretrained_model/config.json)
- [`outputs/train/smolvla_openarm_v9/checkpoints/020000/pretrained_model/train_config.json`](/home/zxro/arena/lerobot/outputs/train/smolvla_openarm_v9/checkpoints/020000/pretrained_model/train_config.json)

이 시점의 runtime 결과에서 확인된 중요한 값:

- `pretrained_path = null`
- `load_vlm_weights = false`
- `batch_size = 8`
- `use_amp = false`
- `save_freq = 20000`
- `log_freq = 200`

반면, 현재 성공 상태 바로 직전의 repository state는 이것보다 훨씬 뒤의 상태였다. 저장된 run과 파일 mtime 기준으로, 성공 직전 OLD는 `smolvla_openarm_v11_refalign` 시기와 가장 가깝다.

근거:

- [`outputs/train/smolvla_openarm_v11_refalign/checkpoints/020000/pretrained_model/train_config.json`](/home/zxro/arena/lerobot/outputs/train/smolvla_openarm_v11_refalign/checkpoints/020000/pretrained_model/train_config.json)
- file mtime
  - `gym_openarm/openarm_env.py`: `2026-08-10 13:55`
  - `src/lerobot/envs/configs.py`: `2026-08-10 13:56`
  - `scripts/train/train_smolvla.py`: `2026-08-10 15:03`
  - `scripts/eval/openarm_smolvla_env.py`: `2026-08-10 16:16`
  - `scripts/eval/eval_smolvla.py`: `2026-08-10 16:20`

즉, 성공에 사용된 wrapper 3개는 OLD 상태에는 없었고, OpenArm native integration 관련 tracked/untracked 코드는 이미 OLD 상태에 존재했다.

## 2. 현재 성공 상태

현재 성공 상태는 다음 command와 wrapper path로 확인됐다.

Train:

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

Eval:

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

근거:

- [`scripts/train/train_smolvla.py`](/home/zxro/arena/lerobot/scripts/train/train_smolvla.py)
- [`scripts/eval/eval_smolvla.py`](/home/zxro/arena/lerobot/scripts/eval/eval_smolvla.py)
- [`outputs/train/random_cube_tilt_30_blue_box_smolvla/checkpoints/020000/pretrained_model/train_config.json`](/home/zxro/arena/lerobot/outputs/train/random_cube_tilt_30_blue_box_smolvla/checkpoints/020000/pretrained_model/train_config.json)
- GUI rollout success는 사용자 확인

## 3. OLD state 복원 가능성

### OLD_STATE_CANDIDATE 1

- source: earliest failed saved run
- commit/reflog/path: run artifact only, not a git commit
- timestamp: checkpoint saved under [`outputs/train/smolvla_openarm_v9/checkpoints/020000/pretrained_model`](/home/zxro/arena/lerobot/outputs/train/smolvla_openarm_v9/checkpoints/020000/pretrained_model)
- confidence: `CONFIRMED` for runtime config, `UNKNOWN` for exact source tree

설명:

- `v9` run은 사용자 original command와 직접 연결된다.
- 하지만 그때의 full source tree는 git commit이나 stash로 보존돼 있지 않다.

### OLD_STATE_CANDIDATE 2

- source: immediate pre-success working tree
- commit/reflog/path: current `HEAD` 기반 local dirty tree에서 wrapper 생성 전 파일만 추출
- timestamp: `2026-08-10 13:55` to `13:56` local file mtimes
- confidence: `LIKELY`

설명:

- 성공 wrapper 파일 3개는 모두 `2026-08-10 15:03` 이후 생성됐다.
- 반면 native OpenArm integration 파일과 `gym_openarm/openarm_env.py`는 그보다 앞서 수정돼 있었다.
- 따라서 "현재 성공 상태 직전" OLD는 wrapper 없는 현재 dirty tree로 보는 것이 가장 안전하다.

### OLD_STATE_CANDIDATE 3

- source: git `HEAD`
- commit/reflog/path: `d324ffe810d17264a0b1e628698aa1fa09aa639c`
- timestamp: current `HEAD`
- confidence: `CONFIRMED` for tracked upstream base, `LOW` as failure-state reconstruction

설명:

- git reflog에는 Codex 작업 전후 commit이 없다.
- tracked 변경 대부분은 local dirty state로만 존재한다.
- 따라서 `HEAD` 단독은 실제 실패 상태를 충분히 설명하지 못한다.

## 4. 안전한 OLD snapshot

비교용 OLD snapshot은 `/tmp/lerobot_old_state`에 생성했다.

구성:

- current working tree에서, wrapper 추가 이전에 이미 존재하던 파일만 복사
- 포함:
  - `src/lerobot/configs/train.py`
  - `src/lerobot/scripts/lerobot_eval.py`
  - `src/lerobot/envs/configs.py`
  - `src/lerobot/envs/factory.py`
  - `src/lerobot/envs/utils.py`
  - `gym_openarm/openarm_env.py`
  - `scripts/openarm_rollout_diagnostic.py`
  - `scripts/openarm_dataset_scale_diagnostic.py`
- 제외:
  - `scripts/train/train_smolvla.py`
  - `scripts/eval/eval_smolvla.py`
  - `scripts/eval/openarm_smolvla_env.py`

정확성:

- tracked/untracked 파일의 선택 기준은 file mtime과 작업 기록이다.
- local modifications의 exact creation author/time은 git으로 복원되지 않으므로 일부는 `LIKELY`다.

## 5. OLD vs CURRENT 코드 차이

가장 중요한 사실은 immediate OLD snapshot과 CURRENT 사이에서 아래 파일들은 바뀌지 않았다는 점이다.

- [`src/lerobot/configs/train.py`](/home/zxro/arena/lerobot/src/lerobot/configs/train.py)
- [`src/lerobot/scripts/lerobot_eval.py`](/home/zxro/arena/lerobot/src/lerobot/scripts/lerobot_eval.py)
- [`src/lerobot/envs/configs.py`](/home/zxro/arena/lerobot/src/lerobot/envs/configs.py)
- [`src/lerobot/envs/factory.py`](/home/zxro/arena/lerobot/src/lerobot/envs/factory.py)
- [`src/lerobot/envs/utils.py`](/home/zxro/arena/lerobot/src/lerobot/envs/utils.py)
- [`gym_openarm/openarm_env.py`](/home/zxro/arena/lerobot/gym_openarm/openarm_env.py)

즉, 현재 성공은 이 tracked/native 파일들을 추가로 고쳐서 생긴 것이 아니다. 성공 직전 OLD와 CURRENT 사이의 실제 코드 변화는 wrapper 추가가 핵심이다.

### A. TRAINING PATH CHANGES

#### [CHANGE T1]

- file: [`scripts/train/train_smolvla.py`](/home/zxro/arena/lerobot/scripts/train/train_smolvla.py)
- function: `main`

OLD:

- 파일 자체가 없었다.
- training은 native `lerobot-train` CLI 또는 그에 준하는 수동 command로 실행됐다.

CURRENT:

- local dataset contract를 먼저 검증한다.
- 내부적으로 `lerobot-train`을 호출한다.
- `--policy.path=lerobot/smolvla_base`
- `--policy.input_features=null`
- `--policy.output_features=null`
- local dataset root/repo id
- `batch_size=4`
- `save_freq=2000`
- `log_freq=100`
- `policy.use_amp=true`

실제 실행 path에 들어갔는가: `YES`

training checkpoint를 바꿀 수 있는가: `YES`

policy rollout behavior를 바꿀 수 있는가: `YES`

성공에 기여했을 가능성: `HIGH`

confidence: `CONFIRMED`

근거:

- wrapper 파일 실존
- 성공 checkpoint [`train_config.json`](/home/zxro/arena/lerobot/outputs/train/random_cube_tilt_30_blue_box_smolvla/checkpoints/020000/pretrained_model/train_config.json)
- `v9`, `v10`, `v11_refalign` run config와 직접 비교 가능

#### [CHANGE T2]

- file: [`src/lerobot/configs/train.py`](/home/zxro/arena/lerobot/src/lerobot/configs/train.py)
- function: `TrainPipelineConfig.validate`

OLD:

- immediate pre-success OLD와 CURRENT가 동일하다.

CURRENT:

- 동일하다.

실제 실행 path에 들어갔는가: `NO` for the successful wrapper command

training checkpoint를 바꿀 수 있는가: `Potentially yes`, but not on the confirmed successful wrapper path

policy rollout behavior를 바꿀 수 있는가: `NO` directly

성공에 기여했을 가능성: `LOW`

confidence: `CONFIRMED`

근거:

- `/tmp/lerobot_old_state/src/lerobot/configs/train.py`와 현재 파일 byte-identical
- current successful command는 wrapper에서 이미 `--policy.path`와 null feature overrides를 넘긴다

### B. EVALUATION PATH CHANGES

#### [CHANGE E1]

- file: [`scripts/eval/eval_smolvla.py`](/home/zxro/arena/lerobot/scripts/eval/eval_smolvla.py)
- function: `main`

OLD:

- 파일 자체가 없었다.
- native [`src/lerobot/scripts/lerobot_eval.py`](/home/zxro/arena/lerobot/src/lerobot/scripts/lerobot_eval.py) 기반 평가 path를 사용했을 가능성이 높다.

CURRENT:

- custom eval loop가 `predict_action()`을 직접 호출한다.
- vector env / native `lerobot-eval` plumbing을 우회한다.
- dataset metadata를 직접 load한다.
- checkpoint config를 직접 load한다.
- custom `OpenArmEnv`를 직접 생성한다.

실제 실행 path에 들어갔는가: `YES`

training checkpoint를 바꿀 수 있는가: `NO`

policy rollout behavior를 바꿀 수 있는가: `YES`

성공에 기여했을 가능성: `MEDIUM`

confidence: `LIKELY`

근거:

- current successful eval은 이 wrapper로 확인됨
- old native eval command는 artifact로 직접 복원되진 않았으므로 일부는 `LIKELY`

#### [CHANGE E2]

- file: [`scripts/eval/openarm_smolvla_env.py`](/home/zxro/arena/lerobot/scripts/eval/openarm_smolvla_env.py)
- function: `OpenArmEnv.reset`, `OpenArmEnv.step`, `OpenArmEnv._get_obs`

OLD:

- 파일 자체가 없었다.
- old direct OpenArm runtime env는 [`gym_openarm/openarm_env.py`](/home/zxro/arena/lerobot/gym_openarm/openarm_env.py)였다.

CURRENT:

- wrapper-local env를 직접 사용한다.
- asset path를 동적으로 resolve한다.
- `HEADLESS`를 env var로 제어한다.
- `reset(seed=...)`마다 cube RNG를 재초기화한다.
- `gripper_override_mode`와 `dataset_schedule` override는 제거돼 있다.
- `observation.state`, `top`, `wrist`, degree contract는 유지한다.

실제 실행 path에 들어갔는가: `YES`

training checkpoint를 바꿀 수 있는가: `NO`

policy rollout behavior를 바꿀 수 있는가: `YES`

성공에 기여했을 가능성: `MEDIUM`

confidence: `CONFIRMED`

근거:

- [`gym_openarm/openarm_env.py`](/home/zxro/arena/lerobot/gym_openarm/openarm_env.py) vs [`scripts/eval/openarm_smolvla_env.py`](/home/zxro/arena/lerobot/scripts/eval/openarm_smolvla_env.py) direct diff

#### [CHANGE E3]

- file: [`src/lerobot/scripts/lerobot_eval.py`](/home/zxro/arena/lerobot/src/lerobot/scripts/lerobot_eval.py), [`src/lerobot/envs/configs.py`](/home/zxro/arena/lerobot/src/lerobot/envs/configs.py), [`src/lerobot/envs/factory.py`](/home/zxro/arena/lerobot/src/lerobot/envs/factory.py), [`src/lerobot/envs/utils.py`](/home/zxro/arena/lerobot/src/lerobot/envs/utils.py)
- function: native eval plumbing

OLD:

- immediate pre-success OLD와 CURRENT가 동일하다.

CURRENT:

- 동일하다.

실제 실행 path에 들어갔는가: `NO` on the current successful wrapper eval

training checkpoint를 바꿀 수 있는가: `NO`

policy rollout behavior를 바꿀 수 있는가: `NO` for current successful path

성공에 기여했을 가능성: `LOW`

confidence: `CONFIRMED`

근거:

- `/tmp/lerobot_old_state`와 current byte-identical

### C. ENVIRONMENT CHANGES

#### [CHANGE C1]

- file: [`gym_openarm/openarm_env.py`](/home/zxro/arena/lerobot/gym_openarm/openarm_env.py)
- function: `dataset_schedule_gripper_value_deg`, `_resolve_gripper_action_deg`

OLD:

- immediate pre-success OLD에 이미 존재했다.

CURRENT:

- 그대로 존재한다.

실제 실행 path에 들어갔는가: `NO` for current successful wrapper eval in normal mode

training checkpoint를 바꿀 수 있는가: `NO`

policy rollout behavior를 바꿀 수 있는가: `YES` when `gripper_override_mode=dataset_schedule`

성공에 기여했을 가능성: `LOW` for current success, `HIGH` as diagnostic evidence

confidence: `CONFIRMED`

근거:

- 기존 실험에서 `dataset_schedule` ablation이 success를 회복했음
- 그러나 current successful eval wrapper는 이 path를 사용하지 않음

## 6. OLD vs CURRENT command 차이

### TRAIN OLD

사용자 original command:

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

### CURRENT SUCCESS

실제 성공 train command:

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

비교:

| 항목 | TRAIN OLD | CURRENT SUCCESS | checkpoint / rollout 영향 |
|---|---|---|---|
| dataset source | HF repo | local dataset root | HIGH |
| dataset root | none | explicit local absolute path | HIGH |
| repo id | `a126-kitech/...` | `local/random_cube_tilt_30_gripper_mapped_box_blue_50_degree` | HIGH |
| pretrained policy loading | `v9` 기준 `pretrained_path=null` | explicit `lerobot/smolvla_base` | HIGH |
| input/output feature override | command에 없음 | wrapper가 null override 강제 | MEDIUM |
| batch size | 8 | 4 | MEDIUM |
| num_workers | implicit/default 4 | explicit 4 | LOW |
| AMP | `v9` 기준 false | true | MEDIUM |
| video backend | pyav | pyav | LOW |
| save/log frequency | 20000 / 200 | 2000 / 100 | LOW |
| seed | 1000 | 1000 | LOW |
| wrapper behavior | 없음 | dataset validation + explicit `policy.path` + null feature override | HIGH |

중요:

- `v11_refalign` 시점까지 오면 이미 `pretrained_path=lerobot/smolvla_base`, `use_amp=true`, `batch_size=4`, `save_freq=2000`, `log_freq=100`은 현재 성공 config와 같아진다.
- 따라서 현재 성공과 가장 직접적으로 남는 train-side 차이는 dataset source/root와 wrapper 경유 실행이다.

## 7. 실제 checkpoint 차이

### `v9` vs current success

강한 차이:

- `v9`: `pretrained_path = null`, `load_vlm_weights = false`, `batch_size = 8`, `use_amp = false`
- current success: `pretrained_path = lerobot/smolvla_base`, `load_vlm_weights = true`, `batch_size = 4`, `use_amp = true`

근거:

- [`outputs/train/smolvla_openarm_v9/checkpoints/020000/pretrained_model/config.json`](/home/zxro/arena/lerobot/outputs/train/smolvla_openarm_v9/checkpoints/020000/pretrained_model/config.json)
- [`outputs/train/random_cube_tilt_30_blue_box_smolvla/checkpoints/020000/pretrained_model/config.json`](/home/zxro/arena/lerobot/outputs/train/random_cube_tilt_30_blue_box_smolvla/checkpoints/020000/pretrained_model/config.json)

### `v11_refalign` vs current success

핵심:

- 저장된 `config.json`은 비교한 범위에서 동일하다.
- 차이는 `train_config.json`의 dataset source/root와 output/job name뿐이다.

동일:

- `input_features`
- `output_features`
- camera keys
- state/action dimensions
- `pretrained_path`
- `load_vlm_weights`
- `chunk_size = 50`
- `n_action_steps = 50`
- optimizer
- scheduler
- `use_amp = true`

차이:

- `dataset.repo_id`
- `dataset.root`
- `output_dir`
- `job_name`

근거:

- [`outputs/train/smolvla_openarm_v11_refalign/checkpoints/020000/pretrained_model/config.json`](/home/zxro/arena/lerobot/outputs/train/smolvla_openarm_v11_refalign/checkpoints/020000/pretrained_model/config.json)
- [`outputs/train/smolvla_openarm_v11_refalign/checkpoints/020000/pretrained_model/train_config.json`](/home/zxro/arena/lerobot/outputs/train/smolvla_openarm_v11_refalign/checkpoints/020000/pretrained_model/train_config.json)
- [`outputs/train/random_cube_tilt_30_blue_box_smolvla/checkpoints/020000/pretrained_model/config.json`](/home/zxro/arena/lerobot/outputs/train/random_cube_tilt_30_blue_box_smolvla/checkpoints/020000/pretrained_model/config.json)
- [`outputs/train/random_cube_tilt_30_blue_box_smolvla/checkpoints/020000/pretrained_model/train_config.json`](/home/zxro/arena/lerobot/outputs/train/random_cube_tilt_30_blue_box_smolvla/checkpoints/020000/pretrained_model/train_config.json)

## 8. eval / env 차이

OLD native eval path는 다음 순서였을 가능성이 높다.

`lerobot-eval`
→ [`src/lerobot/scripts/lerobot_eval.py`](/home/zxro/arena/lerobot/src/lerobot/scripts/lerobot_eval.py)
→ [`src/lerobot/envs/factory.py`](/home/zxro/arena/lerobot/src/lerobot/envs/factory.py)
→ vector env
→ [`src/lerobot/envs/utils.py`](/home/zxro/arena/lerobot/src/lerobot/envs/utils.py)
→ preprocessor
→ policy
→ postprocessor
→ env.step

CURRENT success eval path는 다음이다.

[`scripts/eval/eval_smolvla.py`](/home/zxro/arena/lerobot/scripts/eval/eval_smolvla.py)
→ direct metadata load
→ direct checkpoint config load
→ direct `predict_action()`
→ [`scripts/eval/openarm_smolvla_env.py`](/home/zxro/arena/lerobot/scripts/eval/openarm_smolvla_env.py)
→ env.step

비교:

| 항목 | OLD native path | CURRENT success path | 판정 |
|---|---|---|---|
| observation key | OpenArm native env contract | same contract | same |
| image preprocessing | `preprocess_observation()` + native eval plumbing | `predict_action()` helper path | different |
| state unit | degrees in old env | degrees in wrapper env | same |
| action unit | degree action into env | degree action into env | same |
| task prompt | env/vector task injection path | explicit `TASK` constant injection | different |
| action chunk | policy internal chunking | policy internal chunking | same |
| gripper inverse mapping | old env implements it | wrapper env implements same logic | mostly same |
| reset seed | native env path unclear from artifact | explicit `seed + episode` | different |
| initial robot pose | same joint defaults | same joint defaults | same |
| cube sampling | old env has fixed RNG object | wrapper env reseeds RNG on reset | different |
| episode length | old env default 1000 in `gym_openarm` | wrapper env default 1000 | same |
| success condition | same box/stationary logic | same logic | same |
| terminated/truncated | same | same | same |
| vector env/autoreset | native vector env | no vector env | different |
| physics/control decimation | 4 physics steps per action | same | same |

해석:

- env core dynamics 자체보다는 eval plumbing 구조가 더 크게 달라졌다.
- wrapper env와 `gym_openarm` env는 normal mode 기준으로 상당 부분 동일하다.
- 따라서 "env physics를 고쳐서 성공했다"는 증거는 약하다.

## 9. 성공에 결정적이었던 변경

### ROOT_CAUSE_FIX

1. local verified dataset로 train source를 바꾼 것

- 근거:
  - `v11_refalign`와 current success의 저장 config 차이 중 가장 본질적인 차이는 dataset source/root다.
  - `v11_refalign`는 hub dataset id, `root=None`
  - current success는 local degree/gripper-mapped dataset 절대 경로
- 판정: `CONFIRMED`

2. original fresh SmolVLA path에서 pretrained base path로 이동한 것

- 근거:
  - `v9`는 `pretrained_path=null`, `load_vlm_weights=false`
  - current success는 `pretrained_path=lerobot/smolvla_base`, `load_vlm_weights=true`
- 판정: `CONFIRMED`

### REQUIRED_FOR_REFERENCE_REPRODUCTION

1. [`scripts/train/train_smolvla.py`](/home/zxro/arena/lerobot/scripts/train/train_smolvla.py) 추가
2. [`scripts/eval/eval_smolvla.py`](/home/zxro/arena/lerobot/scripts/eval/eval_smolvla.py) 추가
3. [`scripts/eval/openarm_smolvla_env.py`](/home/zxro/arena/lerobot/scripts/eval/openarm_smolvla_env.py) 추가

판정 이유:

- current confirmed success path는 이 3개를 사용한다.
- 다만 이것만으로 failure 원인을 단정할 수는 없다.

### CONTRIBUTING_FIX

1. `batch_size 8 -> 4`
2. `use_amp false -> true`
3. `save_freq 20000 -> 2000`
4. `log_freq 200 -> 100`
5. wrapper-side dataset validation

판정 이유:

- `v9`와 current success 사이에는 실제 차이다.
- 그러나 `v11_refalign` 시점에는 이미 이 차이들 대부분이 current와 같았고 여전히 실패 checkpoint가 존재했다.
- 따라서 단독 root cause로는 약하다.

### DIAGNOSTIC_ONLY

1. `gym_openarm/openarm_env.py`의 `dataset_schedule` gripper override
2. [`scripts/openarm_rollout_diagnostic.py`](/home/zxro/arena/lerobot/scripts/openarm_rollout_diagnostic.py)
3. [`scripts/openarm_dataset_scale_diagnostic.py`](/home/zxro/arena/lerobot/scripts/openarm_dataset_scale_diagnostic.py)
4. eval wrapper의 progress/debug logs

판정 이유:

- failure 원인 탐지에는 매우 유용했다.
- current 정상 success 자체에는 필수 경로가 아니다.

### IRRELEVANT

1. immediate OLD snapshot 대비 [`src/lerobot/configs/train.py`](/home/zxro/arena/lerobot/src/lerobot/configs/train.py) 추가 변화
2. immediate OLD snapshot 대비 [`src/lerobot/scripts/lerobot_eval.py`](/home/zxro/arena/lerobot/src/lerobot/scripts/lerobot_eval.py) 추가 변화
3. immediate OLD snapshot 대비 [`src/lerobot/envs/configs.py`](/home/zxro/arena/lerobot/src/lerobot/envs/configs.py) 추가 변화
4. immediate OLD snapshot 대비 [`src/lerobot/envs/factory.py`](/home/zxro/arena/lerobot/src/lerobot/envs/factory.py) 추가 변화
5. immediate OLD snapshot 대비 [`src/lerobot/envs/utils.py`](/home/zxro/arena/lerobot/src/lerobot/envs/utils.py) 추가 변화
6. immediate OLD snapshot 대비 [`gym_openarm/openarm_env.py`](/home/zxro/arena/lerobot/gym_openarm/openarm_env.py) 추가 변화

판정 이유:

- `/tmp/lerobot_old_state`와 current가 동일하다.

### UNKNOWN

1. old native `lerobot-eval` exact command line
2. old native eval에서 vector env plumbing이 success failure에 얼마나 직접 기여했는지
3. hub dataset와 local dataset의 실제 row-level 내용 차이가 얼마나 큰지

## 10. 반증된 가설

1. "현재 성공은 `src/lerobot/envs/*`를 추가 수정해서 생겼다"

- 반증 근거:
  - immediate OLD snapshot과 CURRENT가 동일하다.

2. "현재 성공은 `chunk_size` 또는 `n_action_steps` 변경 때문"

- 반증 근거:
  - `v11_refalign`와 current success 모두 `50 / 50`

3. "현재 성공은 old failure checkpoint가 camera1/2/3 schema로 저장돼 있었기 때문"

- 반증 근거:
  - `v11_refalign` checkpoint는 이미 `top/wrist`, `state[8]`, `action[8]`

4. "현재 성공은 optimizer/scheduler 변경 때문"

- 반증 근거:
  - `v11_refalign`와 current success가 동일

## 11. 아직 불확실한 부분

1. `v11_refalign` hub dataset와 current local dataset가 metadata만 다른지, 실제 trajectory/value 내용까지 다른지는 이번 turn에서 row-level 전수 비교를 하지 않았다.
2. old native eval command artifact가 직접 남아 있지 않아서, 정확한 CLI는 `LIKELY` 수준으로만 복원된다.
3. `src/lerobot/configs/train.py`의 SmolVLA fallback patch가 historical runs 중 어느 시점부터 들어갔는지는 git commit으로 특정되지 않는다.

## MOST LIKELY ROOT CAUSE TOP 5

1. 학습 dataset source가 바뀌었다.
   `v11_refalign`는 hub dataset id를 썼고 current success는 local verified dataset root를 썼다. 저장 config에서 확인되는 가장 큰 실질 차이다.

2. original 실패 경로에서는 pretrained base를 쓰지 않았다.
   `v9` checkpoint는 `pretrained_path=null`, `load_vlm_weights=false`였다.

3. successful path는 wrapper가 train contract를 강제했다.
   local dataset validation, explicit `policy.path`, null feature override, fixed batch/save/log/amp가 자동으로 들어간다.

4. successful eval은 native `lerobot-eval`이 아니라 direct wrapper loop였다.
   이것이 rollout plumbing 차이를 제거했을 가능성은 크지만, training 결과를 바꾸지는 않으므로 1위 원인으로 올리긴 어렵다.

5. `batch_size`, AMP, save/log frequency 정렬은 기여 요인일 수 있다.
   다만 `v11_refalign` 시점에는 이미 current와 거의 같아졌으므로 단독 원인으로는 약하다.
