#!/usr/bin/env python3
"""Launch SmolVLA fine-tuning with this project's degree-based dataset."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from pprint import pformat

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_ROOT = (
    PROJECT_ROOT
    / "src/lerobot/datasets/openarm_three_color_transit_tilt_50"
)
EXPECTED_TASKS = {
    "Pick up the red cube and place it in the storage box.",
    "Pick up the blue cube and place it in the storage box.",
    "Pick up the yellow cube and place it in the storage box.",
}
EXPECTED_VECTOR_SHAPE = [8]
EXPECTED_CAMERAS = {"observation.images.top", "observation.images.wrist"}
EXPECTED_TASK_COUNTS = {"red": 50, "blue": 50, "yellow": 50}


def resolve_video_backend(requested_backend: str | None) -> str:
    if requested_backend is not None:
        return requested_backend

    try:
        import torchcodec  # noqa: F401

        return "torchcodec"
    except Exception:
        return "pyav"


def validate_dataset(root: Path) -> None:
    """Fail early when the local dataset does not match this training contract."""
    info_path = root / "meta/info.json"
    tasks_path = root / "meta/tasks.parquet"
    if not info_path.is_file() or not tasks_path.is_file():
        raise FileNotFoundError(f"Incomplete LeRobot v3 metadata under: {root}")

    with info_path.open() as file:
        info = json.load(file)
    features = info.get("features", {})
    for key in ("observation.state", "action"):
        if features.get(key, {}).get("shape") != EXPECTED_VECTOR_SHAPE:
            raise ValueError(f"{key} must have shape {EXPECTED_VECTOR_SHAPE}")
    missing_cameras = EXPECTED_CAMERAS - features.keys()
    if missing_cameras:
        raise ValueError(f"Dataset is missing cameras: {sorted(missing_cameras)}")

    tasks = pd.read_parquet(tasks_path)
    task_names = set(tasks.index.tolist())
    if task_names != EXPECTED_TASKS:
        raise ValueError(
            f"Expected task set {sorted(EXPECTED_TASKS)!r}; dataset contains {sorted(task_names)!r}"
        )

    episodes_dir = root / "meta" / "episodes"
    if not episodes_dir.is_dir():
        raise FileNotFoundError(f"Missing episode metadata under: {episodes_dir}")
    episodes = pd.read_parquet(sorted(episodes_dir.glob("*/*.parquet")))
    counts = {"red": 0, "blue": 0, "yellow": 0}
    for tasks_list in episodes["tasks"]:
        if len(tasks_list) != 1:
            raise ValueError(f"Expected one task string per episode, got {tasks_list!r}")
        task = tasks_list[0]
        if "red cube" in task:
            counts["red"] += 1
        elif "blue cube" in task:
            counts["blue"] += 1
        elif "yellow cube" in task:
            counts["yellow"] += 1
        else:
            raise ValueError(f"Unexpected episode task string: {task!r}")
    if counts != EXPECTED_TASK_COUNTS:
        raise ValueError(f"Expected episode counts {EXPECTED_TASK_COUNTS}, got {counts}")


def inspect_training_contract(dataset_repo_id: str, dataset_root: Path) -> dict:
    import lerobot.policies  # noqa: F401
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
    from lerobot.datasets.utils import dataset_to_policy_features
    from lerobot.policies.factory import make_policy

    meta = LeRobotDatasetMetadata(dataset_repo_id, root=dataset_root)
    dataset_features = meta.info["features"]
    policy_features = dataset_to_policy_features(dataset_features)

    policy_cfg = PreTrainedConfig.from_pretrained(
        "lerobot/smolvla_base",
        cli_overrides=[
            "--input_features=null",
            "--output_features=null",
            "--load_vlm_weights=true",
            "--train_expert_only=false",
            "--freeze_vision_encoder=true",
            "--train_state_proj=true",
            "--attention_mode=cross_attn",
        ],
    )
    policy_cfg.pretrained_path = Path("lerobot/smolvla_base")
    policy_cfg.input_features = None
    policy_cfg.output_features = None
    policy_cfg.device = "cpu"
    policy = make_policy(policy_cfg, ds_meta=meta)

    final_output_features = {
        key: {"type": str(ft.type), "shape": ft.shape}
        for key, ft in policy_features.items()
        if str(ft.type).endswith("ACTION")
    }
    final_input_features = {
        key: {"type": str(ft.type), "shape": ft.shape}
        for key, ft in policy_features.items()
        if key not in final_output_features
    }

    trainable_groups = []
    frozen_groups = []
    group_patterns = [
        ("language_embedding", ("model.vlm_with_expert.vlm.model.text_model.embed_tokens",)),
        ("vlm_transformer", ("model.vlm_with_expert.vlm.model.text_model.layers",)),
        ("vision_connector", ("model.vlm_with_expert.vlm.model.connector",)),
        ("vision_encoder", ("model.vlm_with_expert.vlm.model.vision_model",)),
        ("expert", ("model.vlm_with_expert.lm_expert",)),
        ("state_proj", ("model.state_proj",)),
        ("action_head", ("model.action_in_proj", "model.action_out_proj", "model.action_time_mlp_in", "model.action_time_mlp_out")),
    ]
    for label, prefixes in group_patterns:
        params = [p for name, p in policy.named_parameters() if any(name.startswith(prefix) for prefix in prefixes)]
        if not params:
            continue
        if any(p.requires_grad for p in params):
            trainable_groups.append(label)
        else:
            frozen_groups.append(label)

    return {
        "dataset_features": dataset_features,
        "final_policy_input_features": final_input_features,
        "final_policy_output_features": final_output_features,
        "pretrained_policy": "lerobot/smolvla_base",
        "vlm_weights_loaded": bool(getattr(policy_cfg, "load_vlm_weights", False)),
        "train_expert_only": bool(getattr(policy_cfg, "train_expert_only", True)),
        "freeze_vision_encoder": bool(getattr(policy_cfg, "freeze_vision_encoder", False)),
        "trainable_groups": trainable_groups,
        "frozen_groups": frozen_groups,
    }


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Fine-tune SmolVLA on the OpenArm dual-RealSense dataset.",
    )
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument(
        "--dataset-repo-id",
        default="local/openarm_three_color_transit_tilt_50",
    )
    parser.add_argument("--dataset-source", choices=["local", "hub"], default="local")
    parser.add_argument("--policy-path", default="lerobot/smolvla_base")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs/train/openarm_three_color_transit_tilt_50",
    )
    parser.add_argument("--steps", type=int, default=20_000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--save-freq", type=int, default=2_000)
    parser.add_argument("--log-freq", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--video-backend", choices=["torchcodec", "pyav"], default=None)
    parser.add_argument("--use-amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--grasp-positive-manifest", type=Path, default=None)
    parser.add_argument("--grasp-positive-weight", type=float, default=1.0)
    parser.add_argument("--target-commitment-manifest", type=Path, default=None)
    parser.add_argument("--target-commitment-weight", type=float, default=1.0)
    parser.add_argument("--freeze-vision-encoder", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--train-expert-only", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--train-state-proj", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--attention-mode", default="cross_attn")
    parser.add_argument("--load-vlm-weights", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_known_args()


def main() -> None:
    args, extra = parse_args()
    if args.grasp_positive_weight > 1.0 and args.grasp_positive_manifest is None:
        raise ValueError("--grasp-positive-weight > 1 requires --grasp-positive-manifest")
    if args.target_commitment_weight > 1.0 and args.target_commitment_manifest is None:
        raise ValueError("--target-commitment-weight > 1 requires --target-commitment-manifest")
    if args.dataset_source == "local":
        validate_dataset(args.dataset_root)
    if args.dataset_source == "local" and not args.dry_run:
        report = inspect_training_contract(args.dataset_repo_id, args.dataset_root.resolve())
        print(f"[PRETRAINED POLICY]\n{report['pretrained_policy']}", flush=True)
        print("[DATASET FEATURES]", flush=True)
        print(
            pformat(
                {
                    "observation.state": report["dataset_features"]["observation.state"],
                    "observation.images.top": report["dataset_features"]["observation.images.top"],
                    "observation.images.wrist": report["dataset_features"]["observation.images.wrist"],
                    "action": report["dataset_features"]["action"],
                }
            ),
            flush=True,
        )
        print("[FINAL POLICY INPUT FEATURES]", flush=True)
        print(pformat(report["final_policy_input_features"]), flush=True)
        print("[FINAL POLICY OUTPUT FEATURES]", flush=True)
        print(pformat(report["final_policy_output_features"]), flush=True)
        print(f"[VLM WEIGHTS LOADED]\n{str(report['vlm_weights_loaded']).lower()}", flush=True)
        print(f"[TRAIN EXPERT ONLY]\n{str(report['train_expert_only']).lower()}", flush=True)
        print(f"[FREEZE VISION ENCODER]\n{str(report['freeze_vision_encoder']).lower()}", flush=True)
        print("[TRAINABLE PARAMS]", flush=True)
        for name in report["trainable_groups"]:
            print(name, flush=True)
        print("[FROZEN]", flush=True)
        for name in report["frozen_groups"]:
            print(name, flush=True)
    video_backend = resolve_video_backend(args.video_backend)

    train_exe = shutil.which("lerobot-train")
    if train_exe is None:
        candidate = Path(sys.executable).resolve().parent / "lerobot-train"
        train_exe = str(candidate) if candidate.is_file() else "lerobot-train"

    command = [
        train_exe,
        f"--policy.path={args.policy_path}",
        "--policy.input_features=null",
        "--policy.output_features=null",
        f"--dataset.repo_id={args.dataset_repo_id}",
        f"--dataset.video_backend={video_backend}",
        f"--batch_size={args.batch_size}",
        f"--num_workers={args.num_workers}",
        f"--steps={args.steps}",
        f"--save_freq={args.save_freq}",
        f"--log_freq={args.log_freq}",
        f"--seed={args.seed}",
        f"--output_dir={args.output_dir.resolve()}",
        "--job_name=openarm_three_color_transit_tilt_50_smolvla",
        f"--policy.device={args.device}",
        f"--policy.use_amp={str(args.use_amp).lower()}",
        f"--policy.load_vlm_weights={str(args.load_vlm_weights).lower()}",
        f"--policy.freeze_vision_encoder={str(args.freeze_vision_encoder).lower()}",
        f"--policy.train_expert_only={str(args.train_expert_only).lower()}",
        f"--policy.train_state_proj={str(args.train_state_proj).lower()}",
        f"--policy.attention_mode={args.attention_mode}",
        "--policy.push_to_hub=false",
        f"--wandb.enable={str(args.wandb).lower()}",
        *extra,
    ]
    if args.dataset_source == "local":
        command.insert(5, f"--dataset.root={args.dataset_root.resolve()}")
    if args.grasp_positive_manifest is not None:
        command.append(f"--grasp_positive_manifest={args.grasp_positive_manifest.resolve()}")
        command.append(f"--grasp_positive_weight={args.grasp_positive_weight}")
    if args.target_commitment_manifest is not None:
        command.append(f"--target_commitment_manifest={args.target_commitment_manifest.resolve()}")
        command.append(f"--target_commitment_weight={args.target_commitment_weight}")
    print("[EFFECTIVE CONTRACT]", flush=True)
    print(
        pformat(
            {
                "policy_path": args.policy_path,
                "dataset_source": args.dataset_source,
                "dataset_root": str(args.dataset_root.resolve()) if args.dataset_source == "local" else None,
                "dataset_repo_id": args.dataset_repo_id,
                "output_dir": str(args.output_dir.resolve()),
                "train_expert_only": args.train_expert_only,
                "freeze_vision_encoder": args.freeze_vision_encoder,
                "train_state_proj": args.train_state_proj,
                "load_vlm_weights": args.load_vlm_weights,
                "attention_mode": args.attention_mode,
                "grasp_positive_manifest": (
                    str(args.grasp_positive_manifest.resolve()) if args.grasp_positive_manifest is not None else None
                ),
                "grasp_positive_weight": args.grasp_positive_weight,
                "target_commitment_manifest": (
                    str(args.target_commitment_manifest.resolve())
                    if args.target_commitment_manifest is not None
                    else None
                ),
                "target_commitment_weight": args.target_commitment_weight,
                "batch_size": args.batch_size,
                "steps": args.steps,
                "seed": args.seed,
                "device": args.device,
            }
        ),
        flush=True,
    )
    print("[TRAIN]", " ".join(command), flush=True)
    if args.dry_run:
        return
    os.execvp(command[0], command)


if __name__ == "__main__":
    main()
