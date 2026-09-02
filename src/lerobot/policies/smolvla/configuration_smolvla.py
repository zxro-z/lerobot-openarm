# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass, field
from pathlib import Path

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.optim.optimizers import AdamWConfig
from lerobot.optim.schedulers import (
    CosineDecayWithWarmupSchedulerConfig,
)
from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.utils.constants import OBS_IMAGES


@PreTrainedConfig.register_subclass("smolvla")
@dataclass
class SmolVLAConfig(PreTrainedConfig):
    # Input / output structure.
    n_obs_steps: int = 1
    chunk_size: int = 50
    n_action_steps: int = 50

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        }
    )

    # Shorter state and action vectors will be padded
    max_state_dim: int = 32
    max_action_dim: int = 32

    # Image preprocessing
    resize_imgs_with_padding: tuple[int, int] = (512, 512)

    # Add empty images. Used by smolvla_aloha_sim which adds the empty
    # left and right wrist cameras in addition to the top camera.
    empty_cameras: int = 0

    # Converts the joint and gripper values from the standard Aloha space to
    # the space used by the pi internal runtime which was used to train the base model.
    adapt_to_pi_aloha: bool = False

    # Converts joint dimensions to deltas with respect to the current state before passing to the model.
    # Gripper dimensions will remain in absolute values.
    use_delta_joint_actions_aloha: bool = False

    # Tokenizer
    tokenizer_max_length: int = 48

    # Decoding
    num_steps: int = 10

    # Attention utils
    use_cache: bool = True

    # Finetuning settings
    freeze_vision_encoder: bool = True
    train_expert_only: bool = True
    train_state_proj: bool = True

    # Training presets
    optimizer_lr: float = 1e-4
    optimizer_betas: tuple[float, float] = (0.9, 0.95)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-10
    optimizer_grad_clip_norm: float = 10

    scheduler_warmup_steps: int = 1_000
    scheduler_decay_steps: int = 30_000
    scheduler_decay_lr: float = 2.5e-6

    vlm_model_name: str = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"  # Select the VLM backbone.
    load_vlm_weights: bool = False  # Set to False in case of training the expert from scratch. True when init from pretrained SmolVLA weights

    add_image_special_tokens: bool = False  # Whether to use special image tokens around image features.

    attention_mode: str = "cross_attn"

    prefix_length: int = -1

    pad_language_to: str = "longest"  # "max_length"

    num_expert_layers: int = -1  # Less or equal to 0 is the default where the action expert has the same number of layers of VLM. Otherwise the expert have less layers.
    num_vlm_layers: int = 16  # Number of layers used in the VLM (first num_vlm_layers layers)
    self_attn_every_n_layers: int = 2  # Interleave SA layers each self_attn_every_n_layers
    expert_width_multiplier: float = 0.75  # The action expert hidden size (wrt to the VLM)

    min_period: float = 4e-3  # sensitivity range for the timestep used in sine-cosine positional encoding
    max_period: float = 4.0

    # Real-Time Chunking (RTC) configuration
    rtc_config: RTCConfig | None = None

    compile_model: bool = False  # Whether to use torch.compile for model optimization
    compile_mode: str = "max-autotune"  # Torch compile mode

    early_target_weight: float = 1.0
    early_target_start: int = 36
    early_target_end: int = 100

    counterfactual_lambda: float = 0.0
    counterfactual_chunk_start: int = 16
    counterfactual_chunk_end: int = 50
    counterfactual_triplets_per_batch: int = 0
    counterfactual_triplet_manifest: str | Path | None = None
    counterfactual_gt_weighting: bool = True
    counterfactual_gt_weight_min: float = 0.25
    counterfactual_gt_weight_max: float = 4.0
    counterfactual_gt_weight_eps: float = 1e-6
    counterfactual_gt_min_distance: float = 1e-3

    # Explicit instruction -> visual target-slot grounding.
    target_grounding_enabled: bool = False
    target_grounding_lambda: float = 0.0
    target_grounding_num_classes: int = 3
    target_grounding_action_loss_weight: float = 1.0
    target_grounding_manifest: str | Path | None = None
    # Preservation preset: keep the manipulation expert/head and vision tower fixed,
    # while adapting only the connector, a minimal text-side tail, and the new head.
    target_grounding_preserve_manipulation: bool = True
    target_grounding_train_connector: bool = True
    target_grounding_train_vlm_last_n_layers: int = 1
    # Minimal frozen-policy target conditioning through the existing state prefix token.
    target_conditioning_enabled: bool = False
    target_conditioning_mode: str = "predicted"  # predicted | oracle
    target_conditioning_scale: float = 1.0
    target_conditioning_freeze_grounding_head: bool = True

    def __post_init__(self):
        super().__post_init__()

        """Input validation (not exhaustive)."""
        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"The chunk size is the upper bound for the number of action steps per model invocation. Got "
                f"{self.n_action_steps} for `n_action_steps` and {self.chunk_size} for `chunk_size`."
            )
        if self.use_delta_joint_actions_aloha:
            raise NotImplementedError(
                "`use_delta_joint_actions_aloha` is used by smolvla for aloha real models. It is not ported yet in LeRobot."
            )
        if self.early_target_weight < 1.0:
            raise ValueError(f"early_target_weight must be >= 1.0, got {self.early_target_weight}")
        if self.early_target_end < self.early_target_start:
            raise ValueError(
                f"early_target_end must be >= early_target_start, got {self.early_target_start}>{self.early_target_end}"
            )
        if self.counterfactual_lambda < 0.0:
            raise ValueError(f"counterfactual_lambda must be >= 0.0, got {self.counterfactual_lambda}")
        if self.counterfactual_triplets_per_batch < 0:
            raise ValueError(
                f"counterfactual_triplets_per_batch must be >= 0, got {self.counterfactual_triplets_per_batch}"
            )
        if self.counterfactual_chunk_start < 0:
            raise ValueError(
                f"counterfactual_chunk_start must be >= 0, got {self.counterfactual_chunk_start}"
            )
        if self.counterfactual_chunk_end <= self.counterfactual_chunk_start:
            raise ValueError(
                "counterfactual_chunk_end must be greater than counterfactual_chunk_start, "
                f"got {self.counterfactual_chunk_start}, {self.counterfactual_chunk_end}"
            )
        if self.counterfactual_chunk_end > self.chunk_size:
            raise ValueError(
                f"counterfactual_chunk_end must be <= chunk_size={self.chunk_size}, got {self.counterfactual_chunk_end}"
            )
        if self.counterfactual_gt_weight_min <= 0.0:
            raise ValueError(
                f"counterfactual_gt_weight_min must be > 0.0, got {self.counterfactual_gt_weight_min}"
            )
        if self.counterfactual_gt_weight_max < self.counterfactual_gt_weight_min:
            raise ValueError(
                "counterfactual_gt_weight_max must be >= counterfactual_gt_weight_min, "
                f"got {self.counterfactual_gt_weight_max} < {self.counterfactual_gt_weight_min}"
            )
        if self.counterfactual_gt_weight_eps <= 0.0:
            raise ValueError(
                f"counterfactual_gt_weight_eps must be > 0.0, got {self.counterfactual_gt_weight_eps}"
            )
        if self.counterfactual_gt_min_distance < 0.0:
            raise ValueError(
                "counterfactual_gt_min_distance must be >= 0.0, "
                f"got {self.counterfactual_gt_min_distance}"
            )
        if self.target_grounding_lambda < 0.0:
            raise ValueError(f"target_grounding_lambda must be >= 0, got {self.target_grounding_lambda}")
        if self.target_grounding_num_classes != 3:
            raise ValueError(
                "The fixed-slot target grounding task requires exactly 3 classes "
                f"(A/B/C), got {self.target_grounding_num_classes}"
            )
        if self.target_grounding_action_loss_weight < 0.0:
            raise ValueError(
                "target_grounding_action_loss_weight must be >= 0, "
                f"got {self.target_grounding_action_loss_weight}"
            )
        if self.target_grounding_train_vlm_last_n_layers < 0:
            raise ValueError(
                "target_grounding_train_vlm_last_n_layers must be >= 0, "
                f"got {self.target_grounding_train_vlm_last_n_layers}"
            )
        if self.target_conditioning_mode not in {"predicted", "oracle"}:
            raise ValueError(
                "target_conditioning_mode must be 'predicted' or 'oracle', "
                f"got {self.target_conditioning_mode!r}"
            )
        if self.target_conditioning_scale < 0:
            raise ValueError(f"target_conditioning_scale must be >= 0, got {self.target_conditioning_scale}")

    def validate_features(self) -> None:
        for i in range(self.empty_cameras):
            key = f"{OBS_IMAGES}.empty_camera_{i}"
            empty_camera = PolicyFeature(
                type=FeatureType.VISUAL,
                shape=(3, 480, 640),
            )
            self.input_features[key] = empty_camera

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
            grad_clip_norm=self.optimizer_grad_clip_norm,
        )

    def get_scheduler_preset(self):
        return CosineDecayWithWarmupSchedulerConfig(
            peak_lr=self.optimizer_lr,
            decay_lr=self.scheduler_decay_lr,
            num_warmup_steps=self.scheduler_warmup_steps,
            num_decay_steps=self.scheduler_decay_steps,
        )

    @property
    def observation_delta_indices(self) -> list:
        return [0]

    @property
    def action_delta_indices(self) -> list:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None
