#!/usr/bin/env python

# Copyright 2025 HuggingFace Inc. team. All rights reserved.
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

"""
SmolVLA:

[Paper](https://huggingface.co/papers/2506.01844)

Designed by Hugging Face.

Install smolvla extra dependencies:
```bash
pip install -e ".[smolvla]"
```

Example of finetuning the smolvla pretrained model (`smolvla_base`):
```bash
lerobot-train \
--policy.path=lerobot/smolvla_base \
--dataset.repo_id=<USER>/svla_so100_task1_v3 \
--batch_size=64 \
--steps=200000
```

Example of finetuning a smolVLA. SmolVLA is composed of a pretrained VLM,
and an action expert.
```bash
lerobot-train \
--policy.type=smolvla \
--dataset.repo_id=<USER>/svla_so100_task1_v3 \
--batch_size=64 \
--steps=200000
```

Example of using the smolvla pretrained model outside LeRobot training framework:
```python
policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base")
```

"""

import math
from collections import deque
from typing import TypedDict

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn
from typing_extensions import Unpack

from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.rtc.modeling_rtc import RTCProcessor
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.smolvlm_with_expert import SmolVLMWithExpertModel
from lerobot.policies.utils import (
    populate_queues,
)
from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, OBS_STATE
from lerobot.utils.utils import get_safe_dtype

COUNTERFACTUAL_TRIPLET_KEY = "counterfactual_triplet_id"
COUNTERFACTUAL_COLOR_KEY = "counterfactual_color_id"


class ActionSelectKwargs(TypedDict, total=False):
    inference_delay: int | None
    prev_chunk_left_over: Tensor | None
    execution_horizon: int | None


def create_sinusoidal_pos_embedding(
    time: torch.tensor, dimension: int, min_period: float, max_period: float, device="cpu"
) -> Tensor:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")

    if time.ndim != 1:
        raise ValueError("The time tensor is expected to be of shape `(batch_size, )`.")

    dtype = get_safe_dtype(torch.float64, device.type)
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction

    # Compute the outer product
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    pos_emb = torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)
    return pos_emb


def make_att_2d_masks(pad_masks, att_masks):
    """Copied from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` int[B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: int32[B, N] mask that's 1 where previous tokens cannot depend on
        it and 0 where it shares the same attention mask as the previous token.
    """
    if att_masks.ndim != 2:
        raise ValueError(att_masks.ndim)
    if pad_masks.ndim != 2:
        raise ValueError(pad_masks.ndim)

    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    att_2d_masks = att_2d_masks & pad_2d_masks
    return att_2d_masks


def resize_with_pad(img, width, height, pad_value=-1):
    # assume no-op when width height fits already
    if img.ndim != 4:
        raise ValueError(f"(b,c,h,w) expected, but {img.shape}")

    cur_height, cur_width = img.shape[2:]

    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)
    resized_img = F.interpolate(
        img, size=(resized_height, resized_width), mode="bilinear", align_corners=False
    )

    pad_height = max(0, int(height - resized_height))
    pad_width = max(0, int(width - resized_width))

    # pad on left and top of image
    padded_img = F.pad(resized_img, (pad_width, 0, pad_height, 0), value=pad_value)
    return padded_img


def pad_vector(vector, new_dim):
    """Can be (batch_size x sequence_length x features_dimension)
    or (batch_size x features_dimension)
    """
    if vector.shape[-1] == new_dim:
        return vector
    shape = list(vector.shape)
    current_dim = shape[-1]
    shape[-1] = new_dim
    new_vector = torch.zeros(*shape, dtype=vector.dtype, device=vector.device)
    new_vector[..., :current_dim] = vector
    return new_vector


def normalize(x, min_val, max_val):
    return (x - min_val) / (max_val - min_val)


def unnormalize(x, min_val, max_val):
    return x * (max_val - min_val) + min_val


def safe_arcsin(value):
    # This ensures that the input stays within
    # [−1,1] to avoid invalid values for arcsin
    return torch.arcsin(torch.clamp(value, -1.0, 1.0))


def aloha_gripper_to_angular(value):
    # Aloha transforms the gripper positions into a linear space. The following code
    # reverses this transformation to be consistent with smolvla which is pretrained in
    # angular space.
    #
    # These values are coming from the Aloha code:
    # PUPPET_GRIPPER_POSITION_OPEN, PUPPET_GRIPPER_POSITION_CLOSED
    value = unnormalize(value, min_val=0.01844, max_val=0.05800)

    # This is the inverse of the angular to linear transformation inside the Interbotix code.
    def linear_to_radian(linear_position, arm_length, horn_radius):
        value = (horn_radius**2 + linear_position**2 - arm_length**2) / (2 * horn_radius * linear_position)
        return safe_arcsin(value)

    # The constants are taken from the Interbotix code.
    value = linear_to_radian(value, arm_length=0.036, horn_radius=0.022)

    # Normalize to [0, 1].
    # The values 0.4 and 1.5 were measured on an actual Trossen robot.
    return normalize(value, min_val=0.4, max_val=1.5)


def aloha_gripper_from_angular(value):
    # Convert from the gripper position used by smolvla to the gripper position that is used by Aloha.
    # Note that the units are still angular but the range is different.

    # The values 0.4 and 1.5 were measured on an actual Trossen robot.
    value = unnormalize(value, min_val=0.4, max_val=1.5)

    # These values are coming from the Aloha code:
    # PUPPET_GRIPPER_JOINT_OPEN, PUPPET_GRIPPER_JOINT_CLOSE
    return normalize(value, min_val=-0.6213, max_val=1.4910)


def aloha_gripper_from_angular_inv(value):
    # Directly inverts the gripper_from_angular function.
    value = unnormalize(value, min_val=-0.6213, max_val=1.4910)
    return normalize(value, min_val=0.4, max_val=1.5)


class SmolVLAPolicy(PreTrainedPolicy):
    """Wrapper class around VLAFlowMatching model to train and run inference within LeRobot."""

    config_class = SmolVLAConfig
    name = "smolvla"

    def __init__(
        self,
        config: SmolVLAConfig,
        **kwargs,
    ):
        """
        Args:
            config: Policy configuration class instance or None, in which case the default instantiation of
                    the configuration class is used.
        """

        super().__init__(config)
        config.validate_features()
        self.config = config
        self.init_rtc_processor()
        self.model = VLAFlowMatching(config, rtc_processor=self.rtc_processor)
        self.reset()

    def reset(self):
        """This should be called whenever the environment is reset."""
        self._queues = {
            ACTION: deque(maxlen=self.config.n_action_steps),
        }

    def init_rtc_processor(self):
        """Initialize RTC processor if RTC is enabled in config."""
        self.rtc_processor = None

        # Lets create processor if the config provided
        # If RTC is not enabled - we still can track the denoising data
        if self.config.rtc_config is not None:
            self.rtc_processor = RTCProcessor(self.config.rtc_config)

            # In case of calling init_rtc_processor after the model is created
            # We need to set the rtc_processor to the model
            # During the normal initialization process the model is not created yet
            model_value = getattr(self, "model", None)
            if model_value is not None:
                model_value.rtc_processor = self.rtc_processor

    def get_optim_params(self) -> dict:
        return self.parameters()

    def target_grounding_parameter_report(self) -> dict[str, dict[str, int | list[str]]]:
        """Return auditable parameter names/counts for preservation-sensitive blocks."""
        model = self.model
        vlm_model = model.vlm_with_expert.get_vlm_model()
        last_n = self.config.target_grounding_train_vlm_last_n_layers
        vlm_layers = list(vlm_model.text_model.layers)
        blocks = {
            "vision_encoder": [vlm_model.vision_model],
            "multimodal_connector": [vlm_model.connector],
            "vlm_frozen_prefix": vlm_layers[:-last_n] if last_n else vlm_layers,
            "vlm_trainable_tail": vlm_layers[-last_n:] if last_n else [],
            "language_embeddings": [vlm_model.text_model.embed_tokens],
            "state_projection": [model.state_proj],
            "action_expert": [
                model.vlm_with_expert.lm_expert,
                model.action_in_proj,
                model.action_time_mlp_in,
                model.action_time_mlp_out,
            ],
            "action_head": [model.action_out_proj],
            "target_grounding_head": [model.target_grounding_head],
            "target_conditioning_projection": [model.target_conditioning_proj],
        }
        parameter_names = {id(parameter): name for name, parameter in self.named_parameters()}
        report = {}
        for block_name, modules in blocks.items():
            parameters = []
            seen = set()
            for module in modules:
                if module is None:
                    continue
                for parameter in module.parameters():
                    if id(parameter) not in seen:
                        seen.add(id(parameter))
                        parameters.append(parameter)
            report[block_name] = {
                "total": sum(parameter.numel() for parameter in parameters),
                "trainable": sum(parameter.numel() for parameter in parameters if parameter.requires_grad),
                "trainable_parameter_names": [
                    parameter_names[id(parameter)] for parameter in parameters if parameter.requires_grad
                ],
            }
        return report

    def _get_action_chunk(
        self, batch: dict[str, Tensor], noise: Tensor | None = None, **kwargs: Unpack[ActionSelectKwargs]
    ) -> Tensor:
        # TODO: Check if this for loop is needed.
        # Context: In fact, self.queues contains only ACTION field, and in inference, we don't have action in the batch
        # In the case of offline inference, we have the action in the batch
        # that why without the k != ACTION check, it will raise an error because we are trying to stack
        # on an empty container.
        for k in batch:
            if k in self._queues and k != ACTION:
                batch[k] = torch.stack(list(self._queues[k]), dim=1)

        images, img_masks = self.prepare_images(batch)
        state = self.prepare_state(batch)
        lang_tokens = batch[f"{OBS_LANGUAGE_TOKENS}"]
        lang_masks = batch[f"{OBS_LANGUAGE_ATTENTION_MASK}"]

        actions = self.model.sample_actions(
            images,
            img_masks,
            lang_tokens,
            lang_masks,
            state,
            noise=noise,
            target_slot_label=batch.get("target_slot_label"),
            **kwargs,
        )

        # Unpad actions
        original_action_dim = self.config.action_feature.shape[0]
        actions = actions[:, :, :original_action_dim]

        if self.config.adapt_to_pi_aloha:
            actions = self._pi_aloha_encode_actions(actions)

        return actions

    @torch.no_grad()
    def predict_target_slot_logits(self, batch: dict[str, Tensor]) -> Tensor:
        """Predict A/B/C target-slot logits without running the action denoiser."""
        if not self.config.target_grounding_enabled:
            raise RuntimeError("Target grounding is not enabled for this policy")
        self.eval()
        images, img_masks = self.prepare_images(batch)
        state = self.prepare_state(batch)
        return self.model.predict_target_slot_logits(
            images,
            img_masks,
            batch[OBS_LANGUAGE_TOKENS],
            batch[OBS_LANGUAGE_ATTENTION_MASK],
            state,
        )

    def _prepare_batch(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        if self.config.adapt_to_pi_aloha:
            batch[OBS_STATE] = self._pi_aloha_decode_state(batch[OBS_STATE])

        return batch

    @torch.no_grad()
    def predict_action_chunk(
        self, batch: dict[str, Tensor], noise: Tensor | None = None, **kwargs: Unpack[ActionSelectKwargs]
    ) -> Tensor:
        self.eval()

        batch = self._prepare_batch(batch)
        self._queues = populate_queues(self._queues, batch, exclude_keys=[ACTION])

        actions = self._get_action_chunk(batch, noise, **kwargs)
        return actions

    @torch.no_grad()
    def select_action(
        self, batch: dict[str, Tensor], noise: Tensor | None = None, **kwargs: Unpack[ActionSelectKwargs]
    ) -> Tensor:
        """Select a single action given environment observations.

        This method wraps `select_actions` in order to return one action at a time for execution in the
        environment. It works by managing the actions in a queue and only calling `select_actions` when the
        queue is empty.
        """

        assert not self._rtc_enabled(), (
            "RTC is not supported for select_action, use it with predict_action_chunk"
        )

        self.eval()
        batch = self._prepare_batch(batch)
        self._queues = populate_queues(self._queues, batch, exclude_keys=[ACTION])

        if self._check_get_actions_condition():
            actions = self._get_action_chunk(batch, noise)

            # `self.predict_action_chunk` returns a (batch_size, n_action_steps, action_dim) tensor, but the queue
            # effectively has shape (n_action_steps, batch_size, *), hence the transpose.
            self._queues[ACTION].extend(actions.transpose(0, 1)[: self.config.n_action_steps])

        return self._queues[ACTION].popleft()

    def _check_get_actions_condition(self) -> bool:
        return len(self._queues[ACTION]) == 0

    def _rtc_enabled(self) -> bool:
        return self.config.rtc_config is not None and self.config.rtc_config.enabled

    def forward(
        self, batch: dict[str, Tensor], noise=None, time=None, reduction: str = "mean"
    ) -> dict[str, Tensor]:
        """Do a full training forward pass to compute the loss.

        Args:
            batch: Training batch containing observations and actions.
            noise: Optional noise tensor for flow matching.
            time: Optional time tensor for flow matching.
            reduction: How to reduce the loss. Options:
                - "mean": Return scalar mean loss (default, backward compatible)
                - "none": Return per-sample losses of shape (batch_size,) for RA-BC weighting
        """
        if self.config.adapt_to_pi_aloha:
            batch[OBS_STATE] = self._pi_aloha_decode_state(batch[OBS_STATE])
            batch[ACTION] = self._pi_aloha_encode_actions_inv(batch[ACTION])

        images, img_masks = self.prepare_images(batch)
        state = self.prepare_state(batch)
        lang_tokens = batch[f"{OBS_LANGUAGE_TOKENS}"]
        lang_masks = batch[f"{OBS_LANGUAGE_ATTENTION_MASK}"]
        actions = self.prepare_action(batch)
        actions_is_pad = batch.get("actions_id_pad")
        loss_dict = {}
        forward_out = self.model.forward(
            images,
            img_masks,
            lang_tokens,
            lang_masks,
            state,
            actions,
            noise,
            time,
            return_outputs=True,
            target_slot_label=batch.get("target_slot_label"),
        )
        losses = forward_out["losses"]
        pred_actions = forward_out["pred_actions"]
        loss_dict["losses_after_forward"] = losses.clone().mean().item()

        if actions_is_pad is not None:
            in_episode_bound = ~actions_is_pad
            losses = losses * in_episode_bound.unsqueeze(-1)
            loss_dict["losses_after_in_ep_bound"] = losses.clone().mean().item()

        original_action_dim = self.config.action_feature.shape[0]
        # Remove padding
        losses = losses[:, :, : self.config.max_action_dim]
        pred_actions = pred_actions[:, :, :original_action_dim]
        actions = actions[:, :, :original_action_dim]
        loss_dict["losses_after_rm_padding"] = losses.clone().mean().item()

        losses = self._apply_early_target_weight(batch, losses)
        loss_dict["losses_after_early_weight"] = losses.clone().mean().item()
        aux_loss, aux_metrics = self._compute_counterfactual_loss(batch, pred_actions, actions)
        loss_dict.update(aux_metrics)

        grounding_loss = losses.new_zeros(())
        if self.config.target_grounding_enabled:
            target_slot_label = batch.get("target_slot_label")
            if target_slot_label is None:
                raise ValueError("target_slot_label is required when target_grounding_enabled=true")
            target_slot_label = target_slot_label.to(device=losses.device, dtype=torch.long).reshape(-1)
            target_slot_logits = forward_out["target_slot_logits"]
            grounding_loss = F.cross_entropy(target_slot_logits, target_slot_label)
            predictions = target_slot_logits.argmax(dim=-1)
            loss_dict["target_grounding_loss"] = grounding_loss.item()
            loss_dict["target_grounding_accuracy"] = (predictions == target_slot_label).float().mean().item()
            loss_dict["target_grounding_slot_c_ratio"] = (predictions == 2).float().mean().item()
        else:
            loss_dict["target_grounding_loss"] = 0.0
            loss_dict["target_grounding_accuracy"] = 0.0
            loss_dict["target_grounding_slot_c_ratio"] = 0.0

        if reduction == "none":
            # Return per-sample losses (B,) by averaging over time and action dims
            per_sample_loss = losses.mean(dim=(1, 2))
            loss_dict["base_loss"] = per_sample_loss.mean().item()
            combined_aux_loss = aux_loss + self.config.target_grounding_lambda * grounding_loss
            loss_dict["loss"] = (
                self.config.target_grounding_action_loss_weight * per_sample_loss.mean() + combined_aux_loss
            ).item()
            loss_dict["_aux_loss"] = combined_aux_loss
            return per_sample_loss, loss_dict
        else:
            # Default: return scalar mean loss
            base_loss = losses.mean()
            loss = (
                self.config.target_grounding_action_loss_weight * base_loss
                + aux_loss
                + self.config.target_grounding_lambda * grounding_loss
            )
            loss_dict["base_loss"] = base_loss.item()
            loss_dict["loss"] = loss.item()
            return loss, loss_dict

    def _apply_early_target_weight(self, batch: dict[str, Tensor], losses: Tensor) -> Tensor:
        if self.config.early_target_weight <= 1.0:
            return losses
        frame_index = batch.get("frame_index")
        if frame_index is None:
            return losses
        if frame_index.ndim > 1:
            frame_index = frame_index[:, -1]
        mask = (frame_index >= self.config.early_target_start) & (frame_index <= self.config.early_target_end)
        sample_weights = torch.ones(losses.shape[0], device=losses.device, dtype=losses.dtype)
        sample_weights[mask] = self.config.early_target_weight
        return losses * sample_weights[:, None, None]

    def _compute_counterfactual_loss(
        self, batch: dict[str, Tensor], pred_actions: Tensor, gt_actions: Tensor
    ) -> tuple[Tensor, dict[str, float]]:
        zero = pred_actions.new_zeros(())
        metrics = {
            "counterfactual_loss": 0.0,
            "counterfactual_weighted_loss": 0.0,
            "counterfactual_triplets_in_batch": 0.0,
            "counterfactual_valid_pairs": 0.0,
            "counterfactual_excluded_pairs": 0.0,
            "counterfactual_pred_separation": 0.0,
            "counterfactual_gt_separation": 0.0,
            "counterfactual_sep_ratio": 0.0,
            "counterfactual_gt_mean_norm": 0.0,
            "counterfactual_weight_mean": 0.0,
            "counterfactual_weight_min": 0.0,
            "counterfactual_weight_max": 0.0,
        }
        if self.config.counterfactual_lambda <= 0.0:
            return zero, metrics

        details = self._get_counterfactual_loss_details(batch, pred_actions, gt_actions)
        if details is None:
            return zero, metrics

        metrics["counterfactual_loss"] = details["unweighted_loss"].item()
        metrics["counterfactual_weighted_loss"] = details["weighted_loss"].item()
        metrics["counterfactual_triplets_in_batch"] = float(details["triplet_count"])
        metrics["counterfactual_valid_pairs"] = float(details["valid_pair_count"])
        metrics["counterfactual_excluded_pairs"] = float(details["excluded_pair_count"])
        metrics["counterfactual_pred_separation"] = details["pred_separation"].item()
        metrics["counterfactual_gt_separation"] = details["gt_separation"].item()
        metrics["counterfactual_sep_ratio"] = (
            details["pred_separation"] / details["gt_separation"].clamp_min(1e-6)
        ).item()
        metrics["counterfactual_gt_mean_norm"] = details["mean_gt_norm"].item()
        metrics["counterfactual_weight_mean"] = details["weights"].mean().item()
        metrics["counterfactual_weight_min"] = details["weights"].min().item()
        metrics["counterfactual_weight_max"] = details["weights"].max().item()
        return self.config.counterfactual_lambda * details["weighted_loss"], metrics

    def _get_counterfactual_loss_details(
        self,
        batch: dict[str, Tensor],
        pred_actions: Tensor,
        gt_actions: Tensor,
        *,
        chunk_start: int | None = None,
        chunk_end: int | None = None,
        gt_weighting: bool | None = None,
        gt_weight_min: float | None = None,
        gt_weight_max: float | None = None,
        gt_weight_eps: float | None = None,
        gt_min_distance: float | None = None,
    ) -> dict[str, Tensor | int | slice] | None:
        triplet_ids = batch.get(COUNTERFACTUAL_TRIPLET_KEY)
        color_ids = batch.get(COUNTERFACTUAL_COLOR_KEY)
        if triplet_ids is None or color_ids is None:
            return None

        chunk_start = self.config.counterfactual_chunk_start if chunk_start is None else chunk_start
        chunk_end = self.config.counterfactual_chunk_end if chunk_end is None else chunk_end
        gt_weighting = self.config.counterfactual_gt_weighting if gt_weighting is None else gt_weighting
        gt_weight_min = self.config.counterfactual_gt_weight_min if gt_weight_min is None else gt_weight_min
        gt_weight_max = self.config.counterfactual_gt_weight_max if gt_weight_max is None else gt_weight_max
        gt_weight_eps = self.config.counterfactual_gt_weight_eps if gt_weight_eps is None else gt_weight_eps
        gt_min_distance = (
            self.config.counterfactual_gt_min_distance if gt_min_distance is None else gt_min_distance
        )

        chunk = slice(chunk_start, chunk_end)
        pred_chunk = pred_actions[:, chunk, :].reshape(pred_actions.shape[0], -1)
        gt_chunk = gt_actions[:, chunk, :].reshape(gt_actions.shape[0], -1)

        unique_triplets = torch.unique(triplet_ids)
        pair_losses: list[Tensor] = []
        pred_separations: list[Tensor] = []
        gt_separations: list[Tensor] = []
        gt_norms: list[Tensor] = []
        pair_triplet_ids: list[int] = []
        pair_names: list[str] = []
        triplet_count = 0
        for triplet_id in unique_triplets.tolist():
            if triplet_id < 0:
                continue
            members = triplet_ids == triplet_id
            member_indices = torch.nonzero(members, as_tuple=False).flatten()
            if member_indices.numel() < 3:
                continue

            by_color: dict[int, int] = {}
            for batch_index in member_indices.tolist():
                color_id = int(color_ids[batch_index].item())
                if color_id >= 0 and color_id not in by_color:
                    by_color[color_id] = batch_index
            if tuple(sorted(by_color)) != (0, 1, 2):
                continue

            ordered = [by_color[0], by_color[1], by_color[2]]
            triplet_count += 1
            for pair_name, (left, right) in zip(
                ("red_blue", "red_yellow", "blue_yellow"),
                ((0, 1), (0, 2), (1, 2)),
                strict=True,
            ):
                pred_delta = pred_chunk[ordered[left]] - pred_chunk[ordered[right]]
                gt_delta = (gt_chunk[ordered[left]] - gt_chunk[ordered[right]]).detach()
                pair_losses.append(F.mse_loss(pred_delta, gt_delta))
                pred_separations.append(torch.norm(pred_delta, p=2))
                gt_norm = torch.norm(gt_delta, p=2)
                gt_separations.append(gt_norm)
                gt_norms.append(gt_norm)
                pair_triplet_ids.append(int(triplet_id))
                pair_names.append(pair_name)

        if not pair_losses:
            return None

        pair_losses_t = torch.stack(pair_losses)
        pred_separations_t = torch.stack(pred_separations)
        gt_separations_t = torch.stack(gt_separations)
        gt_norms_t = torch.stack(gt_norms)

        valid_mask = gt_norms_t > gt_min_distance
        excluded_pair_count = int((~valid_mask).sum().item())
        if not torch.any(valid_mask):
            return None

        pair_losses_t = pair_losses_t[valid_mask]
        pred_separations_t = pred_separations_t[valid_mask]
        gt_separations_t = gt_separations_t[valid_mask]
        gt_norms_t = gt_norms_t[valid_mask]
        valid_mask_list = valid_mask.detach().cpu().tolist()
        filtered_pair_triplet_ids = [pair_triplet_ids[idx] for idx, keep in enumerate(valid_mask_list) if keep]
        filtered_pair_names = [pair_names[idx] for idx, keep in enumerate(valid_mask_list) if keep]

        mean_gt_norm = gt_norms_t.detach().mean().clamp_min(gt_weight_eps)
        if gt_weighting:
            raw_weights = gt_norms_t.detach() / mean_gt_norm
            weights = raw_weights.clamp(min=gt_weight_min, max=gt_weight_max)
        else:
            raw_weights = torch.ones_like(gt_norms_t)
            weights = torch.ones_like(gt_norms_t)

        weight_sum = weights.sum().clamp_min(gt_weight_eps)
        unweighted_loss = pair_losses_t.mean()
        weighted_loss = (weights * pair_losses_t).sum() / weight_sum

        return {
            "chunk": chunk,
            "triplet_count": triplet_count,
            "valid_pair_count": int(pair_losses_t.shape[0]),
            "excluded_pair_count": excluded_pair_count,
            "pair_losses": pair_losses_t,
            "gt_norms": gt_norms_t,
            "raw_weights": raw_weights,
            "weights": weights,
            "weighted_contributions": (weights * pair_losses_t) / weight_sum,
            "unweighted_loss": unweighted_loss,
            "weighted_loss": weighted_loss,
            "pred_separation": pred_separations_t.mean(),
            "gt_separation": gt_separations_t.mean(),
            "mean_gt_norm": mean_gt_norm,
            "pair_triplet_ids": filtered_pair_triplet_ids,
            "pair_names": filtered_pair_names,
        }

    def prepare_images(self, batch):
        """Apply SmolVLA preprocessing to the images, like resizing to 224x224 and padding to keep aspect ratio, and
        convert pixel range from [0.0, 1.0] to [-1.0, 1.0] as requested by SigLIP.
        """
        images = []
        img_masks = []
        present_img_keys = [key for key in self.config.image_features if key in batch]
        missing_img_keys = [key for key in self.config.image_features if key not in batch]

        if len(present_img_keys) == 0:
            raise ValueError(
                f"All image features are missing from the batch. At least one expected. (batch: {batch.keys()}) (image_features:{self.config.image_features})"
            )
        # Preprocess image features present in the batch
        for key in present_img_keys:
            img = batch[key][:, -1, :, :, :] if batch[key].ndim == 5 else batch[key]
            if self.config.resize_imgs_with_padding is not None:
                img = resize_with_pad(img, *self.config.resize_imgs_with_padding, pad_value=0)

            # Normalize from range [0,1] to [-1,1] as expacted by siglip
            img = img * 2.0 - 1.0

            bsize = img.shape[0]
            device = img.device
            if f"{key}_padding_mask" in batch:
                mask = batch[f"{key}_padding_mask"].bool()
            else:
                mask = torch.ones(bsize, dtype=torch.bool, device=device)
            images.append(img)
            img_masks.append(mask)

        # Create image features not present in the batch
        # as fully 0 padded images.
        for num_empty_cameras in range(len(missing_img_keys)):
            if num_empty_cameras >= self.config.empty_cameras:
                break
            img = torch.ones_like(img) * -1
            mask = torch.zeros_like(mask)
            images.append(img)
            img_masks.append(mask)
        return images, img_masks

    def _pi_aloha_decode_state(self, state):
        # Flip the joints.
        for motor_idx in [1, 2, 8, 9]:
            state[:, motor_idx] *= -1
        # Reverse the gripper transformation that is being applied by the Aloha runtime.
        for motor_idx in [6, 13]:
            state[:, motor_idx] = aloha_gripper_to_angular(state[:, motor_idx])
        return state

    def _pi_aloha_encode_actions(self, actions):
        # Flip the joints.
        for motor_idx in [1, 2, 8, 9]:
            actions[:, :, motor_idx] *= -1
        # Reverse the gripper transformation that is being applied by the Aloha runtime.
        for motor_idx in [6, 13]:
            actions[:, :, motor_idx] = aloha_gripper_from_angular(actions[:, :, motor_idx])
        return actions

    def _pi_aloha_encode_actions_inv(self, actions):
        # Flip the joints again.
        for motor_idx in [1, 2, 8, 9]:
            actions[:, :, motor_idx] *= -1
        # Reverse the gripper transformation that is being applied by the Aloha runtime.
        for motor_idx in [6, 13]:
            actions[:, :, motor_idx] = aloha_gripper_from_angular_inv(actions[:, :, motor_idx])
        return actions

    def prepare_state(self, batch):
        """Pad state"""
        state = batch[OBS_STATE][:, -1, :] if batch[OBS_STATE].ndim > 2 else batch[OBS_STATE]
        state = pad_vector(state, self.config.max_state_dim)
        return state

    def prepare_action(self, batch):
        """Pad action"""
        actions = pad_vector(batch[ACTION], self.config.max_action_dim)
        return actions

    def _get_default_peft_targets(self) -> dict[str, any]:
        """Return default PEFT target modules for SmolVLA fine-tuning."""
        common_projections = (
            "state_proj|action_in_proj|action_out_proj|action_time_mlp_in|action_time_mlp_out"
        )
        target_modules = rf"(model\.vlm_with_expert\.lm_expert\..*\.(q|v)_proj|model\.({common_projections}))"
        return {
            "target_modules": target_modules,
            "modules_to_save": [],
        }

    def _validate_peft_config(self, peft_config) -> None:
        """Validate PEFT configuration for SmolVLA."""
        super()._validate_peft_config(peft_config)
        if not self.config.load_vlm_weights:
            import logging

            logging.warning(
                "Training SmolVLA from scratch using PEFT. This is unlikely to yield good results. "
                "Set `load_vlm_weights=True` to fine-tune the existing policy."
            )


def pad_tensor(tensor, max_len, pad_value=0):
    """
    Efficiently pads a tensor along sequence dimension to match max_len.

    Args:
        tensor (torch.Tensor): Shape (B, L, ...) or (B, L).
        max_len (int): Fixed sequence length.
        pad_value (int/float): Value for padding.

    Returns:
        torch.Tensor: Shape (B, max_len, ...) or (B, max_len).
    """
    b, d = tensor.shape[:2]

    # Create a padded tensor of max_len and copy the existing values
    padded_tensor = torch.full(
        (b, max_len, *tensor.shape[2:]), pad_value, dtype=tensor.dtype, device=tensor.device
    )
    padded_tensor[:, :d] = tensor  # Efficient in-place copy

    return padded_tensor


class VLAFlowMatching(nn.Module):
    """
    SmolVLA

    [Paper]()

    Designed by Hugging Face.
    ┌──────────────────────────────┐
    │                 actions      │
    │                    ▲         │
    │ ┌─────────┐      ┌─|────┐    │
    │ |         │────► │      │    │
    │ |         │ kv   │      │    │
    │ |         │────► │Action│    │
    │ |   VLM   │cache │Expert│    |
    │ │         │────► |      │    │
    │ │         │      │      │    │
    │ └▲──▲───▲─┘      └───▲──┘    |
    │  │  |   |            │       |
    │  |  |   |          noise     │
    │  │  │ state                  │
    │  │ language tokens           │
    │  image(s)                    │
    └──────────────────────────────┘
    """

    def __init__(self, config: SmolVLAConfig, rtc_processor: RTCProcessor | None = None):
        super().__init__()
        self.config = config

        self.vlm_with_expert = SmolVLMWithExpertModel(
            model_id=self.config.vlm_model_name,
            freeze_vision_encoder=self.config.freeze_vision_encoder,
            train_expert_only=self.config.train_expert_only,
            load_vlm_weights=self.config.load_vlm_weights,
            attention_mode=self.config.attention_mode,
            num_expert_layers=self.config.num_expert_layers,
            num_vlm_layers=self.config.num_vlm_layers,
            self_attn_every_n_layers=self.config.self_attn_every_n_layers,
            expert_width_multiplier=self.config.expert_width_multiplier,
            device=self.config.device if self.config.device is not None else "auto",
        )
        self.state_proj = nn.Linear(
            self.config.max_state_dim, self.vlm_with_expert.config.text_config.hidden_size
        )
        self.action_in_proj = nn.Linear(self.config.max_action_dim, self.vlm_with_expert.expert_hidden_size)
        self.action_out_proj = nn.Linear(self.vlm_with_expert.expert_hidden_size, self.config.max_action_dim)

        hidden_size = self.vlm_with_expert.config.text_config.hidden_size
        self.target_grounding_head = None
        if self.config.target_grounding_enabled:
            self.target_grounding_head = nn.Sequential(
                nn.LayerNorm(hidden_size),
                nn.Linear(hidden_size, self.config.target_grounding_num_classes),
            )
        self.target_conditioning_proj = None
        if self.config.target_conditioning_enabled:
            self.target_conditioning_proj = nn.Linear(
                self.config.target_grounding_num_classes, hidden_size, bias=False
            )
            nn.init.zeros_(self.target_conditioning_proj.weight)

        self.action_time_mlp_in = nn.Linear(
            self.vlm_with_expert.expert_hidden_size * 2, self.vlm_with_expert.expert_hidden_size
        )
        self.action_time_mlp_out = nn.Linear(
            self.vlm_with_expert.expert_hidden_size, self.vlm_with_expert.expert_hidden_size
        )

        self.set_requires_grad()
        self.fake_image_token = self.vlm_with_expert.processor.tokenizer.fake_image_token_id
        self.global_image_token = self.vlm_with_expert.processor.tokenizer.global_image_token_id
        self.global_image_start_token = torch.tensor(
            [self.fake_image_token, self.global_image_token], dtype=torch.long
        )

        self.add_image_special_tokens = self.config.add_image_special_tokens
        self.image_end_token = torch.tensor([self.fake_image_token], dtype=torch.long)
        self.prefix_length = self.config.prefix_length
        self.rtc_processor = rtc_processor

        # Compile model if requested
        if config.compile_model:
            torch.set_float32_matmul_precision("high")
            self.sample_actions = torch.compile(self.sample_actions, mode=config.compile_mode)
            self.forward = torch.compile(self.forward, mode=config.compile_mode)

    def _rtc_enabled(self):
        return self.config.rtc_config is not None and self.config.rtc_config.enabled

    def _normalize_suffix_output(self, suffix_out: Tensor, expected_len: int) -> Tensor:
        if suffix_out.ndim == 2:
            suffix_out = suffix_out[:, None, :].expand(-1, expected_len, -1)
        elif suffix_out.ndim != 3:
            raise ValueError(f"Expected suffix_out to be 2D or 3D, got shape={tuple(suffix_out.shape)}")
        if suffix_out.shape[1] > expected_len:
            suffix_out = suffix_out[:, -expected_len:]
        elif suffix_out.shape[1] < expected_len:
            if suffix_out.shape[1] == 1:
                suffix_out = suffix_out.expand(-1, expected_len, -1)
            else:
                raise ValueError(
                    f"suffix_out sequence length {suffix_out.shape[1]} does not match expected_len={expected_len}"
                )
        return suffix_out

    def set_requires_grad(self):
        for params in self.state_proj.parameters():
            params.requires_grad = self.config.train_state_proj
        if self.config.target_conditioning_enabled and self.config.target_grounding_preserve_manipulation:
            for param in self.parameters():
                param.requires_grad = False
            if self.target_conditioning_proj is None:
                raise RuntimeError("target_conditioning_enabled but projection is not initialized")
            for param in self.target_conditioning_proj.parameters():
                param.requires_grad = True
            if not self.config.target_conditioning_freeze_grounding_head and self.target_grounding_head is not None:
                for param in self.target_grounding_head.parameters():
                    param.requires_grad = True
            return
        if not self.config.target_grounding_enabled or not self.config.target_grounding_preserve_manipulation:
            return

        # Start from a fully frozen policy and open only the deliberately small
        # representation-side set plus the newly initialized classifier.
        for param in self.parameters():
            param.requires_grad = False
        if self.target_grounding_head is not None:
            for param in self.target_grounding_head.parameters():
                param.requires_grad = True
        if self.config.target_grounding_train_connector:
            for param in self.vlm_with_expert.get_vlm_model().connector.parameters():
                param.requires_grad = True
        train_last_n = self.config.target_grounding_train_vlm_last_n_layers
        if train_last_n:
            layers = self.vlm_with_expert.get_vlm_model().text_model.layers
            if train_last_n > len(layers):
                raise ValueError(
                    f"Cannot train last {train_last_n} VLM layers; model only has {len(layers)} layers"
                )
            for layer in layers[-train_last_n:]:
                for param in layer.parameters():
                    param.requires_grad = True

    def sample_noise(self, shape, device):
        noise = torch.normal(
            mean=0.0,
            std=1.0,
            size=shape,
            dtype=torch.float32,
            device=device,
        )
        return noise

    @staticmethod
    def _state_token_indices_from_prefix_masks(
        prefix_pad_masks: Tensor, prefix_att_masks: Tensor
    ) -> Tensor:
        # embed_prefix marks the appended state segment with att_mask=1, while
        # image/text tokens and any fixed-length padding use att_mask=0.
        state_mask = prefix_pad_masks.bool() & prefix_att_masks.bool()
        if not bool(state_mask.any(dim=1).all()):
            raise ValueError("No valid state token found in one or more prefix rows")
        physical_indices = torch.arange(state_mask.shape[1], device=state_mask.device)
        return physical_indices.unsqueeze(0).masked_fill(~state_mask, -1).max(dim=1).values

    def _target_slot_logits_from_prefix(
        self, prefix_out: Tensor, prefix_pad_masks: Tensor, prefix_att_masks: Tensor
    ) -> Tensor:
        if self.target_grounding_head is None:
            raise RuntimeError("Target grounding head is not initialized")
        state_indices = self._state_token_indices_from_prefix_masks(prefix_pad_masks, prefix_att_masks)
        batch_indices = torch.arange(prefix_out.shape[0], device=prefix_out.device)
        fused_state_token = prefix_out[batch_indices, state_indices].to(dtype=torch.float32)
        return self.target_grounding_head(fused_state_token)

    def predict_target_slot_logits(self, images, img_masks, lang_tokens, lang_masks, state) -> Tensor:
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, state=state
        )
        attention_mask = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        (prefix_out, _), _ = self.vlm_with_expert.forward(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=False,
            fill_kv_cache=True,
        )
        return self._target_slot_logits_from_prefix(prefix_out, prefix_pad_masks, prefix_att_masks)

    def sample_time(self, bsize, device):
        beta_dist = torch.distributions.Beta(concentration1=1.5, concentration0=1.0)
        time_beta = beta_dist.sample((bsize,)).to(device=device, dtype=torch.float32)
        time = time_beta * 0.999 + 0.001
        return time

    def _make_target_conditioning(
        self, images, img_masks, lang_tokens, lang_masks, state, target_slot_label
    ) -> Tensor | None:
        if not self.config.target_conditioning_enabled:
            return None
        if self.target_conditioning_proj is None:
            raise RuntimeError("Target conditioning projection is not initialized")
        if self.config.target_conditioning_mode == "oracle":
            if target_slot_label is None:
                raise ValueError("target_slot_label is required for oracle target conditioning")
            target_values = F.one_hot(
                target_slot_label.long().reshape(-1), num_classes=self.config.target_grounding_num_classes
            ).float()
        else:
            # The successful 750-step grounding head and its source representation
            # stay frozen; only the downstream conditioning projection is learned.
            with torch.no_grad():
                logits = self.predict_target_slot_logits(images, img_masks, lang_tokens, lang_masks, state)
                target_values = logits.softmax(dim=-1)
        return self.config.target_conditioning_scale * self.target_conditioning_proj(target_values)

    def embed_prefix(
        self,
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        state: torch.Tensor = None,
        target_conditioning: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Embed images with SigLIP and language tokens with embedding layer to prepare
        for SmolVLM transformer processing.
        """
        embs = []
        pad_masks = []
        att_masks = []
        for _img_idx, (
            img,
            img_mask,
        ) in enumerate(zip(images, img_masks, strict=False)):
            if self.add_image_special_tokens:
                image_start_token = (
                    self.vlm_with_expert.embed_language_tokens(
                        self.global_image_start_token.to(device=self.vlm_with_expert.vlm.device)
                    )
                    .unsqueeze(0)
                    .expand(img.shape[0], -1, -1)
                )
                image_start_mask = torch.ones_like(
                    image_start_token[:, :, 0], dtype=torch.bool, device=image_start_token.device
                )
                att_masks += [0] * (image_start_mask.shape[-1])
                embs.append(image_start_token)
                pad_masks.append(image_start_mask)

            img_emb = self.vlm_with_expert.embed_image(img)
            img_emb = img_emb

            # Normalize image embeddings
            img_emb_dim = img_emb.shape[-1]
            img_emb = img_emb * torch.tensor(img_emb_dim**0.5, dtype=img_emb.dtype, device=img_emb.device)

            bsize, num_img_embs = img_emb.shape[:2]
            img_mask = img_mask[:, None].expand(bsize, num_img_embs)

            embs.append(img_emb)
            pad_masks.append(img_mask)

            att_masks += [0] * (num_img_embs)
            if self.add_image_special_tokens:
                image_end_token = (
                    self.vlm_with_expert.embed_language_tokens(
                        self.image_end_token.to(device=self.vlm_with_expert.vlm.device)
                    )
                    .unsqueeze(0)
                    .expand(img.shape[0], -1, -1)
                )
                image_end_mask = torch.ones_like(
                    image_end_token[:, :, 0], dtype=torch.bool, device=image_end_token.device
                )
                embs.append(image_end_token)
                pad_masks.append(image_end_mask)
                att_masks += [0] * (image_end_mask.shape[1])
        lang_emb = self.vlm_with_expert.embed_language_tokens(lang_tokens)
        # Normalize language embeddings
        lang_emb_dim = lang_emb.shape[-1]
        lang_emb = lang_emb * math.sqrt(lang_emb_dim)

        embs.append(lang_emb)
        pad_masks.append(lang_masks)

        num_lang_embs = lang_emb.shape[1]
        att_masks += [0] * num_lang_embs

        state_emb = self.state_proj(state)
        if target_conditioning is not None:
            state_emb = state_emb + target_conditioning.to(dtype=state_emb.dtype)
        state_emb = state_emb[:, None, :] if state_emb.ndim == 2 else state_emb
        embs.append(state_emb)
        bsize = state_emb.shape[0]
        device = state_emb.device

        states_seq_len = state_emb.shape[1]
        state_mask = torch.ones(bsize, states_seq_len, dtype=torch.bool, device=device)
        pad_masks.append(state_mask)

        # Set attention masks so that image and language inputs do not attend to state or actions
        att_masks += [1] * (states_seq_len)
        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)
        att_masks = att_masks[None, :]

        seq_len = pad_masks.shape[1]
        if seq_len < self.prefix_length:
            embs = pad_tensor(embs, self.prefix_length, pad_value=0)
            pad_masks = pad_tensor(pad_masks, self.prefix_length, pad_value=0)
            att_masks = pad_tensor(att_masks, self.prefix_length, pad_value=0)

        att_masks = att_masks.expand(bsize, -1)

        return embs, pad_masks, att_masks

    def embed_suffix(self, noisy_actions, timestep):
        """Embed state, noisy_actions, timestep to prepare for Expert Gemma processing."""
        embs = []
        pad_masks = []
        att_masks = []

        # Fuse timestep + action information using an MLP
        action_emb = self.action_in_proj(noisy_actions)
        device = action_emb.device
        bsize = action_emb.shape[0]
        dtype = action_emb.dtype
        # Embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = create_sinusoidal_pos_embedding(
            timestep,
            self.vlm_with_expert.expert_hidden_size,
            self.config.min_period,
            self.config.max_period,
            device=device,
        )
        time_emb = time_emb.type(dtype=dtype)

        time_emb = time_emb[:, None, :].expand_as(action_emb)
        action_time_emb = torch.cat([action_emb, time_emb], dim=2)

        action_time_emb = self.action_time_mlp_in(action_time_emb)
        action_time_emb = F.silu(action_time_emb)  # swish == silu
        action_time_emb = self.action_time_mlp_out(action_time_emb)

        # Add to input tokens
        embs.append(action_time_emb)

        bsize, action_time_dim = action_time_emb.shape[:2]
        action_time_mask = torch.ones(bsize, action_time_dim, dtype=torch.bool, device=device)
        pad_masks.append(action_time_mask)

        # Match the actual suffix sequence length instead of assuming config.chunk_size.
        # Some datasets provide fewer action targets than the configured chunk size.
        att_masks += [1] * action_time_dim
        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=embs.dtype, device=embs.device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))
        return embs, pad_masks, att_masks

    def forward(
        self,
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        state,
        actions,
        noise=None,
        time=None,
        return_outputs: bool = False,
        target_slot_label: Tensor | None = None,
    ) -> Tensor | dict[str, Tensor]:
        """Do a full training forward pass and compute the loss (batch_size x num_steps x num_motors)"""
        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)

        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)

        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions
        target_conditioning = self._make_target_conditioning(
            images, img_masks, lang_tokens, lang_masks, state, target_slot_label
        )
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images,
            img_masks,
            lang_tokens,
            lang_masks,
            state=state,
            target_conditioning=target_conditioning,
        )
        suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(x_t, time)

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)

        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1
        (prefix_out, suffix_out), _ = self.vlm_with_expert.forward(
            attention_mask=att_2d_masks,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, suffix_embs],
            use_cache=False,
            fill_kv_cache=False,
        )
        expected_len = actions.shape[1]
        suffix_out = self._normalize_suffix_output(suffix_out, expected_len)
        # Original openpi code, upcast attention output
        suffix_out = suffix_out.to(dtype=torch.float32)
        v_t = self.action_out_proj(suffix_out)
        losses = F.mse_loss(u_t, v_t, reduction="none")
        if return_outputs:
            pred_actions = noise - v_t
            outputs = {"losses": losses, "pred_actions": pred_actions}
            if self.target_grounding_head is not None:
                outputs["target_slot_logits"] = self._target_slot_logits_from_prefix(
                    prefix_out, prefix_pad_masks, prefix_att_masks
                )
            return outputs
        return losses

    def sample_actions(
        self,
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        state,
        noise=None,
        target_slot_label: Tensor | None = None,
        **kwargs: Unpack[ActionSelectKwargs],
    ) -> Tensor:
        """Do a full inference forward and compute the action (batch_size x num_steps x num_motors)"""
        bsize = state.shape[0]
        device = state.device

        if noise is None:
            actions_shape = (bsize, self.config.chunk_size, self.config.max_action_dim)
            noise = self.sample_noise(actions_shape, device)

        target_conditioning = self._make_target_conditioning(
            images, img_masks, lang_tokens, lang_masks, state, target_slot_label
        )
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images,
            img_masks,
            lang_tokens,
            lang_masks,
            state=state,
            target_conditioning=target_conditioning,
        )
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        # Compute image and language key value cache
        _, past_key_values = self.vlm_with_expert.forward(
            attention_mask=prefix_att_2d_masks,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=self.config.use_cache,
            fill_kv_cache=True,
        )
        num_steps = self.config.num_steps
        dt = -1.0 / num_steps

        x_t = noise
        for step in range(num_steps):
            time = 1.0 + step * dt
            time_tensor = torch.tensor(time, dtype=torch.float32, device=device).expand(bsize)

            def denoise_step_partial_call(input_x_t, current_timestep=time_tensor):
                return self.denoise_step(
                    x_t=input_x_t,
                    prefix_pad_masks=prefix_pad_masks,
                    past_key_values=past_key_values,
                    timestep=current_timestep,
                )

            if self._rtc_enabled():
                inference_delay = kwargs.get("inference_delay")
                prev_chunk_left_over = kwargs.get("prev_chunk_left_over")
                execution_horizon = kwargs.get("execution_horizon")

                v_t = self.rtc_processor.denoise_step(
                    x_t=x_t,
                    prev_chunk_left_over=prev_chunk_left_over,
                    inference_delay=inference_delay,
                    time=time,
                    original_denoise_step_partial=denoise_step_partial_call,
                    execution_horizon=execution_horizon,
                )
            else:
                v_t = denoise_step_partial_call(x_t)

            x_t = x_t + dt * v_t

            if self.rtc_processor is not None and self.rtc_processor.is_debug_enabled():
                self.rtc_processor.track(time=time, x_t=x_t, v_t=v_t)

        return x_t

    def denoise_step(
        self,
        prefix_pad_masks,
        past_key_values,
        x_t,
        timestep,
    ):
        """Apply one denoising step of the noise `x_t` at a given timestep."""
        suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(x_t, timestep)

        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]
        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)

        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)

        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)
        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

        outputs_embeds, _ = self.vlm_with_expert.forward(
            attention_mask=full_att_2d_masks,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=self.config.use_cache,
            fill_kv_cache=False,
        )
        suffix_out = outputs_embeds[1]
        expected_len = x_t.shape[1]
        suffix_out = self._normalize_suffix_output(suffix_out, expected_len)
        suffix_out = suffix_out.to(dtype=torch.float32)
        v_t = self.action_out_proj(suffix_out)
        return v_t
