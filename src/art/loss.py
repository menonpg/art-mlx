from collections.abc import Mapping
from typing import Any, Literal, cast

from pydantic import BaseModel, ConfigDict
import torch

from art.utils.group_aggregate import group_aggregate

from . import dev


class Loss(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    reduction: Literal["mean", "sum"]
    policy_loss: torch.Tensor
    entropy: torch.Tensor | None
    policy_loss_sum: torch.Tensor
    probs_corr: torch.Tensor
    kl_policy_ref: torch.Tensor | None = None


class LossInputs(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    assistant_mask: torch.Tensor
    old_logprobs: torch.Tensor
    advantages: torch.Tensor
    weights: torch.Tensor
    group_ids: torch.Tensor | None = None
    original_logprobs: torch.Tensor | None = None
    distributed_group: Any | None = None
    entropies_are_aligned: bool = False

    def group_mean(self, values: torch.Tensor, by: torch.Tensor) -> torch.Tensor:
        if self.distributed_group is None:
            return group_aggregate(values, by=by, reduce="mean")
        return _distributed_group_mean(values, by=by, group=self.distributed_group)

    def masked_mean(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        numerator = values.sum()
        denominator = mask.sum()
        if self.distributed_group is not None:
            torch.distributed.all_reduce(  # ty: ignore[possibly-missing-attribute]
                numerator,
                group=self.distributed_group,
            )
            torch.distributed.all_reduce(  # ty: ignore[possibly-missing-attribute]
                denominator,
                group=self.distributed_group,
            )
        return numerator / (denominator + 1e-18)

    def denominator(self, mask: torch.Tensor, reduction: Literal["mean", "sum"]):
        if reduction == "sum":
            return 1.0
        denominator = mask.sum()
        if self.distributed_group is not None:
            torch.distributed.all_reduce(  # ty: ignore[possibly-missing-attribute]
                denominator,
                group=self.distributed_group,
            )
        return denominator + 1e-18

    def aligned_entropies(self, entropies: torch.Tensor | None) -> torch.Tensor | None:
        if entropies is None:
            return None
        if self.entropies_are_aligned:
            return entropies
        return shift_tensor(entropies, 0.0)


def _tensor_attr(obj: object, name: str) -> torch.Tensor | None:
    value = getattr(obj, name, None)
    return value if isinstance(value, torch.Tensor) else None


def _mapping_tensor(inputs: object, key: str) -> torch.Tensor | None:
    if not isinstance(inputs, Mapping) or key not in inputs:
        return None
    value = cast(Mapping[str, object], inputs)[key]
    return value if isinstance(value, torch.Tensor) else None


def loss_inputs(inputs: object) -> LossInputs:
    aligned_old_logprobs = _tensor_attr(inputs, "old_logprobs")
    aligned_assistant_mask = _tensor_attr(inputs, "assistant_mask")
    aligned_advantages = _tensor_attr(inputs, "advantages")
    aligned_weights = _tensor_attr(inputs, "weights")
    if (
        aligned_old_logprobs is not None
        and aligned_assistant_mask is not None
        and aligned_advantages is not None
        and aligned_weights is not None
    ):
        return LossInputs(
            assistant_mask=aligned_assistant_mask,
            old_logprobs=aligned_old_logprobs,
            advantages=aligned_advantages,
            weights=aligned_weights,
            group_ids=_tensor_attr(inputs, "group_ids"),
            original_logprobs=_tensor_attr(inputs, "original_logprobs"),
            distributed_group=getattr(inputs, "loss_all_reduce_group", None),
            entropies_are_aligned=True,
        )

    logprobs = _mapping_tensor(inputs, "logprobs")
    advantages = _mapping_tensor(inputs, "advantages")
    assistant_mask = _mapping_tensor(inputs, "assistant_mask")
    weights = _mapping_tensor(inputs, "weights")
    if (
        logprobs is None
        or advantages is None
        or assistant_mask is None
        or weights is None
    ):
        raise TypeError("loss inputs must provide packed or aligned loss tensors")

    group_ids = _mapping_tensor(inputs, "group_ids")
    original_logprobs = _mapping_tensor(inputs, "original_logprobs")
    return LossInputs(
        assistant_mask=shift_tensor(assistant_mask, False),
        old_logprobs=shift_tensor(logprobs, float("nan")),
        advantages=shift_tensor(advantages, 0.0),
        weights=shift_tensor(weights, 0.0),
        group_ids=None if group_ids is None else shift_tensor(group_ids, 0),
        original_logprobs=None
        if original_logprobs is None
        else shift_tensor(original_logprobs, 0.0),
    )


def _distributed_group_mean(
    values: torch.Tensor,
    *,
    by: torch.Tensor,
    group: Any,
) -> torch.Tensor:
    flat_values = values.reshape(-1)
    flat_by = by.reshape(-1).to(dtype=torch.float32)
    unique_local = torch.unique(flat_by, sorted=True)
    if int(unique_local.numel()) == 0:
        return values.clone()

    world_size = torch.distributed.get_world_size(group)  # ty: ignore[possibly-missing-attribute]
    local_count = torch.tensor(
        [unique_local.numel()],
        device=values.device,
        dtype=torch.long,
    )
    gathered_counts = [torch.empty_like(local_count) for _ in range(world_size)]
    torch.distributed.all_gather(  # ty: ignore[possibly-missing-attribute]
        gathered_counts,
        local_count,
        group=group,
    )
    max_count = int(torch.stack(gathered_counts).max().item())
    padded_ids = torch.zeros(max_count, device=values.device, dtype=torch.float32)
    padded_ids[: unique_local.numel()] = unique_local
    gathered_ids = [torch.empty_like(padded_ids) for _ in range(world_size)]
    torch.distributed.all_gather(  # ty: ignore[possibly-missing-attribute]
        gathered_ids,
        padded_ids,
        group=group,
    )
    global_ids = torch.unique(
        torch.cat(
            [
                gathered[: int(count.item())]
                for gathered, count in zip(gathered_ids, gathered_counts, strict=True)
            ]
        ),
        sorted=True,
    )
    group_indices = torch.searchsorted(global_ids, flat_by)
    sums = torch.zeros_like(global_ids)
    counts = torch.zeros_like(global_ids)
    sums.scatter_add_(0, group_indices, flat_values.to(dtype=sums.dtype))
    counts.scatter_add_(
        0, group_indices, torch.ones_like(flat_values, dtype=sums.dtype)
    )
    torch.distributed.all_reduce(sums, group=group)  # ty: ignore[possibly-missing-attribute]
    torch.distributed.all_reduce(counts, group=group)  # ty: ignore[possibly-missing-attribute]
    return (sums / (counts + 1e-8)).gather(0, group_indices).reshape_as(values)


def compute_probs_corr(
    old_logprobs: torch.Tensor,
    new_logprobs: torch.Tensor,
) -> torch.Tensor:
    old_logprobs_mask = ~torch.isnan(old_logprobs)
    old_probs = torch.exp(old_logprobs[old_logprobs_mask])
    new_probs = torch.exp(new_logprobs[old_logprobs_mask])
    if old_probs.numel() < 2:
        return new_logprobs.new_zeros(())
    old_std = old_probs.std(unbiased=False)
    new_std = new_probs.std(unbiased=False)
    if (
        not torch.isfinite(old_std).item()
        or not torch.isfinite(new_std).item()
        or old_std.item() == 0.0
        or new_std.item() == 0.0
    ):
        return new_logprobs.new_zeros(())
    return torch.corrcoef(torch.stack([old_probs, new_probs]))[0, 1]


def loss_fn(
    inputs: object,
    new_logprobs: torch.Tensor,
    ref_logprobs: torch.Tensor | None,
    entropies: torch.Tensor | None,
    experimental_config: dev.TrainConfig,
    reduction: Literal["mean", "sum"] = "mean",
) -> Loss:
    aligned_inputs = loss_inputs(inputs)
    old_logprobs = aligned_inputs.old_logprobs
    advantages = aligned_inputs.advantages
    assistant_mask = aligned_inputs.assistant_mask.to(new_logprobs.dtype)
    weights = aligned_inputs.weights
    probs_corr = compute_probs_corr(old_logprobs, new_logprobs)
    # Assume missing old logprobs were sampled under the current policy
    old_logprobs = torch.where(
        torch.isnan(old_logprobs),
        new_logprobs.detach(),
        old_logprobs,
    )
    logprob_diff = new_logprobs - old_logprobs
    importance_sampling_level = experimental_config.get(
        "importance_sampling_level", "token"
    )
    prob_ratio = torch.exp(logprob_diff)
    if importance_sampling_level != "token":
        if aligned_inputs.group_ids is None:
            raise ValueError(
                "group_ids are required for non-token importance sampling."
            )
        sequence_prob_ratio = torch.exp(
            aligned_inputs.group_mean(
                logprob_diff,
                by=aligned_inputs.group_ids * assistant_mask,
            )
        )
        if importance_sampling_level == "sequence":
            prob_ratio = sequence_prob_ratio
        elif importance_sampling_level == "average":
            prob_ratio = (prob_ratio + sequence_prob_ratio) / 2
        elif importance_sampling_level == "geometric_average":
            prob_ratio = (prob_ratio**0.5) * (sequence_prob_ratio**0.5)
    ppo = experimental_config.get("ppo", False)
    if ppo:
        epsilon_default = 0.2
        epsilon_high_default = None
    else:
        epsilon_default = 1.0
        epsilon_high_default = 4.0
    epsilon = experimental_config.get("epsilon", epsilon_default)
    epsilon_high = experimental_config.get("epsilon_high", epsilon_high_default)
    if epsilon_high is None:
        epsilon_high = epsilon
    if max_negative_advantage_importance_sampling_weight := experimental_config.get(
        "max_negative_advantage_importance_sampling_weight", None
    ):
        prob_ratio = torch.clamp(
            prob_ratio, max=max_negative_advantage_importance_sampling_weight
        )
    if experimental_config.get("mask_prob_ratio", False):
        prob_ratio = torch.where(
            (prob_ratio > 1 - epsilon) & (prob_ratio < 1 + epsilon_high),
            prob_ratio,
            0.0,
        )
    if tau := experimental_config.get("kimi_k2_tau", None):
        advantages -= tau * logprob_diff.detach()
    kl_policy_ref: torch.Tensor | None = None
    kl_penalty_coef = experimental_config.get("kl_penalty_coef", 0.0)
    if kl_penalty_coef > 0 and ref_logprobs is not None:
        kl_per_token = (new_logprobs - ref_logprobs).detach() * assistant_mask
        avg_kl = aligned_inputs.masked_mean(kl_per_token, assistant_mask)
        kl_penalty = kl_penalty_coef * (avg_kl - kl_per_token) * assistant_mask
        advantages = advantages + kl_penalty
        kl_policy_ref = avg_kl
    if ppo:
        policy_loss = -torch.min(
            prob_ratio * advantages,
            torch.clip(prob_ratio, 1 - epsilon, 1 + epsilon_high) * advantages,
        )
    else:
        # Modified REINFORCE or Clipped IS-weight Policy Optimization (CISPO)
        policy_loss = -(
            torch.clip(prob_ratio.detach(), 1 - epsilon, 1 + epsilon_high)
            * advantages
            * new_logprobs
        )
    if upper_bound := experimental_config.get("truncated_importance_sampling", None):
        if aligned_inputs.original_logprobs is not None:
            original_logprobs = aligned_inputs.original_logprobs
            original_logprobs = torch.where(
                torch.isnan(original_logprobs),
                new_logprobs.detach(),
                original_logprobs,
            )
            logprob_diff = old_logprobs - original_logprobs
            prob_ratio = torch.exp(logprob_diff)
        policy_loss *= torch.clamp(prob_ratio, max=upper_bound).detach()
    policy_loss = policy_loss * weights * assistant_mask
    denominator = aligned_inputs.denominator(assistant_mask, reduction)
    reduced_policy_loss = policy_loss.sum() / denominator
    # Compute reduced entropy for the current step.
    aligned_entropies = aligned_inputs.aligned_entropies(entropies)
    if aligned_entropies is not None:
        entropy = (aligned_entropies * weights * assistant_mask).sum() / denominator
    else:
        entropy = None
    return Loss(
        reduction=reduction,
        policy_loss=reduced_policy_loss,
        entropy=entropy,
        policy_loss_sum=policy_loss.sum(),
        probs_corr=probs_corr,
        kl_policy_ref=kl_policy_ref,
    )


def shift_tensor(tensor: torch.Tensor, pad: int | float | bool) -> torch.Tensor:
    return torch.nn.functional.pad(tensor[:, 1:], (0, 1), value=pad)
