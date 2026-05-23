from __future__ import annotations

from typing import Literal

import torch

from art import dev
from art.loss import Loss, compute_probs_corr

from .types import DispatchedPackedTensors


def validate_context_parallel_loss_config(
    experimental_config: dev.TrainConfig,
) -> None:
    if experimental_config.get("importance_sampling_level", "token") != "token":
        raise NotImplementedError(
            "CP dispatched loss currently supports token-level importance sampling "
            "only. Add group-id dispatch before enabling sequence-level variants."
        )
    if experimental_config.get("truncated_importance_sampling", None) is not None:
        raise NotImplementedError(
            "CP dispatched loss currently does not dispatch original_logprobs, so "
            "truncated_importance_sampling is disabled for CP training."
        )


def loss_fn_dispatched(
    inputs: DispatchedPackedTensors,
    *,
    new_logprobs: torch.Tensor,
    ref_logprobs: torch.Tensor | None,
    entropies: torch.Tensor | None,
    experimental_config: dev.TrainConfig,
    reduction: Literal["mean", "sum"] = "mean",
) -> Loss:
    assistant_mask = inputs.assistant_mask.to(new_logprobs.dtype)
    old_logprobs = inputs.old_logprobs
    advantages = inputs.advantages
    weights = inputs.weights

    probs_corr = compute_probs_corr(old_logprobs, new_logprobs)
    old_logprobs = torch.where(
        torch.isnan(old_logprobs),
        new_logprobs.detach(),
        old_logprobs,
    )

    logprob_diff = new_logprobs - old_logprobs
    prob_ratio = torch.exp(logprob_diff)
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
        advantages = advantages - tau * logprob_diff.detach()

    kl_policy_ref: torch.Tensor | None = None
    kl_penalty_coef = experimental_config.get("kl_penalty_coef", 0.0)
    if kl_penalty_coef > 0 and ref_logprobs is not None:
        kl_per_token = (new_logprobs - ref_logprobs).detach() * assistant_mask
        avg_kl = kl_per_token.sum() / (assistant_mask.sum() + 1e-6)
        advantages = (
            advantages + kl_penalty_coef * (avg_kl - kl_per_token) * assistant_mask
        )
        kl_policy_ref = avg_kl

    if ppo:
        policy_loss = -torch.min(
            prob_ratio * advantages,
            torch.clip(prob_ratio, 1 - epsilon, 1 + epsilon_high) * advantages,
        )
    else:
        policy_loss = -(
            torch.clip(prob_ratio.detach(), 1 - epsilon, 1 + epsilon_high)
            * advantages
            * new_logprobs
        )

    if ref_logprobs is not None:
        kl_logprob_diff = ref_logprobs - new_logprobs
        kl_div = torch.expm1(kl_logprob_diff) - kl_logprob_diff
    else:
        kl_div = torch.zeros_like(policy_loss)

    policy_loss = policy_loss * weights * assistant_mask
    kl_div = kl_div * weights * assistant_mask
    denominator = assistant_mask.sum() + 1e-6 if reduction == "mean" else 1.0
    reduced_policy_loss = policy_loss.sum() / denominator
    kl = kl_div.sum() / denominator

    if entropies is not None:
        entropy = (entropies * weights * assistant_mask).sum() / denominator
    else:
        entropy = None

    return Loss(
        reduction=reduction,
        policy_loss=reduced_policy_loss,
        kl=kl,
        entropy=entropy,
        policy_loss_sum=policy_loss.sum(),
        probs_corr=probs_corr,
        kl_policy_ref=kl_policy_ref,
    )
