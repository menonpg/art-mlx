from collections import defaultdict
from collections.abc import Iterable
from typing import Any, Literal, cast

from megatron.core import parallel_state as ps
from megatron.core.distributed.finalize_model_grads import finalize_model_grads
from megatron.core.transformer.module import MegatronModule
import torch
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors

GradSyncDomain = Literal["tp_default", "expert_tp"]
GradSyncOp = Literal["none", "sum", "avg"]

TP_DEFAULT_GRAD_SYNC_DOMAIN: GradSyncDomain = "tp_default"
EXPERT_TP_GRAD_SYNC_DOMAIN: GradSyncDomain = "expert_tp"
GRAD_SYNC_OP_NONE: GradSyncOp = "none"
GRAD_SYNC_OP_SUM: GradSyncOp = "sum"
GRAD_SYNC_OP_AVG: GradSyncOp = "avg"
VALID_DOMAINS = (TP_DEFAULT_GRAD_SYNC_DOMAIN, EXPERT_TP_GRAD_SYNC_DOMAIN)
VALID_SYNC_OPS = (GRAD_SYNC_OP_NONE, GRAD_SYNC_OP_SUM, GRAD_SYNC_OP_AVG)


def _iter_named_trainable_parameters(
    model: list[MegatronModule],
) -> Iterable[tuple[str, torch.nn.Parameter]]:
    seen: set[int] = set()
    for chunk_index, model_chunk in enumerate(model):
        for name, param in model_chunk.named_parameters():
            if not param.requires_grad:
                continue
            param_id = id(param)
            if param_id in seen:
                continue
            seen.add(param_id)
            yield f"chunk{chunk_index}.{name}", param


def _resolve_domain_group(
    domain: GradSyncDomain,
) -> Any | None:
    if domain == TP_DEFAULT_GRAD_SYNC_DOMAIN:
        group = ps.get_tensor_model_parallel_group(check_initialized=False)
        if group is None or group.size() <= 1:
            return None
        return group
    if domain != EXPERT_TP_GRAD_SYNC_DOMAIN:
        raise RuntimeError(f"Unknown grad sync domain: {domain}")

    group = ps.get_expert_tensor_parallel_group(check_initialized=False)
    if group is None or group.size() <= 1:
        return None
    return group


def _resolve_reduce_op(op: GradSyncOp) -> Any:
    if op == GRAD_SYNC_OP_SUM:
        return torch.distributed.ReduceOp.SUM  # ty: ignore[possibly-missing-attribute]
    if op == GRAD_SYNC_OP_AVG:
        return torch.distributed.ReduceOp.AVG  # ty: ignore[possibly-missing-attribute]
    raise RuntimeError(f"Unknown grad sync op: {op}")


def finalize_model_grads_extended(
    model: list[MegatronModule],
    num_tokens: torch.Tensor | None = None,
) -> None:
    """Run Megatron finalize, then apply extra LoRA grad-sync reductions.

    Megatron finalize handles DP/CP(via `param.allreduce=True`)(and expert-DP via `param.allreduce=False`) internally.
    This extension handles extra TP/expert-TP reductions for params annotated
    with grad_sync_* metadata.
    """
    # All-reduce all model grads across DP replicas, layernorm grads for sequence parallelism,
    # embedding grads across first and last pipeline stages (if not tied)
    finalize_model_grads(
        cast(list[torch.nn.Module], model),
        num_tokens=num_tokens,
    )

    buckets: dict[
        tuple[GradSyncDomain, GradSyncOp, torch.dtype, torch.device],
        list[tuple[str, torch.Tensor]],
    ] = defaultdict(list)

    for name, param in _iter_named_trainable_parameters(model):
        domain: GradSyncDomain = getattr(
            param, "grad_sync_domain", TP_DEFAULT_GRAD_SYNC_DOMAIN
        )
        if _resolve_domain_group(domain) is None:
            continue

        op: GradSyncOp = getattr(param, "grad_sync_op", GRAD_SYNC_OP_NONE)
        if op not in VALID_SYNC_OPS:
            raise RuntimeError(f"{name}: unsupported grad_sync_op={op}")
        if op == GRAD_SYNC_OP_NONE:
            continue

        if not hasattr(param, "main_grad"):
            raise RuntimeError(
                f"{name}: expected main_grad for domain={domain} reduce_op={op}, but attribute is missing"
            )
        grad = param.main_grad
        if grad is None:
            raise RuntimeError(
                f"{name}: expected non-None main_grad for domain={domain} reduce_op={op}"
            )
        local_grad = cast(  # local part of dtensor
            torch.Tensor, grad._local_tensor if hasattr(grad, "_local_tensor") else grad
        )
        buckets[(domain, op, local_grad.dtype, local_grad.device)].append(
            (name, local_grad)
        )

    for (domain, op, _dtype, _device), entries in buckets.items():
        group = _resolve_domain_group(
            domain
        )  # already checked if the domain is one we are handling

        grads = [grad for _name, grad in entries]
        coalesced = _flatten_dense_tensors(grads)
        reduced = (
            coalesced.float()
            if torch.is_floating_point(coalesced) and coalesced.dtype != torch.float32
            else coalesced
        )
        torch.distributed.all_reduce(  # ty: ignore[possibly-missing-attribute]
            reduced,
            op=_resolve_reduce_op(op),
            group=group,
        )
        if reduced is not coalesced:
            reduced = reduced.to(dtype=coalesced.dtype)
        for grad, synced in zip(grads, _unflatten_dense_tensors(reduced, grads)):
            grad.copy_(synced)
