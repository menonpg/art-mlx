from __future__ import annotations

from collections.abc import (
    Callable,
    Iterable,
    Iterator,
    Mapping,
    MutableMapping,
    Sequence,
)
from dataclasses import dataclass
import os
from typing import TYPE_CHECKING, Generic, Literal, ParamSpec, TypeVar, cast, overload

import torch
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
import torch.distributed as dist

from art.megatron.shared_prefix_packing import (
    estimate_shared_prefix_packed_tokens,
    pack_shared_prefixes,
)

if TYPE_CHECKING:
    from megatron.bridge.models.gpt_provider import GPTModelProvider
    from megatron.core.models.gpt.gpt_model import GPTModel
    from megatron.core.optimizer import MegatronOptimizer, OptimizerConfig
    from megatron.core.packed_seq_params import PackedSeqParams

    from art.megatron.context_parallel.types import (
        ArtContextParallelState,
        ParallelTopology,
    )
    from art.megatron.lora import LoRASlotRef
    from art.megatron.model_support import ModelSupportHandler
    from art.megatron.shared_prefix_state import SharedPrefixAttentionState
    from art.megatron.train import TrainingRuntime


@dataclass(frozen=True)
class AdamParams:
    learning_rate: float
    beta1: float = 0.9
    beta2: float = 0.99
    weight_decay: float = 0.1
    grad_clip_norm: float = 0.1


@dataclass(frozen=True)
class TopK:
    logprobs: torch.Tensor
    tokens: torch.Tensor


LogprobsT = TypeVar("LogprobsT", bound=torch.Tensor | None, covariant=True)
TopKT = TypeVar("TopKT", bound=TopK | None, covariant=True)
LogitsT = TypeVar("LogitsT", bound=torch.Tensor | None, covariant=True)
HiddenStatesT = TypeVar("HiddenStatesT", bound=torch.Tensor | None, covariant=True)
T = TypeVar("T")
P = ParamSpec("P")
R = TypeVar("R")

_COMPILED_FUNCTIONS: dict[Callable[..., object], Callable[..., object]] = {}
_MEMORY_PROFILE_TRUST_GROWTH = 8


class _Unset:
    def __repr__(self) -> str:
        return "Unset"


Unset = _Unset()
type AdapterSelection = str | None | _Unset


@dataclass(frozen=True)
class ForwardOutput(Generic[LogprobsT, TopKT, LogitsT, HiddenStatesT]):
    target_logprobs: LogprobsT
    top_k: TopKT
    logits: LogitsT
    hidden_states: HiddenStatesT


class ForwardInput(Generic[LogprobsT, TopKT, LogitsT, HiddenStatesT]):
    def __init__(
        self,
        *,
        input_tokens: torch.Tensor,
        target_tokens: torch.Tensor | None = None,
        top_k: int | None = None,
        logits: bool = False,
        hidden_states: bool = False,
        checkpoint: AdapterSelection = Unset,
        lora: AdapterSelection = Unset,
    ) -> None:
        if top_k is not None and top_k < 1:
            raise ValueError("top_k must be >= 1")
        if checkpoint is not Unset and lora is not Unset:
            raise ValueError("ForwardInput cannot set both checkpoint and lora")
        self.input_tokens = input_tokens
        self.target_tokens = target_tokens
        self.top_k = top_k
        self.logits = logits
        self.hidden_states = hidden_states
        self.checkpoint = checkpoint
        self.lora = lora


type AnyForwardInput = ForwardInput[
    torch.Tensor | None,
    TopK | None,
    torch.Tensor | None,
    torch.Tensor | None,
]
type AnyForwardOutput = ForwardOutput[
    torch.Tensor | None,
    TopK | None,
    torch.Tensor | None,
    torch.Tensor | None,
]
type ForwardInputs = AnyForwardInput | Iterable["ForwardInputs"]
type ForwardOutputs = AnyForwardOutput | Sequence["ForwardOutputs"]
ForwardInputsT = TypeVar("ForwardInputsT", bound=ForwardInputs)


@dataclass(frozen=True)
class MicroBatch(Generic[ForwardInputsT]):
    inputs: Sequence[ForwardInputsT]
    outputs: Sequence[ForwardOutputs]
    indices: Sequence[int]
    stats: "MicroBatchStats"

    def select(self, xs: Sequence[T]) -> Sequence[T]:
        return [xs[i] for i in self.indices]


@dataclass(frozen=True)
class MicroBatchStats:
    global_start: int
    global_stop: int
    global_count: int
    local_count: int
    packed_tokens: int
    logical_tokens: int
    estimated_required_bytes: int
    available_bytes: int
    rejected_candidates: int
    cold_start: bool


@dataclass(frozen=True)
class _MemoryCheck:
    estimated_required_bytes: int
    available_bytes: int
    fits: bool


@dataclass(frozen=True)
class _MemoryProfile:
    bytes_per_token: float
    packed_tokens: int


@dataclass(frozen=True)
class _CandidateMicroBatch(Generic[ForwardInputsT]):
    inputs: Sequence[ForwardInputsT]
    indices: tuple[int, ...]
    plan: "_FlatForwardPlan"
    check: _MemoryCheck
    stats_global_count: int
    rejected_candidates: int
    cold_start: bool


class TrainerRankMemoryError(RuntimeError):
    pass


@dataclass(frozen=True)
class _PushedSlot:
    trainer: "TrainerRank"
    ref: "LoRASlotRef"

    def __enter__(self) -> "_PushedSlot":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: object,
    ) -> bool:
        if not self.trainer._slot_stack or self.trainer._slot_stack[-1] != self.ref:
            raise RuntimeError(
                "Pushed LoRA/checkpoint stack changed before context exit"
            )
        self.trainer.pop_pushed_lora_or_checkpoint()
        return False


@dataclass(frozen=True)
class _ForwardItem:
    request: AnyForwardInput
    input_ids: torch.Tensor
    labels: torch.Tensor | None


@dataclass(frozen=True)
class _PackedForwardBatch:
    tokens: torch.Tensor
    group_ids: torch.Tensor
    parent_ids: torch.Tensor
    position_ids: torch.Tensor
    positions_by_item: tuple[torch.Tensor, ...]


@dataclass(frozen=True)
class _PreparedPackedForward:
    tokens: torch.Tensor
    position_ids: torch.Tensor
    attention_state: "SharedPrefixAttentionState | ArtContextParallelState"
    packed_seq_params: "PackedSeqParams | None"
    positions_by_item: tuple[torch.Tensor, ...]
    source_positions_by_item: tuple[torch.Tensor, ...]


@dataclass(frozen=True)
class _RowMatch:
    source_offsets: torch.Tensor
    row_offsets: torch.Tensor


@dataclass(frozen=True)
class _MemorySignature:
    topology: tuple[int, int, int, int]
    shared_prefix_max_depth: int
    slot_group_count: int
    request_mix: tuple[str, ...]


@dataclass(frozen=True)
class _ForwardGroupPlan:
    slot_ref: "LoRASlotRef | None"
    request_indices: tuple[int, ...]
    items: tuple[_ForwardItem, ...]
    packed: _PackedForwardBatch


@dataclass(frozen=True)
class _FlatForwardPlan:
    request_count: int
    groups: tuple[_ForwardGroupPlan, ...]
    packed_tokens: int
    logical_tokens: int
    output_bytes: int
    signature: _MemorySignature


type _AdaptivePlanCacheKey = tuple[tuple[int, ...], object, tuple[object, ...], int]


class TrainerRank:
    def __init__(
        self,
        runtime: TrainingRuntime,
        *,
        head_chunk_tokens: int = 512,
        shared_prefix_max_depth: int = 1,
        memory_safety_factor: float = 1.10,
        memory_reserve_fraction: float = 0.03,
    ) -> None:
        if head_chunk_tokens < 1:
            raise ValueError("head_chunk_tokens must be >= 1")
        if shared_prefix_max_depth < 0:
            raise ValueError("shared_prefix_max_depth must be >= 0")
        if memory_safety_factor < 1.0:
            raise ValueError("memory_safety_factor must be >= 1.0")
        if not (0.0 <= memory_reserve_fraction < 1.0):
            raise ValueError("memory_reserve_fraction must be in [0, 1)")
        self.runtime: TrainingRuntime = runtime
        self.head_chunk_tokens = head_chunk_tokens
        self.shared_prefix_max_depth = shared_prefix_max_depth
        self.memory_safety_factor = memory_safety_factor
        self.memory_reserve_fraction = memory_reserve_fraction
        self.device = next(runtime.model[0].parameters()).device
        self._default_slot_ref: LoRASlotRef | None = None
        self._slot_stack: list[LoRASlotRef] = []
        self._dynamic_optimizers: dict[str, torch.optim.Optimizer] = {}
        self._checkpoint_slot_params_by_name: dict[
            str, tuple[torch.nn.Parameter, ...]
        ] = {}
        self._memory_profiles: dict[_MemorySignature, _MemoryProfile] = {}
        self._adaptive_plan_cache: dict[_AdaptivePlanCacheKey, _FlatForwardPlan] = {}
        self._adaptive_plan_cache_top_level_ids: tuple[int, ...] = ()
        self._adaptive_estimate_cache: dict[
            _AdaptivePlanCacheKey, tuple[_MemoryCheck, bool] | None
        ] = {}
        self._last_global_micro_batch_size: int | None = None
        self.zero_grad()

    def zero_grad(self) -> None:
        for chunk in self.runtime.model:
            zero_grad_buffer = getattr(chunk, "zero_grad_buffer", None)
            if callable(zero_grad_buffer):
                zero_grad_buffer()
        optimizer = cast("MegatronOptimizer | None", self.runtime.optimizer)
        if optimizer is not None:
            optimizer.zero_grad()
        for params in self._checkpoint_slot_params_by_name.values():
            for param in params:
                param.grad = None

    def _optimizer(self) -> "MegatronOptimizer":
        optimizer = cast("MegatronOptimizer | None", self.runtime.optimizer)
        if optimizer is None:
            raise RuntimeError("TrainerRank requires a runtime with an optimizer")
        return optimizer

    def _handler(self) -> "ModelSupportHandler":
        return cast("ModelSupportHandler", self.runtime.model_support_handler)

    def _provider(self) -> "GPTModelProvider":
        return cast("GPTModelProvider", self.runtime.provider)

    def set_checkpoint(self, name: str | None) -> None:
        self._set_default_slot(self._slot_ref("checkpoint", name))

    def set_lora(self, name: str | None) -> None:
        self._set_default_slot(self._slot_ref("lora", name))

    def push_checkpoint(self, name: str | None) -> _PushedSlot:
        ref = self._slot_ref("checkpoint", name)
        self._slot_stack.append(ref)
        return _PushedSlot(self, ref)

    def push_lora(self, name: str | None) -> _PushedSlot:
        ref = self._slot_ref("lora", name)
        self._slot_stack.append(ref)
        return _PushedSlot(self, ref)

    def pop_pushed_lora_or_checkpoint(self) -> None:
        if not self._slot_stack:
            raise RuntimeError("No pushed LoRA or checkpoint to pop")
        self._slot_stack.pop()

    def load_checkpoint_slot(
        self,
        name: str,
        adapter_model: dict[str, torch.Tensor],
        *,
        alpha: float | None = None,
    ) -> int:
        loaded = self._load_slot(
            "checkpoint", name, adapter_model, trainable=True, alpha=alpha
        )
        self._checkpoint_slot_params_by_name[name] = (
            self._validate_dynamic_slot_consistency("checkpoint", name, loaded)
        )
        self._dynamic_optimizers.pop(name, None)
        return loaded

    def load_lora_slot(
        self,
        name: str,
        adapter_model: dict[str, torch.Tensor],
        *,
        alpha: float | None = None,
    ) -> int:
        loaded = self._load_slot(
            "lora", name, adapter_model, trainable=False, alpha=alpha
        )
        self._validate_dynamic_slot_consistency("lora", name, loaded)
        return loaded

    def _load_slot(
        self,
        kind: Literal["checkpoint", "lora"],
        name: str,
        adapter_model: dict[str, torch.Tensor],
        *,
        trainable: bool,
        alpha: float | None,
    ) -> int:
        from art.megatron.lora import LORA_ALPHA, load_lora_slot_into_model

        return load_lora_slot_into_model(
            self.runtime.model,
            self._slot_ref(kind, name),
            adapter_model,
            alpha=LORA_ALPHA if alpha is None else alpha,
            requires_grad=trainable,
        )

    def _set_default_slot(self, ref: "LoRASlotRef") -> None:
        if self._slot_stack:
            raise RuntimeError("Cannot set a LoRA/checkpoint while a slot is pushed")
        self._default_slot_ref = ref

    @staticmethod
    def _slot_ref(
        kind: Literal["checkpoint", "lora"], name: str | None
    ) -> "LoRASlotRef":
        from art.megatron.lora import LoRASlotRef

        return LoRASlotRef(kind=kind, name=name)

    def _validate_dynamic_slot_consistency(
        self,
        kind: Literal["checkpoint", "lora"],
        name: str,
        loaded_sites: int,
    ) -> tuple[torch.nn.Parameter, ...]:
        from art.megatron.lora import iter_lora_slot_parameters

        ref = self._slot_ref(kind, name)
        params = tuple(iter_lora_slot_parameters(self.runtime.model, ref))
        if not (dist.is_available() and dist.is_initialized()):
            return params

        local = {
            "rank": dist.get_rank(),
            "loaded_sites": int(loaded_sites),
            "param_count": len(params),
            "numel": sum(int(param.numel()) for param in params),
            "signature": [
                (
                    tuple(int(dim) for dim in param.shape),
                    str(param.dtype),
                    bool(getattr(param, "allreduce", True)),
                    str(getattr(param, "grad_sync_domain", "tp_default")),
                    str(getattr(param, "grad_sync_op", "none")),
                )
                for param in params
            ],
        }
        gathered: list[dict[str, object] | None] = [None] * dist.get_world_size()
        dist.all_gather_object(gathered, local)
        ranks = [rank for rank in gathered if rank is not None]
        reference = ranks[0]
        mismatched = [
            rank
            for rank in ranks
            if rank["loaded_sites"] != reference["loaded_sites"]
            or rank["signature"] != reference["signature"]
        ]
        if not mismatched:
            return params

        summary = [
            {
                "rank": rank["rank"],
                "loaded_sites": rank["loaded_sites"],
                "param_count": rank["param_count"],
                "numel": rank["numel"],
            }
            for rank in ranks
        ]
        raise RuntimeError(
            f"Dynamic LoRA slot {kind}:{name} is not loaded consistently across "
            "distributed ranks. This usually means a sharded/exported LoRA state "
            "dict was passed directly to TrainerRank; gather or materialize the "
            "full adapter state before loading a dynamic slot. "
            f"Rank summary: {summary}."
        )

    def _resolve_slot_ref(self, request: AnyForwardInput) -> "LoRASlotRef | None":
        if request.checkpoint is not Unset:
            return self._slot_ref("checkpoint", cast(str | None, request.checkpoint))
        if request.lora is not Unset:
            return self._slot_ref("lora", cast(str | None, request.lora))
        if self._slot_stack:
            return self._slot_stack[-1]
        return self._default_slot_ref

    def forward_micro_batches(
        self,
        inputs: Iterable[ForwardInputsT],
    ) -> Iterator[MicroBatch[ForwardInputsT]]:
        items = list(inputs)
        self._validate_replicated_top_level_count(len(items))
        start = 0
        while start < len(items):
            candidate = self._select_next_micro_batch(items, start)
            flat_outputs = iter(
                self._run_flat_plan_with_memory_tracking(
                    candidate.plan,
                    context="forward_micro_batches",
                )
            )
            outputs = [_unflatten(item, flat_outputs) for item in candidate.inputs]
            stop = start + candidate.stats_global_count
            self._remember_adaptive_window(
                candidate.stats_global_count,
                is_tail=stop >= len(items),
            )
            yield MicroBatch(
                inputs=candidate.inputs,
                outputs=outputs,
                indices=candidate.indices,
                stats=MicroBatchStats(
                    global_start=start,
                    global_stop=stop,
                    global_count=candidate.stats_global_count,
                    local_count=len(candidate.inputs),
                    packed_tokens=candidate.plan.packed_tokens,
                    logical_tokens=candidate.plan.logical_tokens,
                    estimated_required_bytes=candidate.check.estimated_required_bytes,
                    available_bytes=candidate.check.available_bytes,
                    rejected_candidates=candidate.rejected_candidates,
                    cold_start=candidate.cold_start,
                ),
            )
            start = stop

    @overload
    def dp_rank_forward(
        self,
        inputs: Iterable[ForwardInput[LogprobsT, TopKT, LogitsT, HiddenStatesT]],
    ) -> Sequence[ForwardOutput[LogprobsT, TopKT, LogitsT, HiddenStatesT]]: ...

    @overload
    def dp_rank_forward(
        self,
        inputs: Iterable[
            Iterable[ForwardInput[LogprobsT, TopKT, LogitsT, HiddenStatesT]]
        ],
    ) -> Sequence[
        Sequence[ForwardOutput[LogprobsT, TopKT, LogitsT, HiddenStatesT]]
    ]: ...

    @overload
    def dp_rank_forward(
        self,
        inputs: Iterable[
            Iterable[Iterable[ForwardInput[LogprobsT, TopKT, LogitsT, HiddenStatesT]]]
        ],
    ) -> Sequence[
        Sequence[Sequence[ForwardOutput[LogprobsT, TopKT, LogitsT, HiddenStatesT]]]
    ]: ...

    @overload
    def dp_rank_forward(
        self,
        inputs: Iterable[
            Iterable[
                Iterable[
                    Iterable[ForwardInput[LogprobsT, TopKT, LogitsT, HiddenStatesT]]
                ]
            ]
        ],
    ) -> Sequence[
        Sequence[
            Sequence[Sequence[ForwardOutput[LogprobsT, TopKT, LogitsT, HiddenStatesT]]]
        ]
    ]: ...

    def dp_rank_forward(self, inputs: ForwardInputs) -> ForwardOutputs:
        materialized = _materialize(inputs)
        plan = self._plan_flat_forward(list(_flatten(materialized)))
        check = self._memory_check(plan)
        if not check.fits:
            self._raise_memory_error(
                plan,
                check,
                context="dp_rank_forward",
                message="forward is predicted to exceed available memory",
            )
        outputs = iter(
            self._run_flat_plan_with_memory_tracking(
                plan,
                context="dp_rank_forward",
            )
        )
        return _unflatten(materialized, outputs)

    def dp_reduce(
        self,
        tensor: torch.Tensor,
        *,
        op: dist.ReduceOp.RedOpType = dist.ReduceOp.SUM,
    ) -> None:
        from megatron.core import parallel_state as ps

        dist.all_reduce(
            tensor,
            op=op,
            group=ps.get_data_parallel_group(with_context_parallel=True),
        )

    def optim_step(
        self,
        *,
        params: AdamParams,
        scale_grads: float = 1.0,
        checkpoints: Sequence[str] | None = None,
    ) -> dict[str, float]:
        selected_checkpoints = self._selected_dynamic_checkpoints(checkpoints)
        if selected_checkpoints:
            return self._dynamic_optim_step(
                selected_checkpoints,
                params=params,
                scale_grads=scale_grads,
            )

        from art.megatron.training.finalize_grads import (
            finalize_model_grads_extended,
            flush_param_grads_to_main_grads,
        )
        from art.megatron.training.model_chunks import as_megatron_api_chunks

        optimizer = self._optimizer()
        flush_param_grads_to_main_grads(self.runtime.model)
        finalize_model_grads_extended(
            as_megatron_api_chunks(self.runtime.model),
            num_tokens=None,
        )
        self._scale_main_grads(scale_grads)
        self._configure_optimizer(params)
        update_successful, grad_norm, num_zeros = optimizer.step()
        optimizer.zero_grad()
        self.zero_grad()
        return {
            "learning_rate": float(params.learning_rate),
            "grad_norm": float(grad_norm),
            "update_successful": float(bool(update_successful)),
            "num_zeros_in_grad": float(num_zeros or 0),
        }

    def _selected_dynamic_checkpoints(
        self,
        checkpoints: Sequence[str] | None,
    ) -> tuple[str, ...]:
        if checkpoints is not None:
            if (
                unknown := set(checkpoints)
                - self._checkpoint_slot_params_by_name.keys()
            ):
                raise ValueError(f"Unknown checkpoint slots: {sorted(unknown)}")
            return tuple(dict.fromkeys(checkpoints))
        slots = tuple(sorted(self._checkpoint_slot_params_by_name.items()))
        if not slots:
            return ()
        has_grad = torch.tensor(
            [
                int(any(param.grad is not None for param in params))
                for _, params in slots
            ],
            device=self.device,
            dtype=torch.int32,
        )
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(has_grad, op=dist.ReduceOp.MAX)
        return tuple(name for (name, _), flag in zip(slots, has_grad.tolist()) if flag)

    def _dynamic_optim_step(
        self,
        checkpoint_names: Sequence[str],
        *,
        params: AdamParams,
        scale_grads: float,
    ) -> dict[str, float]:
        all_params: list[torch.nn.Parameter] = []
        for name in checkpoint_names:
            slot_params = self._checkpoint_slot_params_by_name[name]
            for param in slot_params:
                if param.grad is None:
                    param.grad = torch.zeros_like(param)
            self._reduce_dynamic_grads(slot_params)
            if scale_grads != 1.0:
                for param in slot_params:
                    if param.grad is not None:
                        param.grad.mul_(scale_grads)
            all_params.extend(slot_params)

        grad_norm = torch.nn.utils.clip_grad_norm_(
            all_params,
            max_norm=params.grad_clip_norm,
        )
        for name in checkpoint_names:
            optimizer = self._dynamic_optimizer(name, params)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        return {
            "learning_rate": float(params.learning_rate),
            "grad_norm": float(grad_norm),
            "update_successful": 1.0,
            "num_zeros_in_grad": 0.0,
        }

    def _dynamic_optimizer(
        self,
        name: str,
        params: AdamParams,
    ) -> torch.optim.Optimizer:
        optimizer = self._dynamic_optimizers.get(name)
        if optimizer is None:
            optimizer = torch.optim.AdamW(
                self._checkpoint_slot_params_by_name[name],
                lr=params.learning_rate,
                betas=(params.beta1, params.beta2),
                weight_decay=params.weight_decay,
            )
            self._dynamic_optimizers[name] = optimizer
            return optimizer
        for group in optimizer.param_groups:
            group["lr"] = params.learning_rate
            group["betas"] = (params.beta1, params.beta2)
            group["weight_decay"] = params.weight_decay
        return optimizer

    def _reduce_dynamic_grads(self, params: Sequence[torch.nn.Parameter]) -> None:
        from megatron.core import parallel_state as ps

        buckets: dict[
            tuple[int, str, torch.dtype, torch.device],
            tuple[object, dist.ReduceOp.RedOpType, list[torch.Tensor]],
        ] = {}

        def add_to_bucket(
            *,
            group: object,
            op: dist.ReduceOp.RedOpType,
            grad: torch.Tensor,
        ) -> None:
            key = (id(group), str(op), grad.dtype, grad.device)
            bucket = buckets.get(key)
            if bucket is None:
                buckets[key] = (group, op, [grad])
            else:
                bucket[2].append(grad)

        for param in params:
            grad = param.grad
            if grad is None:
                continue
            if bool(getattr(param, "allreduce", True)):
                group = ps.get_data_parallel_group(with_context_parallel=True)
            else:
                group = ps.get_expert_data_parallel_group()
            if group is not None and group.size() > 1:
                add_to_bucket(group=group, op=dist.ReduceOp.SUM, grad=grad)

            op = getattr(param, "grad_sync_op", "none")
            if op == "none":
                continue
            domain = getattr(param, "grad_sync_domain", "tp_default")
            if domain == "expert_tp":
                tp_group = ps.get_expert_tensor_parallel_group(check_initialized=False)
            else:
                tp_group = ps.get_tensor_model_parallel_group(check_initialized=False)
            if tp_group is None or tp_group.size() <= 1:
                continue
            reduce_op = dist.ReduceOp.AVG if op == "avg" else dist.ReduceOp.SUM
            add_to_bucket(group=tp_group, op=reduce_op, grad=grad)

        for group, op, grads in buckets.values():
            self._coalesced_all_reduce(grads, group=group, op=op)

    @staticmethod
    def _coalesced_all_reduce(
        grads: Sequence[torch.Tensor],
        *,
        group: object,
        op: dist.ReduceOp.RedOpType,
    ) -> None:
        if not grads:
            return
        coalesced = _flatten_dense_tensors(grads)
        reduced = (
            coalesced.float()
            if torch.is_floating_point(coalesced) and coalesced.dtype != torch.float32
            else coalesced
        )
        dist.all_reduce(reduced, op=op, group=group)
        if reduced is not coalesced:
            reduced = reduced.to(dtype=coalesced.dtype)
        for grad, synced in zip(grads, _unflatten_dense_tensors(reduced, grads)):
            grad.copy_(synced)

    def _select_next_micro_batch(
        self,
        items: Sequence[ForwardInputsT],
        start: int,
    ) -> _CandidateMicroBatch[ForwardInputsT]:
        dp_rank, dp_size = self._dp_rank_and_size()
        remaining = len(items) - start
        min_width = min(dp_size, remaining)
        if min_width <= 0:
            raise RuntimeError("cannot select an empty microbatch window")

        estimate_cache: dict[int, tuple[_MemoryCheck, bool] | None] = {}
        rejected = 0

        def clamp_width(width: int) -> int:
            return max(min_width, min(width, remaining))

        granularity = self._adaptive_window_granularity(
            remaining=remaining,
            dp_size=dp_size,
        )

        def snap_width(width: int) -> int:
            width = clamp_width(width)
            if width in (min_width, remaining) or granularity <= 1:
                return width
            if width < granularity:
                return width
            return max(min_width, (width // granularity) * granularity)

        def local_slice(width: int) -> tuple[tuple[int, ...], list[ForwardInputsT]]:
            stop = start + clamp_width(width)
            indices = tuple(range(start + dp_rank, stop, dp_size))
            return indices, [items[index] for index in indices]

        def raise_smallest(plan: _FlatForwardPlan, check: _MemoryCheck) -> None:
            self._raise_memory_error(
                plan,
                check,
                context="forward_micro_batches",
                message="smallest DP microbatch is predicted to exceed available memory",
            )

        def candidate(
            width: int,
            estimated_check: _MemoryCheck | None = None,
        ) -> _CandidateMicroBatch[ForwardInputsT]:
            width = clamp_width(width)
            indices, local_inputs = local_slice(width)
            plan = self._cached_adaptive_plan(items, indices, local_inputs)
            return _CandidateMicroBatch(
                inputs=local_inputs,
                indices=indices,
                plan=plan,
                check=estimated_check or self._memory_check(plan),
                stats_global_count=width,
                rejected_candidates=rejected,
                cold_start=not self._all_ranks_have_memory_profile(
                    packed_tokens=plan.packed_tokens,
                    signature=plan.signature,
                ),
            )

        def estimate_check(width: int) -> tuple[_MemoryCheck, bool] | None:
            width = clamp_width(width)
            if width not in estimate_cache:
                indices, local_inputs = local_slice(width)
                estimate_cache[width] = self._cached_adaptive_estimate(
                    items,
                    indices,
                    local_inputs,
                )
            return estimate_cache[width]

        first_estimated = estimate_check(min_width)
        if first_estimated is not None and not first_estimated[0].fits:
            first = candidate(min_width, first_estimated[0])
            raise_smallest(first.plan, first.check)

        if first_estimated is not None and first_estimated[1]:
            best_width = min_width
            best_check: _MemoryCheck | None = first_estimated[0]
        else:
            first = candidate(
                min_width,
                first_estimated[0] if first_estimated is not None else None,
            )
            if not first.check.fits:
                raise_smallest(first.plan, first.check)
            if first.cold_start:
                return first
            best_width = first.stats_global_count
            best_check = None

        def probe(
            width: int,
        ) -> tuple[bool, _MemoryCheck | None]:
            estimated = estimate_check(width)
            if estimated is not None:
                return estimated[1] and estimated[0].fits, estimated[0]
            item = candidate(width)
            return item.check.fits, None

        def remember_fit(
            width: int,
            check: _MemoryCheck | None,
        ) -> None:
            nonlocal best_width, best_check
            best_width = snap_width(width)
            best_check = check

        def search_below(failed_width: int) -> None:
            nonlocal rejected
            low = best_width + 1
            high = failed_width - 1
            while low <= high:
                mid = (low + high) // 2
                fits, check = probe(mid)
                if fits:
                    remember_fit(mid, check)
                    low = mid + 1
                else:
                    rejected += 1
                    high = mid - 1

        stable_width = self._last_global_micro_batch_size
        if stable_width is not None and stable_width >= max(64, granularity * 2):
            stable_capacity = stable_width
            stable_width = clamp_width(stable_capacity)
            fits, check = probe(stable_width)
            if fits:
                grow_multiplier = 4 if stable_capacity < 256 else 2
                grow_capacity = min(remaining, stable_capacity * grow_multiplier)
                if remaining > grow_capacity:
                    grow_width = clamp_width(grow_capacity)
                    if grow_width > stable_width:
                        grow_fits, grow_check = probe(grow_width)
                        if grow_fits:
                            return candidate(grow_width, grow_check)
                        rejected += 1
                        search_below(grow_width)
                        return candidate(best_width, best_check)
                return candidate(stable_width, check)
            rejected += 1
            search_below(stable_width)
            self._last_global_micro_batch_size = best_width
            return candidate(best_width, best_check)

        high_fail: int | None = None
        width = min(
            remaining,
            max(min_width, (self._last_global_micro_batch_size or min_width) * 2),
        )
        while width <= remaining:
            fits, check = probe(width)
            if fits:
                remember_fit(width, check)
                if width == remaining:
                    break
                width = min(remaining, max(width + 1, width * 2))
                continue
            rejected += 1
            high_fail = width
            break

        if high_fail is not None:
            search_below(high_fail)

        return candidate(best_width, best_check)

    @staticmethod
    def _adaptive_window_granularity(*, remaining: int, dp_size: int) -> int:
        if remaining < 64:
            return max(1, dp_size)
        base = 8 if remaining < 256 else 32
        return max(1, ((base + dp_size - 1) // dp_size) * dp_size)

    def _remember_adaptive_window(self, width: int, *, is_tail: bool) -> None:
        if is_tail:
            return
        if self._last_global_micro_batch_size is None:
            self._last_global_micro_batch_size = width
        else:
            self._last_global_micro_batch_size = max(
                self._last_global_micro_batch_size,
                width,
            )

    def _cached_adaptive_plan(
        self,
        items: Sequence[ForwardInputsT],
        indices: tuple[int, ...],
        local_inputs: Sequence[ForwardInputsT],
    ) -> _FlatForwardPlan:
        key = self._adaptive_cache_key(items, indices)
        cached = self._adaptive_plan_cache.get(key)
        if cached is not None:
            return cached
        plan = self._plan_flat_forward(list(_flatten(local_inputs)))
        self._adaptive_plan_cache[key] = plan
        return plan

    def _cached_adaptive_estimate(
        self,
        items: Sequence[ForwardInputsT],
        indices: tuple[int, ...],
        local_inputs: Sequence[ForwardInputsT],
    ) -> tuple[_MemoryCheck, bool] | None:
        key = self._adaptive_cache_key(items, indices)
        if key in self._adaptive_estimate_cache:
            return self._adaptive_estimate_cache[key]
        estimate = self._estimate_flat_forward(list(_flatten(local_inputs)))
        if estimate is not None:
            packed_tokens, output_bytes, signature = estimate
            estimate = (
                self._memory_check_required(
                    self._estimate_required_memory_bytes_from_values(
                        packed_tokens=packed_tokens,
                        output_bytes=output_bytes,
                        signature=signature,
                    )
                ),
                self._all_ranks_have_memory_profile(
                    packed_tokens=packed_tokens,
                    signature=signature,
                ),
            )
        self._adaptive_estimate_cache[key] = estimate
        return estimate

    def _adaptive_cache_key(
        self,
        items: Sequence[ForwardInputsT],
        indices: tuple[int, ...],
    ) -> _AdaptivePlanCacheKey:
        top_level_ids = tuple(id(item) for item in items)
        if top_level_ids != self._adaptive_plan_cache_top_level_ids:
            self._adaptive_plan_cache.clear()
            self._adaptive_estimate_cache.clear()
            self._adaptive_plan_cache_top_level_ids = top_level_ids
        return (
            indices,
            self._default_slot_ref,
            tuple(self._slot_stack),
            self.shared_prefix_max_depth,
        )

    def _validate_replicated_top_level_count(self, count: int) -> None:
        if not (dist.is_available() and dist.is_initialized()):
            return
        counts = [0 for _ in range(dist.get_world_size())]
        dist.all_gather_object(counts, int(count))
        if len(set(counts)) == 1:
            return
        raise ValueError(
            "forward_micro_batches requires the same top-level input count on every "
            "distributed rank. Pass already-DP-local inputs to dp_rank_forward instead. "
            f"Observed counts by rank: {counts}."
        )

    def _dp_rank_and_size(self) -> tuple[int, int]:
        try:
            from megatron.core import parallel_state as ps

            return int(ps.get_data_parallel_rank()), int(
                ps.get_data_parallel_world_size()
            )
        except (AssertionError, ImportError, RuntimeError, ValueError):
            return 0, 1

    def _plan_flat_forward(
        self, requests: Sequence[AnyForwardInput]
    ) -> _FlatForwardPlan:
        plans: list[_ForwardGroupPlan] = []
        output_bytes = 0
        logical_tokens = sum(int(request.input_tokens.numel()) for request in requests)
        groups = self._group_active_request_indices(requests)
        for slot_ref, group_indices in groups:
            items = tuple(
                self._forward_item(requests[index]) for index in group_indices
            )
            packed = _pack_forward_items(items, max_depth=self.shared_prefix_max_depth)
            output_bytes += self._estimate_group_request_output_bytes(
                [item.request for item in items]
            )
            plans.append(
                _ForwardGroupPlan(
                    slot_ref=slot_ref,
                    request_indices=tuple(group_indices),
                    items=items,
                    packed=packed,
                )
            )

        return _FlatForwardPlan(
            request_count=len(requests),
            groups=tuple(plans),
            packed_tokens=sum(int(plan.packed.tokens.numel()) for plan in plans),
            logical_tokens=logical_tokens,
            output_bytes=output_bytes,
            signature=self._memory_signature_from_requests(
                requests,
                slot_group_count=len(plans),
            ),
        )

    def _estimate_flat_forward(
        self, requests: Sequence[AnyForwardInput]
    ) -> tuple[int, int, _MemorySignature] | None:
        groups = self._group_active_request_indices(requests)
        packed_tokens = 0
        output_bytes = 0
        for _, group_indices in groups:
            group_packed_tokens = estimate_shared_prefix_packed_tokens(
                (requests[index].input_tokens for index in group_indices),
                max_depth=self.shared_prefix_max_depth,
            )
            if group_packed_tokens is None:
                return None
            packed_tokens += group_packed_tokens
            output_bytes += self._estimate_group_request_output_bytes(
                [requests[index] for index in group_indices]
            )

        return (
            packed_tokens,
            output_bytes,
            self._memory_signature_from_requests(
                requests,
                slot_group_count=len(groups),
            ),
        )

    def _group_active_request_indices(
        self,
        requests: Sequence[AnyForwardInput],
    ) -> tuple[tuple["LoRASlotRef | None", tuple[int, ...]], ...]:
        groups: dict[LoRASlotRef | None, list[int]] = {}
        for index, request in enumerate(requests):
            if (
                request.target_tokens is not None
                or request.logits
                or request.top_k is not None
                or request.hidden_states
            ):
                groups.setdefault(self._resolve_slot_ref(request), []).append(index)
        return tuple((slot_ref, tuple(indices)) for slot_ref, indices in groups.items())

    def _run_flat_plan_with_memory_tracking(
        self,
        plan: _FlatForwardPlan,
        *,
        context: str,
    ) -> list[AnyForwardOutput]:
        if torch.cuda.is_available() and self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            baseline = int(torch.cuda.memory_allocated(self.device))
            torch.cuda.reset_peak_memory_stats(self.device)
        else:
            baseline = 0
        try:
            outputs = self._execute_flat_plan(plan)
        except torch.cuda.OutOfMemoryError as exc:
            check = self._memory_check(plan)
            self._raise_memory_error(
                plan,
                check,
                context=context,
                message="CUDA OOM occurred despite the planner estimate",
            )
            raise AssertionError("unreachable") from exc
        if torch.cuda.is_available() and self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            peak = int(torch.cuda.max_memory_allocated(self.device))
            self._update_memory_profile(plan, max(0, peak - baseline))
        return outputs

    def _execute_flat_plan(self, plan: _FlatForwardPlan) -> list[AnyForwardOutput]:
        outputs = [
            ForwardOutput(
                target_logprobs=None,
                top_k=None,
                logits=None,
                hidden_states=None,
            )
            for _ in range(plan.request_count)
        ]
        for group in plan.groups:
            from art.megatron.lora import use_lora_slot

            with use_lora_slot(group.slot_ref):
                prepared = self._prepare_packed_forward(group.packed)
                item_outputs = self._forward_packed(group.items, prepared)
            for index, output in zip(group.request_indices, item_outputs, strict=True):
                outputs[index] = output
        return outputs

    def _estimate_group_request_output_bytes(
        self,
        requests: Sequence[AnyForwardInput],
    ) -> int:
        model: GPTModel | None
        try:
            model = _language_model(self.runtime.model[0])
        except RuntimeError:
            model = None
        dtype_size = _dtype_size(next(self.runtime.model[0].parameters()).dtype)
        total = 0
        for request in requests:
            seq_len = int(request.input_tokens.numel())
            if request.target_tokens is not None:
                total += int(request.target_tokens.numel()) * _dtype_size(torch.float32)
            if request.top_k is not None:
                total += (
                    seq_len
                    * int(request.top_k)
                    * (_dtype_size(torch.float32) + _dtype_size(torch.long))
                )
            if request.logits:
                if model is None:
                    raise RuntimeError("logits output memory requires a GPT model")
                total += seq_len * _padded_vocab_size(model) * dtype_size
            if request.hidden_states:
                hidden_size = _hidden_size(model, self.runtime.provider)
                total += seq_len * hidden_size * dtype_size
        return total

    def _memory_signature_from_requests(
        self,
        requests: Sequence[AnyForwardInput],
        *,
        slot_group_count: int,
    ) -> _MemorySignature:
        return _MemorySignature(
            topology=self._topology_key(),
            shared_prefix_max_depth=self.shared_prefix_max_depth,
            slot_group_count=slot_group_count,
            request_mix=tuple(
                sorted({_request_mix_key(request) for request in requests})
            ),
        )

    def _topology_key(self) -> tuple[int, int, int, int]:
        try:
            topology = self._topology()
            return (
                int(topology.dp),
                int(topology.tp),
                int(topology.cp),
                int(topology.pp),
            )
        except (AssertionError, AttributeError, ImportError, RuntimeError, ValueError):
            return (1, 1, 1, 1)

    def _memory_check(
        self,
        forward: _FlatForwardPlan,
    ) -> _MemoryCheck:
        return self._memory_check_required(
            self._estimate_required_memory_bytes_from_values(
                packed_tokens=forward.packed_tokens,
                output_bytes=forward.output_bytes,
                signature=forward.signature,
            )
        )

    def _memory_check_required(self, required: int) -> _MemoryCheck:
        available = self._available_memory_bytes()
        if dist.is_available() and dist.is_initialized():
            values = torch.tensor(
                [float(required), float(available)],
                device=self.device if self.device.type == "cuda" else "cpu",
                dtype=torch.float64,
            )
            dist.all_reduce(values[0], op=dist.ReduceOp.MAX)
            dist.all_reduce(values[1], op=dist.ReduceOp.MIN)
            required = int(values[0].item())
            available = int(values[1].item())
        return _MemoryCheck(
            estimated_required_bytes=required,
            available_bytes=available,
            fits=required <= available,
        )

    def _raise_memory_error(
        self,
        plan: _FlatForwardPlan,
        check: _MemoryCheck,
        *,
        context: str,
        message: str,
    ) -> None:
        raise TrainerRankMemoryError(
            f"{context}: {message}. "
            f"packed_tokens={plan.packed_tokens} "
            f"logical_tokens={plan.logical_tokens} "
            f"output_gb={plan.output_bytes / 1024**3:.3f} "
            f"estimated_required_gb={check.estimated_required_bytes / 1024**3:.3f} "
            f"available_gb={check.available_bytes / 1024**3:.3f}. "
            "Use smaller top-level items, reduce output requests, or call "
            "dp_rank_forward with already-DP-local smaller inputs."
        )

    def _estimate_required_memory_bytes_from_values(
        self,
        *,
        packed_tokens: int,
        output_bytes: int,
        signature: _MemorySignature,
    ) -> int:
        if packed_tokens <= 0:
            return output_bytes
        profiled = self._memory_profiles.get(signature)
        static_compute = self._static_compute_memory_bytes_for_tokens(packed_tokens)
        if profiled is None or not _memory_profile_covers(
            profiled,
            packed_tokens=packed_tokens,
        ):
            compute = static_compute
        else:
            compute = max(static_compute, int(profiled.bytes_per_token * packed_tokens))
        return int((output_bytes + compute) * self.memory_safety_factor)

    def _static_compute_memory_bytes_for_tokens(self, packed_tokens: int) -> int:
        if packed_tokens <= 0:
            return 0
        try:
            model = _language_model(self.runtime.model[0])
        except RuntimeError:
            return 0
        dtype_size = _dtype_size(next(self.runtime.model[0].parameters()).dtype)
        hidden_size = _hidden_size(model, self.runtime.provider)
        layers = int(
            getattr(getattr(model, "config", None), "num_layers", 0)
            or getattr(self.runtime.provider, "num_layers", 1)
            or 1
        )
        activation_factor = max(4, min(16, layers // 4 + 4))
        return int(packed_tokens * hidden_size * dtype_size * activation_factor)

    def _available_memory_bytes(self) -> int:
        if not (torch.cuda.is_available() and self.device.type == "cuda"):
            return 1 << 60
        free, total = torch.cuda.mem_get_info(self.device)
        allocated = int(torch.cuda.memory_allocated(self.device))
        reserved = int(torch.cuda.memory_reserved(self.device))
        reusable_reserved = max(0, reserved - allocated)
        reserve = int(total * self.memory_reserve_fraction)
        return max(0, int(free) + reusable_reserved - reserve)

    def _all_ranks_have_memory_profile(
        self,
        *,
        packed_tokens: int,
        signature: _MemorySignature,
    ) -> bool:
        profile = self._memory_profiles.get(signature)
        local = packed_tokens <= 0 or (
            profile is not None
            and _memory_profile_covers(profile, packed_tokens=packed_tokens)
        )
        if dist.is_available() and dist.is_initialized():
            value = torch.tensor(
                int(local),
                device=self.device if self.device.type == "cuda" else "cpu",
                dtype=torch.int32,
            )
            dist.all_reduce(value, op=dist.ReduceOp.MIN)
            return bool(value.item())
        return local

    def _update_memory_profile(
        self, plan: _FlatForwardPlan, peak_delta_bytes: int
    ) -> None:
        if plan.packed_tokens <= 0:
            return
        compute_delta = max(0, peak_delta_bytes - plan.output_bytes)
        bytes_per_token = compute_delta / max(1, plan.packed_tokens)
        previous = self._memory_profiles.get(plan.signature)
        self._memory_profiles[plan.signature] = _MemoryProfile(
            bytes_per_token=max(
                bytes_per_token,
                0.0 if previous is None else previous.bytes_per_token,
            ),
            packed_tokens=max(
                plan.packed_tokens,
                0 if previous is None else previous.packed_tokens,
            ),
        )

    def _forward_item(self, request: AnyForwardInput) -> _ForwardItem:
        if request.top_k is not None:
            _validate_top_k(request.top_k, _language_model(self.runtime.model[0]))
        input_ids = _as_1d_long(request.input_tokens, name="input_tokens")
        labels = (
            _as_target_tokens(request.target_tokens, request.input_tokens, input_ids)
            if request.target_tokens is not None
            else None
        )
        return _ForwardItem(request=request, input_ids=input_ids, labels=labels)

    def _forward_packed(
        self,
        items: Sequence[_ForwardItem],
        prepared: _PreparedPackedForward,
    ) -> list[AnyForwardOutput]:
        hidden_by_row = self._gather_sequence_parallel_hidden(
            self._decoder_hidden(prepared)
        )
        return self._project_head(items, prepared, hidden_by_row)

    def _decoder_hidden(
        self,
        prepared: _PreparedPackedForward,
    ) -> torch.Tensor:
        from art.megatron.train import _placeholder_attention_mask

        handler = self._handler()
        model = _language_model(self.runtime.model[0])
        attention_mask = _placeholder_attention_mask(self.device)
        forward_kwargs = handler.get_forward_kwargs(
            self.runtime.model[0],
            attention_bias=prepared.attention_state,
        )
        extra_block_kwargs = cast(
            dict[str, object] | None,
            forward_kwargs.pop("extra_block_kwargs", None),
        )
        preprocessed = model._preprocess(
            input_ids=prepared.tokens,
            position_ids=prepared.position_ids,
            packed_seq_params=cast("PackedSeqParams", prepared.packed_seq_params),
        )
        (
            decoder_input,
            rotary_pos_emb,
            rotary_pos_cos,
            rotary_pos_sin,
            sequence_len_offset,
            padding_mask,
        ) = preprocessed[:6]
        rotary_pos_cos_sin = preprocessed[6] if len(preprocessed) == 7 else None
        return cast(
            torch.Tensor,
            model.decoder(
                hidden_states=decoder_input,
                attention_mask=attention_mask,
                rotary_pos_emb=rotary_pos_emb,
                rotary_pos_cos=rotary_pos_cos,
                rotary_pos_sin=rotary_pos_sin,
                rotary_pos_cos_sin=rotary_pos_cos_sin,
                packed_seq_params=prepared.packed_seq_params,
                sequence_len_offset=sequence_len_offset,
                padding_mask=padding_mask,
                **(extra_block_kwargs or {}),
            ),
        )

    def _project_head(
        self,
        items: Sequence[_ForwardItem],
        prepared: _PreparedPackedForward,
        hidden_by_row: torch.Tensor,
    ) -> list[AnyForwardOutput]:
        model = _language_model(self.runtime.model[0])
        output_weight = (
            model.shared_embedding_or_output_weight()
            if bool(model.share_embeddings_and_output_weights)
            else None
        )
        device = hidden_by_row.device
        target_logprobs = [None for _ in items]
        logits: list[torch.Tensor | None] = [None for _ in items]
        top_k: list[TopK | None] = [None for _ in items]
        label_rows: list[torch.Tensor | None] = [None for _ in items]
        full_rows: list[torch.Tensor] = []
        local_rows: list[torch.Tensor] = []

        for index, (item, positions_cpu) in enumerate(
            zip(items, prepared.positions_by_item, strict=True)
        ):
            positions = positions_cpu.to(device=device)
            if item.request.logits:
                full_rows.append(positions)
            elif item.request.top_k is not None:
                local_rows.append(positions)
            if item.labels is not None:
                source_positions = prepared.source_positions_by_item[index].to(device)
                labels = item.labels.to(device=device).index_select(0, source_positions)
                label_rows[index] = labels
                target_logprobs[index] = torch.zeros(
                    tuple(labels.shape),
                    device=device,
                    dtype=torch.float32,
                )
                if item.request.top_k is None and not item.request.logits:
                    valid = labels != -100
                    if labels.ndim > 1:
                        valid = valid.reshape(int(labels.shape[0]), -1).any(dim=1)
                    valid_offsets = torch.nonzero(valid, as_tuple=False).reshape(-1)
                    if int(valid_offsets.numel()):
                        local_rows.append(positions.index_select(0, valid_offsets))
            if item.request.logits:
                logits[index] = torch.empty(
                    (int(positions.numel()), _padded_vocab_size(model)),
                    device=hidden_by_row.device,
                    dtype=hidden_by_row.dtype,
                )

        full_row_tensor = (
            torch.cat(full_rows).unique(sorted=True)
            if full_rows
            else torch.empty(0, dtype=torch.long, device=device)
        )
        local_row_tensor = (
            torch.cat(local_rows).unique(sorted=True)
            if local_rows
            else torch.empty(0, dtype=torch.long, device=device)
        )
        if int(full_row_tensor.numel()) and int(local_row_tensor.numel()):
            local_row_tensor = local_row_tensor[
                ~torch.isin(local_row_tensor, full_row_tensor)
            ]

        if int(full_row_tensor.numel()):
            self._project_full_logits(
                items,
                prepared,
                hidden_by_row,
                full_row_tensor,
                output_weight=output_weight,
                target_logprobs=target_logprobs,
                top_k=top_k,
                logits=logits,
                label_rows=label_rows,
            )

        if int(local_row_tensor.numel()):
            local_row_matches = _row_matches_by_item(
                prepared.positions_by_item,
                local_row_tensor,
                device=device,
            )
            self._project_vocab_parallel(
                items,
                hidden_by_row,
                local_row_tensor,
                row_matches=local_row_matches,
                item_lengths=tuple(
                    int(positions.numel()) for positions in prepared.positions_by_item
                ),
                output_weight=output_weight,
                target_logprobs=target_logprobs,
                top_k=top_k,
                label_rows=label_rows,
            )

        target_logprobs, top_k = _anchor_disconnected_outputs(
            target_logprobs,
            top_k,
            hidden_by_row,
        )
        return [
            ForwardOutput(
                target_logprobs=target_logprobs[index],
                top_k=top_k[index],
                logits=logits[index],
                hidden_states=(
                    _select_positions(hidden_by_row, positions)
                    if item.request.hidden_states
                    else None
                ),
            )
            for index, (item, positions) in enumerate(
                zip(items, prepared.positions_by_item, strict=True)
            )
        ]

    def _project_full_logits(
        self,
        items: Sequence[_ForwardItem],
        prepared: _PreparedPackedForward,
        hidden_by_row: torch.Tensor,
        rows: torch.Tensor,
        *,
        output_weight: torch.Tensor | None,
        target_logprobs: list[torch.Tensor | None],
        top_k: list[TopK | None],
        logits: list[torch.Tensor | None],
        label_rows: list[torch.Tensor | None],
    ) -> None:
        model = _language_model(self.runtime.model[0])
        for start in range(0, int(rows.numel()), self.head_chunk_tokens):
            chunk_rows = rows[start : start + self.head_chunk_tokens]
            chunk_logits = self._logits_from_hidden_rows(
                model,
                _select_positions(hidden_by_row, chunk_rows),
                output_weight=output_weight,
            )
            log_z = None
            if any(
                item.labels is not None or item.request.top_k is not None
                for item in items
            ):
                log_z = torch.logsumexp(chunk_logits.float(), dim=-1)

            for index, item in enumerate(items):
                positions = prepared.positions_by_item[index].to(device=rows.device)
                offsets, chunk_offsets = _matching_offsets(positions, chunk_rows)
                if int(offsets.numel()) == 0:
                    continue
                selected_logits = chunk_logits.index_select(0, chunk_offsets)
                item_logits = logits[index]
                if item_logits is not None:
                    item_logits[offsets] = selected_logits
                labels = label_rows[index]
                item_logprobs = target_logprobs[index]
                if item_logprobs is not None and labels is not None:
                    if log_z is None:
                        raise RuntimeError("target logprobs require logsumexp")
                    item_logprobs[offsets] = _call_compiled(
                        _target_logprobs_from_full_logits,
                        selected_logits,
                        labels.index_select(0, offsets),
                        log_z.index_select(0, chunk_offsets),
                    )
                k = item.request.top_k
                if k is not None:
                    if log_z is None:
                        raise RuntimeError("top_k requires logsumexp")
                    values, tokens = torch.topk(selected_logits.float(), k=k, dim=-1)
                    top_k[index] = _merge_topk(
                        top_k[index],
                        offsets,
                        TopK(
                            logprobs=values
                            - log_z.index_select(0, chunk_offsets).unsqueeze(1),
                            tokens=tokens,
                        ),
                        length=int(positions.numel()),
                    )

    def _project_vocab_parallel(
        self,
        items: Sequence[_ForwardItem],
        hidden_by_row: torch.Tensor,
        rows: torch.Tensor,
        *,
        row_matches: Sequence[_RowMatch],
        item_lengths: Sequence[int],
        output_weight: torch.Tensor | None,
        target_logprobs: list[torch.Tensor | None],
        top_k: list[TopK | None],
        label_rows: list[torch.Tensor | None],
    ) -> None:
        model = _language_model(self.runtime.model[0])
        fused_target_labels = (
            _consistent_row_labels(
                label_rows,
                row_matches,
                row_count=int(rows.numel()),
                device=rows.device,
            )
            if all(item.request.top_k is None for item in items)
            and all(labels is None or labels.ndim == 1 for labels in label_rows)
            else None
        )
        if fused_target_labels is not None:
            row_target_logprobs = torch.empty(
                int(rows.numel()),
                device=rows.device,
                dtype=torch.float32,
            )
            for start in range(0, int(rows.numel()), self.head_chunk_tokens):
                chunk_rows = rows[start : start + self.head_chunk_tokens]
                local_logits = self._local_logits_from_hidden_rows(
                    model,
                    _select_positions(hidden_by_row, chunk_rows),
                    output_weight=output_weight,
                )
                row_target_logprobs[
                    start : start + int(chunk_rows.numel())
                ] = -model.compute_language_model_loss(
                    fused_target_labels[
                        start : start + int(chunk_rows.numel())
                    ].unsqueeze(0),
                    local_logits.unsqueeze(1),
                ).reshape(-1)
            _scatter_row_target_logprobs(
                row_target_logprobs,
                row_matches,
                label_rows,
                target_logprobs,
            )
            return

        max_top_k = max(
            (int(item.request.top_k or 0) for item in items if not item.request.logits),
            default=0,
        )
        for start in range(0, int(rows.numel()), self.head_chunk_tokens):
            chunk_rows = rows[start : start + self.head_chunk_tokens]
            local_logits = self._local_logits_from_hidden_rows(
                model,
                _select_positions(hidden_by_row, chunk_rows),
                output_weight=output_weight,
            )
            topk_stats = _try_triton_local_topk_stats(local_logits, k=max_top_k)
            logsumexp_stats = (
                _try_triton_local_logsumexp_stats(local_logits)
                if topk_stats is None
                else None
            )
            stats = topk_stats if topk_stats is not None else logsumexp_stats
            if stats is not None:
                local_max, local_sum = stats[:2]
                local_max = local_max.detach()
                global_max = _all_reduce_tensor_parallel_max(local_max)
                global_sum = _all_reduce_tensor_parallel_sum(
                    local_sum * torch.exp(local_max - global_max)
                )
                log_z = global_max + torch.log(global_sum)
            else:
                log_z = _vocab_parallel_log_z(local_logits)

            local_topk: tuple[torch.Tensor, torch.Tensor] | None = None
            if topk_stats is not None:
                _, _, local_values, local_tokens = topk_stats
                local_topk = (local_values, local_tokens)
            elif logsumexp_stats is not None and max_top_k > 0:
                local_k = min(max_top_k, int(local_logits.shape[1]))
                local_values, local_tokens = torch.topk(local_logits, k=local_k, dim=-1)
                local_topk = (local_values.float(), local_tokens)

            for index, item in enumerate(items):
                if item.request.logits:
                    continue
                offsets, chunk_offsets = _match_chunk_offsets(
                    row_matches[index],
                    start=start,
                    end=start + int(chunk_rows.numel()),
                )
                if int(offsets.numel()) == 0:
                    continue
                selected_log_z = log_z.index_select(0, chunk_offsets)
                labels = label_rows[index]
                item_logprobs = target_logprobs[index]
                if item_logprobs is not None and labels is not None:
                    item_logprobs[offsets] = _vocab_parallel_target_logprobs(
                        local_logits,
                        labels.index_select(0, offsets),
                        selected_log_z,
                        row_offsets=chunk_offsets,
                    )
                k = item.request.top_k
                if k is not None:
                    if local_topk is not None:
                        local_values, local_tokens = local_topk
                        top_k[index] = _merge_topk(
                            top_k[index],
                            offsets,
                            _vocab_parallel_topk_from_local(
                                local_values.index_select(0, chunk_offsets),
                                local_tokens.index_select(0, chunk_offsets),
                                k=k,
                                log_z=selected_log_z,
                                vocab_start=_vocab_range(local_logits)[0],
                            ),
                            length=item_lengths[index],
                        )
                        continue
                    selected_logits = local_logits.index_select(0, chunk_offsets)
                    top_k[index] = _merge_topk(
                        top_k[index],
                        offsets,
                        _vocab_parallel_topk(
                            selected_logits,
                            k=k,
                            log_z=selected_log_z,
                        ),
                        length=item_lengths[index],
                    )

    def _logits_from_hidden_rows(
        self,
        model: "GPTModel",
        hidden: torch.Tensor,
        *,
        output_weight: torch.Tensor | None,
    ) -> torch.Tensor:
        local_logits = self._local_logits_from_hidden_rows(
            model,
            hidden,
            output_weight=output_weight,
        )
        return _batch_seq_logits(
            self._gather_tensor_parallel_logits(local_logits.unsqueeze(1)),
            seq_len=int(hidden.shape[0]),
        ).squeeze(0)

    def _local_logits_from_hidden_rows(
        self,
        model: "GPTModel",
        hidden: torch.Tensor,
        *,
        output_weight: torch.Tensor | None,
    ) -> torch.Tensor:
        output_layer = model.output_layer
        sequence_parallel = bool(getattr(output_layer, "sequence_parallel", False))
        if sequence_parallel:
            output_layer.sequence_parallel = False
        try:
            logits, _ = output_layer(
                hidden.unsqueeze(1),
                weight=output_weight,
                runtime_gather_output=None,
            )
        finally:
            if sequence_parallel:
                output_layer.sequence_parallel = True
        return _batch_seq_logits(
            model._scale_logits(logits),
            seq_len=int(hidden.shape[0]),
        ).squeeze(0)

    def _gather_sequence_parallel_hidden(self, hidden: torch.Tensor) -> torch.Tensor:
        from megatron.core import parallel_state as ps

        if int(ps.get_tensor_model_parallel_world_size()) <= 1:
            return hidden.squeeze(1)
        from megatron.core import tensor_parallel

        gathered = tensor_parallel.gather_from_sequence_parallel_region(
            hidden,
            tensor_parallel_output_grad=True,
            group=ps.get_tensor_model_parallel_group(check_initialized=False),
        )
        return cast(torch.Tensor, gathered).squeeze(1)

    def _prepare_packed_forward(
        self,
        batch: _PackedForwardBatch,
    ) -> _PreparedPackedForward:
        topology = self._topology()
        batch = _pad_packed_batch(batch, multiple=int(topology.tp))
        if int(topology.cp) > 1:
            return self._prepare_context_parallel_forward(batch, topology=topology)
        from art.megatron.shared_prefix_state import create_shared_prefix_state

        handler = self._handler()
        provider = self._provider()
        return _PreparedPackedForward(
            tokens=batch.tokens.to(self.device),
            position_ids=batch.position_ids.to(self.device),
            attention_state=create_shared_prefix_state(
                group_ids=batch.group_ids,
                parent_ids=batch.parent_ids,
                target_device=self.device,
                build_gdn_execution_spec=handler.build_gdn_execution_spec,
                attention_head_dim=provider.kv_channels,
                attention_value_head_dim=provider.kv_channels,
            ),
            packed_seq_params=None,
            positions_by_item=batch.positions_by_item,
            source_positions_by_item=tuple(
                torch.arange(
                    int(positions.numel()),
                    dtype=torch.long,
                    device=positions.device,
                )
                for positions in batch.positions_by_item
            ),
        )

    def _prepare_context_parallel_forward(
        self,
        batch: _PackedForwardBatch,
        *,
        topology: "ParallelTopology",
    ) -> _PreparedPackedForward:
        from megatron.core import parallel_state as ps

        from art.megatron.context_parallel.runtime import (
            _dispatch_tensor,
            prepare_cp_micro,
        )
        from art.megatron.training.microbatches import (
            _context_parallel_config_for_provider,
        )
        from art.preprocessing.pack import PackedTensors

        assistant_mask = torch.ones_like(batch.tokens, dtype=torch.bool)
        sparse_micro: PackedTensors = {
            "tokens": batch.tokens,
            "group_ids": batch.group_ids,
            "parent_ids": batch.parent_ids,
            "input_pos": batch.position_ids,
            "assistant_mask": assistant_mask,
            "logprobs": torch.full_like(
                batch.tokens, float("nan"), dtype=torch.float32
            ),
            "advantages": torch.zeros_like(batch.tokens, dtype=torch.float32),
            "weights": assistant_mask.to(dtype=torch.float32),
            "pixel_values": [None],
            "image_grid_thw": [None],
            "moe_routing_replay": None,
        }
        handler = self._handler()
        prepared = prepare_cp_micro(
            micro=sparse_micro,
            topology=topology,
            config=_context_parallel_config_for_provider(self._provider(), self.device),
            cp_group=ps.get_context_parallel_group(check_initialized=False),
            cp_rank=ps.get_context_parallel_rank(),
            build_gdn_execution_spec=handler.build_gdn_execution_spec,
            target_device=self.device,
        )
        if prepared.rank_plan is None:
            raise RuntimeError("CP forward preparation did not return a rank plan")
        local_positions = _dispatch_tensor(
            torch.arange(
                int(batch.tokens.shape[1]),
                dtype=torch.long,
            ).unsqueeze(0),
            rank_plan=prepared.rank_plan,
            pad_value=-1,
            pad_multiple=prepared.pad_multiple,
        )
        local_position_pairs = tuple(
            _local_position_pairs(local_positions, positions)
            for positions in batch.positions_by_item
        )
        return _PreparedPackedForward(
            tokens=prepared.tensors.tokens,
            position_ids=prepared.tensors.input_pos,
            attention_state=cast("ArtContextParallelState", prepared.attention_state),
            packed_seq_params=prepared.packed_seq_params,
            positions_by_item=tuple(pair[0] for pair in local_position_pairs),
            source_positions_by_item=tuple(pair[1] for pair in local_position_pairs),
        )

    def _topology(self) -> "ParallelTopology":
        from art.megatron.train import _infer_parallel_topology

        return _infer_parallel_topology(self.runtime.model)

    def _gather_tensor_parallel_logits(self, logits: torch.Tensor) -> torch.Tensor:
        from megatron.core import parallel_state as ps

        if int(ps.get_tensor_model_parallel_world_size()) <= 1:
            return logits
        from megatron.core import tensor_parallel

        return cast(
            torch.Tensor,
            tensor_parallel.gather_from_tensor_model_parallel_region(logits),
        )

    def _configure_optimizer(self, params: AdamParams) -> None:
        optimizer = self._optimizer()
        config = cast("OptimizerConfig | None", optimizer.config)
        if config is not None:
            config.lr = params.learning_rate
            config.adam_beta1 = params.beta1
            config.adam_beta2 = params.beta2
            config.weight_decay = params.weight_decay
            config.clip_grad = params.grad_clip_norm
        for group in optimizer.param_groups:
            param_group = cast(MutableMapping[str, object], group)
            param_group["lr"] = params.learning_rate
            param_group["weight_decay"] = params.weight_decay
            if "betas" in param_group:
                param_group["betas"] = (params.beta1, params.beta2)

    def _scale_main_grads(self, scale: float) -> None:
        if scale == 1.0:
            return
        for chunk in self.runtime.model:
            for param in chunk.parameters():
                grad = getattr(param, "main_grad", None)
                if isinstance(grad, torch.Tensor):
                    grad.mul_(scale)
                elif param.grad is not None:
                    param.grad.mul_(scale)


def _as_1d_long(tensor: torch.Tensor, *, name: str) -> torch.Tensor:
    tensor = tensor.reshape(-1)
    if int(tensor.numel()) == 0:
        raise ValueError(f"{name} must not be empty")
    return tensor.to(dtype=torch.long)


def _as_target_tokens(
    tensor: torch.Tensor,
    input_tokens: torch.Tensor,
    input_ids: torch.Tensor,
) -> torch.Tensor:
    labels = tensor.to(dtype=torch.long)
    if int(labels.numel()) == 0:
        raise ValueError("target_tokens must not be empty")
    if tuple(labels.shape) == tuple(input_tokens.shape):
        return labels.reshape(-1)

    input_shape = tuple(input_tokens.shape)
    if (
        labels.ndim > input_tokens.ndim
        and tuple(labels.shape[: input_tokens.ndim]) == input_shape
    ):
        return labels.reshape(
            int(input_ids.numel()), *labels.shape[input_tokens.ndim :]
        )
    if labels.ndim >= 1 and int(labels.shape[0]) == int(input_ids.numel()):
        return labels
    raise ValueError(
        "target_tokens must match input_tokens or add trailing target dimensions: "
        f"input_tokens={tuple(input_tokens.shape)} target_tokens={tuple(labels.shape)}"
    )


def _validate_top_k(top_k: int | None, model: "GPTModel") -> None:
    if top_k is None:
        return
    if top_k < 1:
        raise ValueError("top_k must be >= 1")
    vocab_size = _padded_vocab_size(model)
    if top_k > vocab_size:
        raise ValueError(f"top_k={top_k} exceeds vocabulary size {vocab_size}")


def _request_mix_key(request: AnyForwardInput) -> str:
    parts = []
    if request.target_tokens is not None:
        target = request.target_tokens
        tail_shape = tuple(target.shape[request.input_tokens.ndim :])
        parts.append(f"target:{tail_shape or 'single'}")
    if request.top_k is not None:
        parts.append(f"topk:{int(request.top_k)}")
    if request.logits:
        parts.append("logits")
    if request.hidden_states:
        parts.append("hidden")
    return "+".join(parts) if parts else "inactive"


def _memory_profile_covers(profile: _MemoryProfile, *, packed_tokens: int) -> bool:
    return profile.packed_tokens * _MEMORY_PROFILE_TRUST_GROWTH >= packed_tokens


def _pack_forward_items(
    items: Sequence[_ForwardItem],
    *,
    max_depth: int,
) -> _PackedForwardBatch:
    input_tensors = tuple(item.input_ids for item in items)
    pack = pack_shared_prefixes(input_tensors, max_depth=max_depth)

    return _PackedForwardBatch(
        tokens=pack.tokens,
        group_ids=pack.group_ids,
        parent_ids=pack.parent_ids,
        position_ids=pack.position_ids,
        positions_by_item=pack.positions_by_sequence,
    )


def _pad_packed_batch(
    batch: _PackedForwardBatch,
    *,
    multiple: int,
) -> _PackedForwardBatch:
    if multiple <= 1:
        return batch
    seq_len = int(batch.tokens.shape[1])
    pad = -seq_len % multiple
    if pad == 0:
        return batch

    device = batch.tokens.device
    next_group = (
        int(batch.group_ids.max().item()) + 1 if int(batch.group_ids.numel()) else 1
    )
    pad_group_ids = torch.arange(
        next_group,
        next_group + pad,
        dtype=batch.group_ids.dtype,
        device=device,
    ).unsqueeze(0)
    return _PackedForwardBatch(
        tokens=torch.cat(
            (
                batch.tokens,
                torch.zeros((1, pad), dtype=batch.tokens.dtype, device=device),
            ),
            dim=1,
        ),
        group_ids=torch.cat((batch.group_ids, pad_group_ids), dim=1),
        parent_ids=torch.cat((batch.parent_ids, pad_group_ids), dim=1),
        position_ids=torch.cat(
            (
                batch.position_ids,
                torch.zeros((1, pad), dtype=batch.position_ids.dtype, device=device),
            ),
            dim=1,
        ),
        positions_by_item=batch.positions_by_item,
    )


def _language_model(model: torch.nn.Module) -> "GPTModel":
    module: object = model
    while hasattr(module, "module"):
        module = getattr(module, "module")
    if hasattr(module, "_preprocess") and hasattr(module, "decoder"):
        return cast("GPTModel", module)
    language_model = getattr(module, "language_model", None)
    if language_model is not None:
        return cast("GPTModel", language_model)
    raise RuntimeError("expected a Megatron GPT model")


def _padded_vocab_size(model: "GPTModel") -> int:
    vocab_size = getattr(getattr(model, "config", None), "padded_vocab_size", None)
    if vocab_size is None:
        vocab_size = getattr(model, "vocab_size", None)
    if vocab_size is None:
        raise RuntimeError("could not determine full padded vocabulary size")
    return int(vocab_size)


def _hidden_size(model: "GPTModel | None", provider: object) -> int:
    for source in (getattr(model, "config", None), model, provider):
        if source is None:
            continue
        hidden_size = getattr(source, "hidden_size", None)
        if hidden_size is not None:
            return int(hidden_size)
    raise RuntimeError("could not determine hidden size")


def _dtype_size(dtype: torch.dtype) -> int:
    return torch.empty((), dtype=dtype).element_size()


def _target_logprobs_from_full_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
    log_z: torch.Tensor,
) -> torch.Tensor:
    flat_labels = labels.clamp_min(0).reshape(int(labels.shape[0]), -1)
    target_logits = logits.gather(1, flat_labels).float().reshape(labels.shape)
    return _finish_target_logprobs(target_logits, labels, log_z)


def _vocab_parallel_target_logprobs(
    local_logits: torch.Tensor,
    labels: torch.Tensor,
    log_z: torch.Tensor,
    *,
    row_offsets: torch.Tensor,
) -> torch.Tensor:
    start, _ = _vocab_range(local_logits)
    target_logits = _call_compiled(
        _owned_target_logits_for_rows,
        local_logits,
        labels,
        start,
        row_offsets,
    )
    target_logits = _all_reduce_tensor_parallel_sum(target_logits)
    return _call_compiled(_finish_target_logprobs, target_logits, labels, log_z)


def _owned_target_logits_for_rows(
    local_logits: torch.Tensor,
    labels: torch.Tensor,
    vocab_start: int,
    row_offsets: torch.Tensor,
) -> torch.Tensor:
    flat_labels = labels.reshape(int(labels.shape[0]), -1)
    local_labels = flat_labels - vocab_start
    owns_label = (
        (flat_labels != -100)
        & (local_labels >= 0)
        & (local_labels < int(local_logits.shape[1]))
    )
    rows = row_offsets.reshape(int(row_offsets.shape[0]), 1).expand_as(flat_labels)
    selected = local_logits[
        rows,
        local_labels.clamp(0, int(local_logits.shape[1]) - 1),
    ].float()
    return selected.masked_fill(~owns_label, 0.0).reshape(labels.shape)


def _finish_target_logprobs(
    target_logits: torch.Tensor,
    labels: torch.Tensor,
    log_z: torch.Tensor,
) -> torch.Tensor:
    log_z = log_z.reshape(int(log_z.shape[0]), *((1,) * (int(labels.ndim) - 1)))
    return (target_logits.float() - log_z).masked_fill(labels == -100, 0.0)


def _consistent_row_labels(
    label_rows: Sequence[torch.Tensor | None],
    row_matches: Sequence[_RowMatch],
    *,
    row_count: int,
    device: torch.device,
) -> torch.Tensor | None:
    labels = torch.full(
        (row_count,),
        -100,
        dtype=torch.long,
        device=device,
    )
    has_label = torch.zeros_like(labels, dtype=torch.bool)
    for item_labels, match in zip(label_rows, row_matches, strict=True):
        if item_labels is None:
            continue
        if int(match.source_offsets.numel()) == 0:
            continue
        selected_labels = item_labels.index_select(0, match.source_offsets)
        keep = selected_labels != -100
        if not bool(keep.any().item()):
            continue
        kept_row_offsets = match.row_offsets[keep]
        kept_labels = selected_labels[keep]
        existing = labels.index_select(0, kept_row_offsets)
        seen = has_label.index_select(0, kept_row_offsets)
        if bool(((existing != kept_labels) & seen).any().item()):
            return None
        labels.index_copy_(0, kept_row_offsets, kept_labels)
        has_label.index_fill_(0, kept_row_offsets, True)
    return labels


def _scatter_row_target_logprobs(
    row_target_logprobs: torch.Tensor,
    row_matches: Sequence[_RowMatch],
    label_rows: Sequence[torch.Tensor | None],
    target_logprobs: list[torch.Tensor | None],
) -> None:
    for match, labels, item_logprobs in zip(
        row_matches,
        label_rows,
        target_logprobs,
        strict=True,
    ):
        if labels is None or item_logprobs is None:
            continue
        if int(match.source_offsets.numel()) == 0:
            continue
        selected = row_target_logprobs.index_select(0, match.row_offsets)
        selected_labels = labels.index_select(0, match.source_offsets)
        item_logprobs[match.source_offsets] = selected.masked_fill(
            selected_labels == -100,
            0.0,
        )


def _anchor_disconnected_outputs(
    target_logprobs: list[torch.Tensor | None],
    top_k: list[TopK | None],
    hidden_by_row: torch.Tensor,
) -> tuple[list[torch.Tensor | None], list[TopK | None]]:
    if not hidden_by_row.requires_grad:
        return target_logprobs, top_k
    anchor: torch.Tensor | None = None

    def anchor_tensor(tensor: torch.Tensor) -> torch.Tensor:
        nonlocal anchor
        if tensor.requires_grad:
            return tensor
        if anchor is None:
            anchor = hidden_by_row.reshape(-1)[:1].float().sum() * 0.0
        return tensor + anchor

    return (
        [
            None if item_logprobs is None else anchor_tensor(item_logprobs)
            for item_logprobs in target_logprobs
        ],
        [
            None
            if item_top_k is None
            else TopK(
                logprobs=anchor_tensor(item_top_k.logprobs),
                tokens=item_top_k.tokens,
            )
            for item_top_k in top_k
        ],
    )


def _vocab_parallel_topk(
    local_logits: torch.Tensor,
    *,
    k: int,
    log_z: torch.Tensor,
) -> TopK:
    start, _ = _vocab_range(local_logits)
    local_k = min(k, int(local_logits.shape[1]))
    local_values, local_tokens = torch.topk(local_logits.float(), k=local_k, dim=-1)
    return _vocab_parallel_topk_from_local(
        local_values,
        local_tokens,
        k=k,
        log_z=log_z,
        vocab_start=start,
    )


def _try_triton_local_topk_stats(
    local_logits: torch.Tensor,
    *,
    k: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None:
    if k <= 0:
        return None
    if k > _triton_fused_topk_max():
        return None
    if not local_logits.is_cuda:
        return None
    if _triton_topk_disabled():
        return None
    if int(local_logits.shape[0]) < _triton_min_rows():
        return None
    try:
        from art.megatron.trainer_rank_topk import local_topk_stats

        stats = local_topk_stats(
            local_logits,
            k=min(k, int(local_logits.shape[1])),
        )
    except Exception:
        if _triton_topk_strict():
            raise
        return None
    return stats.local_max, stats.local_sum, stats.values, stats.tokens


def _try_triton_local_logsumexp_stats(
    local_logits: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    if not local_logits.is_cuda:
        return None
    if _triton_topk_disabled():
        return None
    if int(local_logits.shape[0]) < _triton_min_rows():
        return None
    try:
        from art.megatron.trainer_rank_topk import local_logsumexp_stats

        stats = local_logsumexp_stats(local_logits)
    except Exception:
        if _triton_topk_strict():
            raise
        return None
    return stats.local_max, stats.local_sum


def _triton_topk_disabled() -> bool:
    return os.environ.get("ART_TRAINER_RANK_TRITON_TOPK", "1").lower() in {
        "0",
        "false",
    }


def _triton_topk_strict() -> bool:
    return os.environ.get("ART_TRAINER_RANK_TRITON_TOPK", "1").lower() == "strict"


def _triton_fused_topk_max() -> int:
    return int(os.environ.get("ART_TRAINER_RANK_TRITON_FUSED_TOPK_MAX", "10"))


def _triton_min_rows() -> int:
    return int(os.environ.get("ART_TRAINER_RANK_TRITON_MIN_ROWS", "64"))


def _vocab_parallel_topk_from_local(
    local_values: torch.Tensor,
    local_tokens: torch.Tensor,
    *,
    k: int,
    log_z: torch.Tensor,
    vocab_start: int,
) -> TopK:
    local_k = min(k, int(local_values.shape[1]))
    local_values = local_values[:, :local_k] - log_z.unsqueeze(1)
    local_tokens = local_tokens[:, :local_k] + vocab_start

    from megatron.core import parallel_state as ps

    tp_size = int(ps.get_tensor_model_parallel_world_size())
    if tp_size <= 1:
        if k > int(local_values.shape[1]):
            raise ValueError(
                f"top_k={k} exceeds vocabulary size {int(local_values.shape[1])}"
            )
        return TopK(logprobs=local_values, tokens=local_tokens)

    from torch.distributed.nn.functional import all_gather

    group = ps.get_tensor_model_parallel_group(check_initialized=False)
    gathered_values = cast(tuple[torch.Tensor, ...], all_gather(local_values, group))
    gathered_tokens = [torch.empty_like(local_tokens) for _ in range(tp_size)]
    dist.all_gather(gathered_tokens, local_tokens, group=group)
    values = torch.cat(gathered_values, dim=1)
    tokens = torch.cat(gathered_tokens, dim=1)
    if k > int(values.shape[1]):
        raise ValueError(f"top_k={k} exceeds vocabulary size {int(values.shape[1])}")
    top_values, top_offsets = torch.topk(values, k=k, dim=-1)
    return TopK(logprobs=top_values, tokens=tokens.gather(1, top_offsets))


def _merge_topk(
    current: TopK | None,
    offsets: torch.Tensor,
    values: TopK,
    *,
    length: int,
) -> TopK:
    if current is None:
        current = TopK(
            logprobs=torch.empty(
                (length, int(values.logprobs.shape[1])),
                device=values.logprobs.device,
                dtype=values.logprobs.dtype,
            ),
            tokens=torch.empty(
                (length, int(values.tokens.shape[1])),
                device=values.tokens.device,
                dtype=values.tokens.dtype,
            ),
        )
    current.logprobs[offsets] = values.logprobs
    current.tokens[offsets] = values.tokens
    return current


def _vocab_parallel_log_z(local_logits: torch.Tensor) -> torch.Tensor:
    local_logits = local_logits.float()
    local_max = local_logits.max(dim=-1).values.detach()
    global_max = _all_reduce_tensor_parallel_max(local_max)
    local_sum = _call_compiled(_local_vocab_exp_sum, local_logits, global_max)
    global_sum = _all_reduce_tensor_parallel_sum(local_sum)
    return global_max + torch.log(global_sum)


def _local_vocab_exp_sum(
    local_logits: torch.Tensor,
    global_max: torch.Tensor,
) -> torch.Tensor:
    return torch.exp(local_logits.float() - global_max.unsqueeze(1)).sum(dim=-1)


def _vocab_range(local_logits: torch.Tensor) -> tuple[int, int]:
    from megatron.core import parallel_state as ps

    local_size = int(local_logits.shape[1])
    rank = int(ps.get_tensor_model_parallel_rank())
    start = rank * local_size
    return start, start + local_size


def _all_reduce_tensor_parallel_sum(tensor: torch.Tensor) -> torch.Tensor:
    from megatron.core import parallel_state as ps

    if int(ps.get_tensor_model_parallel_world_size()) <= 1:
        return tensor
    from torch.distributed.nn.functional import all_reduce

    return cast(
        torch.Tensor,
        all_reduce(
            tensor,
            op=dist.ReduceOp.SUM,
            group=ps.get_tensor_model_parallel_group(check_initialized=False),
        ),
    )


def _all_reduce_tensor_parallel_max(tensor: torch.Tensor) -> torch.Tensor:
    from megatron.core import parallel_state as ps

    if int(ps.get_tensor_model_parallel_world_size()) <= 1:
        return tensor
    output = tensor.clone()
    dist.all_reduce(
        output,
        op=dist.ReduceOp.MAX,
        group=ps.get_tensor_model_parallel_group(check_initialized=False),
    )
    return output


def _call_compiled(fn: Callable[P, R], *args: P.args, **kwargs: P.kwargs) -> R:
    if os.environ.get("ART_TRAINER_RANK_COMPILE", "0").lower() in {"0", "false"}:
        return fn(*args, **kwargs)
    compiled = _COMPILED_FUNCTIONS.get(fn)
    if compiled is None:
        compiled = cast(Callable[..., object], torch.compile(fn, dynamic=True))
        _COMPILED_FUNCTIONS[fn] = compiled
    try:
        return cast(Callable[P, R], compiled)(*args, **kwargs)
    except Exception:
        return fn(*args, **kwargs)


def _matching_offsets(
    positions: torch.Tensor,
    chunk_rows: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if int(positions.numel()) == 0 or int(chunk_rows.numel()) == 0:
        empty = torch.empty(0, dtype=torch.long, device=positions.device)
        return empty, empty
    sorted_rows, order = chunk_rows.sort()
    indices = torch.searchsorted(sorted_rows, positions)
    in_bounds = indices < int(sorted_rows.numel())
    source_offsets = torch.arange(
        int(positions.numel()),
        device=positions.device,
        dtype=torch.long,
    )[in_bounds]
    found = indices[in_bounds]
    keep = sorted_rows.index_select(0, found) == positions.index_select(
        0,
        source_offsets,
    )
    return source_offsets[keep], order.index_select(0, found[keep])


def _row_matches_by_item(
    positions_by_item: Sequence[torch.Tensor],
    rows: torch.Tensor,
    *,
    device: torch.device,
) -> tuple[_RowMatch, ...]:
    return tuple(
        _row_match(positions.to(device=device), rows) for positions in positions_by_item
    )


def _row_match(positions: torch.Tensor, rows: torch.Tensor) -> _RowMatch:
    source_offsets, row_offsets = _matching_offsets(positions, rows)
    if int(row_offsets.numel()) > 1:
        order = row_offsets.argsort()
        source_offsets = source_offsets.index_select(0, order)
        row_offsets = row_offsets.index_select(0, order)
    return _RowMatch(source_offsets=source_offsets, row_offsets=row_offsets)


def _match_chunk_offsets(
    match: _RowMatch,
    *,
    start: int,
    end: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    keep = (match.row_offsets >= start) & (match.row_offsets < end)
    source_offsets = match.source_offsets[keep]
    return source_offsets, match.row_offsets[keep] - start


def _local_position_pairs(
    local_global_positions: torch.Tensor,
    item_positions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    flat = local_global_positions.reshape(-1).to(device=item_positions.device)
    local_positions = torch.nonzero(flat >= 0, as_tuple=False).reshape(-1)
    global_positions = flat.index_select(0, local_positions)
    source_offsets, local_offsets = _matching_offsets(item_positions, global_positions)
    return (
        local_positions.index_select(0, local_offsets).to("cpu"),
        source_offsets.to("cpu"),
    )


def _select_positions(values: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    if int(positions.numel()) == 0:
        return values[:0]
    return values.index_select(0, positions.to(device=values.device))


def _batch_seq_logits(logits: torch.Tensor, *, seq_len: int) -> torch.Tensor:
    if int(logits.ndim) != 3:
        raise RuntimeError(
            f"expected logits with shape [B, S, V] or [S, B, V], got {tuple(logits.shape)}"
        )
    if int(logits.shape[0]) == 1 and int(logits.shape[1]) == seq_len:
        return logits
    if int(logits.shape[0]) == seq_len and int(logits.shape[1]) == 1:
        return logits.transpose(0, 1).contiguous()
    raise RuntimeError(
        f"logits do not match sequence length {seq_len}: {tuple(logits.shape)}"
    )


def _materialize(inputs: ForwardInputs) -> ForwardInputs:
    if isinstance(inputs, ForwardInput):
        return inputs
    return [_materialize(item) for item in _nested_forward_children(inputs)]


def _flatten(inputs: ForwardInputs) -> Iterator[AnyForwardInput]:
    if isinstance(inputs, ForwardInput):
        yield inputs
        return
    for item in _nested_forward_children(inputs):
        yield from _flatten(item)


def _unflatten(
    template: ForwardInputs, outputs: Iterator[AnyForwardOutput]
) -> ForwardOutputs:
    if isinstance(template, ForwardInput):
        return next(outputs)
    return [_unflatten(item, outputs) for item in _nested_forward_children(template)]


def _nested_forward_children(inputs: ForwardInputs) -> Iterator[ForwardInputs]:
    if isinstance(inputs, Mapping):
        raise TypeError(
            "dict was passed directly to TrainerRank; gather or materialize the "
            "values into a list/tuple so nested forward output ordering is explicit"
        )
    if isinstance(inputs, str | bytes):
        raise TypeError(
            "TrainerRank forward inputs must be ForwardInput objects or nested "
            "iterables of ForwardInput objects, not strings"
        )
    try:
        return iter(cast(Iterable[ForwardInputs], inputs))
    except TypeError as exc:
        raise TypeError(
            "TrainerRank forward inputs must be ForwardInput objects or nested "
            "iterables of ForwardInput objects"
        ) from exc


__all__ = [
    "AdamParams",
    "ForwardInput",
    "ForwardOutput",
    "MicroBatch",
    "MicroBatchStats",
    "TopK",
    "TrainerRank",
    "TrainerRankMemoryError",
]
