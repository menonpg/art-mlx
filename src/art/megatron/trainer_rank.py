from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, MutableMapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from itertools import zip_longest
import os
from typing import TYPE_CHECKING, Generic, Literal, ParamSpec, TypeVar, cast, overload

import torch
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
import torch.distributed as dist

from art.megatron.shared_prefix_packing import pack_shared_prefixes

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

    @overload
    def __new__(
        cls,
        *,
        input_tokens: torch.Tensor,
        target_tokens: None = None,
        top_k: None = None,
        logits: Literal[False] = False,
        hidden_states: Literal[False] = False,
    ) -> "ForwardInput[None, None, None, None]": ...

    @overload
    def __new__(
        cls,
        *,
        input_tokens: torch.Tensor,
        target_tokens: torch.Tensor,
        top_k: None = None,
        logits: Literal[False] = False,
        hidden_states: Literal[False] = False,
    ) -> "ForwardInput[torch.Tensor, None, None, None]": ...

    @overload
    def __new__(
        cls,
        *,
        input_tokens: torch.Tensor,
        target_tokens: None = None,
        top_k: int,
        logits: Literal[False] = False,
        hidden_states: Literal[False] = False,
    ) -> "ForwardInput[None, TopK, None, None]": ...

    @overload
    def __new__(
        cls,
        *,
        input_tokens: torch.Tensor,
        target_tokens: None = None,
        top_k: None = None,
        logits: Literal[True],
        hidden_states: Literal[False] = False,
    ) -> "ForwardInput[None, None, torch.Tensor, None]": ...

    @overload
    def __new__(
        cls,
        *,
        input_tokens: torch.Tensor,
        target_tokens: None = None,
        top_k: None = None,
        logits: Literal[False] = False,
        hidden_states: Literal[True],
    ) -> "ForwardInput[None, None, None, torch.Tensor]": ...

    @overload
    def __new__(
        cls,
        *,
        input_tokens: torch.Tensor,
        target_tokens: torch.Tensor,
        top_k: int,
        logits: Literal[False] = False,
        hidden_states: Literal[False] = False,
    ) -> "ForwardInput[torch.Tensor, TopK, None, None]": ...

    @overload
    def __new__(
        cls,
        *,
        input_tokens: torch.Tensor,
        target_tokens: torch.Tensor,
        top_k: None = None,
        logits: Literal[True],
        hidden_states: Literal[False] = False,
    ) -> "ForwardInput[torch.Tensor, None, torch.Tensor, None]": ...

    @overload
    def __new__(
        cls,
        *,
        input_tokens: torch.Tensor,
        target_tokens: torch.Tensor,
        top_k: None = None,
        logits: Literal[False] = False,
        hidden_states: Literal[True],
    ) -> "ForwardInput[torch.Tensor, None, None, torch.Tensor]": ...

    @overload
    def __new__(
        cls,
        *,
        input_tokens: torch.Tensor,
        target_tokens: None = None,
        top_k: int,
        logits: Literal[True],
        hidden_states: Literal[False] = False,
    ) -> "ForwardInput[None, TopK, torch.Tensor, None]": ...

    @overload
    def __new__(
        cls,
        *,
        input_tokens: torch.Tensor,
        target_tokens: None = None,
        top_k: int,
        logits: Literal[False] = False,
        hidden_states: Literal[True],
    ) -> "ForwardInput[None, TopK, None, torch.Tensor]": ...

    @overload
    def __new__(
        cls,
        *,
        input_tokens: torch.Tensor,
        target_tokens: None = None,
        top_k: None = None,
        logits: Literal[True],
        hidden_states: Literal[True],
    ) -> "ForwardInput[None, None, torch.Tensor, torch.Tensor]": ...

    @overload
    def __new__(
        cls,
        *,
        input_tokens: torch.Tensor,
        target_tokens: torch.Tensor,
        top_k: int,
        logits: Literal[True],
        hidden_states: Literal[False] = False,
    ) -> "ForwardInput[torch.Tensor, TopK, torch.Tensor, None]": ...

    @overload
    def __new__(
        cls,
        *,
        input_tokens: torch.Tensor,
        target_tokens: torch.Tensor,
        top_k: int,
        logits: Literal[False] = False,
        hidden_states: Literal[True],
    ) -> "ForwardInput[torch.Tensor, TopK, None, torch.Tensor]": ...

    @overload
    def __new__(
        cls,
        *,
        input_tokens: torch.Tensor,
        target_tokens: torch.Tensor,
        top_k: None = None,
        logits: Literal[True],
        hidden_states: Literal[True],
    ) -> "ForwardInput[torch.Tensor, None, torch.Tensor, torch.Tensor]": ...

    @overload
    def __new__(
        cls,
        *,
        input_tokens: torch.Tensor,
        target_tokens: None = None,
        top_k: int,
        logits: Literal[True],
        hidden_states: Literal[True],
    ) -> "ForwardInput[None, TopK, torch.Tensor, torch.Tensor]": ...

    @overload
    def __new__(
        cls,
        *,
        input_tokens: torch.Tensor,
        target_tokens: torch.Tensor,
        top_k: int,
        logits: Literal[True],
        hidden_states: Literal[True],
    ) -> "ForwardInput[torch.Tensor, TopK, torch.Tensor, torch.Tensor]": ...

    @overload
    def __new__(
        cls,
        *,
        input_tokens: torch.Tensor,
        target_tokens: torch.Tensor | None = None,
        top_k: int | None = None,
        logits: bool = False,
        hidden_states: bool = False,
        checkpoint: AdapterSelection = Unset,
        lora: AdapterSelection = Unset,
    ) -> "AnyForwardInput": ...

    def __new__(
        cls,
        *,
        input_tokens: torch.Tensor,
        target_tokens: torch.Tensor | None = None,
        top_k: int | None = None,
        logits: bool = False,
        hidden_states: bool = False,
        checkpoint: AdapterSelection = Unset,
        lora: AdapterSelection = Unset,
    ) -> "AnyForwardInput":
        return super().__new__(cls)


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
    indices: Sequence[int]

    def select(self, xs: Sequence[T]) -> Sequence[T]:
        return [xs[i] for i in self.indices]


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
class _HeadOutputs:
    target_logprobs: list[torch.Tensor | None]
    top_k: list[TopK | None]
    logits: list[torch.Tensor | None]


@dataclass(frozen=True)
class _RowMatch:
    source_offsets: torch.Tensor
    row_offsets: torch.Tensor


class TrainerRank:
    def __init__(
        self,
        runtime: TrainingRuntime,
        *,
        micro_batch_size: int = 1,
        head_chunk_tokens: int = 512,
        shared_prefix_max_depth: int = 1,
    ) -> None:
        if micro_batch_size < 1:
            raise ValueError("micro_batch_size must be >= 1")
        if head_chunk_tokens < 1:
            raise ValueError("head_chunk_tokens must be >= 1")
        if shared_prefix_max_depth < 0:
            raise ValueError("shared_prefix_max_depth must be >= 0")
        self.runtime: TrainingRuntime = runtime
        self.micro_batch_size = micro_batch_size
        self.head_chunk_tokens = head_chunk_tokens
        self.shared_prefix_max_depth = shared_prefix_max_depth
        self.device = next(runtime.model[0].parameters()).device
        self._default_slot_ref: LoRASlotRef | None = None
        self._slot_stack: list[LoRASlotRef] = []
        self._dynamic_optimizers: dict[str, torch.optim.Optimizer] = {}
        self._checkpoint_slot_names: set[str] = set()
        self.zero_grad()

    def zero_grad(self) -> None:
        for chunk in self.runtime.model:
            zero_grad_buffer = getattr(chunk, "zero_grad_buffer", None)
            if callable(zero_grad_buffer):
                zero_grad_buffer()
        optimizer = cast("MegatronOptimizer | None", self.runtime.optimizer)
        if optimizer is not None:
            optimizer.zero_grad()
        for name in self._checkpoint_slot_names:
            for param in self._checkpoint_slot_params(name):
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
        self._validate_dynamic_slot_consistency("checkpoint", name, loaded)
        self._checkpoint_slot_names.add(name)
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
    ) -> None:
        if not (dist.is_available() and dist.is_initialized()):
            return

        from art.megatron.lora import iter_lora_slot_parameters

        ref = self._slot_ref(kind, name)
        params = list(iter_lora_slot_parameters(self.runtime.model, ref))
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
            return

        first_mismatch = None
        for left, right in zip_longest(
            cast(list[object], reference["signature"]),
            cast(list[object], mismatched[0]["signature"]),
            fillvalue=None,
        ):
            if left != right:
                first_mismatch = {"expected": left, "actual": right}
                break
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
            f"Rank summary: {summary}. First mismatch: {first_mismatch}."
        )

    def _resolve_slot_ref(self, request: AnyForwardInput) -> "LoRASlotRef | None":
        if request.checkpoint is not Unset:
            return self._slot_ref("checkpoint", cast(str | None, request.checkpoint))
        if request.lora is not Unset:
            return self._slot_ref("lora", cast(str | None, request.lora))
        if self._slot_stack:
            return self._slot_stack[-1]
        return self._default_slot_ref

    def _set_current_slot(self, ref: "LoRASlotRef | None") -> object:
        from art.megatron.lora import set_lora_slot_context

        return set_lora_slot_context(ref)

    def _reset_current_slot(self, token: object) -> None:
        from art.megatron.lora import reset_lora_slot_context

        reset_lora_slot_context(token)  # type: ignore[arg-type]

    @contextmanager
    def _use_slot(self, ref: "LoRASlotRef | None") -> Iterator[None]:
        token = self._set_current_slot(ref)
        try:
            yield
        finally:
            self._reset_current_slot(token)

    def micro_batches(
        self,
        inputs: Iterable[ForwardInputsT],
    ) -> Sequence[MicroBatch[ForwardInputsT]]:
        items = list(inputs)
        from megatron.core import parallel_state as ps

        dp_rank = int(ps.get_data_parallel_rank())
        dp_size = int(ps.get_data_parallel_world_size())
        global_micro_size = self.micro_batch_size * dp_size
        batches: list[MicroBatch[ForwardInputsT]] = []
        for start in range(0, len(items), global_micro_size):
            stop = min(start + global_micro_size, len(items))
            indices = list(range(start + dp_rank, stop, dp_size))
            batches.append(MicroBatch([items[i] for i in indices], indices))
        return batches

    @overload
    def forward(
        self,
        inputs: Iterable[ForwardInput[LogprobsT, TopKT, LogitsT, HiddenStatesT]],
    ) -> Sequence[ForwardOutput[LogprobsT, TopKT, LogitsT, HiddenStatesT]]: ...

    @overload
    def forward(
        self,
        inputs: Iterable[
            Iterable[ForwardInput[LogprobsT, TopKT, LogitsT, HiddenStatesT]]
        ],
    ) -> Sequence[
        Sequence[ForwardOutput[LogprobsT, TopKT, LogitsT, HiddenStatesT]]
    ]: ...

    @overload
    def forward(
        self,
        inputs: Iterable[
            Iterable[Iterable[ForwardInput[LogprobsT, TopKT, LogitsT, HiddenStatesT]]]
        ],
    ) -> Sequence[
        Sequence[Sequence[ForwardOutput[LogprobsT, TopKT, LogitsT, HiddenStatesT]]]
    ]: ...

    @overload
    def forward(
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

    def forward(self, inputs: ForwardInputs) -> ForwardOutputs:
        materialized = _materialize(inputs)
        outputs = iter(self._forward_flat(list(_flatten(materialized))))
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
            unknown = set(checkpoints) - self._checkpoint_slot_names
            if unknown:
                raise ValueError(f"Unknown checkpoint slots: {sorted(unknown)}")
            return tuple(dict.fromkeys(checkpoints))
        names = []
        for name in sorted(self._checkpoint_slot_names):
            local_has_grad = any(
                param.grad is not None for param in self._checkpoint_slot_params(name)
            )
            has_grad = torch.tensor(
                int(local_has_grad),
                device=self.device,
                dtype=torch.int32,
            )
            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(has_grad, op=dist.ReduceOp.MAX)
            if bool(has_grad.item()):
                names.append(name)
        return tuple(names)

    def _dynamic_optim_step(
        self,
        checkpoint_names: Sequence[str],
        *,
        params: AdamParams,
        scale_grads: float,
    ) -> dict[str, float]:
        all_params: list[torch.nn.Parameter] = []
        for name in checkpoint_names:
            slot_params = self._checkpoint_slot_params(name)
            self._ensure_dynamic_grads(slot_params)
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
                self._checkpoint_slot_params(name),
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

    def _checkpoint_slot_params(self, name: str) -> list[torch.nn.Parameter]:
        from art.megatron.lora import iter_lora_slot_parameters

        return list(
            iter_lora_slot_parameters(
                self.runtime.model,
                self._slot_ref("checkpoint", name),
            )
        )

    @staticmethod
    def _ensure_dynamic_grads(params: Sequence[torch.nn.Parameter]) -> None:
        for param in params:
            if param.grad is None:
                param.grad = torch.zeros_like(param)

    def _reduce_dynamic_grads(self, params: Sequence[torch.nn.Parameter]) -> None:
        from megatron.core import parallel_state as ps

        buckets: list[
            tuple[
                object,
                dist.ReduceOp.RedOpType,
                torch.dtype,
                torch.device,
                list[torch.Tensor],
            ]
        ] = []

        def add_to_bucket(
            *,
            group: object,
            op: dist.ReduceOp.RedOpType,
            grad: torch.Tensor,
        ) -> None:
            for (
                bucket_group,
                bucket_op,
                bucket_dtype,
                bucket_device,
                bucket_grads,
            ) in buckets:
                if (
                    bucket_group is group
                    and bucket_op == op
                    and bucket_dtype == grad.dtype
                    and bucket_device == grad.device
                ):
                    bucket_grads.append(grad)
                    return
            buckets.append((group, op, grad.dtype, grad.device, [grad]))

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

        for group, op, _dtype, _device, grads in buckets:
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

    def _forward_flat(
        self, requests: Sequence[AnyForwardInput]
    ) -> list[AnyForwardOutput]:
        outputs = [
            ForwardOutput(
                target_logprobs=None,
                top_k=None,
                logits=None,
                hidden_states=None,
            )
            for _ in requests
        ]
        active_indices = [
            index
            for index, request in enumerate(requests)
            if request.target_tokens is not None
            or request.logits
            or request.top_k is not None
            or request.hidden_states
        ]
        if not active_indices:
            return outputs

        groups: dict[LoRASlotRef | None, list[int]] = {}
        for index in active_indices:
            groups.setdefault(self._resolve_slot_ref(requests[index]), []).append(index)

        for slot_ref, group_indices in groups.items():
            items = [self._forward_item(requests[index]) for index in group_indices]
            packed = _pack_forward_items(items, max_depth=self.shared_prefix_max_depth)
            with self._use_slot(slot_ref):
                prepared = self._prepare_packed_forward(packed)
                item_outputs = self._forward_packed(items, prepared)
            for index, output in zip(group_indices, item_outputs, strict=True):
                outputs[index] = output
        return outputs

    def _forward_item(self, request: AnyForwardInput) -> _ForwardItem:
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
        if _is_native_target_only(items):
            labels = self._consistent_packed_labels(items, prepared)
            if labels is not None:
                return self._forward_native_target_logprobs(items, prepared, labels)

        hidden_by_row = self._gather_sequence_parallel_hidden(
            self._decoder_hidden(prepared)
        )
        head_outputs = self._project_head(items, prepared, hidden_by_row)
        outputs: list[AnyForwardOutput] = []
        for index, (item, positions) in enumerate(
            zip(items, prepared.positions_by_item, strict=True)
        ):
            hidden_states = (
                _select_positions(hidden_by_row, positions)
                if item.request.hidden_states
                else None
            )
            outputs.append(
                ForwardOutput(
                    target_logprobs=head_outputs.target_logprobs[index],
                    top_k=head_outputs.top_k[index],
                    logits=head_outputs.logits[index],
                    hidden_states=hidden_states,
                )
            )
        return outputs

    def _forward_native_target_logprobs(
        self,
        items: Sequence[_ForwardItem],
        prepared: _PreparedPackedForward,
        labels: torch.Tensor,
    ) -> list[AnyForwardOutput]:
        from art.megatron.train import _placeholder_attention_mask

        per_token_loss = self.runtime.model[0](
            input_ids=prepared.tokens,
            position_ids=prepared.position_ids,
            attention_mask=_placeholder_attention_mask(self.device),
            labels=labels,
            packed_seq_params=prepared.packed_seq_params,
            **self._handler().get_forward_kwargs(
                self.runtime.model[0],
                attention_bias=prepared.attention_state,
            ),
        )
        flat_logprobs = -per_token_loss.reshape(-1)
        outputs: list[AnyForwardOutput] = []
        for item, positions, source_positions in zip(
            items,
            prepared.positions_by_item,
            prepared.source_positions_by_item,
            strict=True,
        ):
            if item.labels is None:
                raise RuntimeError("native target path requires labels")
            item_labels = item.labels.to(device=self.device).index_select(
                0,
                source_positions.to(device=self.device),
            )
            target_logprobs = _select_positions(flat_logprobs, positions).masked_fill(
                item_labels == -100,
                0.0,
            )
            outputs.append(
                ForwardOutput(
                    target_logprobs=target_logprobs,
                    top_k=None,
                    logits=None,
                    hidden_states=None,
                )
            )
        return outputs

    def _consistent_packed_labels(
        self,
        items: Sequence[_ForwardItem],
        prepared: _PreparedPackedForward,
    ) -> torch.Tensor | None:
        labels = torch.full_like(prepared.tokens, -100)
        flat_labels = labels.reshape(-1)
        has_label = torch.zeros_like(flat_labels, dtype=torch.bool)
        for item, positions, source_positions in zip(
            items,
            prepared.positions_by_item,
            prepared.source_positions_by_item,
            strict=True,
        ):
            if item.labels is None:
                continue
            item_positions = positions.to(device=labels.device)
            item_labels = item.labels.to(device=labels.device).index_select(
                0,
                source_positions.to(device=labels.device),
            )
            keep = item_labels != -100
            if not bool(keep.any().item()):
                continue
            kept_positions = item_positions[keep]
            kept_labels = item_labels[keep]
            existing = flat_labels.index_select(0, kept_positions)
            seen = has_label.index_select(0, kept_positions)
            if bool(((existing != kept_labels) & seen).any().item()):
                return None
            flat_labels.index_copy_(0, kept_positions, kept_labels)
            has_label.index_fill_(0, kept_positions, True)
        return labels

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
    ) -> "_HeadOutputs":
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
                    valid_offsets = _valid_target_offsets(labels)
                    if int(valid_offsets.numel()):
                        local_rows.append(positions.index_select(0, valid_offsets))
            if item.request.logits:
                logits[index] = _empty_logits_like_positions(
                    positions,
                    model,
                    hidden_by_row,
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

        return _HeadOutputs(target_logprobs, top_k, logits)

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
                    item_logprobs[offsets] = _target_logprobs_from_full_logits(
                        selected_logits,
                        labels.index_select(0, offsets),
                        log_z.index_select(0, chunk_offsets),
                    )
                k = item.request.top_k
                if k is not None:
                    if log_z is None:
                        raise RuntimeError("top_k requires logsumexp")
                    top_k[index] = _merge_topk(
                        top_k[index],
                        offsets,
                        _topk_from_full_logits(
                            selected_logits,
                            k=k,
                            log_z=log_z.index_select(0, chunk_offsets),
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
        use_fused_target_ce = _can_use_fused_target_ce(items, label_rows)
        fused_target_labels = (
            _consistent_row_labels(
                label_rows,
                row_matches,
                row_count=int(rows.numel()),
                device=rows.device,
            )
            if use_fused_target_ce
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

        reference_target_labels = (
            _reference_row_labels(
                label_rows,
                row_matches,
                row_count=int(rows.numel()),
                device=rows.device,
            )
            if _can_use_reference_target_ce(items, label_rows)
            else None
        )
        if reference_target_labels is not None:
            for start in range(0, int(rows.numel()), self.head_chunk_tokens):
                chunk_rows = rows[start : start + self.head_chunk_tokens]
                local_logits = self._local_logits_from_hidden_rows(
                    model,
                    _select_positions(hidden_by_row, chunk_rows),
                    output_weight=output_weight,
                )
                chunk_reference_labels = reference_target_labels[
                    start : start + int(chunk_rows.numel())
                ]
                reference_loss = model.compute_language_model_loss(
                    chunk_reference_labels.unsqueeze(0),
                    local_logits.unsqueeze(1),
                ).reshape(-1)
                reference_logits = _vocab_parallel_target_logits(
                    local_logits,
                    chunk_reference_labels,
                )
                log_z = reference_logits + reference_loss
                for index, item_logprobs in enumerate(target_logprobs):
                    labels = label_rows[index]
                    if item_logprobs is None or labels is None:
                        continue
                    offsets, chunk_offsets = _match_chunk_offsets(
                        row_matches[index],
                        start=start,
                        end=start + int(chunk_rows.numel()),
                    )
                    if int(offsets.numel()) == 0:
                        continue
                    item_logprobs[offsets] = _vocab_parallel_target_logprobs(
                        local_logits,
                        labels.index_select(0, offsets),
                        log_z.index_select(0, chunk_offsets),
                        row_offsets=chunk_offsets,
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
            if topk_stats is not None:
                local_max, local_sum, _, _ = topk_stats
                local_max = local_max.detach()
                global_max = _all_reduce_tensor_parallel_max(local_max)
                global_sum = _all_reduce_tensor_parallel_sum(
                    local_sum * torch.exp(local_max - global_max)
                )
                log_z = global_max + torch.log(global_sum)
            elif logsumexp_stats is not None:
                local_max, local_sum = logsumexp_stats
                local_max = local_max.detach()
                global_max = _all_reduce_tensor_parallel_max(local_max)
                global_sum = _all_reduce_tensor_parallel_sum(
                    local_sum * torch.exp(local_max - global_max)
                )
                log_z = global_max + torch.log(global_sum)
            else:
                log_z = _vocab_parallel_log_z(local_logits)

            logits_topk: tuple[torch.Tensor, torch.Tensor] | None = None
            if logsumexp_stats is not None and max_top_k > 0:
                local_k = min(max_top_k, int(local_logits.shape[1]))
                local_values, local_tokens = torch.topk(local_logits, k=local_k, dim=-1)
                logits_topk = (local_values.float(), local_tokens)

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
                    if topk_stats is not None:
                        _, _, local_values, local_tokens = topk_stats
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
                    if logits_topk is not None:
                        local_values, local_tokens = logits_topk
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


def _is_native_target_only(items: Sequence[_ForwardItem]) -> bool:
    return all(
        item.labels is not None
        and item.labels.ndim == 1
        and item.request.top_k is None
        and not item.request.logits
        and not item.request.hidden_states
        for item in items
    )


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


def _empty_logits_like_positions(
    positions: torch.Tensor,
    model: "GPTModel",
    like: torch.Tensor,
) -> torch.Tensor:
    return torch.empty(
        (int(positions.numel()), _padded_vocab_size(model)),
        device=like.device,
        dtype=like.dtype,
    )


def _padded_vocab_size(model: "GPTModel") -> int:
    vocab_size = getattr(getattr(model, "config", None), "padded_vocab_size", None)
    if vocab_size is None:
        vocab_size = getattr(model, "vocab_size", None)
    if vocab_size is None:
        raise RuntimeError("could not determine full padded vocabulary size")
    return int(vocab_size)


def _target_logprobs_from_full_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
    log_z: torch.Tensor,
) -> torch.Tensor:
    return _call_compiled(_target_logprobs_from_full_logits_impl, logits, labels, log_z)


def _target_logprobs_from_full_logits_impl(
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
    row_offsets: torch.Tensor | None = None,
) -> torch.Tensor:
    target_logits = _vocab_parallel_target_logits(
        local_logits,
        labels,
        row_offsets=row_offsets,
    )
    return _call_compiled(_finish_target_logprobs, target_logits, labels, log_z)


def _vocab_parallel_target_logits(
    local_logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    row_offsets: torch.Tensor | None = None,
) -> torch.Tensor:
    start, _ = _vocab_range(local_logits)
    if row_offsets is None:
        local_target_logits = _call_compiled(
            _owned_target_logits,
            local_logits,
            labels,
            start,
        )
    else:
        local_target_logits = _call_compiled(
            _owned_target_logits_for_rows,
            local_logits,
            labels,
            start,
            row_offsets,
        )
    return _all_reduce_tensor_parallel_sum(local_target_logits)


def _owned_target_logits(
    local_logits: torch.Tensor,
    labels: torch.Tensor,
    vocab_start: int,
) -> torch.Tensor:
    flat_labels = labels.reshape(int(labels.shape[0]), -1)
    local_labels = flat_labels - vocab_start
    owns_label = (
        (flat_labels != -100)
        & (local_labels >= 0)
        & (local_labels < int(local_logits.shape[1]))
    )
    selected = local_logits.gather(
        1,
        local_labels.clamp(0, int(local_logits.shape[1]) - 1),
    ).float()
    return selected.masked_fill(~owns_label, 0.0).reshape(labels.shape)


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


def _valid_target_offsets(labels: torch.Tensor) -> torch.Tensor:
    if int(labels.shape[0]) == 0:
        return torch.empty(0, dtype=torch.long, device=labels.device)
    valid = labels != -100
    if labels.ndim > 1:
        valid = valid.reshape(int(labels.shape[0]), -1).any(dim=1)
    return torch.nonzero(valid, as_tuple=False).reshape(-1)


def _can_use_fused_target_ce(
    items: Sequence[_ForwardItem],
    label_rows: Sequence[torch.Tensor | None],
) -> bool:
    return all(item.request.top_k is None for item in items) and all(
        labels is None or labels.ndim == 1 for labels in label_rows
    )


def _can_use_reference_target_ce(
    items: Sequence[_ForwardItem],
    label_rows: Sequence[torch.Tensor | None],
) -> bool:
    return (
        os.environ.get("ART_TRAINER_RANK_REFERENCE_TARGET_CE", "0").lower()
        not in {"0", "false"}
        and all(
            item.request.top_k is None and not item.request.logits for item in items
        )
        and any(labels is not None and labels.ndim > 1 for labels in label_rows)
    )


def _reference_row_labels(
    label_rows: Sequence[torch.Tensor | None],
    row_matches: Sequence[_RowMatch],
    *,
    row_count: int,
    device: torch.device,
) -> torch.Tensor | None:
    references = torch.full((row_count,), -100, dtype=torch.long, device=device)
    for labels, match in zip(label_rows, row_matches, strict=True):
        if labels is None or int(match.source_offsets.numel()) == 0:
            continue
        selected = labels.index_select(0, match.source_offsets).reshape(
            int(match.source_offsets.numel()),
            -1,
        )
        valid = selected != -100
        has_label = valid.any(dim=1)
        if not bool(has_label.any()):
            continue
        candidates = selected.gather(
            1,
            valid.to(torch.int64).argmax(dim=1, keepdim=True),
        ).squeeze(1)
        row_offsets = match.row_offsets.index_select(
            0,
            torch.nonzero(has_label, as_tuple=False).reshape(-1),
        )
        candidates = candidates.masked_select(has_label)
        unset = references.index_select(0, row_offsets) == -100
        if bool(unset.any()):
            references[row_offsets.masked_select(unset)] = candidates.masked_select(
                unset
            )
    if bool((references == -100).any()):
        return None
    return references


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
        item_logprobs[match.source_offsets] = row_target_logprobs.index_select(
            0,
            match.row_offsets,
        )


def _topk_from_full_logits(
    logits: torch.Tensor,
    *,
    k: int,
    log_z: torch.Tensor,
) -> TopK:
    if k > int(logits.shape[1]):
        raise ValueError(f"top_k={k} exceeds vocabulary size {int(logits.shape[1])}")
    values, tokens = torch.topk(logits.float(), k=k, dim=-1)
    return TopK(logprobs=values - log_z.unsqueeze(1), tokens=tokens)


def _vocab_parallel_topk(
    local_logits: torch.Tensor,
    *,
    k: int,
    log_z: torch.Tensor,
) -> TopK:
    start, _ = _vocab_range(local_logits)
    local_k = min(k, int(local_logits.shape[1]))
    local_values, local_tokens = torch.topk(local_logits.float(), k=local_k, dim=-1)
    local_values = local_values - log_z.unsqueeze(1)
    local_tokens = local_tokens + start

    from megatron.core import parallel_state as ps

    tp_size = int(ps.get_tensor_model_parallel_world_size())
    if tp_size <= 1:
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
    # H200 measurements: fused top-k wins through k=10; above that the
    # logsumexp-only Triton path plus torch.topk scales better.
    return int(os.environ.get("ART_TRAINER_RANK_TRITON_FUSED_TOPK_MAX", "10"))


def _triton_min_rows() -> int:
    # Below this, Triton launch overhead usually costs more than the memory saved.
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
    sort_order = global_positions.argsort()
    sorted_global_positions = global_positions.index_select(0, sort_order)
    sorted_local_positions = local_positions.index_select(0, sort_order)

    indices = torch.searchsorted(sorted_global_positions, item_positions)
    in_bounds = indices < int(sorted_global_positions.numel())
    source_offsets = torch.arange(
        int(item_positions.numel()),
        device=item_positions.device,
        dtype=torch.long,
    )[in_bounds]
    found = indices[in_bounds]
    keep = sorted_global_positions.index_select(
        0, found
    ) == item_positions.index_select(
        0,
        source_offsets,
    )
    return (
        sorted_local_positions.index_select(0, found[keep]).to("cpu"),
        source_offsets[keep].to("cpu"),
    )


def _select_positions(values: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    if int(positions.numel()) == 0:
        return values[:0]
    return values.index_select(0, positions.to(device=values.device))


def _gather_target_logprobs(
    logprobs: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    if int(labels.shape[0]) == 0:
        return torch.empty(labels.shape, device=logprobs.device, dtype=logprobs.dtype)
    flat_labels = labels.clamp_min(0).reshape(int(labels.shape[0]), -1)
    selected = logprobs.gather(1, flat_labels).reshape(labels.shape)
    return selected.masked_fill(labels == -100, 0.0)


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
    return [_materialize(item) for item in inputs]


def _flatten(inputs: ForwardInputs) -> Iterator[AnyForwardInput]:
    if isinstance(inputs, ForwardInput):
        yield inputs
        return
    for item in inputs:
        yield from _flatten(item)


def _unflatten(
    template: ForwardInputs, outputs: Iterator[AnyForwardOutput]
) -> ForwardOutputs:
    if isinstance(template, ForwardInput):
        return next(outputs)
    return [_unflatten(item, outputs) for item in template]


__all__ = [
    "AdamParams",
    "ForwardInput",
    "ForwardOutput",
    "MicroBatch",
    "TopK",
    "TrainerRank",
]
