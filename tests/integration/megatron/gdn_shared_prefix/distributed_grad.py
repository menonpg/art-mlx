from __future__ import annotations

from collections import defaultdict

import torch


def all_reduce_parameter_grads_coalesced(
    module: torch.nn.Module, *, group: object | None = None
) -> None:
    grad_entries: dict[
        tuple[torch.device, torch.dtype],
        list[tuple[torch.nn.Parameter, torch.Tensor | None, int]],
    ] = defaultdict(list)
    main_grad_entries: dict[tuple[torch.device, torch.dtype], list[torch.Tensor]] = (
        defaultdict(list)
    )
    for parameter in module.parameters():
        grad_entries[(parameter.device, parameter.dtype)].append(
            (parameter, parameter.grad, 1 if parameter.grad is not None else 0)
        )
        main_grad = getattr(parameter, "main_grad", None)
        if main_grad is not None:
            main_grad_entries[(main_grad.device, main_grad.dtype)].append(main_grad)
    for entries in grad_entries.values():
        _all_reduce_parameter_grad_group(entries, group=group)
    for entries in main_grad_entries.values():
        _all_reduce_tensor_group(entries, group=group)


def _all_reduce_parameter_grad_group(
    entries: list[tuple[torch.nn.Parameter, torch.Tensor | None, int]],
    *,
    group: object | None,
) -> None:
    if not entries:
        return
    has_grad = torch.tensor(
        [entry_has_grad for _, _, entry_has_grad in entries],
        device=entries[0][0].device,
        dtype=torch.int32,
    )
    torch.distributed.all_reduce(has_grad, group=group)  # ty: ignore[possibly-missing-attribute]
    flat = torch.cat(
        [
            torch.zeros(
                parameter.numel(), device=parameter.device, dtype=parameter.dtype
            )
            if grad is None
            else grad.reshape(-1)
            for parameter, grad, _ in entries
        ]
    )
    torch.distributed.all_reduce(flat, group=group)  # ty: ignore[possibly-missing-attribute]
    offset = 0
    for index, (parameter, grad, _) in enumerate(entries):
        size = parameter.numel()
        reduced = flat.narrow(0, offset, size).view_as(parameter)
        if int(has_grad[index].item()) > 0:
            if grad is None:
                parameter.grad = torch.empty_like(parameter)
                grad = parameter.grad
            grad.copy_(reduced)
        offset += size


def _all_reduce_tensor_group(
    entries: list[torch.Tensor], *, group: object | None
) -> None:
    if not entries:
        return
    flat = torch.cat([tensor.reshape(-1) for tensor in entries])
    torch.distributed.all_reduce(flat, group=group)  # ty: ignore[possibly-missing-attribute]
    offset = 0
    for tensor in entries:
        size = tensor.numel()
        tensor.copy_(flat.narrow(0, offset, size).view_as(tensor))
        offset += size
