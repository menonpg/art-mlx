from __future__ import annotations

from typing import Any

import torch

TRACE_ROW_TOKEN_UIDS_ATTR = "_art_trace_row_token_uids"
TRACE_UID_SPAN_ATTR = "_art_trace_uid_span"


def extract_tensor_attr(value: Any, attr_name: str) -> Any:
    if isinstance(value, torch.Tensor):
        return getattr(value, attr_name, None)
    if isinstance(value, dict):
        for item in value.values():
            attr_value = extract_tensor_attr(item, attr_name)
            if attr_value is not None:
                return attr_value
    if isinstance(value, (list, tuple)):
        for item in value:
            attr_value = extract_tensor_attr(item, attr_name)
            if attr_value is not None:
                return attr_value
    return None


def normalize_row_token_uids(value: Any) -> torch.Tensor | None:
    if not isinstance(value, torch.Tensor):
        return None
    return value.detach().to(device="cpu", dtype=torch.int64).reshape(-1)


def positive_uid_span(value: Any) -> int | None:
    return int(value) if isinstance(value, int) and value > 0 else None


def row_token_uids_from_trace_sources(
    *,
    inputs: Any,
    output: Any,
    module: Any,
    row_count: int | None = None,
    prefer_uid_span: bool = False,
) -> tuple[torch.Tensor | None, int | None]:
    candidates = (
        (
            extract_tensor_attr(output, TRACE_ROW_TOKEN_UIDS_ATTR),
            extract_tensor_attr(output, TRACE_UID_SPAN_ATTR),
        ),
        (
            extract_tensor_attr(inputs, TRACE_ROW_TOKEN_UIDS_ATTR),
            extract_tensor_attr(inputs, TRACE_UID_SPAN_ATTR),
        ),
        (
            getattr(module, TRACE_ROW_TOKEN_UIDS_ATTR, None),
            getattr(module, TRACE_UID_SPAN_ATTR, None),
        ),
    )
    row_count_matches: list[tuple[torch.Tensor, int | None]] = []
    tensor_candidates: list[tuple[torch.Tensor, int | None]] = []
    for row_token_uids, uid_span in candidates:
        row_token_uids = normalize_row_token_uids(row_token_uids)
        if row_token_uids is None:
            continue
        candidate = (row_token_uids, positive_uid_span(uid_span))
        tensor_candidates.append(candidate)
        if row_count is None or int(row_token_uids.numel()) == int(row_count):
            row_count_matches.append(candidate)
    if not tensor_candidates:
        return None, None

    def _select(
        options: list[tuple[torch.Tensor, int | None]],
    ) -> tuple[torch.Tensor, int | None] | None:
        if prefer_uid_span:
            for row_token_uids, uid_span in options:
                if uid_span is not None:
                    return row_token_uids, uid_span
        return options[0] if options else None

    selected = _select(row_count_matches) or _select(tensor_candidates)
    return selected if selected is not None else (None, None)


def expand_token_uids_for_heads(
    token_uids: torch.Tensor,
    *,
    head_count: int,
) -> torch.Tensor:
    if token_uids.ndim != 1:
        raise RuntimeError(
            f"Expected 1D token UID tensor, got shape={tuple(token_uids.shape)}"
        )
    if head_count <= 0:
        raise RuntimeError(f"Expected positive head_count, got {head_count}")
    return token_uids.repeat_interleave(head_count).contiguous()
