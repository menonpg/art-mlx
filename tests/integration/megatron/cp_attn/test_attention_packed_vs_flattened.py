from __future__ import annotations

from contextlib import ExitStack
import math
from typing import Any

import pytest

torch = pytest.importorskip("torch")
from torch.nn.attention.flex_attention import BlockMask, create_block_mask

from art.megatron.flex_attn.attention import FlexAttentionWrapper
from art.megatron.shared_prefix_state import create_shared_prefix_state
from tests.integration.megatron.gdn_shared_prefix.cases import default_phase0_cases
from tests.integration.megatron.gdn_shared_prefix.metrics import (
    GDN_CORRECTNESS_DTYPE,
    MEAN_ABS_PCT_MISMATCH_THRESHOLD,
    MEAN_ABS_PCT_THRESHOLD,
    assert_mean_abs_pct,
    mean_abs_pct,
)
from tests.integration.megatron.gdn_shared_prefix.packed_layout import (
    build_phase0_packed_tensors,
)
from tests.integration.megatron.gdn_shared_prefix.parser_import import (
    parse_gdn_shared_prefix_segments,
)
from tests.integration.megatron.model_support.oracle_harness import (
    TEST_DEFAULT_FLEX_BACKEND,
)
from tests.integration.megatron.model_support.oracle_worker import (
    _apply_requested_flex_backend_patch,
    _apply_test_attention_full_fp32_patch,
    _apply_test_flex_inner_fp32_patch,
)


@pytest.fixture(autouse=True)
def _fp32_test_flex_backend():
    with ExitStack() as stack:
        stack.enter_context(
            _apply_requested_flex_backend_patch(TEST_DEFAULT_FLEX_BACKEND)
        )
        stack.enter_context(
            _apply_test_flex_inner_fp32_patch(TEST_DEFAULT_FLEX_BACKEND)
        )
        stack.enter_context(
            _apply_test_attention_full_fp32_patch(TEST_DEFAULT_FLEX_BACKEND)
        )
        yield


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is required for compiled flex-attention shared-prefix coverage.",
)
def test_shared_prefix_attention_matches_flattened_grad_accumulation() -> None:
    case = next(
        item for item in default_phase0_cases() if item.name == "multi_family_repeated"
    )
    tensors = build_phase0_packed_tensors(case)
    group_ids = tensors["group_ids"].cuda()
    parent_ids = tensors["parent_ids"].cuda()
    spec = parse_gdn_shared_prefix_segments(
        group_ids.cpu(), parent_ids.cpu(), min_completions_per_family=1
    )
    q, k, v = _attention_inputs(group_ids.shape, seed=20260425)
    q_ref = q.detach().clone().requires_grad_(True)
    k_ref = k.detach().clone().requires_grad_(True)
    v_ref = v.detach().clone().requires_grad_(True)
    output_grad = _packed_output_grad(spec, q.shape, seed=20260426)

    attention_state = create_shared_prefix_state(group_ids, parent_ids)
    packed_out = FlexAttentionWrapper()(
        q,
        k,
        v,
        block_mask=attention_state.block_mask,
        scale=1.0 / math.sqrt(q.shape[-1]),
        enable_gqa=False,
    )
    (packed_out * output_grad).sum().backward()

    ref_out = torch.zeros_like(packed_out)
    ref_loss = q_ref.new_zeros(())
    for family in spec.families:
        prefix = family.prefix
        prefix_grad_used = False
        for completion in family.completions:
            indices = torch.tensor(
                [
                    *range(prefix.start, prefix.end),
                    *range(completion.start, completion.end),
                ],
                device=q.device,
                dtype=torch.long,
            )
            row = family.row_index
            q_slice = q_ref[row : row + 1].index_select(2, indices)
            k_slice = k_ref[row : row + 1].index_select(2, indices)
            v_slice = v_ref[row : row + 1].index_select(2, indices)
            flat_out = _dense_causal_attention(q_slice, k_slice, v_slice)

            ref_out[row, :, completion.start : completion.end] = flat_out[
                0, :, prefix.length :
            ]
            flat_grad = torch.zeros_like(flat_out)
            flat_grad[0, :, prefix.length :] = output_grad[
                row, :, completion.start : completion.end
            ]
            if not prefix_grad_used:
                ref_out[row, :, prefix.start : prefix.end] = flat_out[
                    0, :, : prefix.length
                ]
                flat_grad[0, :, : prefix.length] = output_grad[
                    row, :, prefix.start : prefix.end
                ]
                prefix_grad_used = True
            ref_loss = ref_loss + (flat_out * flat_grad).sum()
    ref_loss.backward()

    real_mask = _real_token_mask(spec, q.shape, device=q.device)
    assert_mean_abs_pct(ref_out[real_mask], packed_out[real_mask], "attention_output")
    assert q.grad is not None
    assert k.grad is not None
    assert v.grad is not None
    assert q_ref.grad is not None
    assert k_ref.grad is not None
    assert v_ref.grad is not None
    assert_mean_abs_pct(q_ref.grad, q.grad, "q_grad")
    assert_mean_abs_pct(k_ref.grad, k.grad, "k_grad")
    assert_mean_abs_pct(v_ref.grad, v.grad, "v_grad")


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is required for compiled flex-attention shared-prefix coverage.",
)
def test_physical_causal_attention_leaks_across_siblings() -> None:
    case = next(
        item for item in default_phase0_cases() if item.name == "multi_family_repeated"
    )
    tensors = build_phase0_packed_tensors(case)
    group_ids = tensors["group_ids"].cuda()
    parent_ids = tensors["parent_ids"].cuda()
    spec = parse_gdn_shared_prefix_segments(
        group_ids.cpu(), parent_ids.cpu(), min_completions_per_family=1
    )
    q, k, v = _attention_inputs(group_ids.shape, seed=20260427)
    attention_state = create_shared_prefix_state(group_ids, parent_ids)
    packed_out = FlexAttentionWrapper()(
        q,
        k,
        v,
        block_mask=attention_state.block_mask,
        scale=1.0 / math.sqrt(q.shape[-1]),
        enable_gqa=False,
    )
    physical_out = _dense_causal_attention(q, k, v)
    completion_mask = _completion_token_mask(spec, q.shape, device=q.device)
    assert (
        mean_abs_pct(
            packed_out[completion_mask],
            physical_out[completion_mask],
        )
        > MEAN_ABS_PCT_MISMATCH_THRESHOLD
    )


def test_sparse_shared_prefix_block_mask_matches_create_block_mask_metadata() -> None:
    for case_name, group_ids, parent_ids in _block_mask_metadata_cases():
        attention_state = create_shared_prefix_state(
            group_ids,
            parent_ids,
            target_device=torch.device("cpu"),
            attention_head_dim=16,
            attention_value_head_dim=16,
        )
        old_mask = _create_block_mask_reference(
            group_ids,
            parent_ids,
            block_size=attention_state.block_mask.BLOCK_SIZE,
        )
        _assert_block_mask_metadata_equal(
            case_name,
            attention_state.block_mask,
            old_mask,
        )


def _attention_inputs(
    shape: torch.Size, *, seed: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size, sequence_length = shape
    generator = torch.Generator(device="cuda").manual_seed(seed)
    q = torch.randn(
        batch_size,
        2,
        sequence_length,
        16,
        device="cuda",
        dtype=GDN_CORRECTNESS_DTYPE,
        generator=generator,
        requires_grad=True,
    )
    k = torch.randn(
        q.shape, device="cuda", dtype=GDN_CORRECTNESS_DTYPE, generator=generator
    )
    v = torch.randn(
        q.shape, device="cuda", dtype=GDN_CORRECTNESS_DTYPE, generator=generator
    )
    return q, k.requires_grad_(True), v.requires_grad_(True)


def _dense_causal_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> torch.Tensor:
    scores = torch.matmul(q, k.transpose(-1, -2)) * (1.0 / math.sqrt(q.shape[-1]))
    length = int(q.shape[-2])
    causal_mask = torch.ones(length, length, device=q.device, dtype=torch.bool).tril()
    scores = scores.masked_fill(~causal_mask, float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    return torch.matmul(probs, v)


def _packed_output_grad(spec: Any, shape: torch.Size, *, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    grad = torch.randn(
        shape,
        device="cuda",
        dtype=GDN_CORRECTNESS_DTYPE,
        generator=generator,
    )
    return grad * _real_token_mask(spec, shape, device=grad.device) * 0.1


def _real_token_mask(
    spec: Any, shape: torch.Size, *, device: torch.device
) -> torch.Tensor:
    mask = torch.zeros(shape, device=device, dtype=torch.bool)
    for row_index, valid_length in enumerate(spec.valid_lengths):
        mask[row_index, :, :valid_length] = True
    return mask


def _completion_token_mask(
    spec: Any, shape: torch.Size, *, device: torch.device
) -> torch.Tensor:
    mask = torch.zeros(shape, device=device, dtype=torch.bool)
    for family in spec.families:
        for completion in family.completions:
            mask[
                family.row_index,
                :,
                completion.start : completion.end,
            ] = True
    return mask


def _block_mask_metadata_cases() -> tuple[tuple[str, torch.Tensor, torch.Tensor], ...]:
    return tuple(
        (name, *_group_parent_from_segments(segments, total_len))
        for name, segments, total_len in (
            ("causal_only", ((0, 0, 17),), 17),
            (
                "block_boundary_shared_prefix",
                ((40, 40, 130), (41, 40, 95), (42, 40, 70)),
                384,
            ),
            (
                "non_multiple_length",
                ((50, 50, 129), (51, 50, 2), (52, 50, 127), (53, 50, 13)),
                291,
            ),
            (
                "long_many_blocks_padding",
                ((60, 60, 513), (61, 60, 257), (62, 60, 601), (63, 60, 311)),
                2048,
            ),
        )
    )


def _group_parent_from_segments(
    segments: tuple[tuple[int, int, int], ...],
    total_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    group_ids = torch.full((1, total_len), -1, dtype=torch.int64)
    parent_ids = torch.full_like(group_ids, -1)
    cursor = 0
    for group_id, parent_id, length in segments:
        end = cursor + int(length)
        group_ids[0, cursor:end] = int(group_id)
        parent_ids[0, cursor:end] = int(parent_id)
        cursor = end
    return group_ids, parent_ids


def _create_block_mask_reference(
    group_ids: torch.Tensor,
    parent_ids: torch.Tensor,
    *,
    block_size: tuple[int, int],
) -> BlockMask:
    def shared_prefix_mask(
        batch_idx: torch.Tensor,
        head_idx: torch.Tensor,
        query_idx: torch.Tensor,
        kv_idx: torch.Tensor,
    ) -> torch.Tensor:
        del head_idx
        same_group = group_ids[batch_idx, query_idx] == group_ids[batch_idx, kv_idx]
        parent_prefix = parent_ids[batch_idx, query_idx] == group_ids[batch_idx, kv_idx]
        return (query_idx >= kv_idx) & (same_group | parent_prefix)

    seq_len = int(group_ids.shape[1])
    return create_block_mask(
        shared_prefix_mask,
        1,
        None,
        seq_len,
        seq_len,
        device=group_ids.device,
        BLOCK_SIZE=block_size,
    )


def _assert_block_mask_metadata_equal(
    case_name: str,
    new_mask: BlockMask,
    old_mask: BlockMask,
) -> None:
    assert new_mask.seq_lengths == old_mask.seq_lengths, case_name
    assert new_mask.BLOCK_SIZE == old_mask.BLOCK_SIZE, case_name
    for count_attr, index_attr in (
        ("kv_num_blocks", "kv_indices"),
        ("full_kv_num_blocks", "full_kv_indices"),
        ("q_num_blocks", "q_indices"),
        ("full_q_num_blocks", "full_q_indices"),
    ):
        _assert_sparse_block_rows_equal(
            case_name,
            count_attr,
            index_attr,
            getattr(new_mask, count_attr),
            getattr(new_mask, index_attr),
            getattr(old_mask, count_attr),
            getattr(old_mask, index_attr),
        )


def _assert_sparse_block_rows_equal(
    case_name: str,
    count_attr: str,
    index_attr: str,
    new_counts: torch.Tensor | None,
    new_indices: torch.Tensor | None,
    old_counts: torch.Tensor | None,
    old_indices: torch.Tensor | None,
) -> None:
    assert new_counts is not None and old_counts is not None, (
        f"{case_name}: missing {count_attr}"
    )
    assert new_indices is not None and old_indices is not None, (
        f"{case_name}: missing {index_attr}"
    )
    new_counts = new_counts.cpu()
    old_counts = old_counts.cpu()
    assert torch.equal(new_counts, old_counts), (
        f"{case_name}: {count_attr} mismatch\nnew={new_counts}\nold={old_counts}"
    )
    new_indices = new_indices.cpu()
    old_indices = old_indices.cpu()
    assert new_indices.shape == old_indices.shape, (
        f"{case_name}: {index_attr} shape mismatch "
        f"{tuple(new_indices.shape)} != {tuple(old_indices.shape)}"
    )
    counts = new_counts.reshape(-1, new_counts.shape[-1])
    new_rows = new_indices.reshape(-1, new_indices.shape[-2], new_indices.shape[-1])
    old_rows = old_indices.reshape(-1, old_indices.shape[-2], old_indices.shape[-1])
    for batch_head_idx in range(int(new_rows.shape[0])):
        for row_idx in range(int(new_rows.shape[1])):
            count = int(counts[batch_head_idx, row_idx])
            assert torch.equal(
                new_rows[batch_head_idx, row_idx, :count],
                old_rows[batch_head_idx, row_idx, :count],
            ), (
                f"{case_name}: {index_attr} row {row_idx} mismatch\n"
                f"new={new_rows[batch_head_idx, row_idx, :count]}\n"
                f"old={old_rows[batch_head_idx, row_idx, :count]}"
            )
