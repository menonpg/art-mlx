from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("megatron.bridge")
pytest.importorskip("megatron.bridge.models.qwen_vl.qwen35_vl_provider")

from art.megatron.context_parallel.layout_index import TokenLayoutIndex
from art.megatron.gdn.operator import gdn_shared_prefix_forward

from .cases import default_phase0_cases
from .metrics import (
    GDN_CORRECTNESS_DTYPE,
    REAL_GDN_GRAD_MEAN_ABS_PCT_THRESHOLD,
    REAL_GDN_OUTPUT_MEAN_ABS_PCT_THRESHOLD,
    assert_mean_abs_pct,
    assert_scalar_loss_close,
    parameter_grad_mean_abs_pct_with_name,
)
from .packed_layout import build_phase0_packed_tensors
from .real_gdn_oracle import (
    run_real_gdn_local_fork_reference,
    zero_parameter_grads,
)
from .test_real_gdn_cp1_packed_vs_flattened import (
    _make_matching_qwen35_gdn_pair,
    _single_rank_model_parallel,
)


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is required for real Megatron/FLA GDN local-fork coverage.",
)
@pytest.mark.parametrize("cp_size", (2, 4, 8))
def test_real_qwen35_gdn_cp_local_fork_matches_cp1(cp_size: int) -> None:
    selected_names = {"ragged_family_mix", "conv_tail_boundary", "padding_tail"}
    cases = [
        case
        for case in default_phase0_cases(conv_width=2)
        if case.name in selected_names
    ]
    with _single_rank_model_parallel():
        cp1_gdn, local_fork_gdn = _make_matching_qwen35_gdn_pair()
        device = torch.device("cuda")
        for case_index, case in enumerate(cases):
            zero_parameter_grads(cp1_gdn)
            zero_parameter_grads(local_fork_gdn)
            tensors = build_phase0_packed_tensors(case)
            group_ids = tensors["group_ids"].to(device)
            parent_ids = tensors["parent_ids"].to(device)
            real_token_mask = (group_ids != -1).transpose(0, 1).unsqueeze(-1)
            hidden_states = torch.randn(
                case.sequence_length,
                len(case.rows),
                64,
                device=device,
                dtype=GDN_CORRECTNESS_DTYPE,
                generator=torch.Generator(device=device).manual_seed(
                    20260425 + cp_size * 100 + case_index
                ),
            )
            output_grad = (
                torch.randn(
                    hidden_states.shape,
                    device=device,
                    dtype=GDN_CORRECTNESS_DTYPE,
                    generator=torch.Generator(device=device).manual_seed(
                        20270425 + cp_size * 100 + case_index
                    ),
                )
                * real_token_mask
            )
            cp1_hidden = hidden_states.clone().detach().requires_grad_(True)
            local_hidden = hidden_states.clone().detach().requires_grad_(True)

            cp1_out, _ = gdn_shared_prefix_forward(
                cp1_gdn,
                cp1_hidden,
                group_ids=group_ids,
                parent_ids=parent_ids,
            )
            local_out = run_real_gdn_local_fork_reference(
                local_fork_gdn,
                local_hidden,
                group_ids=group_ids,
                parent_ids=parent_ids,
                cp_size=cp_size,
                attention_token_layout_index=_layout_index_from_rank_indices(
                    _striped_rank_indices(
                        tuple(reversed(_real_token_indices(tensors["group_ids"]))),
                        cp_size=cp_size,
                    )
                ),
            )
            cp1_loss = (cp1_out * output_grad).sum()
            local_loss = (local_out * output_grad).sum()
            cp1_loss.backward()
            local_loss.backward()

            param_name, param_pct = parameter_grad_mean_abs_pct_with_name(
                cp1_gdn, local_fork_gdn
            )
            assert_scalar_loss_close(cp1_loss.detach(), local_loss.detach(), case.name)
            assert_mean_abs_pct(
                cp1_out.detach(),
                local_out.detach(),
                case.name,
                threshold=REAL_GDN_OUTPUT_MEAN_ABS_PCT_THRESHOLD,
            )
            assert cp1_hidden.grad is not None
            assert local_hidden.grad is not None
            assert_mean_abs_pct(
                cp1_hidden.grad,
                local_hidden.grad,
                case.name,
                threshold=REAL_GDN_GRAD_MEAN_ABS_PCT_THRESHOLD,
            )
            assert param_pct <= REAL_GDN_GRAD_MEAN_ABS_PCT_THRESHOLD, (
                f"{case.name}:{param_name}"
            )


def _real_token_indices(group_ids: torch.Tensor) -> tuple[int, ...]:
    sequence_length = int(group_ids.shape[1])
    return tuple(
        row * sequence_length + position
        for row in range(int(group_ids.shape[0]))
        for position in torch.nonzero(group_ids[row] != -1, as_tuple=False)
        .flatten()
        .tolist()
    )


def _striped_rank_indices(
    token_indices: tuple[int, ...],
    *,
    cp_size: int,
) -> tuple[tuple[int, ...], ...]:
    ranks: list[list[int]] = [[] for _ in range(cp_size)]
    for offset, token_index in enumerate(token_indices):
        ranks[offset % cp_size].append(token_index)
    return tuple(tuple(rank_indices) for rank_indices in ranks)


def _layout_index_from_rank_indices(
    rank_indices: tuple[tuple[int, ...], ...],
) -> TokenLayoutIndex:
    return TokenLayoutIndex(
        ownership_ranges_by_rank=tuple(
            _ranges_from_tokens(tokens) for tokens in rank_indices
        ),
        token_counts_by_rank=tuple(len(tokens) for tokens in rank_indices),
    )


def _ranges_from_tokens(tokens: tuple[int, ...]) -> tuple[tuple[int, int, int], ...]:
    if not tokens:
        return ()
    ranges: list[tuple[int, int, int]] = []
    start = tokens[0]
    end = start + 1
    position = 0
    for offset, token in enumerate(tokens[1:], start=1):
        if token == end:
            end += 1
            continue
        ranges.append((start, end, position))
        start = token
        end = token + 1
        position = offset
    ranges.append((start, end, position))
    return tuple(ranges)
