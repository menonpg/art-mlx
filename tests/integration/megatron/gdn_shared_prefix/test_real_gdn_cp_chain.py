from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("megatron.bridge")
pytest.importorskip("megatron.bridge.models.qwen_vl.qwen35_vl_provider")

from art.megatron.gdn.operator import gdn_shared_prefix_forward

from .cases import (
    GdnFamilyShape,
    GdnPackedRowShape,
    GdnPhase0Case,
    default_phase0_cases,
)
from .metrics import (
    GDN_CORRECTNESS_DTYPE,
    MEAN_ABS_PCT_MISMATCH_THRESHOLD,
    REAL_GDN_GRAD_MEAN_ABS_PCT_THRESHOLD,
    REAL_GDN_OUTPUT_MEAN_ABS_PCT_THRESHOLD,
    assert_mean_abs_pct,
    assert_scalar_loss_close,
    mean_abs_pct,
    parameter_grad_mean_abs_pct_with_name,
    stable_output_mse_loss,
)
from .packed_layout import build_phase0_packed_tensors
from .real_gdn_oracle import (
    run_real_gdn_chunk_native_reference,
    run_real_gdn_mixed_cp_reference,
    run_real_gdn_physical_stream,
    run_real_gdn_suffix_only_chain_reference,
    zero_parameter_grads,
)
from .test_real_gdn_cp1_packed_vs_flattened import (
    _make_matching_qwen35_gdn_pair,
    _single_rank_model_parallel,
)


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is required for real Megatron/FLA GDN chunk-native coverage.",
)
@pytest.mark.parametrize("cp_size", (2, 4, 8))
def test_real_qwen35_gdn_chunk_native_reference_matches_cp1(cp_size: int) -> None:
    selected_names = {"cp_boundary_prefix", "cp_boundary_suffix", "dominant_family"}
    cases = [
        case
        for case in default_phase0_cases(conv_width=2)
        if case.name in selected_names
    ]
    with _single_rank_model_parallel():
        cp1_gdn, chunk_gdn = _make_matching_qwen35_gdn_pair()
        device = torch.device("cuda")
        for case_index, case in enumerate(cases):
            zero_parameter_grads(cp1_gdn)
            zero_parameter_grads(chunk_gdn)
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
                    20260426 + cp_size * 100 + case_index
                ),
            )
            output_grad = (
                torch.randn(
                    hidden_states.shape,
                    device=device,
                    dtype=GDN_CORRECTNESS_DTYPE,
                    generator=torch.Generator(device=device).manual_seed(
                        20270426 + cp_size * 100 + case_index
                    ),
                )
                * real_token_mask
            )
            loss_denominator = real_token_mask.expand_as(output_grad).sum()
            cp1_hidden = hidden_states.clone().detach().requires_grad_(True)
            chunk_hidden = hidden_states.clone().detach().requires_grad_(True)
            cp1_out, _ = gdn_shared_prefix_forward(
                cp1_gdn,
                cp1_hidden,
                group_ids=group_ids,
                parent_ids=parent_ids,
            )
            chunk_out = run_real_gdn_chunk_native_reference(
                chunk_gdn,
                chunk_hidden,
                group_ids=group_ids,
                parent_ids=parent_ids,
            )
            cp1_loss = stable_output_mse_loss(
                cp1_out,
                output_grad,
                mask=real_token_mask,
                denominator=loss_denominator,
            )
            chunk_loss = stable_output_mse_loss(
                chunk_out,
                output_grad,
                mask=real_token_mask,
                denominator=loss_denominator,
            )
            cp1_loss.backward()
            chunk_loss.backward()

            param_name, param_pct = parameter_grad_mean_abs_pct_with_name(
                cp1_gdn, chunk_gdn
            )
            assert_scalar_loss_close(cp1_loss.detach(), chunk_loss.detach(), case.name)
            assert_mean_abs_pct(
                cp1_out.detach(),
                chunk_out.detach(),
                case.name,
                threshold=REAL_GDN_OUTPUT_MEAN_ABS_PCT_THRESHOLD,
            )
            assert cp1_hidden.grad is not None
            assert chunk_hidden.grad is not None
            assert_mean_abs_pct(
                cp1_hidden.grad,
                chunk_hidden.grad,
                case.name,
                threshold=REAL_GDN_GRAD_MEAN_ABS_PCT_THRESHOLD,
            )
            assert param_pct <= REAL_GDN_GRAD_MEAN_ABS_PCT_THRESHOLD, (
                f"{case.name}:{param_name}"
            )


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is required for real Megatron/FLA GDN chain-shard coverage.",
)
def test_real_qwen35_gdn_cp_chain_known_bad_mutations_fail() -> None:
    cases_by_name = {case.name: case for case in default_phase0_cases(conv_width=2)}
    with _single_rank_model_parallel():
        cp1_gdn, bad_gdn = _make_matching_qwen35_gdn_pair()
        device = torch.device("cuda")
        boundary_case = cases_by_name["cp_boundary_suffix"]
        boundary_tensors = build_phase0_packed_tensors(boundary_case)
        boundary_group_ids = boundary_tensors["group_ids"].to(device)
        boundary_parent_ids = boundary_tensors["parent_ids"].to(device)
        boundary_hidden = torch.randn(
            boundary_case.sequence_length,
            len(boundary_case.rows),
            64,
            device=device,
            dtype=GDN_CORRECTNESS_DTYPE,
            generator=torch.Generator(device=device).manual_seed(20280426),
        )
        with torch.no_grad():
            cp1_out, _ = gdn_shared_prefix_forward(
                cp1_gdn,
                boundary_hidden,
                group_ids=boundary_group_ids,
                parent_ids=boundary_parent_ids,
            )
            bad_conv_out = run_real_gdn_suffix_only_chain_reference(
                bad_gdn,
                boundary_hidden,
                group_ids=boundary_group_ids,
                parent_ids=boundary_parent_ids,
                cp_size=4,
                mutation="zero_conv_tail",
            )
            bad_rec_out = run_real_gdn_suffix_only_chain_reference(
                bad_gdn,
                boundary_hidden,
                group_ids=boundary_group_ids,
                parent_ids=boundary_parent_ids,
                cp_size=4,
                mutation="zero_recurrent_parent",
            )
        assert (
            _real_token_mean_abs_pct(cp1_out, bad_conv_out, boundary_group_ids)
            > MEAN_ABS_PCT_MISMATCH_THRESHOLD
        )
        assert (
            _real_token_mean_abs_pct(cp1_out, bad_rec_out, boundary_group_ids)
            > MEAN_ABS_PCT_MISMATCH_THRESHOLD
        )

        ragged_case = cases_by_name["ragged_family_mix"]
        ragged_tensors = build_phase0_packed_tensors(ragged_case)
        ragged_group_ids = ragged_tensors["group_ids"].to(device)
        ragged_parent_ids = ragged_tensors["parent_ids"].to(device)
        ragged_hidden = torch.randn(
            ragged_case.sequence_length,
            len(ragged_case.rows),
            64,
            device=device,
            dtype=GDN_CORRECTNESS_DTYPE,
            generator=torch.Generator(device=device).manual_seed(20290426),
        )
        with torch.no_grad():
            ragged_cp1_out, _ = gdn_shared_prefix_forward(
                cp1_gdn,
                ragged_hidden,
                group_ids=ragged_group_ids,
                parent_ids=ragged_parent_ids,
            )
            physical_out = run_real_gdn_physical_stream(
                bad_gdn,
                ragged_hidden,
                group_ids=ragged_group_ids,
            )
        assert (
            _real_token_mean_abs_pct(ragged_cp1_out, physical_out, ragged_group_ids)
            > MEAN_ABS_PCT_MISMATCH_THRESHOLD
        )


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is required for real Megatron/FLA GDN chain-shard coverage.",
)
def test_real_qwen35_gdn_cp_chain_detached_prefix_state_loses_gradients() -> None:
    case = next(
        case
        for case in default_phase0_cases(conv_width=2)
        if case.name == "ragged_family_mix"
    )
    with _single_rank_model_parallel():
        cp1_gdn, bad_gdn = _make_matching_qwen35_gdn_pair()
        device = torch.device("cuda")
        tensors = build_phase0_packed_tensors(case)
        group_ids = tensors["group_ids"].to(device)
        parent_ids = tensors["parent_ids"].to(device)
        suffix_mask = (group_ids != parent_ids).transpose(0, 1).unsqueeze(-1)
        hidden_states = torch.randn(
            case.sequence_length,
            len(case.rows),
            64,
            device=device,
            dtype=GDN_CORRECTNESS_DTYPE,
            generator=torch.Generator(device=device).manual_seed(20320426),
        )
        output_grad = (
            torch.randn(
                hidden_states.shape,
                device=device,
                dtype=GDN_CORRECTNESS_DTYPE,
                generator=torch.Generator(device=device).manual_seed(20330426),
            )
            * suffix_mask
        )
        loss_denominator = suffix_mask.expand_as(output_grad).sum()
        cp1_hidden = hidden_states.clone().detach().requires_grad_(True)
        bad_hidden = hidden_states.clone().detach().requires_grad_(True)

        cp1_out, _ = gdn_shared_prefix_forward(
            cp1_gdn,
            cp1_hidden,
            group_ids=group_ids,
            parent_ids=parent_ids,
        )
        bad_out = run_real_gdn_suffix_only_chain_reference(
            bad_gdn,
            bad_hidden,
            group_ids=group_ids,
            parent_ids=parent_ids,
            cp_size=4,
            mutation="detach_prefix_state",
        )
        cp1_loss = stable_output_mse_loss(
            cp1_out,
            output_grad,
            mask=suffix_mask,
            denominator=loss_denominator,
        )
        bad_loss = stable_output_mse_loss(
            bad_out,
            output_grad,
            mask=suffix_mask,
            denominator=loss_denominator,
        )
        cp1_loss.backward()
        bad_loss.backward()

        assert_mean_abs_pct(cp1_out.detach(), bad_out.detach(), case.name)
        assert_mean_abs_pct(cp1_loss.detach(), bad_loss.detach(), case.name)
        assert cp1_hidden.grad is not None
        assert bad_hidden.grad is not None
        assert (
            mean_abs_pct(cp1_hidden.grad, bad_hidden.grad)
            > MEAN_ABS_PCT_MISMATCH_THRESHOLD
        )
        _, param_pct = parameter_grad_mean_abs_pct_with_name(cp1_gdn, bad_gdn)
        assert param_pct > MEAN_ABS_PCT_MISMATCH_THRESHOLD


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is required for real Megatron/FLA GDN sibling-order coverage.",
)
def test_real_qwen35_gdn_sibling_outputs_are_order_independent() -> None:
    case = GdnPhase0Case(
        name="sibling_swap",
        sequence_length=16,
        rows=(
            GdnPackedRowShape(
                families=(GdnFamilyShape(prefix_length=5, suffix_lengths=(3, 4)),)
            ),
        ),
        seed=59,
    )
    with _single_rank_model_parallel():
        gdn, _ = _make_matching_qwen35_gdn_pair()
        device = torch.device("cuda")
        tensors = build_phase0_packed_tensors(case)
        group_ids = tensors["group_ids"].to(device)
        parent_ids = tensors["parent_ids"].to(device)
        hidden_states = torch.randn(
            case.sequence_length,
            1,
            64,
            device=device,
            dtype=GDN_CORRECTNESS_DTYPE,
            generator=torch.Generator(device=device).manual_seed(20340426),
        )
        swapped_hidden = hidden_states.clone()
        swapped_hidden[5:9] = hidden_states[8:12]
        swapped_hidden[9:12] = hidden_states[5:8]
        swapped_group_ids = torch.full_like(group_ids, -1)
        swapped_parent_ids = torch.full_like(parent_ids, -1)
        swapped_group_ids[0, :5] = 0
        swapped_parent_ids[0, :5] = 0
        swapped_group_ids[0, 5:9] = 1
        swapped_parent_ids[0, 5:9] = 0
        swapped_group_ids[0, 9:12] = 2
        swapped_parent_ids[0, 9:12] = 0

        with torch.no_grad():
            original_out, _ = gdn_shared_prefix_forward(
                gdn,
                hidden_states,
                group_ids=group_ids,
                parent_ids=parent_ids,
            )
            swapped_out, _ = gdn_shared_prefix_forward(
                gdn,
                swapped_hidden,
                group_ids=swapped_group_ids,
                parent_ids=swapped_parent_ids,
            )

        assert_mean_abs_pct(original_out[:5], swapped_out[:5], "prefix")
        assert_mean_abs_pct(original_out[8:12], swapped_out[5:9], "sibling_1")
        assert_mean_abs_pct(original_out[5:8], swapped_out[9:12], "sibling_2")


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is required for real Megatron/FLA GDN mixed CP coverage.",
)
@pytest.mark.parametrize("cp_size", (2, 4, 8))
def test_real_qwen35_gdn_mixed_local_fork_and_chain_matches_cp1(
    cp_size: int,
) -> None:
    case = GdnPhase0Case(
        name="mixed_local_fork_and_chain",
        sequence_length=128,
        rows=(
            GdnPackedRowShape(
                families=(
                    GdnFamilyShape(prefix_length=4, suffix_lengths=(2, 3, 2)),
                    GdnFamilyShape(prefix_length=30, suffix_lengths=(35, 5)),
                )
            ),
        ),
        seed=41,
    )
    with _single_rank_model_parallel():
        cp1_gdn, mixed_gdn = _make_matching_qwen35_gdn_pair()
        device = torch.device("cuda")
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
            generator=torch.Generator(device=device).manual_seed(20300426 + cp_size),
        )
        output_grad = (
            torch.randn(
                hidden_states.shape,
                device=device,
                dtype=GDN_CORRECTNESS_DTYPE,
                generator=torch.Generator(device=device).manual_seed(
                    20310426 + cp_size
                ),
            )
            * real_token_mask
        )
        loss_denominator = real_token_mask.expand_as(output_grad).sum()
        cp1_hidden = hidden_states.clone().detach().requires_grad_(True)
        mixed_hidden = hidden_states.clone().detach().requires_grad_(True)

        cp1_out, _ = gdn_shared_prefix_forward(
            cp1_gdn,
            cp1_hidden,
            group_ids=group_ids,
            parent_ids=parent_ids,
        )
        mixed_out = run_real_gdn_mixed_cp_reference(
            mixed_gdn,
            mixed_hidden,
            group_ids=group_ids,
            parent_ids=parent_ids,
            cp_size=cp_size,
            local_fork_max_tokens=16,
        )
        cp1_loss = stable_output_mse_loss(
            cp1_out,
            output_grad,
            mask=real_token_mask,
            denominator=loss_denominator,
        )
        mixed_loss = stable_output_mse_loss(
            mixed_out,
            output_grad,
            mask=real_token_mask,
            denominator=loss_denominator,
        )
        cp1_loss.backward()
        mixed_loss.backward()

        param_name, param_pct = parameter_grad_mean_abs_pct_with_name(
            cp1_gdn, mixed_gdn
        )
        assert_scalar_loss_close(cp1_loss.detach(), mixed_loss.detach(), case.name)
        assert_mean_abs_pct(
            cp1_out.detach(),
            mixed_out.detach(),
            case.name,
            threshold=REAL_GDN_OUTPUT_MEAN_ABS_PCT_THRESHOLD,
        )
        assert cp1_hidden.grad is not None
        assert mixed_hidden.grad is not None
        assert_mean_abs_pct(
            cp1_hidden.grad,
            mixed_hidden.grad,
            case.name,
            threshold=REAL_GDN_GRAD_MEAN_ABS_PCT_THRESHOLD,
        )
        assert param_pct <= REAL_GDN_GRAD_MEAN_ABS_PCT_THRESHOLD, param_name


def _real_token_mean_abs_pct(
    left: torch.Tensor,
    right: torch.Tensor,
    group_ids: torch.Tensor,
) -> float:
    real_mask = (group_ids != -1).transpose(0, 1)
    return mean_abs_pct(left[real_mask], right[real_mask])
