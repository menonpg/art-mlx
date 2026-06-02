from __future__ import annotations

import json

from artifacts import Dsv4Metric, write_manifest, write_readable_summary
from cases import default_validation_cases, randomized_repeated_case
from packed_layout import (
    build_dsv4_packed_tensors,
    format_case_summary,
    summarize_case,
)
import torch


def test_default_validation_cases_cover_required_diagnostics() -> None:
    names = {case.name for case in default_validation_cases()}
    assert {
        "single_family_two_branches",
        "multi_family_repeated",
        "ragged_family_mix",
        "dominant_family",
        "long_sibling",
        "padding_tail",
        "cp_boundary_prefix",
        "cp_boundary_completion",
        "family_boundary_at_partition",
        "empty_trailing_rank",
        "csa_ratio_boundary",
        "hca_ratio_boundary",
        "swa_boundary",
        "topk_tie_or_near_tie",
        "no_stage_keys",
    }.issubset(names)


def test_randomized_case_is_deterministic_and_varies_completion_lengths() -> None:
    case_a = randomized_repeated_case(
        name="weak_scale_probe",
        sequence_length=1024,
        seed=123,
        row_count=2,
        prefix_length=64,
        completion_count=6,
        completion_mean=24,
        completion_jitter=8,
        prefix_jitter=8,
    )
    case_b = randomized_repeated_case(
        name="weak_scale_probe",
        sequence_length=1024,
        seed=123,
        row_count=2,
        prefix_length=64,
        completion_count=6,
        completion_mean=24,
        completion_jitter=8,
        prefix_jitter=8,
    )
    case_c = randomized_repeated_case(
        name="weak_scale_probe",
        sequence_length=1024,
        seed=124,
        row_count=2,
        prefix_length=64,
        completion_count=6,
        completion_mean=24,
        completion_jitter=8,
        prefix_jitter=8,
    )
    assert case_a == case_b
    assert case_a != case_c
    per_family_lengths = [
        family.completion_lengths for row in case_a.rows for family in row.families
    ]
    assert any(len(set(lengths)) > 1 for lengths in per_family_lengths)
    assert len({lengths for lengths in per_family_lengths}) > 1
    assert summarize_case(case_a).completion_lengths_vary


def test_packed_tensors_preserve_masked_context_token_mechanic() -> None:
    case = next(
        case
        for case in default_validation_cases()
        if case.name == "single_family_two_branches"
    )
    tensors = build_dsv4_packed_tensors(case)
    family = case.rows[0].families[0]
    first_completion_start = family.prefix_length
    first_completion_end = first_completion_start + family.completion_lengths[0]
    assert tensors["tokens"].shape == (1, case.sequence_length)
    assert tensors["assistant_mask"][0, : family.prefix_length].sum().item() == 0
    assert not bool(tensors["assistant_mask"][0, first_completion_start])
    assert tensors["assistant_mask"][
        0, first_completion_start + 1 : first_completion_end
    ].all()
    assert torch.isnan(tensors["logprobs"][0, first_completion_start])
    assert torch.isfinite(
        tensors["logprobs"][0, first_completion_start + 1 : first_completion_end]
    ).all()
    assert (
        tensors["parent_ids"][0, first_completion_start:first_completion_end]
        == tensors["group_ids"][0, 0]
    ).all()
    assert tensors["group_ids"][0, first_completion_end:].eq(-1).any()


def test_case_summaries_are_readable_and_flag_boundary_cases() -> None:
    cases = {case.name: case for case in default_validation_cases()}
    prefix_summary = summarize_case(cases["cp_boundary_prefix"])
    completion_summary = summarize_case(cases["cp_boundary_completion"])
    hca_summary = summarize_case(cases["hca_ratio_boundary"])
    text = format_case_summary(completion_summary)
    assert prefix_summary.cp_boundary_prefix
    assert completion_summary.cp_boundary_completion
    assert hca_summary.hca_ratio_boundary
    assert "cp_boundary_completion" in text
    assert "completion_lengths=" in text
    assert "valid_lengths=" in text


def test_artifact_manifest_and_summary_are_readable(tmp_path) -> None:
    case = next(
        case for case in default_validation_cases() if case.name == "swa_boundary"
    )
    case_summary = summarize_case(case)
    metric = Dsv4Metric(
        name="mean_abs_pct",
        value=0.12,
        unit="%",
        threshold=0.5,
        passed=True,
    )
    manifest_path = write_manifest(
        tmp_path,
        kind="correctness",
        command=["pytest", "tests/integration/dsv4_context_parallel"],
        configs={"topology": "cp4", "dtype": "fp32", "seed": case.seed},
        cases=(case_summary,),
        metrics=(metric,),
        caveats=("cpu scaffold only",),
    )
    summary_path = write_readable_summary(
        tmp_path,
        title="DSV4 CP validation scaffold",
        status="pass",
        manifest_path=manifest_path,
        case_summaries=(case_summary,),
        metrics=(metric,),
        caveats=("cpu scaffold only",),
    )
    manifest = json.loads(manifest_path.read_text())
    summary = summary_path.read_text()
    assert manifest["kind"] == "correctness"
    assert manifest["configs"]["topology"] == "cp4"
    assert "status: pass" in summary
    assert "DSV4 CP validation scaffold" in summary
    assert "swa_boundary" in summary
    assert "mean_abs_pct" in summary
    assert "cpu scaffold only" in summary
