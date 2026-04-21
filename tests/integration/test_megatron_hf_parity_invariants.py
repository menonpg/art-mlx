from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch

from art.megatron.model_support.spec import MinimalLayerCoverageReport

from . import megatron_hf_parity as hf_parity_module
from . import megatron_hf_parity_worker as hf_parity_worker_module
from .megatron_hf_parity import (
    HF_PARITY_OUTPUT_DIRNAME,
    HF_PARITY_REPORT_FILENAME,
    HfParityReport,
    HfParityRunRequest,
    build_parity_sample_indices,
    build_tensor_map_metric_rows,
    run_hf_parity,
    set_hf_config_num_layers,
)
from .megatron_hf_parity_worker import (
    _build_megatron_runtime,
    _filter_language_only_tensor_map,
    _is_language_hf_param_name,
    _mapping_supports_derivative_parity,
    _normalize_hf_grads_for_bridge,
    _normalize_hf_tensor_map_for_bridge,
)
from .megatron_oracle_harness import DiskPackedTensorsSpec, OracleCaseConfig


def test_build_parity_sample_indices_pads_with_none() -> None:
    assert build_parity_sample_indices(
        num_sequences=2,
        global_grad_accumulation_sequences=4,
    ) == [0, 1, None, None]


def test_set_hf_config_num_layers_updates_supported_field() -> None:
    config = SimpleNamespace(num_hidden_layers=28)

    field = set_hf_config_num_layers(config, 4)

    assert field == "num_hidden_layers"
    assert config.num_hidden_layers == 4


def test_set_hf_config_num_layers_updates_nested_text_config() -> None:
    text_config = SimpleNamespace(
        num_hidden_layers=40,
        layer_types=["linear_attention", "linear_attention", "full_attention"] * 2,
        mlp_only_layers=[1, 4, 7],
    )
    config = SimpleNamespace(text_config=text_config)

    field = set_hf_config_num_layers(config, 4)

    assert field == "text_config.num_hidden_layers"
    assert text_config.num_hidden_layers == 4
    assert text_config.layer_types == [
        "linear_attention",
        "linear_attention",
        "full_attention",
        "linear_attention",
    ]
    assert text_config.mlp_only_layers == [1]


def test_run_hf_parity_rejects_uncovered_toy_model(monkeypatch) -> None:
    monkeypatch.setattr(
        hf_parity_module,
        "assess_minimal_layer_coverage",
        lambda **_: SimpleNamespace(
            covered=False,
            missing_layer_families=["standard_attention"],
            unresolved_risks=[],
        ),
    )

    with pytest.raises(
        AssertionError,
        match="HF parity toy model does not cover required layer families",
    ):
        run_hf_parity(
            case_config=OracleCaseConfig(
                base_model="Qwen/Qwen3.5-35B-A3B",
                num_layers=2,
            )
        )


def test_run_hf_parity_always_reruns_existing_report(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    coverage = MinimalLayerCoverageReport(
        base_model="Qwen/Qwen3.5-35B-A3B",
        model_key="qwen3_5_moe",
        requested_num_layers=4,
        recommended_min_layers=4,
        covered=True,
    )
    case_dir = tmp_path / "case"
    output_dir = case_dir / HF_PARITY_OUTPUT_DIRNAME
    output_dir.mkdir(parents=True)
    stale_report = HfParityReport(
        case_id="stale",
        base_model="Qwen/Qwen3.5-35B-A3B",
        model_key="qwen3_5_moe",
        requested_num_layers=4,
        coverage=coverage,
        signal="pass",
        pass_count=99,
        fail_count=0,
    )
    (output_dir / HF_PARITY_REPORT_FILENAME).write_text(
        stale_report.model_dump_json(indent=2),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        hf_parity_module,
        "assess_minimal_layer_coverage",
        lambda **_: coverage,
    )
    monkeypatch.setattr(
        hf_parity_module,
        "ensure_case_artifacts",
        lambda _: SimpleNamespace(
            case_id="fresh-case",
            case_dir=str(case_dir),
            packed_tensors=DiskPackedTensorsSpec(
                dir=str(case_dir / "packed"),
                num_sequences=4,
                sequence_length=8,
            ),
        ),
    )
    calls: list[str] = []

    def _fake_subprocess(request, run_output_dir):
        calls.append(request.case_id)
        fresh_report = HfParityReport(
            case_id=request.case_id,
            base_model=request.case_config.base_model,
            model_key=request.coverage.model_key,
            requested_num_layers=request.case_config.num_layers,
            coverage=request.coverage,
            signal="pass",
            pass_count=1,
            fail_count=0,
        )
        (run_output_dir / HF_PARITY_REPORT_FILENAME).write_text(
            fresh_report.model_dump_json(indent=2),
            encoding="utf-8",
        )

    monkeypatch.setattr(hf_parity_module, "run_hf_parity_subprocess", _fake_subprocess)

    report = run_hf_parity(
        case_config=OracleCaseConfig(base_model="Qwen/Qwen3.5-35B-A3B")
    )

    assert calls == ["fresh-case"]
    assert report.case_id == "fresh-case"
    assert report.pass_count == 1


def test_run_hf_parity_subprocess_does_not_override_recompute(monkeypatch, tmp_path) -> None:
    request = HfParityRunRequest(
        case_id="case-id",
        case_config=OracleCaseConfig(base_model="Qwen/Qwen3.5-35B-A3B"),
        packed_tensors=DiskPackedTensorsSpec(
            dir=str(tmp_path / "packed"),
            num_sequences=4,
            sequence_length=8,
        ),
        output_dir=str(tmp_path),
        coverage=MinimalLayerCoverageReport(
            base_model="Qwen/Qwen3.5-35B-A3B",
            model_key="qwen3_5_moe",
            requested_num_layers=4,
            recommended_min_layers=4,
            covered=True,
        ),
    )
    captured: dict[str, Any] = {}

    def _fake_run(*args, **kwargs):
        del args
        captured.update(kwargs)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(hf_parity_module.subprocess, "run", _fake_run)

    hf_parity_module.run_hf_parity_subprocess(request, tmp_path)

    env = cast(dict[str, str], captured["env"])
    assert "ART_MEGATRON_RECOMPUTE_GRANULARITY" not in env
    assert "ART_MEGATRON_RECOMPUTE_METHOD" not in env
    assert "ART_MEGATRON_RECOMPUTE_NUM_LAYERS" not in env
    assert "ART_MEGATRON_RECOMPUTE_MODULES" not in env


def test_normalize_hf_tensor_map_for_bridge_adds_language_model_prefix() -> None:
    normalized = _normalize_hf_tensor_map_for_bridge(
        {
            "model.layers.0.input_layernorm.weight": torch.ones(1),
            "lm_head.weight": torch.ones(1),
        },
        {
            "model.language_model.layers.0.input_layernorm.weight",
            "lm_head.weight",
        },
    )

    assert set(normalized) == {
        "model.language_model.layers.0.input_layernorm.weight",
        "lm_head.weight",
    }


def test_build_tensor_map_metric_rows_rejects_tensor_set_mismatch() -> None:
    rows = build_tensor_map_metric_rows(
        phase="grads",
        reference={"a": torch.ones(1)},
        candidate={"b": torch.ones(1)},
    )

    assert len(rows) == 1
    assert rows[0].param == "__tensor_set__"
    assert rows[0].pass_signal is False
    assert "missing=['a'] extra=['b']" in rows[0].failure_reasons[0]


def test_build_tensor_map_metric_rows_enforces_nonzero_per_tensor() -> None:
    rows = build_tensor_map_metric_rows(
        phase="grads",
        reference={"all_zero": torch.zeros(2), "active": torch.ones(2)},
        candidate={"all_zero": torch.zeros(2), "active": torch.ones(2)},
    )
    by_param = {row.param: row for row in rows}

    assert by_param["all_zero"].pass_signal is False
    assert by_param["active"].pass_signal is True


def test_language_hf_param_filter_keeps_text_and_drops_visual() -> None:
    assert _is_language_hf_param_name("model.layers.0.self_attn.q_proj.weight") is True
    assert _is_language_hf_param_name("model.visual.blocks.0.attn.qkv.weight") is False
    filtered = _filter_language_only_tensor_map(
        {
            "model.layers.0.self_attn.q_proj.weight": torch.ones(1),
            "model.visual.blocks.0.attn.qkv.weight": torch.ones(1),
        }
    )
    assert set(filtered) == {"model.layers.0.self_attn.q_proj.weight"}
    assert torch.equal(
        filtered["model.layers.0.self_attn.q_proj.weight"],
        torch.ones(1),
    )


def test_normalize_hf_grads_for_bridge_keeps_expected_key_set() -> None:
    normalized = _normalize_hf_grads_for_bridge(
        {
            "model.layers.0.input_layernorm.weight": torch.ones(1),
            "lm_head.weight": torch.ones(1),
            "model.visual.blocks.0.attn.qkv.weight": torch.ones(1),
        },
        expected_grad_keys={
            "model.language_model.layers.0.input_layernorm.weight",
            "lm_head.weight",
        },
    )

    assert set(normalized) == {
        "model.language_model.layers.0.input_layernorm.weight",
        "lm_head.weight",
    }


def test_build_megatron_runtime_uses_training_provider_bundle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []
    runtime = SimpleNamespace(provider="provider", model=["model"])

    monkeypatch.setattr(
        hf_parity_worker_module.megatron_train,
        "build_training_runtime",
        lambda **kwargs: calls.append(kwargs) or runtime,
    )

    request = HfParityRunRequest(
        case_id="case",
        case_config=OracleCaseConfig(base_model="Qwen/Qwen3.5-35B-A3B"),
        packed_tensors=DiskPackedTensorsSpec(
            dir="/tmp", num_sequences=4, sequence_length=8
        ),
        output_dir="/tmp/out",
        coverage=MinimalLayerCoverageReport(
            base_model="Qwen/Qwen3.5-35B-A3B",
            model_key="qwen3_5_moe",
            requested_num_layers=4,
            recommended_min_layers=4,
            covered=True,
        ),
    )

    built_runtime = _build_megatron_runtime(request)

    assert built_runtime is runtime
    assert len(calls) == 1
    kwargs = calls[0]
    assert kwargs["model_identifier"] == "Qwen/Qwen3.5-35B-A3B"
    assert kwargs["provider_torch_dtype"] == torch.float32
    assert kwargs["provider_bundle_configure"] is hf_parity_worker_module._install_bridge_timing_debug
    assert kwargs["print_env"] is False
    assert kwargs["trainable_parameter_mode"] == "base_model"
    configured_provider = SimpleNamespace()
    kwargs["provider_configure"](configured_provider)
    optimizer_config = kwargs["optimizer_config"]
    assert configured_provider.num_layers == request.case_config.num_layers
    assert optimizer_config.params_dtype == torch.float32


def test_mapping_supports_derivative_parity_rejects_affine_weight_exports() -> None:
    from megatron.bridge.models.conversion.param_mapping import (
        AutoMapping,
        RMSNorm2ZeroCenteredRMSNormMapping,
    )

    assert _mapping_supports_derivative_parity(AutoMapping("a", "b")) is True
    assert (
        _mapping_supports_derivative_parity(
            RMSNorm2ZeroCenteredRMSNormMapping("a", "b")
        )
        is False
    )
