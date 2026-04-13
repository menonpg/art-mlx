from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch

from art.megatron.model_support.spec import MinimalLayerCoverageReport

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
        "integration.megatron_hf_parity.assess_minimal_layer_coverage",
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
        "integration.megatron_hf_parity.assess_minimal_layer_coverage",
        lambda **_: coverage,
    )
    monkeypatch.setattr(
        "integration.megatron_hf_parity.ensure_case_artifacts",
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

    monkeypatch.setattr(
        "integration.megatron_hf_parity.run_hf_parity_subprocess",
        _fake_subprocess,
    )

    report = run_hf_parity(
        case_config=OracleCaseConfig(base_model="Qwen/Qwen3.5-35B-A3B")
    )

    assert calls == ["fresh-case"]
    assert report.case_id == "fresh-case"
    assert report.pass_count == 1


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


def test_build_megatron_runtime_uses_single_gpu_parity_provider_bundle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, object]] = []
    fake_model = torch.nn.Linear(1, 1)
    fake_model.config = SimpleNamespace(num_layers=4)  # type: ignore[attr-defined]

    class _FakeProvider:
        def provide_distributed_model(self, **kwargs):
            return [fake_model]

    fake_provider = _FakeProvider()
    fake_bundle = SimpleNamespace(
        provider=fake_provider,
        bridge="bridge",
        handler="handler",
        spec="spec",
    )

    monkeypatch.setattr(
        "integration.megatron_hf_parity_worker.get_provider_bundle",
        lambda *args, **kwargs: (
            calls.append(("bundle", {"args": args, "kwargs": kwargs})) or fake_bundle
        ),
    )
    monkeypatch.setattr(
        "integration.megatron_hf_parity_worker._configure_provider",
        lambda provider, topology, case_config: calls.append(
            (
                "configure",
                {
                    "provider": provider,
                    "topology": topology,
                    "case_config": case_config,
                },
            )
        ),
    )
    monkeypatch.setattr(
        "integration.megatron_hf_parity_worker.megatron_train._install_gpt_preprocess_hook",
        lambda model: None,
    )
    monkeypatch.setattr(
        "integration.megatron_hf_parity_worker.megatron_train._build_optimizer",
        lambda model, optimizer_config: "optimizer",
    )
    monkeypatch.setattr(
        "integration.megatron_hf_parity_worker.megatron_train.TrainingRuntime",
        lambda **kwargs: SimpleNamespace(**kwargs),
    )
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 1)

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

    runtime = _build_megatron_runtime(request)

    assert runtime.provider is fake_provider
    bundle_call = next(payload for name, payload in calls if name == "bundle")
    assert bundle_call["kwargs"]["runtime_profile"] == "single_gpu_parity"
    assert [name for name, _ in calls] == ["bundle", "configure"]
    assert calls[0][1] == {
        "args": ("Qwen/Qwen3.5-35B-A3B",),
        "kwargs": {
            "torch_dtype": torch.float32,
            "runtime_profile": "single_gpu_parity",
        },
    }
    configured = cast(dict[str, Any], calls[1][1])
    assert configured["provider"] is fake_provider


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
