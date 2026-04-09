from types import SimpleNamespace

import torch

from art.megatron import merged_weight_export


def test_build_merged_weight_export_dispatches_through_handler(monkeypatch) -> None:
    chunk = torch.nn.Linear(1, 1)
    chunk.config = object()  # type: ignore[attr-defined]
    model = [chunk]
    handler = SimpleNamespace(
        build_adapter_weights_by_base=lambda model_chunks: {
            "layer.weight": [model_chunks]
        }
    )
    monkeypatch.setattr(
        merged_weight_export,
        "build_art_conversion_tasks",
        lambda *, bridge, model: ["task", bridge, model],
    )

    weight_export = merged_weight_export.build_merged_weight_export(
        bridge="bridge",
        model=model,
        model_support_handler=handler,
    )

    assert weight_export.bridge == "bridge"
    assert len(weight_export.model) == 1
    assert weight_export.model[0] is chunk
    assert weight_export.model_config_value is chunk.config
    assert weight_export.conversion_tasks == ["task", "bridge", model]
    assert weight_export.adapter_weights_by_base == {"layer.weight": [model]}


def test_iter_merged_vllm_weights_merges_adapter_weights() -> None:
    tensor = torch.ones(2)
    task = SimpleNamespace(
        global_param_name="layer.weight",
        param_weight=tensor,
        megatron_module=object(),
    )

    class Mapping:
        is_grouped_export = False

        def megatron_to_hf(self, param_weight, megatron_module):
            del megatron_module
            return {"hf.weight": param_weight + 1}

    task.mapping = Mapping()

    class FakeModelBridge:
        def _merge_lora_adapter_weights(
            self,
            model,
            converted_weights_dict,
            adapter_weights,
        ):
            del model, adapter_weights
            return {"hf.weight": converted_weights_dict["hf.weight"] + 2}

        def maybe_modify_converted_hf_weight(
            self,
            task,
            converted_weights_dict,
            hf_state_dict,
        ):
            del task, hf_state_dict
            return {"hf.weight": converted_weights_dict["hf.weight"] + 3}

    weight_export = merged_weight_export.MergedWeightExport(
        bridge=SimpleNamespace(
            _model_bridge=FakeModelBridge(),
            hf_pretrained=SimpleNamespace(state=object()),
        ),
        model=[torch.nn.Linear(1, 1)],
        model_config_value=object(),
        conversion_tasks=[task],
        adapter_weights_by_base={"layer.weight": [object()]},
    )

    weights = dict(merged_weight_export.iter_merged_vllm_weights(weight_export))

    assert torch.equal(weights["hf.weight"], torch.full((2,), 7.0))
