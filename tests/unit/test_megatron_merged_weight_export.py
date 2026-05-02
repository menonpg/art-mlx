import sys
from types import ModuleType, SimpleNamespace

import torch

from art.megatron import merged_weight_export
from art.megatron.jobs import MergedWeightTransferInitInfo, MergedWeightTransferSpec


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


def test_ensure_merged_weight_transfer_group_short_circuits_on_matching_init() -> None:
    spec = MergedWeightTransferSpec(
        init_info=MergedWeightTransferInitInfo(
            master_address="127.0.0.1",
            master_port=2345,
            rank_offset=1,
            world_size=2,
        ),
        vllm_base_url="http://127.0.0.1:8000",
        served_model_name="test-model@1",
    )

    group, init_info = merged_weight_export.ensure_merged_weight_transfer_group(
        rank=0,
        world_size=1,
        merged_weight_transfer_group="group",
        merged_weight_transfer_init_info=spec.init_info,
        spec=spec,
    )

    assert group == "group"
    assert init_info == spec.init_info


def test_sync_merged_weights_to_vllm_posts_update_payload(
    monkeypatch,
) -> None:
    sent_weights: list[list[tuple[str, torch.Tensor]]] = []
    http_calls: list[tuple[str, dict | None, dict | None]] = []

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

    class FakeClient:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            del exc_type, exc, tb
            return None

        def post(
            self,
            url: str,
            json: dict | None = None,
            params: dict | None = None,
            timeout: float | None = None,
        ) -> FakeResponse:
            del timeout
            http_calls.append((url, json, params))
            return FakeResponse()

    httpx_module = ModuleType("httpx")
    setattr(httpx_module, "Client", FakeClient)

    monkeypatch.setitem(sys.modules, "httpx", httpx_module)
    monkeypatch.setattr(
        merged_weight_export,
        "trainer_send_weights",
        lambda iterator, options: sent_weights.append(list(iterator)),
    )
    monkeypatch.setattr(
        merged_weight_export,
        "ensure_merged_weight_transfer_group",
        lambda **_: ("group", "init"),
    )
    monkeypatch.setattr(
        merged_weight_export,
        "build_merged_weight_export",
        lambda **_: "export",
    )
    monkeypatch.setattr(
        merged_weight_export,
        "iter_merged_vllm_weights",
        lambda export: iter(
            [
                ("a", torch.zeros(2, dtype=torch.float32)),
                ("b", torch.ones(1, dtype=torch.bfloat16)),
            ]
        ),
    )
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)

    spec = MergedWeightTransferSpec(
        init_info=MergedWeightTransferInitInfo(
            master_address="127.0.0.1",
            master_port=2345,
            rank_offset=1,
            world_size=2,
        ),
        vllm_base_url="http://127.0.0.1:8000",
        served_model_name="test-model@1",
    )

    group, init_info = merged_weight_export.sync_merged_weights_to_vllm(
        bridge="bridge",
        model=[torch.nn.Linear(1, 1)],
        model_support_handler="handler",
        rank=0,
        world_size=1,
        merged_weight_transfer_group=None,
        merged_weight_transfer_init_info=None,
        spec=spec,
        pause_generation=True,
    )

    assert group == "group"
    assert init_info == "init"
    assert len(sent_weights) == 1
    assert len(sent_weights[0]) == 2
    assert sent_weights[0][0][0] == "a"
    assert torch.equal(sent_weights[0][0][1], torch.zeros(2, dtype=torch.float32))
    assert sent_weights[0][1][0] == "b"
    assert torch.equal(sent_weights[0][1][1], torch.ones(1, dtype=torch.bfloat16))
    assert http_calls == [
        ("http://127.0.0.1:8000/pause", None, {"mode": "wait"}),
        (
            "http://127.0.0.1:8000/update_weights",
            {
                "update_info": {
                    "names": ["a", "b"],
                    "dtype_names": ["float32", "bfloat16"],
                    "shapes": [[2], [1]],
                    "is_checkpoint_format": True,
                    "packed": True,
                    "packed_buffer_size_bytes": merged_weight_export.DEFAULT_PACKED_BUFFER_SIZE_BYTES,
                    "packed_num_buffers": merged_weight_export.DEFAULT_PACKED_NUM_BUFFERS,
                }
            },
            None,
        ),
        (
            "http://127.0.0.1:8000/art/set_served_model_name",
            {"name": "test-model@1"},
            None,
        ),
        ("http://127.0.0.1:8000/resume", None, None),
    ]
