"""Probe stock vLLM native LoRA key handling for ART canonical adapters.

Run with the vLLM runtime interpreter, not ART's venv:
  ./vllm_runtime/.venv/bin/python tests/integration/vllm_separation/probe_native_vllm_lora_layout.py
"""

from __future__ import annotations

import json
from tempfile import TemporaryDirectory

from safetensors.torch import save_file
import torch
from transformers import AutoConfig
from vllm.lora.lora_model import LoRAModel
from vllm.lora.peft_helper import PEFTHelper
from vllm.lora.utils import parse_fine_tuned_lora_name
from vllm.model_executor.models.qwen3_vl import Qwen3VLForConditionalGeneration

MODELS = (
    "Qwen/Qwen3.5-4B",
    "Qwen/Qwen3.5-35B-A3B",
    "Qwen/Qwen3.6-27B",
    "Qwen/Qwen3.6-35B-A3B",
)


def _parse(key: str) -> str:
    return parse_fine_tuned_lora_name(
        key,
        Qwen3VLForConditionalGeneration.hf_to_vllm_mapper,
    )[0]


def _load_modules(tensors: dict[str, torch.Tensor]) -> tuple[str, list[str]]:
    with TemporaryDirectory() as tmpdir:
        with open(f"{tmpdir}/adapter_config.json", "w") as handle:
            json.dump(
                {
                    "r": 2,
                    "lora_alpha": 2,
                    "target_modules": ["experts"],
                    "bias": "none",
                },
                handle,
            )
        save_file(tensors, f"{tmpdir}/adapter_model.safetensors")
        peft = PEFTHelper.from_local_dir(tmpdir, max_position_embeddings=None)
        try:
            lora = LoRAModel.from_local_checkpoint(
                tmpdir,
                {"experts"},
                peft,
                lora_model_id=1,
                device="cpu",
                weights_mapper=Qwen3VLForConditionalGeneration.hf_to_vllm_mapper,
            )
        except Exception as exc:
            return type(exc).__name__, [str(exc)]
        return "ok", sorted(lora.loras)


def main() -> None:
    print("hf_architectures")
    for model in MODELS:
        config = AutoConfig.from_pretrained(model, trust_remote_code=True)
        print(
            model,
            getattr(config, "architectures", None),
            getattr(config, "model_type", None),
        )

    canonical_dense = "base_model.model.model.layers.0.mlp.down_proj.lora_A.weight"
    qwen_wrapper_dense = (
        "base_model.model.model.language_model.layers.0.mlp.down_proj.lora_A.weight"
    )
    print("dense_key_parse")
    print("canonical", canonical_dense, "->", _parse(canonical_dense))
    print("qwen_wrapper", qwen_wrapper_dense, "->", _parse(qwen_wrapper_dense))

    canonical_moe = {
        "base_model.model.model.layers.0.mlp.experts.0.gate_proj.lora_A.weight": torch.zeros(
            2, 4
        ),
        "base_model.model.model.layers.0.mlp.experts.0.gate_proj.lora_B.weight": torch.zeros(
            4, 2
        ),
        "base_model.model.model.layers.0.mlp.experts.0.up_proj.lora_A.weight": torch.zeros(
            2, 4
        ),
        "base_model.model.model.layers.0.mlp.experts.0.up_proj.lora_B.weight": torch.zeros(
            4, 2
        ),
        "base_model.model.model.layers.0.mlp.experts.0.down_proj.lora_A.weight": torch.zeros(
            2, 4
        ),
        "base_model.model.model.layers.0.mlp.experts.0.down_proj.lora_B.weight": torch.zeros(
            4, 2
        ),
    }
    fused_runtime_moe = {
        "base_model.model.model.language_model.layers.0.mlp.experts.base_layer.lora_A.weight": torch.zeros(
            4, 4
        ),
        "base_model.model.model.language_model.layers.0.mlp.experts.base_layer.lora_B.weight": torch.zeros(
            8, 4
        ),
        "base_model.model.model.language_model.layers.0.mlp.experts.lora_A.weight": torch.zeros(
            4, 4
        ),
        "base_model.model.model.language_model.layers.0.mlp.experts.lora_B.weight": torch.zeros(
            4, 4
        ),
    }
    print("moe_checkpoint_load")
    print("canonical_per_expert", _load_modules(canonical_moe))
    print("fused_runtime", _load_modules(fused_runtime_moe))


if __name__ == "__main__":
    main()
