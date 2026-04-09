from collections import Counter

import torch

from art.megatron.model_support.spec import ArchitectureReport, LayerFamilyInstance
from art.megatron.provider import get_provider_bundle


def summarize_layer_families(
    layer_families: list[LayerFamilyInstance],
) -> list[LayerFamilyInstance]:
    counts = Counter(family.key for family in layer_families)
    return [
        LayerFamilyInstance(key=key, count=count)
        for key, count in sorted(counts.items())
    ]


def inspect_architecture(
    base_model: str,
    *,
    torch_dtype: torch.dtype = torch.bfloat16,
) -> ArchitectureReport:
    provider_bundle = get_provider_bundle(base_model, torch_dtype=torch_dtype)
    discovered = provider_bundle.handler.collect_layer_families(
        provider_bundle.provider
    )
    summarized = summarize_layer_families(discovered)
    unresolved_risks: list[str] = []
    if not summarized:
        unresolved_risks.append(
            "handler did not report any layer families; codex review is required"
        )
    return ArchitectureReport(
        base_model=base_model,
        model_key=provider_bundle.spec.key,
        handler_key=provider_bundle.handler.key,
        bridge_type=type(provider_bundle.bridge._model_bridge).__name__,
        provider_type=type(provider_bundle.provider).__name__,
        layer_families=summarized,
        recommended_min_layers=max(len(summarized), 1),
        unresolved_risks=unresolved_risks,
    )
