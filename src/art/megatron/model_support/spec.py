from typing import Any, Literal, Protocol, Sequence

from pydantic import BaseModel, Field

RolloutWeightsMode = Literal["lora", "merged"]
NativeVllmLoraStatus = Literal["disabled", "wip", "validated"]


class DependencyFloor(BaseModel):
    transformers: str | None = None
    vllm: str | None = None
    megatron_bridge: str | None = None


class ValidationManifest(BaseModel):
    require_hf_parity: bool = True
    require_oracle_correctness: bool = True
    require_non_zero_forwards: bool = True
    require_non_zero_grads: bool = True
    require_non_zero_deltas: bool = True
    require_chat_template_validation: bool = True
    require_yes_no_trainability: bool = True


class LayerFamilyInstance(BaseModel):
    key: str
    count: int = 1


class ModelSupportSpec(BaseModel):
    key: str
    handler_key: str
    model_names: tuple[str, ...] = ()
    default_target_modules: tuple[str, ...]
    default_rollout_weights_mode: RolloutWeightsMode = "lora"
    native_vllm_lora_status: NativeVllmLoraStatus = "disabled"
    dependency_floor: DependencyFloor = Field(default_factory=DependencyFloor)
    validation: ValidationManifest = Field(default_factory=ValidationManifest)


class ModelSupportHandler(Protocol):
    key: str

    def patch_provider(self, provider: Any, bridge: Any) -> None: ...

    def collect_layer_families(self, provider: Any) -> list[LayerFamilyInstance]: ...

    def apply_lora_adapters(
        self,
        model_chunks: Sequence[Any],
        provider: Any,
        *,
        target_modules: list[str],
        rank: int,
        alpha: int,
    ) -> None: ...

    def build_adapter_weights(self, model_chunks: Sequence[Any]) -> dict[str, Any]: ...

    def get_forward_kwargs(self, model: Any, **kwargs: Any) -> dict[str, Any]: ...
