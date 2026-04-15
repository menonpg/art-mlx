from typing import Any, Literal, Protocol, Sequence

from pydantic import BaseModel, Field

RolloutWeightsMode = Literal["lora", "merged"]
NativeVllmLoraStatus = Literal["disabled", "wip", "validated"]


class DependencyFloor(BaseModel):
    transformers: str | None = None
    vllm: str | None = None
    megatron_bridge: str | None = None


class LayerFamilyInstance(BaseModel):
    key: str
    count: int = 1
    layer_index: int | None = None
    module_path: str | None = None
    module_type: str | None = None


class ArchitectureReport(BaseModel):
    base_model: str
    model_key: str
    handler_key: str
    bridge_type: str | None = None
    provider_type: str | None = None
    layer_families: list[LayerFamilyInstance] = Field(default_factory=list)
    recommended_min_layers: int = 1
    unresolved_risks: list[str] = Field(default_factory=list)


class MinimalLayerCoverageReport(BaseModel):
    base_model: str
    model_key: str
    requested_num_layers: int
    recommended_min_layers: int
    covered: bool
    missing_layer_families: list[str] = Field(default_factory=list)
    unresolved_risks: list[str] = Field(default_factory=list)


class ValidationStageResult(BaseModel):
    name: str
    passed: bool = False
    metrics: dict[str, Any] = Field(default_factory=dict)
    artifact_dir: str | None = None


class ValidationReport(BaseModel):
    base_model: str
    model_key: str
    dependency_versions: dict[str, str] = Field(default_factory=dict)
    stages: list[ValidationStageResult] = Field(default_factory=list)


class ModelSupportSpec(BaseModel):
    key: str
    handler_key: str
    model_names: tuple[str, ...] = ()
    default_target_modules: tuple[str, ...]
    default_rollout_weights_mode: RolloutWeightsMode = "lora"
    native_vllm_lora_status: NativeVllmLoraStatus = "disabled"
    dependency_floor: DependencyFloor = Field(default_factory=DependencyFloor)


class ModelSupportHandler(Protocol):
    key: str

    def patch_provider(self, provider: Any, bridge: Any) -> None: ...

    def install_preprocess_patch(self, model_chunks: Sequence[Any]) -> None: ...

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

    def build_adapter_weights_by_base(
        self,
        model_chunks: Sequence[Any],
    ) -> dict[str, list[Any]]: ...

    def get_forward_kwargs(self, model: Any, **kwargs: Any) -> dict[str, Any]: ...
