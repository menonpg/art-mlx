from __future__ import annotations

import os
from pathlib import Path

from ..metrics import DEFAULT_MEAN_ABS_PCT_THRESHOLD
from ..model_support.oracle_harness import (
    FlexBackend,
    LoraConfig,
    MetricThresholdRule,
    OracleCaseConfig,
    PackedTensorConfig,
    PhasePassFn,
    SensitivityMutation,
    StepTrace,
    Topology,
    VariantReport,
    VariantRunner,
    VariantSpec,
    WorkerRunRequest,
)
from .megatron_attention_oracle_worker import (
    run_worker_subprocess as run_attention_worker_subprocess,
)

ATTN_SENSITIVITY_MUTATION_ENV = "ART_ATTN_SENSITIVITY_MUTATIONS"
ATTN_TOPOLOGY_INDICES_ENV = "ART_ATTN_TOPOLOGY_INDICES"

ATTN_SENSITIVITY_MUTATIONS = (
    "attn_kv_fetch_pack_on_comm_stream",
    "attn_skip_nested_grad_sanitize",
    "attn_skip_flash_lse_normalize",
)

ATTN_TOPOLOGIES = [
    Topology(tp=1, ep=1, etp=1, dp=1, cp=2, sp=False),
    Topology(tp=2, ep=1, etp=1, dp=1, cp=2, sp=True),
    Topology(tp=1, ep=1, etp=1, dp=1, cp=4, sp=False),
    Topology(tp=2, ep=1, etp=1, dp=1, cp=4, sp=True),
    Topology(tp=1, ep=1, etp=1, dp=1, cp=8, sp=False),
]

ATTN_SENSITIVITY_TOPOLOGY_BY_MUTATION = {
    "attn_kv_fetch_pack_on_comm_stream": Topology(
        tp=2, ep=1, etp=1, dp=1, cp=2, sp=True
    ),
    "attn_skip_nested_grad_sanitize": Topology(tp=1, ep=1, etp=1, dp=1, cp=2, sp=False),
    "attn_skip_flash_lse_normalize": Topology(tp=1, ep=1, etp=1, dp=1, cp=4, sp=False),
}


def attention_case_config(
    base_model: str = "Qwen/Qwen3-30B-A3B-Instruct-2507",
) -> OracleCaseConfig:
    return OracleCaseConfig(
        base_model=base_model,
        precision="bf16",
        num_layers=1,
        packed_tensors=PackedTensorConfig(
            num_sequences=4,
            sequence_length=1024,
            prefill_tokens=256,
            completion_branches_per_prefix=2,
            decode_tokens=128,
            decode_tokens_jitter=32,
            packing_mode="stop_early",
            vocab_high=8192,
        ),
        lora=LoraConfig(
            rank=1,
            alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        ),
    )


def attention_sensitivity_mutations() -> list[str]:
    raw = os.environ.get(ATTN_SENSITIVITY_MUTATION_ENV)
    if raw is None or raw.strip() == "":
        return []
    normalized = raw.strip().lower()
    if normalized == "all":
        return list(ATTN_SENSITIVITY_MUTATIONS)
    mutations = [item.strip().lower() for item in raw.split(",") if item.strip()]
    unsupported = [
        mutation for mutation in mutations if mutation not in ATTN_SENSITIVITY_MUTATIONS
    ]
    if unsupported:
        supported = ", ".join(ATTN_SENSITIVITY_MUTATIONS)
        raise ValueError(
            f"Unsupported {ATTN_SENSITIVITY_MUTATION_ENV} value '{raw}'. "
            f"Supported values: {supported}, CSV of supported values, or all."
        )
    return mutations


def attention_sensitivity_enabled() -> bool:
    return bool(attention_sensitivity_mutations())


def attention_required_world_size(mutations: list[str]) -> int:
    return max(
        ATTN_SENSITIVITY_TOPOLOGY_BY_MUTATION[mutation].world_size()
        for mutation in mutations
    )


def _selected_attention_topologies() -> list[tuple[int, Topology]]:
    raw = os.environ.get(ATTN_TOPOLOGY_INDICES_ENV, "all")
    normalized = raw.strip().lower()
    if normalized in {"", "all"}:
        return list(enumerate(ATTN_TOPOLOGIES))
    selected: list[int] = []
    seen: set[int] = set()
    for item in raw.split(","):
        stripped = item.strip()
        if not stripped:
            continue
        index = int(stripped)
        if index in seen:
            continue
        selected.append(index)
        seen.add(index)
    invalid = [index for index in selected if index not in range(len(ATTN_TOPOLOGIES))]
    if invalid:
        available = ", ".join(
            f"{index}:{topology.slug()}"
            for index, topology in enumerate(ATTN_TOPOLOGIES)
        )
        raise ValueError(
            f"Unsupported {ATTN_TOPOLOGY_INDICES_ENV} indices {invalid}. "
            f"Available topology candidates: {available}"
        )
    return [(index, ATTN_TOPOLOGIES[index]) for index in selected]


def _attention_phase_pass_fns() -> dict[str, PhasePassFn]:
    metric_rule = MetricThresholdRule(
        limits={"mean_abs_pct": DEFAULT_MEAN_ABS_PCT_THRESHOLD}
    )
    return {
        "forward": metric_rule,
        "outputs": metric_rule,
        "losses": metric_rule,
        "grads": metric_rule,
        "deltas": metric_rule,
    }


class AttentionVariantRunner(VariantRunner):
    """Runs the attention-only oracle with its dedicated worker and no routing replay."""

    def _run_topology(
        self,
        *,
        topology: Topology,
        output_slug: str,
        mutation: SensitivityMutation | None,
        replay_bundle_dir: Path | None,
        capture_bundle_dir: Path | None,
        regenerate: bool,
        flex_backend: FlexBackend | None = None,
    ) -> Path:
        del replay_bundle_dir, capture_bundle_dir
        topology_dir = self.case_dir / output_slug
        manifest_path = topology_dir / "manifest.json"
        if manifest_path.exists() and not regenerate:
            return topology_dir
        from ..model_support.oracle_harness import REPO_ROOT, _replace_topology_dir

        _replace_topology_dir(topology_dir)
        request = WorkerRunRequest(
            case_id=self.case_id,
            objective=self.objective,
            case_config=self.case_config,
            topology=topology,
            topology_dir=str(topology_dir),
            packed_tensors=self.case_artifacts.packed_tensors,
            shared_init_adapter_path=str(self.shared_init_path),
            mutation=mutation,
            moe_routing_replay_path=None,
            moe_routing_replay_strict=True,
            capture_moe_routing_bundle_path=None,
            flex_backend=flex_backend,
        )
        run_attention_worker_subprocess(request, topology_dir, repo_root=REPO_ROOT)
        return topology_dir


def run_attention_suite(
    *,
    case_config: OracleCaseConfig,
    max_world_size: int | None = None,
) -> list[VariantReport]:
    phase_pass = _attention_phase_pass_fns()
    variants: list[VariantSpec] = []
    for _, topology in _selected_attention_topologies():
        if max_world_size is not None and topology.world_size() > max_world_size:
            continue
        variants.append(
            VariantSpec(
                name=f"attention_{topology.slug()}",
                topology=topology,
                output_slug=f"{topology.slug()}__flash_attention",
                pass_fn_by_phase=phase_pass,
                flex_backend="FLASH",
            )
        )
    runner = AttentionVariantRunner(
        case_config=case_config,
        oracle_flex_backend="TRITON_LEGACY",
    )
    return runner.run_suite(variants)


def run_attention_sensitivity_suite(
    *,
    case_config: OracleCaseConfig,
    mutations: list[str],
) -> list[VariantReport]:
    phase_pass = _attention_phase_pass_fns()
    variants = [
        VariantSpec(
            name=f"attention_sensitivity_{mutation}",
            topology=ATTN_SENSITIVITY_TOPOLOGY_BY_MUTATION[mutation],
            mutation=mutation,
            output_slug=f"{ATTN_SENSITIVITY_TOPOLOGY_BY_MUTATION[mutation].slug()}__{mutation}",
            expected_signal="fail",
            pass_fn_by_phase=phase_pass,
            flex_backend="FLASH",
        )
        for mutation in mutations
    ]
    runner = AttentionVariantRunner(
        case_config=case_config,
        oracle_flex_backend="TRITON_LEGACY",
    )
    return runner.run_suite(variants)
