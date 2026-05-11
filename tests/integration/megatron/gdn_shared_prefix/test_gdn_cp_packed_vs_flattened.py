from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("megatron.bridge")
pytest.importorskip("megatron.bridge.models.qwen_vl.qwen35_vl_provider")

from megatron.core import parallel_state as ps  # noqa: E402
from torch.distributed import destroy_process_group, init_process_group  # noqa: E402
import torch.multiprocessing as mp  # noqa: E402

from art.megatron.gdn.gdn_shared_prefix import (  # noqa: E402
    build_gdn_rank_execution_plan,
    parse_gdn_shared_prefix_segments,
)
from art.megatron.gdn.operator import run_gdn_layer  # noqa: E402

from .packed_layout import build_phase0_packed_tensors  # noqa: E402
from .real_gdn_oracle import (  # noqa: E402
    run_real_gdn_flattened_reference,
    zero_parameter_grads,
)
from .test_gdn_cp_packed_correctness import (  # noqa: E402
    _assert_cp_matches_reference,
    _find_free_port,
    _hidden_and_grad,
    _packed_correctness_cases,
    _planner_config_for_case,
    _skip_without_gpus,
)
from .test_real_gdn_native_fla_cp import _make_matching_gdn_pair  # noqa: E402


@pytest.mark.parametrize("cp_size", (2, 4, 8))
def test_gdn_cp_packed_matches_flattened_all_edge_cases(
    cp_size: int, tmp_path: Path
) -> None:
    _skip_without_gpus(cp_size)
    port = _find_free_port()
    mp.spawn(
        _packed_vs_flattened_worker,
        args=(cp_size, port, str(tmp_path)),
        nprocs=cp_size,
        join=True,
    )
    for rank in range(cp_size):
        assert (tmp_path / f"packed_vs_flattened_rank_{rank}.ok").read_text() == "ok\n"


def _packed_vs_flattened_worker(
    rank: int,
    cp_size: int,
    port: int,
    output_dir: str,
) -> None:
    torch.cuda.set_device(rank)
    init_process_group(
        backend="nccl",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=cp_size,
    )
    try:
        ps.initialize_model_parallel(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            context_parallel_size=cp_size,
            expert_model_parallel_size=1,
        )
        flat_gdn, cp_gdn = _make_matching_gdn_pair(cp_size=cp_size)
        for case_index, case in enumerate(_packed_correctness_cases()):
            zero_parameter_grads(flat_gdn)
            zero_parameter_grads(cp_gdn)
            tensors = build_phase0_packed_tensors(case)
            group_ids = tensors["group_ids"].cuda()
            parent_ids = tensors["parent_ids"].cuda()
            spec = parse_gdn_shared_prefix_segments(
                group_ids, parent_ids, min_completions_per_family=0
            )
            plan = build_gdn_rank_execution_plan(
                spec,
                device=group_ids.device,
                cp_rank=rank,
                cp_size=cp_size,
                planner_config=_planner_config_for_case(case),
            )
            hidden, output_grad = _hidden_and_grad(
                case,
                seed=20530426 + 1000 * cp_size + case_index,
            )
            real_mask = (group_ids != -1).transpose(0, 1).unsqueeze(-1)
            output_grad = output_grad * real_mask
            flat_hidden = hidden.clone().detach().requires_grad_(True)
            flat_out = run_real_gdn_flattened_reference(
                flat_gdn,
                flat_hidden,
                group_ids=group_ids,
                parent_ids=parent_ids,
                execution_spec=spec,
            )
            flat_loss = (flat_out * output_grad).sum()
            flat_loss.backward()

            hidden_flat = hidden.transpose(0, 1).reshape(-1, hidden.shape[-1])
            grad_flat = output_grad.transpose(0, 1).reshape(-1, output_grad.shape[-1])
            local_index = torch.tensor(
                plan.attention_token_indices,
                device=hidden.device,
                dtype=torch.long,
            )
            local_hidden = (
                hidden_flat.index_select(0, local_index)
                .unsqueeze(1)
                .contiguous()
                .detach()
                .requires_grad_(True)
            )
            local_output_grad = (
                grad_flat.index_select(0, local_index).unsqueeze(1).contiguous()
            )
            cp_out, _ = run_gdn_layer(
                cp_gdn,
                local_hidden,
                group_ids=group_ids,
                parent_ids=parent_ids,
                execution_spec=spec,
                execution_plan=plan,
                cp_group=torch.distributed.group.WORLD,
            )
            cp_loss = (cp_out * local_output_grad).sum()
            cp_loss.backward()
            _assert_cp_matches_reference(
                case.name,
                flat_gdn,
                cp_gdn,
                flat_hidden,
                flat_out,
                flat_loss.detach(),
                local_hidden,
                cp_out,
                cp_loss.detach(),
                local_index,
            )
        Path(output_dir, f"packed_vs_flattened_rank_{rank}.ok").write_text("ok\n")
    finally:
        if getattr(ps, "model_parallel_is_initialized", lambda: False)():
            ps.destroy_model_parallel()
        destroy_process_group()
