import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[4]


def test_runtime_project_imports_in_its_own_project_env(artifact_dir: Path) -> None:
    result = subprocess.run(
        [
            "uv",
            "run",
            "--project",
            str(ROOT / "vllm_runtime"),
            "python",
            "-c",
            (
                "import importlib.util, json; "
                "import art_vllm_runtime; "
                "print(json.dumps({"
                "'runtime_ok': True, "
                "'has_vllm': importlib.util.find_spec('vllm') is not None"
                "}))"
            ),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    (artifact_dir / "stdout.txt").write_text(result.stdout)
    (artifact_dir / "stderr.txt").write_text(result.stderr)
    payload = json.loads(result.stdout.strip())
    assert payload == {"runtime_ok": True, "has_vllm": True}


def test_runtime_server_source_contains_only_required_custom_routes() -> None:
    source = (
        ROOT / "vllm_runtime" / "src" / "art_vllm_runtime" / "dedicated_server.py"
    ).read_text()
    for route in ("/sleep", "/wake_up", "/is_sleeping", "/art/set_served_model_name"):
        assert route in source


def test_runtime_general_plugin_loads_full_patch_set() -> None:
    pyproject = (ROOT / "vllm_runtime" / "pyproject.toml").read_text()
    assert (
        'art = "art_vllm_runtime.patches:apply_vllm_runtime_patches"' in pyproject
    )


def test_runtime_project_restores_nccl_unique_id_from_raw_bytes(
    artifact_dir: Path,
) -> None:
    result = subprocess.run(
        [
            "uv",
            "run",
            "--project",
            str(ROOT / "vllm_runtime"),
            "python",
            "-c",
            (
                "import ctypes, json; "
                "from art_vllm_runtime.patches import _restore_nccl_unique_id_payload; "
                "from vllm.distributed.device_communicators.pynccl_wrapper import ncclUniqueId; "
                "payload = bytes(range(128)); "
                "restored = _restore_nccl_unique_id_payload(payload, ncclUniqueId()); "
                "print(json.dumps({"
                "'type': type(restored).__name__, "
                "'matches': ctypes.string_at(ctypes.byref(restored), ctypes.sizeof(restored)).hex() == payload.hex()"
                "}))"
            ),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    (artifact_dir / "restore_stdout.txt").write_text(result.stdout)
    (artifact_dir / "restore_stderr.txt").write_text(result.stderr)
    payload = json.loads(result.stdout.strip())
    assert payload == {"type": "ncclUniqueId", "matches": True}


def test_runtime_project_nccl_wrapper_accepts_raw_bytes(artifact_dir: Path) -> None:
    result = subprocess.run(
        [
            "uv",
            "run",
            "--project",
            str(ROOT / "vllm_runtime"),
            "python",
            "-c",
            (
                "import json; "
                "from art_vllm_runtime.patches import _normalize_nccl_comm_init_rank_unique_id; "
                "FakeLibrary = type('FakeLibrary', (), {'unique_id_from_bytes': lambda self, data: {'restored': len(data)}}); "
                "restored = _normalize_nccl_comm_init_rank_unique_id(FakeLibrary(), bytes(range(128))); "
                "print(json.dumps(restored))"
            ),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    (artifact_dir / "nccl_wrapper_stdout.txt").write_text(result.stdout)
    (artifact_dir / "nccl_wrapper_stderr.txt").write_text(result.stderr)
    payload = json.loads(result.stdout.strip())
    assert payload == {"restored": 128}


def test_runtime_project_localizes_ep_moe_lora_experts(artifact_dir: Path) -> None:
    result = subprocess.run(
        [
            "uv",
            "run",
            "--project",
            str(ROOT / "vllm_runtime"),
            "python",
            "-c",
            (
                "import json, torch; "
                "from art_vllm_runtime.patches import _ep_local_expert_global_indices, _slice_ep_local_experts; "
                "expert_map = torch.tensor([1, -1, 0, -1], dtype=torch.int32); "
                "weights = torch.arange(12, dtype=torch.float32).reshape(4, 3); "
                "indices = _ep_local_expert_global_indices(expert_map).tolist(); "
                "local = _slice_ep_local_experts(weights, expert_map, 2).tolist(); "
                "print(json.dumps({'indices': indices, 'local': local}))"
            ),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    (artifact_dir / "ep_localize_stdout.txt").write_text(result.stdout)
    (artifact_dir / "ep_localize_stderr.txt").write_text(result.stderr)
    payload = json.loads(result.stdout.strip())
    assert payload == {
        "indices": [2, 0],
        "local": [[6.0, 7.0, 8.0], [0.0, 1.0, 2.0]],
    }


def test_runtime_project_passes_ep_expert_map_into_moe_lora_alignment(
    artifact_dir: Path,
) -> None:
    result = subprocess.run(
        [
            "uv",
            "run",
            "--project",
            str(ROOT / "vllm_runtime"),
            "python",
            "-c",
            (
                "import json, torch; "
                "from art_vllm_runtime.patches import patch_punica_ep_moe_lora_alignment; "
                "from vllm.lora.punica_wrapper import punica_gpu; "
                "patch_punica_ep_moe_lora_alignment(); "
                "captured = {}; "
                "FakeMeta = type('FakeMeta', (), {'meta_args': staticmethod(lambda num_tokens, specialize: (torch.zeros(num_tokens, dtype=torch.int32), None, None, None, torch.zeros(1, dtype=torch.int32), None, None))}); "
                "FakeConfig = type('FakeConfig', (), {'specialize_active_lora': False}); "
                "FakeWrapper = type('FakeWrapper', (), {'token_mapping_meta': FakeMeta(), 'lora_config': FakeConfig()}); "
                "exec(\"def fake_align(topk_ids, token_lora_mapping, num_experts, block_size, max_loras, max_num_tokens_padded, max_num_m_blocks, sorted_ids, expert_ids, num_tokens_post_pad, adapter_enabled, lora_ids, expert_map=None):\\n"
                "    captured['num_experts'] = int(num_experts)\\n"
                "    captured['expert_map_shape'] = None if expert_map is None else list(expert_map.shape)\\n"
                "    expert_ids.fill_(-1)\\n"
                "    expert_ids[:2] = torch.tensor([0, 1], device=expert_ids.device, dtype=expert_ids.dtype)\\n"
                "    num_tokens_post_pad.zero_()\", globals(), locals()); "
                "punica_gpu.ops.moe_lora_align_block_size = fake_align; "
                "wrapper = FakeWrapper(); "
                "expert_map = torch.full((128,), -1, dtype=torch.int32); "
                "expert_map[64] = 0; "
                "expert_map[65] = 1; "
                "_, _, expert_ids, _ = punica_gpu.PunicaWrapperGPU.moe_lora_align_block_size(wrapper, torch.tensor([[64, 65]], dtype=torch.int32), 1, 16, 2, 2, torch.tensor([1, 1], dtype=torch.int32), expert_map=expert_map); "
                "print(json.dumps({'num_experts': captured['num_experts'], 'expert_map_shape': captured['expert_map_shape'], 'expert_ids': expert_ids[:2].tolist()}))"
            ),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    (artifact_dir / "ep_align_stdout.txt").write_text(result.stdout)
    (artifact_dir / "ep_align_stderr.txt").write_text(result.stderr)
    payload = json.loads(result.stdout.strip())
    assert payload == {
        "num_experts": 2,
        "expert_map_shape": [128],
        "expert_ids": [0, 1],
    }
