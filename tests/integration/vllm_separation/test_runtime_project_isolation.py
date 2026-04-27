import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[3]


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
