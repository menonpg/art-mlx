import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[3]


def _run(
    command: list[str],
    *,
    artifact_dir: Path,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        command,
        cwd=ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    (artifact_dir / "stdout.txt").write_text(result.stdout)
    (artifact_dir / "stderr.txt").write_text(result.stderr)
    return result


def test_art_import_does_not_require_vllm_or_mutate_compile_threads(
    artifact_dir: Path,
) -> None:
    env = dict(os.environ)
    env.pop("TORCHINDUCTOR_COMPILE_THREADS", None)
    result = _run(
        [
            sys.executable,
            "-c",
            (
                "import importlib.util, json, os; "
                "before = os.environ.get('TORCHINDUCTOR_COMPILE_THREADS'); "
                "import art; "
                "after = os.environ.get('TORCHINDUCTOR_COMPILE_THREADS'); "
                "print(json.dumps({"
                "'before': before, "
                "'after': after, "
                "'has_vllm': importlib.util.find_spec('vllm') is not None"
                "}))"
            ),
        ],
        artifact_dir=artifact_dir,
        env=env,
    )
    payload = json.loads(result.stdout.strip())
    assert payload["has_vllm"] is False
    assert payload["before"] is None
    assert payload["after"] is None
