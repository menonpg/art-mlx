import os
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[3]


def test_art_import_with_unsloth_enabled_blocks_broken_mamba() -> None:
    env = os.environ.copy()
    env["IMPORT_UNSLOTH"] = "1"
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import importlib.util; "
                "import art; "
                "print('art_ok'); "
                "print(importlib.util.find_spec('mamba_ssm'))"
            ),
        ],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + "\n" + completed.stderr
    assert "art_ok" in completed.stdout
    assert "None" in completed.stdout
