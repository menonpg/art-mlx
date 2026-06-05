from __future__ import annotations

from pathlib import Path

from ..artifacts import REPO_ROOT
from ..artifacts import create_artifact_dir as _create_artifact_dir
from ..artifacts import require_clean_git_state as _require_clean_git_state

TEST_ROOT = Path(__file__).resolve().parent
ARTIFACTS_ROOT = TEST_ROOT / "artifacts"
SUITE_NAME = "Megatron runtime-isolation tests"


def require_clean_git_state() -> str:
    return _require_clean_git_state(SUITE_NAME)


def create_artifact_dir(test_nodeid: str) -> Path:
    return _create_artifact_dir(
        test_nodeid,
        artifacts_root=ARTIFACTS_ROOT,
        suite_name=SUITE_NAME,
    )
