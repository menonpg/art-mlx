from pathlib import Path

import pytest

from .artifacts import create_artifact_dir, require_clean_git_state


TEST_ROOT = Path(__file__).resolve().parent
ARTIFACTS_ROOT = TEST_ROOT / "artifacts"


@pytest.fixture(scope="session", autouse=True)
def _require_clean_commit_state() -> None:
    require_clean_git_state()


@pytest.fixture
def artifact_dir(request: pytest.FixtureRequest) -> Path:
    return create_artifact_dir(request.node.nodeid)
