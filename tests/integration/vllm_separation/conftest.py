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


def pytest_collection_modifyitems(
    session: pytest.Session,
    config: pytest.Config,
    items: list[pytest.Item],
) -> None:
    del session, config
    yes_no_order = {
        "test_megatron_dedicated_yes_no_trainability_live": 0,
        "test_megatron_shared_yes_no_trainability_live": 1,
        "test_unsloth_dedicated_yes_no_trainability_live": 2,
    }

    def _sort_key(item: pytest.Item) -> tuple[int, str]:
        return (yes_no_order.get(item.name, 99), item.nodeid)

    items.sort(key=_sort_key)
