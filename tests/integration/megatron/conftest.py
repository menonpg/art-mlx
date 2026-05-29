from pathlib import Path
from typing import Any

import pytest

import art  # noqa: F401

from .artifacts import create_megatron_artifact_dir, write_pytest_result

_ARTIFACT_DIR_ATTR = "_megatron_integration_artifact_dir"
_REPORTS_ATTR = "_megatron_integration_reports"


def _artifact_dir_for_item(item: pytest.Item) -> Path:
    artifact_dir = getattr(item, _ARTIFACT_DIR_ATTR, None)
    if artifact_dir is None:
        artifact_dir = create_megatron_artifact_dir(item.nodeid)
        setattr(item, _ARTIFACT_DIR_ATTR, artifact_dir)
    return artifact_dir


def pytest_runtest_setup(item: pytest.Item) -> None:
    _artifact_dir_for_item(item)


@pytest.fixture
def artifact_dir(request: pytest.FixtureRequest) -> Path:
    return _artifact_dir_for_item(request.node)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item: pytest.Item, call: pytest.CallInfo[Any]):
    del call
    outcome = yield
    report = outcome.get_result()
    reports = list(getattr(item, _REPORTS_ATTR, []))
    reports.append(report)
    setattr(item, _REPORTS_ATTR, reports)
    write_pytest_result(
        _artifact_dir_for_item(item),
        test_nodeid=item.nodeid,
        reports=reports,
    )
