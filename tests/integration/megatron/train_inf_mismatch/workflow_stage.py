import os
from pathlib import Path
import re
import subprocess
import sys

from pydantic import BaseModel

from .artifacts import REPO_ROOT, TEST_ROOT, create_artifact_dir


class TrainInfMismatchReport(BaseModel):
    base_model: str
    passed: bool
    returncode: int
    artifact_dir: str
    test_root: str
    stdout_path: str
    stderr_path: str
    passed_count: int
    failed_count: int
    skipped_count: int


def _pytest_counts(output: str) -> dict[str, int]:
    counts = {"passed": 0, "failed": 0, "skipped": 0}
    for line in reversed(output.splitlines()):
        matches = re.findall(r"(\d+) (passed|failed|skipped|error|errors)", line)
        if not matches:
            continue
        for count, kind in matches:
            if kind in {"error", "errors"}:
                counts["failed"] += int(count)
            else:
                counts[kind] += int(count)
        return counts
    return counts


def run_train_inf_mismatch(*, base_model: str) -> TrainInfMismatchReport:
    artifact_dir = create_artifact_dir("workflow::train_inf_mismatch")
    stdout_path = artifact_dir / "pytest_stdout.txt"
    stderr_path = artifact_dir / "pytest_stderr.txt"
    env = os.environ.copy()
    env["BASE_MODEL"] = base_model
    env["ART_TRAIN_INF_MISMATCH_BASE_MODEL"] = base_model
    existing_pythonpath = env.get("PYTHONPATH")
    tests_dir = str(REPO_ROOT / "tests")
    env["PYTHONPATH"] = (
        tests_dir
        if not existing_pythonpath
        else f"{tests_dir}{os.pathsep}{existing_pythonpath}"
    )
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            str(TEST_ROOT),
            f"--ignore={TEST_ROOT / 'artifacts'}",
            "--tb=short",
        ],
        cwd=Path(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    stdout_path.write_text(result.stdout, encoding="utf-8")
    stderr_path.write_text(result.stderr, encoding="utf-8")
    counts = _pytest_counts(result.stdout + "\n" + result.stderr)
    return TrainInfMismatchReport(
        base_model=base_model,
        passed=result.returncode == 0,
        returncode=result.returncode,
        artifact_dir=str(artifact_dir),
        test_root=str(TEST_ROOT),
        stdout_path=str(stdout_path),
        stderr_path=str(stderr_path),
        passed_count=counts["passed"],
        failed_count=counts["failed"],
        skipped_count=counts["skipped"],
    )
