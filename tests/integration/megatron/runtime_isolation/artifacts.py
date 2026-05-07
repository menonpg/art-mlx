from __future__ import annotations

from datetime import datetime, timezone
import os
from pathlib import Path
import re
import subprocess
import sys
import uuid

from pydantic import BaseModel

TEST_ROOT = Path(__file__).resolve().parent
ARTIFACTS_ROOT = TEST_ROOT / "artifacts"
REPO_ROOT = Path(
    subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        cwd=TEST_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
)


class ArtifactMetadata(BaseModel):
    commit: str
    branch: str
    test_nodeid: str
    created_at_utc: str
    python_executable: str
    artifact_dir: str


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _dirty_lines() -> list[str]:
    output = _git("status", "--porcelain=v1", "--untracked-files=all")
    return [line for line in output.splitlines() if line]


def require_clean_git_state() -> str:
    dirty = _dirty_lines()
    if dirty:
        rendered = "\n".join(dirty)
        raise RuntimeError(
            "Megatron runtime-isolation tests require a fully committed worktree.\n"
            "Commit or remove these changes before running tests:\n"
            f"{rendered}"
        )
    return _git("rev-parse", "HEAD")


def _sanitize_nodeid(nodeid: str) -> str:
    collapsed = re.sub(r"[^A-Za-z0-9_.-]+", "_", nodeid.strip())
    return collapsed.strip("._") or "unnamed_test"


def create_artifact_dir(test_nodeid: str) -> Path:
    commit = require_clean_git_state()
    branch = _git("branch", "--show-current")
    test_name = _sanitize_nodeid(test_nodeid)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{timestamp}_{os.getpid()}_{uuid.uuid4().hex[:8]}"
    artifact_dir = ARTIFACTS_ROOT / test_name / commit[:12] / run_id
    artifact_dir.mkdir(parents=True, exist_ok=False)

    metadata = ArtifactMetadata(
        commit=commit,
        branch=branch,
        test_nodeid=test_nodeid,
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        python_executable=sys.executable,
        artifact_dir=str(artifact_dir),
    )
    (artifact_dir / "run_metadata.json").write_text(
        metadata.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    return artifact_dir
