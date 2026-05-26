"""Shared helpers for integration tests that need durable per-run artifacts.

These helpers create a suite-owned artifacts/ directory keyed by test node id,
git commit, and run id, then write metadata that ties logs and JSON outputs back
to the exact committed code. They do not replace repo .local logs used by oracle
workflows that intentionally keep mutable local development output.
"""

from __future__ import annotations

from datetime import datetime, timezone
import os
from pathlib import Path
import re
import subprocess
import sys
import uuid

from pydantic import BaseModel

REPO_ROOT = Path(__file__).resolve().parents[3]


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


def _sanitize_nodeid(nodeid: str) -> str:
    collapsed = re.sub(r"[^A-Za-z0-9_.-]+", "_", nodeid.strip())
    return collapsed.strip("._") or "unnamed_test"


def require_clean_git_state(suite_name: str) -> str:
    """Return the current commit after checking artifacts can be tied to clean code."""
    dirty = _git("status", "--porcelain=v1", "--untracked-files=all").splitlines()
    if dirty:
        rendered = "\n".join(dirty)
        raise RuntimeError(
            f"{suite_name} require a fully committed worktree.\n"
            "Commit or remove these changes before running tests:\n"
            f"{rendered}"
        )
    return _git("rev-parse", "HEAD")


def create_artifact_dir(
    test_nodeid: str,
    *,
    artifacts_root: Path,
    suite_name: str,
) -> Path:
    """Create a durable, git-addressed artifact directory for one test invocation."""
    commit = require_clean_git_state(suite_name)
    branch = _git("branch", "--show-current")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{timestamp}_{os.getpid()}_{uuid.uuid4().hex[:8]}"
    artifact_dir = artifacts_root / _sanitize_nodeid(test_nodeid) / commit[:12] / run_id
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
