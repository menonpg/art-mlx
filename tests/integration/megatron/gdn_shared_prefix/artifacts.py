from __future__ import annotations

from datetime import UTC, datetime
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys

from pydantic import BaseModel, ConfigDict, Field


class GitRepoState(BaseModel):
    model_config = ConfigDict(frozen=True)

    path: str
    commit: str
    dirty: bool
    status: tuple[str, ...] = Field(default_factory=tuple)


class RuntimeInfo(BaseModel):
    model_config = ConfigDict(frozen=True)

    python: str
    platform: str
    torch: str | None = None
    cuda_available: bool | None = None
    cuda: str | None = None
    cudnn: int | None = None
    gpu_names: tuple[str, ...] = Field(default_factory=tuple)
    package_versions: dict[str, str] = Field(default_factory=dict)


class GdnArtifactManifest(BaseModel):
    model_config = ConfigDict(frozen=True)

    created_at: str
    kind: str
    command: tuple[str, ...]
    art: GitRepoState
    project_tracking: GitRepoState | None = None
    runtime: RuntimeInfo
    configs: dict[str, object] = Field(default_factory=dict)
    cases: tuple[dict[str, object], ...] = Field(default_factory=tuple)
    caveats: tuple[str, ...] = Field(default_factory=tuple)


def write_manifest(
    output_dir: Path,
    *,
    kind: str,
    command: list[str],
    configs: dict[str, object] | None = None,
    cases: tuple[dict[str, object], ...] = (),
    caveats: tuple[str, ...] = (),
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = GdnArtifactManifest(
        created_at=datetime.now(UTC).isoformat(),
        kind=kind,
        command=tuple(command),
        art=git_state(Path(__file__).resolve().parents[4]),
        project_tracking=_optional_git_state(
            Path("/root/ws/project_tracking/art/megatron_bridge_model_support_skill")
        ),
        runtime=runtime_info(),
        configs={} if configs is None else configs,
        cases=cases,
        caveats=caveats,
    )
    path = output_dir / "manifest.json"
    path.write_text(json.dumps(manifest.model_dump(), indent=2, sort_keys=True) + "\n")
    return path


def git_state(path: Path) -> GitRepoState:
    commit = _git(path, "rev-parse", "HEAD")
    status = tuple(
        line for line in _git(path, "status", "--short").splitlines() if line
    )
    return GitRepoState(
        path=str(path),
        commit=commit,
        dirty=bool(status),
        status=status,
    )


def runtime_info() -> RuntimeInfo:
    torch_version: str | None = None
    cuda_available: bool | None = None
    cuda_version: str | None = None
    cudnn_version: int | None = None
    gpu_names: tuple[str, ...] = ()
    try:
        import torch

        torch_version = torch.__version__
        cuda_available = torch.cuda.is_available()
        cuda_version = torch.version.cuda
        cudnn_version = torch.backends.cudnn.version()
        if cuda_available:
            gpu_names = tuple(
                torch.cuda.get_device_name(index)
                for index in range(torch.cuda.device_count())
            )
    except Exception:
        pass
    packages = {
        name: version
        for name in (
            "triton",
            "flash-linear-attention",
            "fla",
            "megatron-core",
            "transformer-engine",
            "causal-conv1d",
        )
        if (version := _dist_version(name)) is not None
    }
    return RuntimeInfo(
        python=sys.version.split()[0],
        platform=platform.platform(),
        torch=torch_version,
        cuda_available=cuda_available,
        cuda=cuda_version,
        cudnn=cudnn_version,
        gpu_names=gpu_names,
        package_versions=packages,
    )


def _optional_git_state(path: Path) -> GitRepoState | None:
    if not path.exists():
        return None
    try:
        return git_state(path)
    except subprocess.CalledProcessError:
        return None


def _dist_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _git(path: Path, *args: str) -> str:
    result = subprocess.run(
        ("git", "-C", str(path), *args),
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout.strip()
