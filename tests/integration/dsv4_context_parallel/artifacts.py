from __future__ import annotations

from datetime import UTC, datetime
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys

from packed_layout import Dsv4CaseSummary
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


class Dsv4Metric(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    value: float
    unit: str = ""
    threshold: float | None = None
    passed: bool | None = None


class Dsv4ArtifactManifest(BaseModel):
    model_config = ConfigDict(frozen=True)

    created_at: str
    kind: str
    command: tuple[str, ...]
    art: GitRepoState
    project_tracking: GitRepoState | None = None
    runtime: RuntimeInfo
    configs: dict[str, object] = Field(default_factory=dict)
    cases: tuple[dict[str, object], ...] = Field(default_factory=tuple)
    metrics: tuple[dict[str, object], ...] = Field(default_factory=tuple)
    caveats: tuple[str, ...] = Field(default_factory=tuple)


def write_manifest(
    output_dir: Path,
    *,
    kind: str,
    command: list[str],
    configs: dict[str, object] | None = None,
    cases: tuple[Dsv4CaseSummary, ...] = (),
    metrics: tuple[Dsv4Metric, ...] = (),
    caveats: tuple[str, ...] = (),
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = Dsv4ArtifactManifest(
        created_at=datetime.now(UTC).isoformat(),
        kind=kind,
        command=tuple(command),
        art=git_state(_art_repo_root()),
        project_tracking=_optional_git_state(_project_tracking_root()),
        runtime=runtime_info(),
        configs={} if configs is None else configs,
        cases=tuple(case.model_dump() for case in cases),
        metrics=tuple(metric.model_dump() for metric in metrics),
        caveats=caveats,
    )
    path = output_dir / "manifest.json"
    path.write_text(json.dumps(manifest.model_dump(), indent=2, sort_keys=True) + "\n")
    return path


def write_readable_summary(
    output_dir: Path,
    *,
    title: str,
    status: str,
    manifest_path: Path,
    case_summaries: tuple[Dsv4CaseSummary, ...],
    metrics: tuple[Dsv4Metric, ...] = (),
    caveats: tuple[str, ...] = (),
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = Dsv4ArtifactManifest.model_validate_json(manifest_path.read_text())
    lines = [
        title,
        f"status: {status}",
        f"kind: {manifest.kind}",
        f"command: {' '.join(manifest.command)}",
        f"created_at: {manifest.created_at}",
        (f"art: {manifest.art.commit}{' dirty' if manifest.art.dirty else ' clean'}"),
    ]
    if manifest.project_tracking is not None:
        lines.append(
            f"project_tracking: {manifest.project_tracking.commit}"
            f"{' dirty' if manifest.project_tracking.dirty else ' clean'}"
        )
    lines.extend(
        (
            f"python: {manifest.runtime.python}",
            f"torch: {manifest.runtime.torch}",
            f"cuda_available: {manifest.runtime.cuda_available}",
            f"gpu_names: {', '.join(manifest.runtime.gpu_names) or 'none'}",
            f"configs: {json.dumps(manifest.configs, sort_keys=True)}",
        )
    )
    if case_summaries:
        lines.append("cases:")
        lines.extend(f"- {_format_case(case)}" for case in case_summaries)
    if metrics:
        lines.append("metrics:")
        lines.extend(f"- {_format_metric(metric)}" for metric in metrics)
    if caveats:
        lines.append("caveats:")
        lines.extend(f"- {caveat}" for caveat in caveats)
    path = output_dir / "summary.txt"
    path.write_text("\n".join(lines) + "\n")
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
            "tilelang",
            "triton",
            "flash-attn-4",
            "megatron-core",
            "transformer-engine",
            "torch",
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


def _format_case(case: Dsv4CaseSummary) -> str:
    flags = [
        name
        for name in (
            "completion_lengths_vary",
            "cp_boundary_prefix",
            "cp_boundary_completion",
            "family_boundary_at_partition",
            "empty_trailing_rank",
            "csa_ratio_boundary",
            "hca_ratio_boundary",
            "swa_boundary",
            "topk_tie_or_near_tie",
            "no_stage_keys",
        )
        if getattr(case, name)
    ]
    return (
        f"{case.name} tokens={case.total_tokens} families={case.family_count} "
        f"completions={case.completion_count} "
        f"completion_lengths={case.min_completion_length}"
        f"..{case.max_completion_length} "
        f"flags={','.join(flags) or 'none'}"
    )


def _format_metric(metric: Dsv4Metric) -> str:
    pieces = [metric.name, f"value={metric.value:g}{metric.unit}"]
    if metric.threshold is not None:
        pieces.append(f"threshold={metric.threshold:g}{metric.unit}")
    if metric.passed is not None:
        pieces.append(f"passed={metric.passed}")
    return " ".join(pieces)


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


def _art_repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _project_tracking_root() -> Path:
    return _art_repo_root().parents[3] / "project_tracking" / "art" / "dsv4"
