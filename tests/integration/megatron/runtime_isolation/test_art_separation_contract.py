from pathlib import Path
import tomllib

ROOT = Path(__file__).resolve().parents[4]


def test_art_source_has_no_vllm_imports() -> None:
    offenders: list[str] = []
    for path in sorted((ROOT / "src" / "art").rglob("*.py")):
        for line_number, line in enumerate(path.read_text().splitlines(), start=1):
            stripped = line.strip()
            if stripped.startswith("import vllm") or stripped.startswith("from vllm"):
                offenders.append(f"{path.relative_to(ROOT)}:{line_number}")
    assert offenders == []


def test_art_pyproject_has_no_vllm_dependency_or_plugin_entrypoint() -> None:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())
    project = pyproject["project"]
    backend = project["optional-dependencies"]["backend"]
    megatron = project["optional-dependencies"]["megatron"]
    dev = pyproject["dependency-groups"]["dev"]

    def _contains_vllm(values: list[str]) -> bool:
        return any(
            value.startswith("vllm") or value == "art-vllm-runtime" for value in values
        )

    assert not _contains_vllm(backend)
    assert not _contains_vllm(megatron)
    assert not _contains_vllm(dev)
    assert "entry-points" not in project or "vllm.general_plugins" not in project.get(
        "entry-points", {}
    )
