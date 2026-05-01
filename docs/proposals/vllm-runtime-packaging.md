# Proposal: Package the ART vLLM Runtime as a Managed Separate Environment

## Summary

Separate ART's Python environment from vLLM's Python environment while keeping the user experience close to:

```bash
pip install "openpipe-art[backend]"
```

The root `openpipe-art` package should not declare or install `vllm`. Instead, it should bundle the small ART-owned `art-vllm-runtime` wheel as package data, then install and launch that runtime in a separate managed virtual environment when dedicated vLLM serving is needed.

This keeps vLLM's strict dependency constraints out of the main ART environment without requiring normal users to manually create a second venv or set `ART_VLLM_RUNTIME_BIN`.

## Goals

- Keep `openpipe-art[backend]` installable without resolving or installing vLLM.
- Keep vLLM in a separate Python environment from ART.
- Make package installs work without a source checkout.
- Keep source checkout development convenient by using repo-relative `vllm_runtime/.venv` when it exists.
- Keep the managed runtime cache bounded by default, because vLLM runtime envs are large.
- Keep release builds explicit and auditable through scripts rather than hidden build magic.
- Keep the first implementation small: no user-facing CLI and no non-uv fallback path.

## Non-Goals

- Do not install vLLM into the root ART environment.
- Do not require normal package users to set `ART_VLLM_RUNTIME_BIN`.
- Do not make the root project and `vllm_runtime/` a single uv workspace with one lockfile.
- Do not rely on a repo-relative `vllm_runtime/` directory for wheel installs.
- Do not add runtime management CLI commands in the first implementation.
- Do not support a non-uv installer path.

## Package Shape

Build two distribution artifacts:

1. `openpipe-art`
2. `art-vllm-runtime`

`art-vllm-runtime` remains its own package with the runtime server console script:

```text
art-vllm-runtime-server = art_vllm_runtime.dedicated_server:main
```

For the managed-runtime packaging path, `art-vllm-runtime` does not need to be published as a public PyPI project. It can be built during `openpipe-art` packaging and bundled inside the root wheel. This matters because the runtime package may contain strict/direct vLLM dependency metadata that is fine for a local bundled wheel install, but may not be acceptable as public package-index metadata.

The root `openpipe-art` wheel includes the runtime wheel as inert package data:

```text
openpipe_art-*.whl
  art/
    vllm_runtime.py
    _vllm_runtime/
      manifest.json
      pyproject.toml
      uv.lock
      art_vllm_runtime-*.whl
```

The bundled runtime wheel is not listed in `openpipe-art` dependency metadata. `pip` therefore does not install it into the ART environment. ART installs it later into a separate managed venv.

The runtime manifest should describe the runtime ART expects:

```json
{
  "runtime_package": "art-vllm-runtime",
  "runtime_version": "0.5.18",
  "protocol_version": 1,
  "python": ">=3.11,<3.13",
  "runtime_wheel": "art_vllm_runtime-0.5.18-py3-none-any.whl",
  "runtime_wheel_sha256": "...",
  "lockfile": "uv.lock"
}
```

`vllm_runtime/uv.lock` is the source of truth for strict runtime dependencies such as torch, transformers, and the pinned vLLM wheel URL or index requirement. This matches ART's existing uv-based dependency management and keeps those constraints out of root package metadata.

The managed runtime installer should create a venv from the bundled lock project, then install the bundled runtime wheel into that venv:

```text
uv sync --project <bundled-lock-project> --frozen --no-install-project
uv pip install --python <runtime-venv-python> <bundled art-vllm-runtime wheel>
```

## Runtime Resolution

ART should resolve the vLLM runtime binary in this order:

1. `ART_VLLM_RUNTIME_BIN`
2. Repo-relative source checkout runtime:

   ```text
   <repo>/vllm_runtime/.venv/bin/art-vllm-runtime-server
   ```

3. Managed cache runtime matching the bundled manifest.
4. Install the managed cache runtime from the bundled runtime artifacts, then use it.
5. Hard error with actionable context about the resolved paths and failed install/validation step.

Step 2 is intentionally retained for local development. It should only apply when the repo-relative runtime binary exists. In wheel installs, that path will not exist and ART should continue to the managed cache path.

## Managed Cache

The cache should be keyed by the runtime manifest hash:

```text
~/.cache/art/vllm_runtime/
  <manifest_hash>/
    .venv/
    install.json
```

Install flow:

1. If the matching cache entry exists and validates, reuse it.
2. If not, install into a temporary staging directory under the same cache root.
3. Validate that `art-vllm-runtime-server` exists and can report its runtime/protocol version.
4. Atomically promote the staging directory to the manifest-hash directory.
5. Delete old sibling runtime cache directories by default.

Default cache retention should keep only the current runtime env. vLLM environments are large, so retaining every old manifest hash is not acceptable by default.

Useful overrides:

```text
ART_VLLM_RUNTIME_CACHE_DIR=/custom/cache
ART_VLLM_RUNTIME_KEEP_OLD=1
ART_VLLM_RUNTIME_BIN=/custom/runtime/bin/art-vllm-runtime-server
```

Cleanup should happen only after the new runtime validates. Because `ART_VLLM_RUNTIME_CACHE_DIR` is user-controlled, cleanup must be conservative:

- Only delete sibling directories under the selected cache root.
- Only delete directories that contain an ART runtime install marker, for example `install.json` with the expected package name plus a matching `.venv/pyvenv.cfg`.
- Refuse to delete the cache root itself.
- Refuse to delete paths that are not directories.
- Skip active-looking or locked runtime directories and try again on a later install.

The default policy is still one current cached runtime, but ART must not delete arbitrary directories even if environment variables are set adversarially.

## Local Development

Local development should keep two uv projects:

```bash
cd /path/to/art
uv sync --extra backend
```

```bash
cd /path/to/art/vllm_runtime
uv sync
```

With `vllm_runtime/.venv/bin/art-vllm-runtime-server` present, ART should use the source checkout runtime through resolver step 2. Developers should not need to rebuild the root wheel while iterating on runtime code.

For custom experiments, developers can still force a runtime:

```bash
export ART_VLLM_RUNTIME_BIN=/path/to/runtime/.venv/bin/art-vllm-runtime-server
```

## Build Process Integration

ART currently builds packages directly with Hatch:

- `scripts/publish.sh` runs `uv run hatch build`.
- `.github/workflows/release.yml` runs `uv run hatch build`.
- `.github/workflows/package-install.yml` runs `uv build --wheel --out-dir dist`.

Replace these direct build calls with a single explicit build script:

```text
scripts/build_package.py
```

The script should:

1. Clean generated runtime bundle artifacts.
2. Read `openpipe-art` version from root `pyproject.toml`.
3. Read `art-vllm-runtime` version from `vllm_runtime/pyproject.toml` and record both versions in the manifest.
4. Check `vllm_runtime/uv.lock` is current with `uv lock --project vllm_runtime --check`.
5. Build `vllm_runtime/` into a wheel.
6. Compute sha256 for the runtime wheel.
7. Generate `manifest.json`.
8. Copy `vllm_runtime/pyproject.toml` and `vllm_runtime/uv.lock` into a stable package-data directory under `src/art/_vllm_runtime/`.
9. Copy `manifest.json` and the runtime wheel into `src/art/_vllm_runtime/`.
10. Build the root `openpipe-art` wheel and sdist.
11. Verify the built root wheel includes the runtime bundle.
12. Verify root wheel metadata has no `vllm` or `art-vllm-runtime` dependency.
13. Verify the sdist includes the same runtime bundle data so it does not depend on a source-tree `vllm_runtime/`.

Update build call sites:

```text
scripts/publish.sh
  python scripts/build_package.py

.github/workflows/release.yml
  python scripts/build_package.py

.github/workflows/package-install.yml
  python scripts/build_package.py --wheel
```

The release workflow can keep uploading and publishing `dist/*` after the script populates `dist/`.

## Maintainer Publishing Without vLLM

Maintainers should be able to publish `openpipe-art` from a machine that cannot install or run vLLM dependencies. Publishing should require only:

- Python
- uv
- build-system dependencies such as Hatchling
- the committed `vllm_runtime/pyproject.toml`
- the committed `vllm_runtime/uv.lock`

The build script must not run any command that creates the runtime venv or installs vLLM dependencies. In particular, release/package builds should not run:

```text
uv sync --project vllm_runtime
any managed-runtime install helper
```

The release build should only build the small runtime package artifact and bundle its lock metadata:

```text
uv build --wheel vllm_runtime --out-dir <runtime-dist>
```

This wheel build should require only the runtime package build backend, not runtime dependencies. The managed vLLM environment is created later on the user or production machine when ART actually needs to launch vLLM.

If `vllm_runtime/pyproject.toml` changes in a way that requires lockfile updates, refreshing `vllm_runtime/uv.lock` is a separate maintainer task. The package build should treat the committed lock as frozen and fail with a clear message if it is stale, rather than silently resolving or installing vLLM during publishing.

## sdist Policy

The sdist must not depend on an unbundled source-tree `vllm_runtime/` directory. Include the generated runtime bundle artifacts in both the wheel and sdist. This should be part of the normal Hatch package-data configuration used by the build script, not a separate fallback path.

## Release Runtime Smoke Test

The official release workflow should validate runtime installability, but this does not need to run in normal PR CI.

Split `.github/workflows/release.yml` into three jobs:

1. `build-package` on `ubuntu-latest`
2. `runtime-smoke` on `art-large-runner`
3. `publish` on `ubuntu-latest`

`build-package` should build `dist/*` once and upload it as a workflow artifact. `runtime-smoke` should download that exact artifact, install `openpipe-art[backend]` into a clean env, trigger the managed runtime install path, and verify imports such as:

```text
import art_vllm_runtime
import vllm
import torch
```

The smoke test should not start a vLLM server because the runner does not have GPUs. `publish` should depend on `runtime-smoke` and publish the exact artifact built by `build-package`; it should not rebuild.

Tag creation should move to the final `publish` job after validation succeeds.

## Validation

Keep code-level tests focused on the resolution and safety properties that are cheap to check locally:

- Root `openpipe-art` metadata contains no `vllm` dependency.
- Root `openpipe-art` metadata contains no `art-vllm-runtime` dependency.
- Built root wheel contains `art/_vllm_runtime/manifest.json`.
- Built root wheel contains `art/_vllm_runtime/uv.lock`.
- Built root wheel contains the bundled `art-vllm-runtime` wheel.
- Source checkout resolution still prefers `vllm_runtime/.venv/bin/art-vllm-runtime-server` when present.
- `ART_VLLM_RUNTIME_BIN` overrides all other resolution paths.
- Cache cleanup only deletes ART-managed runtime venv directories with the expected marker and `.venv/pyvenv.cfg`.

The expensive end-to-end managed runtime install should be covered by the official release smoke test instead of normal CI.

## Open Questions

- Whether runtime version should exactly match `openpipe-art` version or use an independent version plus protocol compatibility.
- Whether the pinned ART vLLM wheel should remain a direct URL in `vllm_runtime/uv.lock` or move to an internal/package index.
- Whether auto-install should be enabled by default in all environments or require an explicit opt-out for hermetic production jobs.
