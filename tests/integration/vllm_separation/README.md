# vLLM Separation Tests

All vLLM-separation integration tests live in this directory.

Rules:

- Put every test for this effort under `tests/integration/vllm_separation/`.
- Write all test artifacts under `tests/integration/vllm_separation/artifacts/`.
- Do not run these tests from a dirty worktree.
- Any code involved in a test run must be committed before the test starts.
- Every artifact set must include the exact commit hash it ran from.

Use the `artifact_dir` fixture from [conftest.py](./conftest.py) for artifact output.

That fixture:

- refuses to run when the worktree is dirty
- creates a per-test artifact directory under `artifacts/`
- writes `run_metadata.json` with the exact commit hash and test node id

Artifact directories are git-ignored by design so reproducible outputs do not dirty the worktree.
