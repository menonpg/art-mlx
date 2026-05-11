# GDN Shared-Prefix Validation

This directory is the home for ART integration tests, probes, and benchmarks for shared-prefix GDN and future GDN CP work.

Authoritative planning docs:

- `/root/ws/project_tracking/art/megatron_bridge_model_support_skill/2026_04_24_qwen35_gdn_shared_prefix_cp_plan.md`
- `/root/ws/project_tracking/art/megatron_bridge_model_support_skill/2026_04_24_qwen35_gdn_validation_plan.md`
- `/root/ws/project_tracking/art/megatron_bridge_model_support_skill/technical_guide_gdn_shared_prefix_cp.md`

Implemented layout:

- `cases.py`: pydantic workload and topology case models.
- `packed_layout.py`: deterministic packed-row generation and segment-DAG assertions.
- `artifacts.py`: manifest writing with git commit and dirty-state capture.
- `nsys_profile_tables.py`: nsys SQLite export parser that writes JSON, CSV, and Markdown profile tables.
- `oracles.py`: CPU toy-state oracle for validating packed-vs-flattened mechanics.
- `real_gdn_oracle.py`: real Megatron/FLA GDN CP1 packed-vs-flattened and CP reference oracle helpers.
- `src/art/megatron/gdn/layout.py`: reusable CP boundary token-layout plan for attention-order to GDN-order exchange.
- `parser_import.py`: direct source import for CPU parser tests without Megatron extras.
- `test_segment_dag.py`: parser, malformed-input, and generated-case coverage.
- `test_gdn_cp_layout.py`: CP2/CP4/CP8 layout/all-to-all roundtrip reference, including gradients and empty ranks.
- `test_gdn_cp1_packed_vs_flattened.py`: CPU toy-state CP1 oracle and known-bad physical-stream sensitivity.
- `test_real_gdn_cp1_packed_vs_flattened.py`: CUDA real-GDN CP1 oracle and physical-stream sensitivity.
- `test_real_gdn_tp_lora.py`: CUDA real-GDN LoRA gradient and TP2 gradient oracle coverage.
- `test_real_gdn_cp_chain.py`: CP chain reference, boundary-state, and known-bad mutation coverage. This is a semantic reference until native FLA CP summary scan supports ART parent-state injection and final-state emission.
- `test_fla_cp_native_recurrent.py`: native FLA CP recurrent summary-scan coverage for CP2/CP4/CP8, including external `h0`, emitted `hT`, backward gradients, and an affine summary debug check.
- `test_real_gdn_native_fla_cp.py`: native FLA CP full-Qwen GDN segment coverage for CP2/CP4/CP8, including conv-tail exchange, recurrent state transport, input grads, and GDN parameter grads.
- `test_qwen35_full_model_cp1_packed_vs_flattened.py`: CUDA Qwen3.5 full-model CP1 packed-vs-flattened gradient oracle.
- `bench_single_gdn_operation.py`: Phase 2 single-operation lab for dry-run, correctness, timing, nsys profiling, profile parsing, memory-debug, baseline, and CP layout topology dispatch modes.
- `bench_gdn_cp_layout_exchange.py`: spawned CP2/CP4/CP8 layout exchange benchmark with NVTX-labelled CP layout/communication ranges.

Expected future layout:

- Native FLA CP packed-planner integration: route long shared-prefix chain segments through the native CP segment runtime instead of the semantic sequential chain reference.
- `test_gdn_topology_oracle.py`: integrated CP2/CP4/CP8 topology invariance tests.
- `test_attention_packed_vs_flattened.py`: attention invariant extension.
- `bench_stacked_training_proxy.py`: stacked training-style benchmark entrypoint.
- `configs/`: frozen config snapshots.
- `scratch/`: run artifacts for validation and benchmark outputs.

Current CPU checks:

```
env -u VIRTUAL_ENV uv run pytest tests/integration/megatron/gdn_shared_prefix/test_segment_dag.py
env -u VIRTUAL_ENV uv run pytest tests/integration/megatron/gdn_shared_prefix/test_gdn_cp_layout.py
env -u VIRTUAL_ENV uv run pytest tests/integration/megatron/gdn_shared_prefix/test_gdn_cp1_packed_vs_flattened.py
env -u VIRTUAL_ENV uv run pytest tests/integration/megatron/gdn_shared_prefix/test_real_gdn_cp1_packed_vs_flattened.py
env -u VIRTUAL_ENV uv run pytest tests/integration/megatron/gdn_shared_prefix/test_real_gdn_tp_lora.py
env -u VIRTUAL_ENV uv run pytest tests/integration/megatron/gdn_shared_prefix/test_qwen35_full_model_cp1_packed_vs_flattened.py
env -u VIRTUAL_ENV uv run python -m tests.integration.megatron.gdn_shared_prefix.bench_single_gdn_operation --dry-run-cases
env -u VIRTUAL_ENV uv run python -m tests.integration.megatron.gdn_shared_prefix.bench_single_gdn_operation --correctness-only --case-name all
env -u VIRTUAL_ENV uv run python -m tests.integration.megatron.gdn_shared_prefix.bench_single_gdn_operation --benchmark --case-name ragged_family_mix
env -u VIRTUAL_ENV uv run python -m tests.integration.megatron.gdn_shared_prefix.bench_single_gdn_operation --benchmark-baselines --case-name repeated_family --target-seq-len 40960 --prefix-len 5000 --suffix-len 100 --completions-per-family 16 --warmup-iters 1 --iters 3 --output-dir tests/integration/megatron/gdn_shared_prefix/scratch/phase2_baselines_repeated_5k_16x100
env -u VIRTUAL_ENV uv run python -m tests.integration.megatron.gdn_shared_prefix.bench_single_gdn_operation --benchmark --topology cp2-layout --target-seq-len 40960 --prefix-len 5000 --suffix-len 100 --completions-per-family 16 --warmup-iters 1 --iters 3 --output-dir tests/integration/megatron/gdn_shared_prefix/scratch/phase3_cp2_layout
env -u VIRTUAL_ENV uv run python -m tests.integration.megatron.gdn_shared_prefix.bench_single_gdn_operation --memory-debug --case-name ragged_family_mix
env -u VIRTUAL_ENV uv run python -m tests.integration.megatron.gdn_shared_prefix.bench_single_gdn_operation --nsys-profile --case-name ragged_family_mix --warmup-iters 1 --iters 1 --output-dir tests/integration/megatron/gdn_shared_prefix/scratch/phase2_nsys_profile
env -u VIRTUAL_ENV uv run python -m tests.integration.megatron.gdn_shared_prefix.bench_single_gdn_operation --parse-profile-sqlite tests/integration/megatron/gdn_shared_prefix/scratch/phase2_nsys_profile/nsys_gdn_profile.sqlite --output-dir tests/integration/megatron/gdn_shared_prefix/scratch/phase2_nsys_parse
```

The nsys profile mode writes `profile_tables/profile_report.md` for human review plus CSV and JSON tables. The key tables are:

- `Top-Level Lab Ranges`: host NVTX duration and inclusive CUDA work for forward/loss/backward.
- `Operator NVTX Ranges`: host NVTX duration and inclusive CUDA work for internal GDN stages.
- `Kernel Time By Deepest NVTX Range`: each kernel counted once under the narrowest matching range.
- `Top CUDA Kernels`: highest-total GPU kernels in the trace.

ART-realistic throughput benchmarks should use one packed row (`batch_size == 1`). Some fast correctness cases intentionally include more than one row to stress parser and oracle mechanics, but ART training packs more trajectory groups into one longer row instead of running multiple packed rows in a batch.

Rules:

- Use pydantic `BaseModel` for structured cases, manifests, and metrics.
- Do not add dataclasses for ART-owned validation additions.
- Do not simplify cases to one prompt family per packed row.
- Do not report accepted results from dirty code without marking them provisional.
- Keep durable interpretation in project tracking, not only in local logs.
