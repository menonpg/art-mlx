# Model Support Follow-Up Review

## Signal forwarding / cleanup on interrupt

Implemented in `service.py`.

- The parent now installs SIGINT and SIGTERM handlers after starting the Megatron and dedicated vLLM child processes.
- On interrupt, the handler calls `MegatronService.close()`, which tears down both child trees, then re-raises the original signal behavior.
- Dedicated vLLM now also starts in its own session and is killed by process group, matching Megatron.

This keeps the earlier `start_new_session=True` isolation, but removes the downside where a raw parent interrupt would not clean up the detached child group.

## Server probing and `/health`

The relevant vLLM OpenAI-compatible health endpoint is in:

- `vllm/entrypoints/serve/instrumentator/health.py`

That endpoint calls `engine_client(raw_request).check_health()` and returns:

- `200` when the engine is healthy
- `503` on `EngineDeadError`

So `/health` is meaningful for engine liveness, not just a trivial process heartbeat.

Current monitor behavior in `local/backend.py` is now:

1. check `/health`
2. check `/metrics`
3. if idle, issue a real generation probe

The generation probe still matters because it proves request handling and model readiness. The first idle probe now has an extended timeout through `ART_SERVER_MONITOR_INITIAL_TIMEOUT`.

## `streams::sync_dealloc`

The implementation is in Torch Dynamo stream tracing code:

- `torch/_dynamo/variables/streams.py`

Torch defines:

- `@custom_op("streams::sync_dealloc", mutates_args=())`

Its purpose is to wait on a stream event and move the last use of a tensor until after that wait, so the tensor cannot be deallocated or memory-reused before the side stream is finished with it.

This is a stream-lifetime / memory-safety op for compiled execution. It is not model math.

Why it showed up in compile workarounds:

- compiled graph capture encountered the op
- FakeTensor tracing needed a fake implementation registered for it

Why we removed it from `offload.py`:

- the duplicate fake registration there was redundant
- `compile_workarounds.py` is the right place for compile-only fake registrations

Risk assessment:

- correctness: the fake registration does not change runtime math, it only lets tracing reason about the op
- performance: the fake registration itself is not a runtime perf issue
- real risk: if we needed to fake-register this because some compiled path does not yet model the op cleanly, it is still a sign of compiler integration debt, but not a reason to keep duplicate registrations in runtime offload code

## Offload and colocation default

The intended behavior is now restored in `train.py`.

- non-dedicated Megatron service uses offload/reload around training jobs again
- dedicated mode remains enabled by this PR
- dedicated mode is not being made the default current RL path

So the current default remains training/inference colocation with offload for Megatron service.

## `_run_merged_vllm_serving()` startup flow

The merged-serving validator is doing the intended flow, but indirectly through `MegatronService.start_openai_server()`.

The actual sequence is:

1. start dedicated vLLM with the base model
2. wait for server readiness
3. call `_sync_dedicated_merged_weights(...)`
4. that triggers the Megatron-side merged-weight sync into the running vLLM server

The base-model startup is visible in `runtime_project.py`, where the dedicated runtime command is built with `--model=<base_model>`.

## `adapter_a` / `adapter_b` and moving off `_fused_gdn_adapter_weight`

The old fused GDN export no longer matches the current Bridge canonical adapter merge path.

Current Bridge merge wants canonical adapter entries keyed by suffix, not one ART-specific fused payload. For Qwen3.5 GDN that means:

- `adapter_qkv`
- `adapter_z`
- `adapter_b`
- `adapter_a`

Why zero `adapter_a` / `adapter_b` are present:

- Bridge canonical merge expects those suffix slots to exist for the base parameter shape it is merging
- Qwen3.5 GDN only has learned LoRA content for the qkv and z branches in our current wrapper/export path
- zero placeholders let us satisfy canonical merge structure without inventing non-zero weights for unsupported branches

Why the Qwen-specific adapter-name map belongs in the handler:

- it is Qwen3.5-specific Bridge integration knowledge
- shared export code should not mutate Bridge global mapping tables for one model family

That handler move is now done.

## Inductor / Triton cache overrides

The runtime-dir overrides in `service.py` were reverted.

Current persistent cache behavior remains in `runtime_env.py`:

- `TORCHINDUCTOR_CACHE_DIR=~/.cache/torchinductor`
- `TRITON_CACHE_DIR=~/.triton/cache`

That is the right final behavior.

## Position IDs

The suspicious early return in `train.py` is removed.

What is now added:

- realistic oracle packed-sequence construction pulled over from `codex_official_magi_attention_for_art`
- unit coverage for `stop_early` and `truncate`
- a new integration/runtime stage `packed_position_ids`

That stage:

- uses realistic packed sequences with multiple whole prompt families and multiple completion branches
- instantiates the real reduced Megatron provider/model path
- compares the unhooked real GPT `_preprocess` output against the hooked real `_preprocess` output on the same packed tensors
- validates that the hook either gathers correctly from a lookup-table rotary output or correctly no-ops on already batch-aligned Qwen3.5 mRoPE output

This is now wired into the model-support workflow as a mandatory stage.

## `shifted_labels`

No new follow-up action was needed here.

The earlier change was correct because the parity and SFT paths needed to derive labels from the same packed-tensor/SFT input contract used by the oracle code. That change was about aligning the shared SFT path, not about the position-id hook.

## Yes/no trainability disabling compile / server monitor

Those temporary disables are removed from `megatron_yes_no_trainability.py`.

The yes/no gate now runs with:

- server monitor enabled
- Megatron compile enabled

That is closer to the real system behavior and is the right final validation.

## `ART_FAST_DEBUG_DISABLE_FLEX_MAX_AUTOTUNE`

Completed wiring is:

- `flex_attention.py` now honors the env var directly and disables only max autotune options, not compiled flex attention itself
- workflow subprocesses explicitly inherit the parent environment
- Megatron child launch explicitly passes `env=os.environ.copy()`
- dedicated vLLM subprocess launch also now passes `env=os.environ.copy()`

So the flag now propagates through the workflow and the dedicated runtime paths, while keeping compiled flex attention enabled.
