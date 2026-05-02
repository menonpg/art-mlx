# Review Findings Since faeca8a715bab2530b7513eb8e8bee7b8caf6ccd

Running review focus:

- Introduced fallback behavior
- Error or behavior hiding
- Silent failure modes
- Poor or brittle code design
- Unintentional or unnecessary changes

## Findings

1. High: `import art` now requires Unsloth even though Unsloth is still only an optional backend dependency.
   - `src/art/__init__.py:41-44` unconditionally calls `disable_broken_mamba_ssm()` and imports `unsloth`.
   - `pyproject.toml:7-16` base dependencies do not include Unsloth; it remains under the `backend` extra at `pyproject.toml:21-42`.
   - This breaks base/serverless installs and makes the old `IMPORT_UNSLOTH` guard path effectively dead.
   - user: do we need to import unsloth here initially for any reason or can we use the old path? 
   - codex: We do not need to import `unsloth` from `art.__init__`. The real requirement is only that the Unsloth backend process imports it before `transformers`/`peft`/`trl`, so the old guarded path or an equivalent service-local lazy import is the right shape.
   - user: we do that then

2. High: the vLLM runtime launcher is source-tree dependent and not package-installable.
   - `src/art/vllm_runtime.py:28-47` defaults to `vllm_runtime/.venv/bin/art-vllm-runtime-server`.
   - Root packaging only includes `src/art` and `src/mp_actors`; `art-vllm-runtime` was removed from root extras.
   - A wheel install of `openpipe-art[backend]` cannot follow the runtime error's instruction to run `uv sync` in `vllm_runtime` because that project is not shipped.
   - user: respond below with a plan to correctly package art with the new vllm separation change
   - codex: The clean packaging is two distributions: `openpipe-art` and `art-vllm-runtime`. ART should not default to a source-tree `.venv` path in production; instead it should resolve an explicit runtime binary, with `ART_VLLM_RUNTIME_BIN` as the primary override, and the install docs/error path should tell users to install `art-vllm-runtime` in a separate venv and point ART at that console script.
   - user: this has been solved now

3. Medium-high: corrupted or unreadable Megatron adapter files can be silently replaced with identity LoRA weights.
   - `src/art/megatron/service.py:282-294` catches every exception while reading `adapter_model.safetensors` and returns `False`.
   - `_ensure_identity_lora()` then creates a new identity adapter at `src/art/megatron/service.py:303-306`.
   - `_resolve_active_lora_path()` now calls this unconditionally, including merged rollout mode.
   - user: this we definitely don't want. I think the only check we should do is if the path exists, we should expect to load the weights, strictly, fully, correctly. Anything that doesn't succeed in that throws an error
   - codex: Agreed. Identity LoRA creation should only be the intentional bootstrap path when no adapter exists yet; if `adapter_model.safetensors` exists, loading it should be strict and any corruption, parse error, shape mismatch, or partial-read condition should raise immediately.
   - user: confirmed

4. Medium: runtime readiness treats any non-5xx `/health` response as ready.
   - `src/art/vllm_runtime.py:80-83` returns on status codes below 500, including 3xx/4xx.
   - This can hide a wrong server bound to the selected port or auth/config errors.
   - user: respond if we can use 200 or should simply use raise_for_status.
   - codex: We should use exact `200` for `/health`. `raise_for_status()` is appropriate for the separate `/v1/models` follow-up probe, but `/health` itself should be a strict readiness check rather than “anything non-error-ish”.
   - user: ok then switch both, 200 for health and raise_for_status where appropriate

5. Medium: the Mamba import guard globally hides a real binary/import error.
   - `src/art/utils/optional_import_guards.py:61-88` patches `importlib.util.find_spec` and installs a `sys.meta_path` blocker.
   - `disable_broken_mamba_ssm()` makes `mamba_ssm` look missing after detecting one specific import error pattern.
   - This fallback is silent and process-wide.
   - user: this is an issue, agents.md explicitly requests no fallbacks. This seems like an error due to a broken venv install. Probably `rm -rf .venv` and `uv sync --all-extras` will fix it. If not, we need to investigate why it is failing. Respond below if you are successful in fixing this.
   - codex: I have not fixed this yet, and the current ART worktree venv does not actually have `mamba_ssm` installed at all, so `import mamba_ssm` fails with `ModuleNotFoundError` rather than the broken symbol error. That makes the guard even less defensible; the right production shape is to remove the fallback and let a genuinely broken backend env fail loudly.
   - user: confirmed

6. Medium-low: tests are stale around `fast_inference`.
   - `src/art/dev/validate.py:45-50` now rejects `fast_inference` globally.
   - `tests/unit/test_dedicated_config.py:100-145` still expects the old dedicated-only message and expects shared config to set `fast_inference=False`.
   - user: let's fix this mismatch
   - codex: Agreed. The tests should be updated to the hard-cutover semantics: `fast_inference` is rejected globally, shared mode should stop asserting an injected `fast_inference=False`, and the expected error text should match the new external-runtime-only design.
   - user: confirmed

7. Low-medium: newly added NCCL/vLLM patch paths use `assert` for runtime data validation.
   - `src/art/weight_transfer/nccl.py:187` validates NCCL unique-id payload length with `assert`.
   - `src/art/weight_transfer/nccl.py:281` and `:295` validate tensor device placement with `assert`.
   - `vllm_runtime/src/art_vllm_runtime/patches.py:117`, `:223-224`, and `:344` validate EP LoRA and NCCL payload invariants with `assert`.
   - Under `python -O`, these checks disappear and invalid payload/device state can proceed into NCCL or tensor slicing code.
   - user: ignore this, it's fine
   - codex: Acknowledged, leaving this alone.

8. Low/unnecessary: `.gitignore` adds `scratch/`, which appears unrelated to vLLM separation. 
   - user: we can keep this
   - codex: Acknowledged, fine to keep.

9.  Compatibility risk: deleting `src/art/vllm/*` removes the old `art.vllm` import surface.
   - This may be intentional, but it is a breaking change for downstream code that imported `art.vllm`.
   - user: this is fine, intentional 
   - codex: Acknowledged, intentional break.

10. Medium: shared-mode runtime launch now silently drops configured API auth.
    - `src/art/local/backend.py:491-495` still returns `api_key = server_args.get("api_key") or "default"` to callers.
    - `src/art/unsloth/service.py:164-174` and `src/art/megatron/service.py:249-261` remove `api_key` before launching the external runtime.
    - This changes shared-mode auth semantics silently: clients can be configured with a custom key while the runtime is started without enforcing that key.
    - user: we should respect this, not drop it. 
    - codex: Agreed. The runtime launch path should preserve `api_key`, and ART should use that same key consistently for its own internal calls to `/v1/models`, adapter load, sleep/wake, pause/resume, merged-weight endpoints, and alias updates.
    - user: confirmed

11. Medium-high: packed merged-weight transfer can return before the final NCCL broadcast completes.
    - `src/art/weight_transfer/packed_tensor.py:32-60` enqueues packed broadcasts on side CUDA streams and breaks after the final broadcast without synchronizing that stream.
    - `src/art/unsloth/service.py:406-438` awaits `trainer_send_weights()`, then unmerges the PEFT adapter in `finally`; that default-stream mutation can race the final side-stream broadcast.
    - This can silently publish partially stale or corrupted merged rollout weights, especially because the HTTP `/update_weights` success is treated as the synchronization point.
    - user: we should ensure this is synchronized by the end
    - codex: Agreed. `trainer_send_weights()` should not return until all producer-side side-stream broadcasts have completed; the clean fix is to explicitly join or synchronize the producer streams before returning so callers can safely mutate or unmerge the source weights afterward.
    - user: confirmed

12. Medium-high: multi-rank Megatron merged-weight export can deadlock on rank-0 HTTP failures.
    - Non-sender ranks wait at barriers in `src/art/megatron/merged_weight_export.py:247` and `:305-309`.
    - Rank 0 can raise before those barriers while posting `/init_weight_transfer_engine` at `src/art/megatron/merged_weight_export.py:238-246` or `/pause` at `:311-319`.
    - A transient runtime HTTP failure can therefore become a hung distributed job with no propagated error on the waiting ranks.
    - user: respond with a clean solution for this
    - codex: The clean fix is explicit distributed error propagation. Rank 0 should wrap each HTTP phase, serialize success/failure state plus the error text, broadcast that status to all trainer ranks at the matching sync point, and every rank should raise the same failure instead of some ranks blocking indefinitely at barriers.
    - user: sounds good to me. We should also be somewhat robust against potentially transient errors, perhaps with a simple retry policy and waiting up to 10s

13. Medium: routing replay now hides unexpected extra router calls by reusing the last route for the active microbatch.
    - `src/art/megatron/routing_replay.py:1372-1385` reuses `last_call_index` when the active call key is unchanged and the next recorded call belongs to another key.
    - Because the cursor is not incremented in that branch, `finalize_step()` at `src/art/megatron/routing_replay.py:1325-1338` can still pass after extra router invocations.
    - This is useful for recompute, but it also masks real forward-graph or dispatcher changes that add an extra router call for the same microbatch.
    - user: I am generally ok with this, but respond if you want to improve this functionality 
    - codex: I am okay with the intent, but I would improve it by making reuse explicit rather than silent. A good production shape is to keep the recompute-friendly reuse path, but count and surface these reuses in metrics/logs and keep a stricter mode available for debugging graph drift.
    - user: this is a good idea

14. Medium: shared-mode training can leave the external runtime asleep after any training or checkpoint failure.
    - Unsloth shared RL/SFT sleeps the runtime at `src/art/unsloth/service.py:636` and `:692`, but wake/reload happens only after the full training and checkpoint path completes at `:649-662` and `:710-722`.
    - Megatron shared RL/SFT sleeps the runtime in `_prepare_for_training()` at `src/art/megatron/service.py:611-618`, but wake happens only in `_publish_training_checkpoint()` at `:634-643`.
    - There is no `try/finally` around the train/job/checkpoint path, so an exception can silently strand inference in sleep mode.
    - user: what's a clean way to centralize the takedown in art? i.e. the training services, things moved to separate processes, vllm, etc. and get this to trigger on art death, finish, closing, etc
    - codex: This wants a real lifecycle owner. I would introduce a per-service async lifecycle manager that owns runtime sleep/wake state, subprocesses, temp files, wake locks, and transfer groups, and wire cleanup through `async with`, `close()`, signal handlers, and `atexit` so teardown runs on normal finish, exceptions, and process death paths rather than being open-coded in success-only flows.
    - user: now handled

15. Medium: Megatron sleep-before-start ordering can hide setup failures behind a sleeping runtime.
    - `_prepare_for_training()` sleeps the vLLM runtime before it calls `_ensure_megatron_running()` at `src/art/megatron/service.py:611-615`.
    - `_ensure_megatron_running()` now hard-fails if `megatron.bridge` is missing at `src/art/megatron/service.py:532-539`.
    - If the dependency check fails, the runtime has already been put to sleep and is not woken.
    - user: correct me if I'm wrong, but this would be addressed by a solution to 14?
    - codex: Yes, 14 addresses the stranded-runtime part, but I would still reorder this. Dependency validation and Megatron process startup checks should happen before sleep so setup failures do not disrupt inference at all.
    - user: sounds good

16. Medium: GDN distributed rank/group helpers silently fall back after any Megatron parallel-state error.
    - `src/art/megatron/gdn/operator.py:1919-1930` catches all exceptions from Megatron tensor-parallel state and falls back to process-group attrs or rank 0.
    - `src/art/megatron/gdn/operator.py:2738-2765` catches all exceptions from context-parallel state and falls back to global distributed rank or `WORLD`.
    - These fallbacks can mask partially initialized or mismatched TP/CP state and then use the wrong rank/group for masking or collectives.
    - user: for all locations where topology is accessed, unconditionally and without catching any exceptions, use megatron's parallel state. There is no reason to have guards and fallbacks around this
    - codex: Agreed. All topology lookups in these paths should use Megatron `parallel_state` directly and fail immediately if it is not initialized correctly; the current broad exception handlers are masking exactly the class of bugs we want surfaced.
    - user: confirmed

17. Medium: Qwen3.5 linear-attention support guesses a layer pattern when the Megatron helper import is missing.
    - `src/art/megatron/model_support/handlers/qwen3_5_moe.py:587-599` catches `ImportError` and derives the linear-attention pattern from `linear_attention_freq` and `num_layers`.
    - That can hide a Megatron Bridge version mismatch even though the Qwen3.5 spec declares a dependency floor.
    - user: don't catch import errors in any handler, these should be installed, not protected
    - codex: Agreed. Handler imports should be strict; if required Megatron Bridge support is missing, ART should fail loudly instead of guessing a fallback pattern.
    - user: confirmed

18. Compatibility risk: `runtime_profile="single_gpu_parity"` was removed from the Megatron provider API.
    - The `RuntimeProfile` type and `runtime_profile` arguments were removed from `prepare_provider_bundle()` and `get_provider_bundle()` in `src/art/megatron/provider.py:269-315`.
    - Callers can no longer request the single-GPU parity defaults; all provider construction now gets ART training defaults, FlexAttention, runtime env overrides, and finalize-time DeepEP defaults.
    - user: this is intentional
    - codex: Acknowledged, intentional.

19. Medium-low: provider import now installs global Megatron Bridge monkey patches.
    - `src/art/megatron/provider.py:12-27` imports and immediately calls `install_art_bridge_runtime_patches()`.
    - The patch function replaces global Megatron Bridge/model-provider methods at `src/art/megatron/bridge_runtime.py:350-367`.
    - This makes behavior depend on import order for any code sharing the same Python process with ART's Megatron provider.
    - user: maybe we can install these patches at __init__ time? What do you think?
    - codex: I would not move this to `__init__`, because that is still an import-time global side effect. A better shape is an explicit one-time bootstrap call from the Megatron runtime entrypoints so the patch is installed deliberately when a Megatron process starts, not when some module happens to be imported.
    - user: ok, that sounds good

20. High: tracked GDN code depends on untracked source files that are not in `HEAD` or the combined diff.
    - `src/art/megatron/gdn/gdn_shared_prefix.py:452`, `:702`, `:841`, and `:1604` import `art.megatron.gdn.layout`.
    - `src/art/megatron/gdn/layout.py` and `src/art/megatron/context_parallel/layout_index.py` exist in the working tree but are untracked (`git ls-files` returns nothing for them).
    - A clean checkout of `HEAD` or the saved `base..HEAD` patch will miss these modules, so GDN CP paths can fail with `ModuleNotFoundError`.
    - user: we need to include these, since the code relies on them. Preferably a minimal set of functions from these as well, rather than the whole files as they pertain to general cp execution.
    - codex: This finding is stale now; those files are tracked in the current worktree. I still agree with the minimization goal though: if we trim later, we should keep only the CP layout/index primitives that GDN actually uses rather than broad general-purpose files.
    - user: confirmed, this needs a trimming pass

21. Medium-low: shared Unsloth runtime now uses the subprocess lifecycle, but cleanup still only terminates the parent process.
    - Shared mode now calls `_start_vllm_subprocess()` at `src/art/unsloth/service.py:516-529`.
    - The subprocess is started without a new process group at `src/art/unsloth/service.py:217-223`.
    - `close()` only calls `terminate()`/`kill()` on the parent process at `src/art/unsloth/service.py:483-493`, unlike Megatron's process-group cleanup in `src/art/megatron/service.py:810-830`.
    - This was already a dedicated-mode risk, but the change extends it to shared mode, which is now the normal external-runtime path.
    - user: addressed by 14 decision?
    - codex: Mostly yes, but it still needs one concrete invariant: the Unsloth runtime subprocess should start in its own process group and be torn down via group termination, just like Megatron. The lifecycle manager from 14 is the right place to own that.
    - user: this sounds like a good idea

22. Low/unintentional: FlexAttention tuning behavior and its debug override were removed.
    - `src/art/megatron/flex_attention.py:34-39` now always compiles `flex_attention` with `_compile_options = None`.
    - The previous `max_autotune`, `coordinate_descent_tuning`, and `triton.cudagraphs=False` options are gone, and `ART_FAST_DEBUG_DISABLE_FLEX_MAX_AUTOTUNE` no longer appears anywhere in `src` or `tests`.
    - This looks unrelated to vLLM separation and can change performance/debug behavior silently.
    - user: I did this myself, don't change it and I will determine if I want to add it back in the final version.
    - codex: Acknowledged, no action here.

23. Compatibility risk: the top-level vLLM `GuidedDecodingParams` shim was removed without a replacement.
    - `src/art/__init__.py` no longer installs the old `vllm.sampling_params.GuidedDecodingParams` shim that was documented as needed for TRL compatibility.
    - `GuidedDecodingParams` no longer appears anywhere else in the repo.
    - This may be intentional for import separation, but it is still a behavior change for environments that have vLLM installed in the ART process.
    - user: do we need this plausibly at all? Or was it just a stray cleanup? The patches I don't believe are well validated, so it may not be necessary
    - codex: I do not see evidence that ART itself still needs this shim. Given the separation goal, a top-level ART compatibility shim for an in-process vLLM/TRl quirk looks like stray cleanup territory; if a real need still exists, it belongs on the runtime side, not in ART core.
    - user: ok, we'll leave it out then

10. Medium: `megatron/service.py` no longer calles `setup.sh` which installs system dependencies that are necessary
    - user: respond with why this was done and if you see any reason not to call `setup.sh`
    - codex: I think removing automatic `setup.sh` calls from runtime code was correct. `setup.sh` does environment provisioning work like `apt-get` and `uv sync`, so it should be an image/bootstrap step, not something ART tries to run dynamically when a service starts; runtime behavior should fail loudly with install instructions rather than mutating the machine.
    - user: ok let's have this fail if a megatron backend is instantiated and the packages aren't installed already. The error message can say to run setup.sh

11. Medium: `unsloth/service.py` `_train_sft_dedicated()`, not sure if this is necessary, if it isn't in main, is it used by tests? I don't think SFT with any inference makes sense, SFT doesn't need rollouts
    - user: respond to this question
    - codex: It is not used by the yes/no trainability path. I agree with the design concern too: unless ART main explicitly wants dedicated Unsloth SFT as a supported product path, I would rather remove or hard-error this mode than keep an extra inference-coupled SFT path around.
    - user: handled

12. Medium: `tests/integration/yes_no_trainability.py` just imports from `.vllm_separation`, but it should be the other way around, trainability code in `tests/integration` and `vllm_separation` imports that with any patches or modifiers specific to it as patches/extensions of the main test. 
    - user: fix this
    - codex: Agreed. The shared yes/no harness should live in `tests/integration`, and `tests/integration/vllm_separation` should wrap or extend it with separation-specific configuration rather than owning the canonical implementation.
    - user: confirmed

13. Medium: `tests/integration/test_megatron_qwen35_lora_wrapping.py`, this test seems unnecessary and we shouldn't write whole files to test each model. Remove it
    - user: fix this
    - codex: Agreed. The plan should be to delete this standalone model-specific file and keep only minimal coverage in existing generic LoRA/model-support tests so we do not accumulate one-off whole-file tests per model family.
    - user: you can literally just delete it, no need to preserve its code

14. Medium: `yes_no_trainability.py` uses `in_process=True` for megatron, this is not default in ART and thus should not be set
    - user: we should follow defaults, rather than using in_process=True. Are there any issues with not using in_process?
    - codex: We should follow defaults. There is no fundamental issue with not using `in_process`; the main difference is that subprocess mode exercises the real process boundary and serialization path, which is actually better validation for the production ART flow.
    - user: confirmed

## Verification Notes

- `uv run ruff check` completed successfully for the touched Python files.
- `uv run python -m py_compile src/art/unsloth/service.py src/art/megatron/service.py tests/integration/vllm_separation/yes_no_trainability.py tests/integration/yes_no_trainability.py` completed successfully.
- `uv run python -m pytest tests/unit/test_megatron_merged_weight_export.py tests/unit/test_megatron_service_dedicated.py tests/unit/test_dedicated_config.py tests/unit/test_moe_routing_replay.py` completed successfully: 48 passed.
- `uv run python -m pytest tests/integration/vllm_separation/test_megatron_merged_weight_export.py tests/integration/vllm_separation/test_runtime_launcher.py tests/integration/vllm_separation/test_yes_no_trainability_config.py tests/integration/vllm_separation/test_service_runtime_boundary.py` completed successfully after committing the test-update patch: 23 passed.
- `git diff --check` completed with no whitespace errors.

## Applied Diffs

### Finding 1

```diff
diff --git a/src/art/__init__.py b/src/art/__init__.py
@@
-from .utils.optional_import_guards import disable_broken_mamba_ssm
-
-disable_broken_mamba_ssm()
-import unsloth  # noqa: F401
+if os.environ.get("IMPORT_UNSLOTH", "0") == "1":
+    import unsloth  # noqa: F401
```

### Finding 3

```diff
diff --git a/src/art/megatron/service.py b/src/art/megatron/service.py
@@
-    def _adapter_has_weights(self, lora_path: str) -> bool:
+    def _adapter_exists_and_loads(self, lora_path: str) -> bool:
         adapter_path = os.path.join(lora_path, "adapter_model.safetensors")
         if not os.path.exists(adapter_path):
             return False
-        try:
-            with safe_open(adapter_path, framework="pt") as adapter_file:
-                for key in adapter_file.keys():
-                    tensor = adapter_file.get_tensor(key)
-                    if torch.any(tensor != 0):
-                        return True
-        except Exception:
-            return False
-        return False
+        with safe_open(adapter_path, framework="pt") as adapter_file:
+            keys = list(adapter_file.keys())
+            if not keys:
+                raise RuntimeError(f"LoRA adapter contains no tensors: {adapter_path}")
+            for key in keys:
+                adapter_file.get_tensor(key)
+        return True
```

### Finding 4

```diff
diff --git a/src/art/vllm_runtime.py b/src/art/vllm_runtime.py
@@
-                if response.status_code < 500:
+                if response.status_code == 200:
                     return
```

### Finding 5

```diff
diff --git a/src/art/unsloth/train.py b/src/art/unsloth/train.py
@@
-    from ..utils.optional_import_guards import disable_broken_mamba_ssm
-
-    disable_broken_mamba_ssm()
     import unsloth
diff --git a/src/art/preprocessing/tokenize.py b/src/art/preprocessing/tokenize.py
@@
-    from ..utils.optional_import_guards import disable_broken_mamba_ssm
-
-    disable_broken_mamba_ssm()
     import unsloth  # noqa: F401 - Must be imported first to set UNSLOTH_IS_PRESENT env var
diff --git a/src/art/utils/optional_import_guards.py b/src/art/utils/optional_import_guards.py
deleted file mode 100644
```

### Finding 6

```diff
diff --git a/src/art/dev/validate.py b/src/art/dev/validate.py
@@
-    if config.get("init_args", {}).get("fast_inference"):
+    if "fast_inference" in config.get("init_args", {}):
         raise ValueError(
             "fast_inference is no longer supported; ART always uses an external "
             "vLLM runtime"
diff --git a/tests/unit/test_dedicated_config.py b/tests/unit/test_dedicated_config.py
@@
-        ValueError, match="fast_inference is incompatible with dedicated"
+        ValueError, match="fast_inference is no longer supported"
@@
-        assert result["init_args"].get("fast_inference") is False
+        assert "fast_inference" not in result["init_args"]
```

### Finding 10

```diff
diff --git a/src/art/unsloth/service.py b/src/art/unsloth/service.py
@@
-        for key in ("port", "host", "lora_modules", "api_key"):
+        for key in ("port", "host", "lora_modules"):
             server_args.pop(key, None)
         return server_args
+
+    def _runtime_request_kwargs(self) -> dict[str, dict[str, str]]:
+        headers = self._runtime_headers()
+        return {"headers": headers} if headers else {}
diff --git a/src/art/megatron/service.py b/src/art/megatron/service.py
@@
-        for key in ("port", "host", "lora_modules", "api_key"):
+        for key in ("port", "host", "lora_modules"):
             server_args.pop(key, None)
         return server_args
@@
         return MergedWeightTransferSpec(
             init_info=init_info,
             vllm_base_url=self._vllm_base_url,
             served_model_name=f"{self.model_name}@{step}",
+            api_key=self._vllm_api_key,
         )
diff --git a/src/art/megatron/jobs.py b/src/art/megatron/jobs.py
@@
 class MergedWeightTransferSpec(BaseModel):
     init_info: MergedWeightTransferInitInfo
     vllm_base_url: str
     served_model_name: str
+    api_key: str | None = None
```

### Finding 11

```diff
diff --git a/src/art/weight_transfer/packed_tensor.py b/src/art/weight_transfer/packed_tensor.py
@@
                 if packing_tensor_list[buffer_idx]:
                     packed_tensors[buffer_idx] = torch.cat(
                         packing_tensor_list[buffer_idx], dim=0
                     )
                     group.broadcast(packed_tensors[buffer_idx], src=src)
                 break
+    for stream in streams:
+        stream.synchronize()
```

### Finding 12

```diff
diff --git a/src/art/megatron/merged_weight_export.py b/src/art/megatron/merged_weight_export.py
@@
+def _post_with_retry(...):
+    ...
+    raise RuntimeError(f"{phase} failed after retrying for {retry_seconds:g}s")
+
+def _sync_rank_zero_status(...):
+    torch.distributed.broadcast_object_list(payload, src=0)
+    if payload[0] is not None:
+        raise RuntimeError(f"{phase} failed on rank 0: {payload[0]}")
@@
-    _maybe_distributed_barrier(world_size)
+    _sync_rank_zero_status(
+        rank=rank,
+        world_size=world_size,
+        phase="initialize merged weight transfer",
+        error=error,
+    )
@@
-        _maybe_distributed_barrier(world_size)
+        _sync_rank_zero_status(..., phase="pause generation", error=pause_error)
@@
-            _maybe_distributed_barrier(world_size)
+            _sync_rank_zero_status(..., phase="update merged weights", error=update_error)
+            _sync_rank_zero_status(..., phase="resume generation", error=resume_error)
diff --git a/tests/integration/vllm_separation/test_megatron_merged_weight_export.py b/tests/integration/vllm_separation/test_megatron_merged_weight_export.py
@@
-    assert barriers == [2]
+    assert barriers == []
@@
-    assert barrier_calls == [2, 2, 2]
+    assert barrier_calls == [2]
```

### Finding 13

```diff
diff --git a/src/art/megatron/routing_replay.py b/src/art/megatron/routing_replay.py
@@
         strict: bool,
         local_token_indexer: LocalTokenIndexer | None = None,
+        allow_recompute_reuse: bool = True,
@@
+        self._router_reuse_counts: dict[str, int] = {}
@@
+        if self._router_reuse_counts:
+            logger.info(
+                "Routing replay reused routes for recompute: step=%s counts=%s",
+                self._active_step_index,
+                dict(sorted(self._router_reuse_counts.items())),
+            )
@@
+            if not self.allow_recompute_reuse:
+                raise RuntimeError("Routing replay recompute reuse is disabled: ...")
             route = router_calls[last_call_index]
+            self._router_reuse_counts[router_key] = (
+                self._router_reuse_counts.get(router_key, 0) + 1
+            )
```

### Finding 15

```diff
diff --git a/src/art/megatron/service.py b/src/art/megatron/service.py
@@
     async def _prepare_for_training(self) -> str:
         self._validate_megatron_dependencies()
-        await self._sleep_runtime()
-        gc_and_empty_cuda_cache()
-
         await self._ensure_megatron_running()
+        await self._sleep_runtime()
+        gc_and_empty_cuda_cache()
```

### Finding 16

```diff
diff --git a/src/art/megatron/gdn/operator.py b/src/art/megatron/gdn/operator.py
@@
-    try:
-        from megatron.core import parallel_state as ps
-        if getattr(ps, "model_parallel_is_initialized", lambda: False)():
-            return int(ps.get_tensor_model_parallel_rank())
-    except Exception:
-        pass
-    ...
-    return int(getattr(projection, "tp_rank", 0))
+    del projection
+    from megatron.core import parallel_state as ps
+    return int(ps.get_tensor_model_parallel_rank())
@@
-    if torch.distributed.is_available() and torch.distributed.is_initialized():
-        return torch.distributed.group.WORLD
-    raise RuntimeError("CP GDN execution requires torch.distributed initialization")
+    del cp_size
+    from megatron.core import parallel_state as ps
+    return ps.get_context_parallel_group()
```

### Finding 17

```diff
diff --git a/src/art/megatron/model_support/handlers/qwen3_5_moe.py b/src/art/megatron/model_support/handlers/qwen3_5_moe.py
@@
-    try:
-        from megatron.bridge.models.qwen_vl.qwen35_vl_bridge import Qwen35VLMoEBridge
-    except ImportError:
-        return bridge_types
-    return bridge_types + (Qwen35VLMoEBridge,)
+    from megatron.bridge.models.qwen_vl.qwen35_vl_bridge import Qwen35VLMoEBridge
+    return (Qwen3MoEBridge, Qwen35VLMoEBridge)
@@
-    except ImportError:
-        frequency = int(getattr(provider, "linear_attention_freq", 1) or 1)
-        layer_count = int(getattr(provider, "num_layers", 1) or 1)
-        return [...]
+    from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
+        get_linear_attention_pattern,
+    )
```

### Finding 19

```diff
diff --git a/src/art/megatron/provider.py b/src/art/megatron/provider.py
@@
-from art.megatron.bridge_runtime import install_art_bridge_runtime_patches
@@
-install_art_bridge_runtime_patches()
diff --git a/src/art/megatron/train.py b/src/art/megatron/train.py
@@
+from art.megatron.bridge_runtime import install_art_bridge_runtime_patches
+
+install_art_bridge_runtime_patches()
```

### Finding 20

```diff
diff --git a/src/art/megatron/gdn/gdn_shared_prefix.py b/src/art/megatron/gdn/gdn_shared_prefix.py
@@
-try:
-    from art.megatron.context_parallel.layout_index import TokenLayoutIndex
-except ModuleNotFoundError:
-    class TokenLayoutIndex(BaseModel):
-        ...
+from art.megatron.context_parallel.layout_index import TokenLayoutIndex
diff --git a/src/art/megatron/gdn/layout.py b/src/art/megatron/gdn/layout.py
@@
-class GdnCpLayoutPlan(BaseModel):
-    ...
-
-def build_gdn_cp_layout_plan(...):
-    ...
-
-def build_gdn_token_order(...):
-    ...
-
-def split_gdn_families_by_rank(...):
-    ...
```

### Finding 21

```diff
diff --git a/src/art/unsloth/service.py b/src/art/unsloth/service.py
@@
             except RuntimeError as exc:
+                returncode = self._vllm_process.returncode
+                self.close()
                 raise RuntimeError(
-                    f"vLLM subprocess exited with code {self._vllm_process.returncode}. "
+                    f"vLLM subprocess exited with code {returncode}. "
                     f"Check logs at {log_dir}/vllm-runtime.log"
                 ) from exc
diff --git a/src/art/megatron/service.py b/src/art/megatron/service.py
@@
             except RuntimeError as exc:
+                returncode = self._vllm_process.returncode
+                self._stop_vllm_subprocess()
                 raise RuntimeError(
-                    "vLLM subprocess exited with code "
-                    f"{self._vllm_process.returncode}. "
+                    f"vLLM subprocess exited with code {returncode}. "
                     f"Check logs at {log_dir}/vllm-runtime.log"
                 ) from exc
```

### Additional Finding 10

```diff
diff --git a/src/art/megatron/service.py b/src/art/megatron/service.py
@@
+    def __post_init__(self) -> None:
+        self._validate_megatron_dependencies()
@@
                 "Megatron dependencies are not available in the active ART environment. "
-                "Build the project venv with `uv sync --extra backend --extra megatron` "
-                "before starting Megatron training."
+                "Run `setup.sh` for this worktree or build the project venv with "
+                "`uv sync --extra backend --extra megatron` before starting Megatron "
+                "training."
```

### Additional Finding 12

```diff
diff --git a/tests/integration/vllm_separation/yes_no_trainability.py b/tests/integration/yes_no_trainability.py
similarity index 99%
rename from tests/integration/vllm_separation/yes_no_trainability.py
rename to tests/integration/yes_no_trainability.py
@@
-from ..megatron_oracle_harness import ORACLE_TOPOLOGY, Topology
-from ..megatron_oracle_worker import provider_topology_env
+from .megatron_oracle_harness import ORACLE_TOPOLOGY, Topology
+from .megatron_oracle_worker import provider_topology_env
diff --git a/tests/integration/vllm_separation/yes_no_trainability.py b/tests/integration/vllm_separation/yes_no_trainability.py
new file mode 100644
@@
+from ..yes_no_trainability import (...)
```

### Additional Finding 13

```diff
diff --git a/tests/integration/test_megatron_qwen35_lora_wrapping.py b/tests/integration/test_megatron_qwen35_lora_wrapping.py
deleted file mode 100644
```

### Additional Finding 14

```diff
diff --git a/tests/integration/vllm_separation/test_live_megatron_backend_smoke.py b/tests/integration/vllm_separation/test_live_megatron_backend_smoke.py
@@
-            async with MegatronBackend(path=str(backend_root), in_process=True) as backend:
+            async with MegatronBackend(
+                path=str(backend_root), in_process=False
+            ) as backend:
                 yield backend
```
