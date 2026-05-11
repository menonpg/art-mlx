# GDN Shared-Prefix Run Artifacts

This directory is for run outputs from GDN shared-prefix tests, probes, and benchmarks:

```
scratch/<run_id>/
```

Run outputs here are not disposable in the usual unit-test sense. If a run supports a claim, preserve its artifact path and record the claim in:

- `/root/ws/project_tracking/art/megatron_bridge_model_support_skill/achievement_index.md`

Large run directories are ignored by default. Commit compact manifests, config snapshots, and durable summaries in the appropriate tracked locations when they become part of an accepted result.

