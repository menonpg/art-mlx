# GDN Shared-Prefix Validation

This directory tracks correctness tests for the Qwen3.5 GDN shared-prefix and
context-parallel training path.

The main coverage is:

- `test_real_gdn_native_fla_cp.py`: production bf16 native FLA CP GDN path for
  outputs, recurrent state transport, input grads, and parameter grads.
- `test_qwen35_gdn_topology_oracle.py`: integrated Qwen3.5 GDN-only CP topology
  oracle through the model-support harness.
- `test_qwen35_full_model_cp1_packed_vs_flattened.py`: full-model fp32
  packed-vs-flattened oracle with the test-only GDN fp32 reference.
- `test_gdn_cp_packed_correctness.py`: CP2/4/8 packed edge cases against CP1.
- `test_gdn_cp_layout_distributed.py`: distributed layout exchange, including
  zero-token collective participation.
- `test_gdn_cp_train_prepare.py`: CP train microbatch preparation and main loss
  compatibility.
- `test_gdn_conv_gelu.py`: compact varlen causal conv kernel coverage.
- `test_real_gdn_tp_lora.py`: isolated GDN LoRA and TP gradient coverage.

The full-model oracle remains fp32 where a narrow test reference is available.
The real GDN CP tests intentionally exercise production bf16 kernels and CP
collectives. Do not change that split without discussing the coverage tradeoff.
