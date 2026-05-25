# Phase 2 (future work): dynamic LoRA at CUTLASS speeds on Spark

This document parks the design + scope of a follow-up project that
addresses the one capability the Phase 1 release does NOT deliver:
**dynamic LoRA adapter swap at native NVFP4 CUTLASS speeds on DGX Spark
for the Super-120B model**.

Phase 1 ships a merge-then-serve workflow that gives ~11-14 tok/s for
each FT adapter, but loses the ability to swap adapters at runtime
without re-loading the model. Phase 2's goal: keep the speed and add
back the dynamic-swap.

## Why this is a separate project

The Phase 1 investigation, summarized in [serve/diagnostics/README.md](../serve/diagnostics/README.md),
found that vLLM 0.21 on sm_121:

- `CutlassExpertsFp4.supports_lora()` returns `False`. The CUTLASS NVFP4
  MoE kernel does not have a LoRA-aware forward path.
- `Nvfp4QuantizationEmulationTritonExperts` claims LoRA support but
  vLLM's Triton `_fused_moe_lora_expand` kernel crashes with
  `illegal memory access` during warmup on EMULATION backend.
- Marlin (which has LoRA) doesn't fit on Spark due to weight-repack
  transient.

So the natural vLLM paths for "fast + dynamic LoRA on Super" don't
exist. Adding them is an upstream engineering project estimated at
~1-2 weeks of focused work, with a probable 5-10 tok/s ceiling (not
the full 11-14 of base) for the working POC.

## Approach

Add a **post-MoE LoRA delta hook** to `FusedMoEModularMethod.apply()`:

```
y = cutlass_moe(hidden_states, w1_nvfp4, w2_nvfp4, ...)  # base @ 11-14 tok/s
if active_lora:
    y += moe_lora_delta(hidden_states, topk_ids, lora_ids, A, B, scaling)
return y
```

Architecture: keep CUTLASS doing the heavy quantized MoE matmul; add a
small bf16 LoRA delta after using the existing fused Triton
`_fused_moe_lora_expand` kernel (which must be fixed first - same bug
as the EMULATION path).

Expected throughput: ~5-10 tok/s achievable (some loss from the extra
LoRA compute + synchronization vs base CUTLASS).

## Scope

1. **Fix the upstream `_fused_moe_lora_expand` kernel** to work with
   EMULATION-format and CUTLASS-format weights. This is a vLLM bug
   independent of Phase 2 but is a prerequisite. The issue is summarized in
   [docs/TROUBLESHOOTING.md](TROUBLESHOOTING.md).

2. **Implement the post-MoE hook** as a vLLM monkey-patch:
   `VLLM_EXPERIMENTAL_CUTLASS_LORA_POST=1` env flag enables it. Run
   numerical validation against EMULATION+LoRA as reference.

3. **Benchmark + harden**. Target: >5 tok/s with dynamic adapter swap.
   Acceptable: 5-10 tok/s for the working POC.

4. **Upstream PRs**:
   - Kernel bug fix (high priority, blocks LoRA on multiple paths)
   - CUTLASS-LoRA integration (lower priority, larger discussion)

## Status

PARKED as of 2026-05-24. Phase 1 ships first; Phase 2 begins after
Phase 1 is published.

For related published context, see [docs/PERFORMANCE_ROADMAP.md](PERFORMANCE_ROADMAP.md)
and [serve/diagnostics/README.md](../serve/diagnostics/README.md).
