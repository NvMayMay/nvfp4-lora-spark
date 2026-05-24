# Phase 2 (future work): dynamic LoRA at CUTLASS speeds on Spark

This document parks the design + scope of a follow-up project that
addresses the one capability the Phase 1 release does NOT deliver:
**dynamic LoRA adapter swap at native NVFP4 CUTLASS speeds on DGX Spark
for the Super-120B model**.

Phase 1 ships a merge-then-serve workflow that gives ~12-14 tok/s for
each FT adapter, but loses the ability to swap adapters at runtime
without re-loading the model. Phase 2's goal: keep the speed and add
back the dynamic-swap.

## Why this is a separate project

The Phase 1 investigation discovered (see [DECISION_LOG D024-D025](../../Research/nvfp4_lora_spark/DECISION_LOG.md))
that vLLM 0.21 on sm_121:

- `CutlassExpertsFp4.supports_lora()` returns `False`. The CUTLASS NVFP4
  MoE kernel does not have a LoRA-aware forward path.
- `Nvfp4QuantizationEmulationTritonExperts` claims LoRA support but
  vLLM's Triton `_fused_moe_lora_expand` kernel crashes with
  `illegal memory access` during warmup on EMULATION backend.
- Marlin (which has LoRA) doesn't fit on Spark due to weight-repack
  transient.

So the natural vLLM paths for "fast + dynamic LoRA on Super" don't
exist. Adding them is an upstream engineering project that the codex
audits in round 8 + round 9 estimated at ~1-2 weeks of focused work
with a probable 5-15 tok/s ceiling (not the full 12-14 of base).

## Approach (per round-8 codex consensus)

Add a **post-MoE LoRA delta hook** to `FusedMoEModularMethod.apply()`:

```
y = cutlass_moe(hidden_states, w1_nvfp4, w2_nvfp4, ...)  # base @ 12-14 tok/s
if active_lora:
    y += moe_lora_delta(hidden_states, topk_ids, lora_ids, A, B, scaling)
return y
```

Architecture: keep CUTLASS doing the heavy quantized MoE matmul; add a
small bf16 LoRA delta after using the existing fused Triton
`_fused_moe_lora_expand` kernel (which must be fixed first - same bug
as the EMULATION path).

Codex prediction: ~5-10 tok/s achievable (some throughput loss from the
extra LoRA compute + synchronization).

## Scope

1. **Fix the upstream `_fused_moe_lora_expand` kernel** to work with
   EMULATION-format and CUTLASS-format weights. This is a vLLM bug
   independent of Phase 2 but is a prerequisite. Bug report draft:
   [`Research/nvfp4_lora_spark/vllm_issue_draft_fused_moe_lora_emulation_bug.md`](../../Research/nvfp4_lora_spark/vllm_issue_draft_fused_moe_lora_emulation_bug.md).

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

For full scope + codex audit history see
[round-8 audit](../../Research/nvfp4_lora_spark/round8_gpt55_20260524.md)
and [round-8 audit (codex-5.3)](../../Research/nvfp4_lora_spark/round8_codex53_20260524.md).
