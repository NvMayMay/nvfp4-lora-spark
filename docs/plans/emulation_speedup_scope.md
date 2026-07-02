# Emulation expert-LoRA serving speedup -- scoping (Claude + gpt-5.5, 2026-06-29)

Scopes fixes for the ~10-20x slow NVFP4 expert-LoRA serving on one GB10 (vLLM 0.22.1, sm_121).
Feeds the comprehensiveness-roadmap item "fast expert-LoRA serving" (gated on the value experiment).

## Bottleneck (confirmed from vLLM source)
`fused_moe/experts/nvfp4_emulation_moe.py` `Nvfp4QuantizationEmulationTritonExperts.apply()`
dequantizes the FULL `w1` [E,2*inter,hidden//2] and `w2` [E,hidden,inter//2] to bf16 EVERY forward
(`:117-136`, before `topk_ids` is used), then calls `TritonExperts.apply()` which uses only the top-k
ROUTED experts (`:148-164`). Two wastes: ALL experts (not routed) + EVERY forward (static frozen
weights). On-the-fly dequant exists because full-bf16 experts (~210GB) exceed 128GB while 4-bit (~53GB) fit.

## Ranked fixes (one box)
1. **Routed-only dequant** -- M effort, ~4-12x decode (≈128/unique_routed), collapses on big prefill.
   Gather unique experts from `topk_ids`, dequant a compact `[k,...]` tensor (+ matching scales/global
   scales), call TritonExperts with compact weights. `expert_map` plumbing exists (`:93/:157`,
   `fused_moe.py:1351-1379`) and likely supports the global->local remap; RISK = id remap correctness +
   `expert_tokens_meta` staleness + dynamic shapes vs CUDA graphs. EXACT if scales gathered identically.
   **Highest-ROI code fix.**
2. **Batching / concurrency** -- S effort (config + concurrent clients), ~2-8x THROUGHPUT, ZERO kernel
   change. The full dequant is a FIXED per-forward cost independent of token count, so more tokens/step
   amortize it. REFRAMES the "10-20x" as a single-stream (max_num_seqs 1, sequential client) worst case;
   batched serving is far better. (Our eval was single-stream -> worst case.)
3. **Routed BF16 LRU cache** -- M (paired with #1), ~1.5-5x if routing has reuse. Cache dequantized hot
   experts (static weights). Can't cache all (memory); routing-skew dependent. LoRA does not invalidate
   the base-expert cache (LoRA weights are separate, `lora/layers/fused_moe.py:115-170`).
4. **Faster dequant kernel** -- LOW ROI (~1.1-1.5x). CORRECTION: `dequantize_to_dtype` is ALREADY a tuned
   Triton kernel (packed loads, nibble unpack, coalesced stores, `nvfp4_emulation_utils.py:50-128`), NOT
   naive. Only sm_121 tile/warp tuning left; can't fix the core "write huge bf16 then immediately read".
5. **CUDA graph** -- ~1.0-1.2x; helps launch overhead, not the dominant bf16 memory traffic. Secondary.

## Architectural option (the "real" cure) -- L/XL, NOT a cheap hook
"Keep base experts on fast cutlass + apply only the low-rank LoRA delta post-MoE" is NOT exact for
general expert LoRA. MoE LoRA modifies BOTH `gate_up` (BEFORE the gated nonlinearity) and `down` (after)
(`lora/layers/fused_moe.py:115-170`, 509-530). A post-MoE correction cannot reproduce the `gate_up`
change inside the nonlinear MLP. vLLM requires the selected MoE kernel ITSELF to support LoRA
(`supports_lora()` asserted, non-supporting quantized backends rejected, `:49-66`); cutlass/flashinfer
don't on sm_121. So exact fast-backend LoRA = large kernel work. A cheap post-MoE delta is exact ONLY for
down-projection-only (`w2`) adapters.

## Recommendation
If the value experiment says experts earn their keep: implement **#1 routed-only dequant** first (the real
10-20x attack), add **#3 bounded routed LRU cache**, and use **#2 concurrency** for serving throughput.
Kernel tuning (#4) and CUDA graph (#5) are secondary. The exact cutlass-LoRA backend is the long-term cure.
NET: emulation slowness is FIXABLE (medium effort), not fundamental -- this de-risks expert-LoRA serving.
