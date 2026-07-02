# Emulation expert-LoRA speedup -- implementation plan

From emulation_speedup_scope.md (Claude + gpt-5.5). Implements the routed-only dequant (primary),
concurrency batching (free), and routed LRU cache (compounding). These PATCH vLLM 0.22.1's emulation
MoE path -- shipped as a repo monkey-patch in the serve venv (precedent: serve/vllm_patches/).
GPU-validation gated on a free box + the expert-LoRA GO/NO-GO (don't ship a fix for a parked feature).

## Target
vllm/model_executor/layers/fused_moe/experts/nvfp4_emulation_moe.py
  `Nvfp4QuantizationEmulationTritonExperts.apply()` (:83-164) -- today dequants FULL w1/w2 (:117-136)
  before topk_ids is used (:148-164).

## F1 -- Routed-only dequant (PRIMARY; ~4-12x decode; effort M)
Mechanism (inside apply(), before the dequant):
  1. `routed = torch.unique(topk_ids)` ; drop padding/invalid (>= global_num_experts or < 0).
  2. Gather the compact expert set: `w1_c = w1[routed]`, `w2_c = w2[routed]`, and the MATCHING scale
     tensors `w1_scale_val[routed]`, `w2_scale_val[routed]`, and per-expert global scales
     `g1_alphas[routed]`, `g2_alphas[routed]` (dim0 = expert; scope confirmed global scale indexed by
     dim0, nvfp4_emulation_utils.py:71-75/267-273). Gather scales IDENTICALLY to weights or math breaks.
  3. dequantize_to_dtype on the COMPACT w1_c/w2_c only.
  4. Remap routing: build `remap = full(-1, global_num_experts); remap[routed] = arange(len(routed))`.
     Two integration options (pick whichever TritonExperts honors -- validate both):
       (a) pass `expert_map=remap` (already accepted at :93 and forwarded :157; fused_moe.py:1351-1379
           threads expert_map) and keep topk_ids global; OR
       (b) remap topk_ids -> compact (`topk_ids_c = remap[topk_ids]`) and pass global_num_experts=len(routed).
  5. Force `expert_tokens_meta=None` (or recompute for the compact set) so stale full-E metadata isn't
     reused -- codex's main risk flag.
  6. `super().apply(..., w1=w1_c, w2=w2_c, global_num_experts=len(routed), <remap per chosen option>)`.
Correctness: EXACT vs full-dequant (same math, fewer experts materialized). enforce-eager is on, so the
dynamic unique() shape doesn't fight CUDA graphs.
Caveats: prefill with many tokens routes to most experts -> gain shrinks (still correct). LoRA path
unaffected (LoRA weights separate, lora/layers/fused_moe.py:115-170).

## F2 -- Concurrency batching (FREE; ~2-8x throughput; effort S)
The full(or routed) dequant is a FIXED per-forward cost; amortize over more tokens/step.
  - Serve: raise `--max-num-seqs` (e.g. 1 -> 16) and gpu-mem-util as headroom allows.
  - Client: the eval/serving must issue CONCURRENT requests (eval_retention.py is sequential today ->
    add an optional `--concurrency N` thread-pool over rows, or shard). No kernel change.
This is the cheapest win and also gives an honest "throughput at concurrency" serving number (vs the
single-stream worst case we measured).

## F3 -- Routed BF16 LRU cache (compounding; ~1.5-5x w/ reuse; effort M, after F1)
Cache dequantized bf16 experts keyed by (layer_id, expert_id, dtype); bound by a mem budget (NOT all --
~210GB won't fit). On F1's gather, dequant only the MISSING routed experts, reuse cached. Evict LRU.
Static base weights => cache valid for the whole serve; LoRA does not invalidate it. Win is routing-skew
dependent (narrow tasks reuse a stable expert subset).

## Validation harness (GPU-gated; the gate before shipping)
1. CORRECTNESS (decisive): on a fixed input + adapter, assert routed-only output == full-dequant output
   within tight tol (ideally bit-exact; the only diff is which experts are materialized). Run base AND
   myft (with expert LoRA). A divergence = remap/scale-gather bug.
2. SPEEDUP: time the Spider eval (or a fixed token budget) full-dequant vs routed-only, single-stream;
   then F2 throughput at --max-num-seqs 16 + concurrent client.
3. Re-run the arm-B/C Spider EM eval with the patched serve -> must match the unpatched EM exactly.
Wire as a repo test/script under serve/vllm_patches/ + a parity check (ties into credibility B1).

## Sequencing
Write F1 patch + F2 client concurrency now (CPU). REVIEW (codex). GPU-validate F1 correctness + measure
F1/F2 speedup when a box frees AND the GO/NO-GO says experts ship. F3 only if F1+F2 leave it too slow.
