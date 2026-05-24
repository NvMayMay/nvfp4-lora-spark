# Performance roadmap: closing the NVFP4-to-BF16 throughput gap

The current `NVFP4LoRALinear` runs at roughly **11x slower per step** than a stock bf16 LoRA path on Nano-30B (measured: 4.2 s/step bf16 vs 43.8 s/step NVFP4 at batch=1, max_len=1536, identical hyperparams). The gap isn't fundamental, it's an implementation choice. This document records the five viable routes to close it, ordered by effort-to-payoff ratio, so future work can pick up cleanly.

## Why the gap exists today

Per training step, `NVFP4LoRALinear` does:

1. Read uint8 packed E2M1 nibbles + fp8_e4m3fn group scales + fp32 per-tensor scale (~`0.55 * numel` bytes).
2. Dequant arithmetic per element (cheap).
3. **Write** the materialized bf16 weight to HBM (`2 * numel` bytes).
4. Read the bf16 weight back for the tensor-core GEMM (`2 * numel` bytes).
5. Tensor-core math at full throughput.

Steps 3 and 4 are pure waste relative to the bf16 path, which goes straight to step 5 with the weight already in HBM. Backward through the chain repeats roughly the same dance. On top of that, the custom autograd Function has small per-call Python and CUDA dispatch overhead.

The gap is **memory bandwidth, not compute**. The dequant arithmetic is tiny compared to the GEMM. What's slowing us down is round-tripping the dequanted weights through HBM instead of streaming them through tensor-core registers.

## Route 1: Bigger micro-batch (low effort, ~1.5-3x)

Current scripts hardcode `batch_size=1`. At small batch, the GEMM is closer to memory-bound and the dequant write/read overhead dominates. Larger batch amortizes the dequant cost across more compute.

- For Nano-30B: trivial. ~22 GB used of 120 GB budget at batch=1, so there is ~100 GB headroom. Should fit batch=8 at max_len=4096 comfortably.
- For Super-120B: tighter. ~92 GB at batch=1, ~30 GB headroom under the ceiling. Batch=2 at max_len=1024 to 1536 likely fits; batch=2 at max_len=2048+ probably OOMs.

No kernel work required, just a script change and a memory-fit smoke. The 4x4 decision-map sweep in `Research/nvfp4_lora_spark/sweep_train_nvfp4.py` characterizes the viable region.

## Route 2: torch.compile / Inductor (low effort, unclear win)

The custom autograd Function makes Inductor compilation awkward, but worth trying. May fuse some of the dequant scaffolding. Probably a 1.2 to 1.5x gain at best because Inductor can't fuse uint8 bit-unpacking into a tensor-core GEMM, only the surrounding glue.

Risk: torch.compile on hybrid Mamba2 + MoE architectures has been flaky in past PyTorch versions. Worth a 30-minute experiment to see if the compile graph builds at all before committing to deeper work.

## Route 3: Triton fused dequant-GEMM kernel (1-2 weeks, ~5-8x)

This is the real answer. Write a Triton kernel that:

- Loads NVFP4 weights + fp8 group scales + fp32 per-tensor scale in tiles.
- Dequants in registers, never materializing the full bf16 weight in HBM.
- Feeds the tensor cores directly with the dequanted tile.

For backward wrt input (the only backward we need, since the NVFP4 weights are frozen), the kernel can be reused with a transposed access pattern.

Existing references:

- vLLM's marlin kernel does this for inference (forward only).
- AutoGPTQ / exllama have similar patterns for GPTQ formats.
- A draft NVFP4 Triton kernel may already exist in NVIDIA's open repos; worth scanning before writing from scratch.

Estimated residual gap to native bf16: ~1.5 to 2x, dominated by the irreducible cost of reading the scale tensors.

## Route 4: Transformer Engine NVFP4 path (medium effort, unknown ceiling)

NVIDIA Transformer Engine has Blackwell FP4 GEMM primitives. The catch is that TE expects to manage quantization itself: you give it bf16 weights and it quantizes on the forward pass. We have pre-quantized NVFP4 weights from disk in the exact wire format NVIDIA Model Optimizer produces.

Wrapping our pre-quantized NVFP4 weights as TE-compatible inputs requires confirming that TE's internal scale scheme matches the NVFP4 wire format (block-scaled E2M1 with fp8 group scales + fp32 per-tensor scale). If it does, this could be the lowest-effort path to native FP4 compute. If it doesn't, we would have to dequant-then-requant which defeats the point.

Worth an afternoon of investigation against the TE source code before committing.

## Route 5: Adapt marlin itself (3-4 weeks, ~native bf16)

vLLM's marlin kernel is hand-written CUDA optimized for inference forward. Extending it to support a training backward pass would give us inference-grade throughput during training.

Heavy lift. The marlin code is intricate (warp-level synchronization, ping-pong shared memory, hand-tuned for sm_80 and later). Adding a backward kernel that handles the same wire format is achievable but is multi-week kernel work for someone comfortable with CUDA tensor-core programming.

Side benefit: the marlin maintainers might be interested in an upstream contribution. A training-aware marlin would be useful to the broader vLLM community, not just this repo. Profile-raising as a side effect.

## Recommended order

1. **Route 1 first** because it is free. The 4x4 decision-map sweep gives us the per-cell throughput in a few hours and tells us how much headroom exists.
2. **Route 2 in parallel** as a 30-minute experiment. If torch.compile works on the model graph, ship a `--compile` flag. If it errors out, document the failure and move on.
3. **Route 3 as the v2 release.** A Triton fused kernel is the cleanest way to get most of the gap back without a multi-week CUDA project. Good story for a follow-up blog post.
4. **Route 4 as scoping work in parallel with route 3.** If TE turns out to accept the on-disk NVFP4 format directly, it might dominate route 3. Cheap to investigate.
5. **Route 5 only if there is a clear demand signal.** Don't pre-invest weeks of kernel work without users asking. The right time is once the v2 (Triton) release lands and someone says "still too slow for my use case."

## What we are shipping in v1

The current `NVFP4LoRALinear` is the v1 baseline. It is correct, memory-safe, and fits Super-120B on a single GB10 box. It is honest about the cost: see [README.md](../README.md) "Performance" section for the time/memory tradeoff.

Subsequent releases will work through this list in the order above.
