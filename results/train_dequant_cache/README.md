# Train-time bf16 dequant cache (P1-3): speedup + correctness

`--train-dequant-cache-gb` keeps the frozen dequantized base weight resident and reuses it with a
plain F.linear, instead of re-dequantizing every step. Benchmarked on Box A (GB10/sm_121).

## Result (2026-07-02, Llama-3.1-8B-Instruct-NVFP4, 20 steps, bs1 x ga16, seq<=1024)

| | cache OFF (recompute) | cache ON (40 GB budget) | speedup |
|---|---|---|---|
| updates/s (steady) | ~0.075 | ~0.13 | **~1.7x** |
| supervised tok/s | ~35 | ~60 | ~1.7x |
| CUDA allocated | 6.6 GB | 20.6 GB | +14 GB (the bf16 cache) |

So on an 8B model with UMA headroom, the cache buys ~1.7x training step throughput for ~14 GB of
resident bf16 weights. 120B: leave the flag 0 (the cache would not fit; it falls back to recompute
per-module once the budget is exhausted, so it stays memory-safe either way).

## Correctness

- **Algebraically identical**: proven bit-for-bit on CPU (forward + every gradient, atol=0) in
  `tests/test_train_dequant_cache.py`. The base weight is frozen, so F.linear(x, W) yields the same
  dx = dy @ W and no grad to W as the recompute autograd.
- **On GPU**: the two paths dispatch different bf16 kernels (cached F.linear vs the recompute
  autograd), so tiny floating-point differences appear and compound over steps -- loss tracks the
  recompute run within ~2e-2 over 20 steps (steps 1-2 identical, then fp-noise divergence). This is
  the normal envelope of nondeterministic GPU training, not a logic difference; the trajectories
  match (e.g. step 3: 0.619 vs 0.600). Do not rely on step-for-step bit-identity on GPU.
