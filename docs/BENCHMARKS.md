# Measured on GB10

Every number here was measured on a single NVIDIA DGX Spark (GB10, sm_121, ~128 GB
unified LPDDR5x). The honesty rules: the *delta* base→adapter is the signal (absolute
scores are scorer-dependent); the committed eval JSON under [../results/](../results/)
backs each quoted number; the "frontier" rows are fit-tested one-shot, **not** safety-certified.

## Contents

- [Capability: Spider text-to-SQL before/after](#capability-spider-text-to-sql-beforeafter)
- [Training fits on a single GB10](#training-fits-on-a-single-gb10)
- [Long-context training: validated configurations](#long-context-training-validated-configurations)
- [Quantization tax on adapter quality](#quantization-tax-on-adapter-quality)
- [Serving throughput (single-stream)](#serving-throughput-single-stream)
- [Concurrency scaling (the headline serve case)](#concurrency-scaling-the-headline-serve-case)
- [Dequant kernel throughput](#dequant-kernel-throughput)

---

## Capability: Spider text-to-SQL before/after

Public base + public dataset + deterministic metric. Reproduce end to end with
[../REPRODUCE_SPIDER.md](../REPRODUCE_SPIDER.md). Trained *and* served on one GB10;
the adapter is attached to the 4-bit base at request time (never merged, never re-quantized).

| Spider dev (Llama-3.1-8B-NVFP4) | base | adapter | delta |
|---|---|---|---|
| gold-SQL NLL (lower is better) | 0.889 | 0.850 | -0.039 |
| exact-set-match | 36.8% | 52.0% | +15.3 pp |

Full 1034-row dev, 2 epochs, deterministic (no sampling, no DB execution). The *delta*
is the signal; the absolute exact-set-match is scorer-dependent (strict component
set-match, value-insensitive). Per-row eval JSON: [../results/spider/](../results/spider/).

**Generalizes across families** (same recipe, swap `--model-dir`):

| Base (NVFP4) | eval n / epochs | exact-set-match | NLL | artifact |
|---|---|---|---|---|
| Llama-3.1-8B   | 1034 / 2 | 36.8% -> 52.0% (+15.3 pp) | 0.889 -> 0.850 | `results/spider/spider_retention_llama8b_e2.json` |
| Mistral-Small-3.2-24B | 200 / 2 | 24.5% -> 61.0% (+36.5 pp) | 1.37 -> 0.57 | `results/spider/spider_retention_mistral24b_e2.json` |
| Qwen3-32B      | 1034 / 1 | 46.1% -> 43.4% (saturated) | 1.29 -> 0.26 (large) | `results/spider/spider_retention_qwen32b_e1.json` |

The capability lift scales with base headroom: a base that already near-saturates the
strict set-match (Qwen3-32B) shows its gain as a large NLL/calibration improvement rather
than +EM. Only the Llama row is the full 1034-row, 2-epoch headline.

---

## Training fits on a single GB10

Measured at batch=1, max_len=1536, grad_accum=4, AdamW lr=1e-4, gradient checkpointing on,
over 1 epoch on a chat-format JSONL dataset. Step time and final loss are production-run
averages from the v1.0 ICH-v3.1 reference runs (1081 forward/backward steps). Memory is
`torch.cuda.max_memory_allocated` for the NVFP4 rows; `max(cuda_alloc)` sampled at log time
for the BF16 row (a lower bound, no torch-peak available).

| Run | Base dtype | LoRA targets | Trainable params | GPU memory | Step time | Final loss (ICH-v3.1) |
|---|---|---|---:|---:|---:|---:|
| Super-120B (NVFP4) | NVFP4 | up_proj, down_proj (r=8) | 1216.4 M | 93.2 GB peak | 135.6 s | 0.81 |
| Nano-30B (NVFP4) | NVFP4 | up_proj, down_proj (r=8) | 216.4 M | 36.1 GB peak | 43.8 s | 1.00 |
| Nano-30B (BF16) | BF16 | up_proj, down_proj (r=8) | 216.4 M | 67.5 GB sampled | 4.2 s | 0.98 |

![Training memory and throughput](../plots/04_throughput_and_memory.png)

Super-120B NVFP4 training peaks at 93.2 GB on a 128 GB box, leaving headroom for longer
sequences. Nano-30B NVFP4 peaks at ~36 GB. The sampled per-step memory series (109 logged
points across the 1081-step run) is flat for both models, with no leak or creep:

![GPU memory timeline during training](../plots/03_memory_timeline.png)

A (batch × max_len) sweep characterized the feasible region and identified `b=4, max_len=1024`
as the throughput-optimal config that still fits the Spark memory budget:

| Model | Conservative (b=1, ml=1536) | Throughput-optimal (b=4, ml=1024) | Per-sample speedup |
|---|---|---|---:|
| Super-120B (NVFP4) | 137.94 s/sample, 93.2 GB peak | **36.69 s/sample**, 99.2 GB peak | **3.76x** |
| Nano-30B (NVFP4) | 49.25 s/sample, 36.1 GB peak | **18.55 s/sample**, 60.5 GB peak | **2.65x** |

The sweep cells are 3-step warm-state measurements; per-step rates are stable across the
feasible region (full per-cell data in [../results/training_throughput_sweep/](../results/training_throughput_sweep/)).
Loss curves for the v1.0 production runs are in [../plots/train_metrics.json](../plots/train_metrics.json):

![Training loss curves for Nano-30B and Super-120B](../plots/01_loss_curves.png)

**Operational note (Super training on GB10).** Launch Super-120B training from a clean boot.
Loading the 75 GB NVFP4 base produces a burst of self-resolving `NV_ERR_NO_MEMORY` lines in
`/var/log/kern.log` during the load phase; if a fresh NVRM burst or any `NVRM: Xid` appears
*after* training starts, abort and reboot. Full failure-signature playbook in
[TROUBLESHOOTING.md](TROUBLESHOOTING.md).

---

## Long-context training: validated configurations

Two training modes cover short SFT up to 262k-token cached-context adaptation on one Spark,
all with the standard r=8 LoRA on `up_proj,down_proj`.

- **Exact full-sequence** (default `--training-mode full_sequence`): all tokens backprop.
- **Cached-prefix + trainable suffix** (`--training-mode cached_prefix_suffix --train-suffix-len N`):
  the prefix is prefilled under `torch.no_grad()` into a read-only K/V + Mamba SSM-state cache;
  only the trailing N tokens receive gradients (at most **N − 1** supervised targets per step).

Single-step results on Super-120B-NVFP4 on a 130.66 GB GB10 (`--batch 1 --grad-accum 1`):

| Training mode | Total context | Trainable suffix | CUDA peak (backward) | Step wall (post-load) | Use case |
|---|---:|---:|---:|---:|---|
| Exact full-sequence | 16,384 | (full) | 101.9 GB | 3.5 min | Standard SFT, contexts ≤16k |
| Cached-prefix + suffix | 4,096 | 2,048 | 85.7 GB | 3.1 min | Smallest cached shape; smoke |
| Cached-prefix + suffix | 16,384 | 2,048 | 85.8 GB | 6.9 min | ICH-v3.1-style records fit fully |
| Cached-prefix + suffix | 65,536 | 2,048 | 87.1 GB | 23.3 min | Retrieval / document-level pretexts |
| Cached-prefix + suffix | 262,144 | 1,024 | 90.8 GB | 22.4 min | Longest validated context (suffix reduced to clear the backward allocator cliff) |

Recommended flag set for cached-prefix rows (the certified-run values):

```
--batch 1 --grad-accum 1
--training-mode cached_prefix_suffix --train-suffix-len <SUFFIX> --prefix-chunk-len <CHUNK>
--loss-mode chunked_frozen_ce --loss-chunk-tokens 512
--optimizer adafactor
--sdpa-causal-no-mask --pooled-loader-buffers --moe-sparse-no-one-hot --mamba-cached-multitoken
--watchdog-min-available-gb 2 --watchdog-nvrm-errors --profile-memory-phases
```

Use `--prefix-chunk-len 2048` up to 16k, `4096` for 64k, `8192` for 256k. Full per-row CLI
invocations: [LONG_CONTEXT_EXPERIMENTS.md](LONG_CONTEXT_EXPERIMENTS.md) (SUPER-LC-032, LC-046,
LC-048, LC-050, LC-060).

**Trainable-suffix ceiling.** Suffix ≥ ~3,000 tokens fails at backward with NVRM
`NV_ERR_NO_MEMORY` at `mem_desc.c:1359` despite byte-level headroom — a per-process CUDA
descriptor-pool ceiling, not byte-OOM. Suffix=2,048 covers the ICH-v3.1 distribution; larger
suffixes are future-work.

### Frontier (fit-tested, not certified)

These predate the safety watchdog. They demonstrate that a single training step fits one-shot,
but are explicitly **not** a recipe: host headroom during the step is well under 3 GiB on every
row, the later watchdog-instrumented test at 20,480 tokens failed with NVRM `mem_desc.c:1359`
before backward, and the higher rows use reduced LoRA rank / target set (collapsing adapter capacity).

| Total context | LoRA config | Status | Peak CUDA reserved | Host avail during step | Journal entry |
|---:|---|---|---:|---:|---|
| 24,576 | r=8, `up_proj+down_proj` | one-step fit | 117.4 GB | ~2.6 GiB | [SUPER-LC-013](LONG_CONTEXT_EXPERIMENTS.md) |
| 28,672 | r=8, `up_proj+down_proj` | one-step fit | 121.5 GB | <1 GiB | [SUPER-LC-016](LONG_CONTEXT_EXPERIMENTS.md) |
| 32,768 | r=4 alpha=8, `up_proj+down_proj` | one-step fit | 124.3 GB | nearly exhausted | [SUPER-LC-022](LONG_CONTEXT_EXPERIMENTS.md) |
| 34,816 | r=2 alpha=4, **`down_proj` only** | one-step fit | (tiny-adapter probe) | (tiny-adapter probe) | [SUPER-LC-023](LONG_CONTEXT_EXPERIMENTS.md) |

For production training, the cached-prefix path at 65k/256k delivers larger effective context
with much more headroom; for exact full-sequence, 16,384 remains the safe ceiling.

---

## Quantization tax on adapter quality

The Nano-30B BF16 row above is an ablation: identical LoRA hyperparameters trained against the
BF16 Nano base. Final losses match to within 0.02 (1.00 vs 0.98) and the loss curves overlap
step-for-step:

![NVFP4 vs BF16 ablation: loss matches, NVFP4 uses 1/3 the memory](../plots/02_quant_ablation_training.png)

NVFP4 LoRA uses 36.1 GB peak vs 67.5 GB sampled at BF16 for the same workload — **roughly half
the memory with no detectable training-loss penalty** on this 1081-example dataset. Broader-domain
evaluation is on the roadmap.

---

## Serving throughput (single-stream)

Single-stream decode on one Spark is **bandwidth-bound** (~273 GB/s LPDDR5x): a single request
decodes slowly. This is the community's known Spark limitation and is *not* how you should run it
in production — see [concurrency scaling](#concurrency-scaling-the-headline-serve-case) below.

Merged-FT Super-120B serving via vLLM 0.21 CUTLASS runs at 13.2 tok/s mean across 5 workload cells
`(prompt, output)` of (12,32), (54,64), (54,256), (228,256), (456,128), landing 12.2-13.7 tok/s — at
parity with base CUTLASS on the same cells (11.3-13.9 tok/s, mean 12.9). The EMULATION fallback
measures 0.70-0.73 tok/s (~18x slower), retained only for the case where CUTLASS breaks.

![Inference throughput: merged-FT matches base, CUTLASS is 18x EMULATION](../plots/05_inference_throughput.png)

---

## Concurrency scaling (the headline serve case)

The 128 GB unified pool is the architectural advantage for **batched** serving: KV cache for
concurrent requests fits inside the same memory the weights already live in, with no PCIe
transfer to bottleneck batching. Discrete-GPU local systems (16-32 GB VRAM) saturate much earlier
because KV cache competes with weights for a small pool.

**Nano-30B-FT**, measured at concurrency 1/2/4/8 (`--max-num-seqs 8`, ctx=4096). Short-prompt cells
(the headline batched case):

| Workload | conc=1 | conc=2 | conc=4 | conc=8 | speedup at conc=8 |
|---|---:|---:|---:|---:|---:|
| prompt=512, output=64 | 30 tok/s | 91 tok/s | 137 tok/s | 200 tok/s | 6.7x |
| prompt=512, output=256 | 54 tok/s | 99 tok/s | 206 tok/s | 312 tok/s | 5.7x |
| prompt=512, output=2048 | 55 tok/s | 116 tok/s | 191 tok/s | **339 tok/s** | 6.2x |

![Nano-30B-FT aggregate throughput vs concurrency on Spark](../plots/06_inference_concurrency.png)

Aggregate throughput across the full sweep ranges from ~5 tok/s (long prompt + short output,
TTFT-dominated) up to ~339 tok/s (short prompt + long output at conc=8) — a ~120x span between the
worst single-stream cell and the best batched cell. Long-prompt cells (prompt ≥ 2048) become
compute-bound at prefill (TTFT 0.1-3 s, up to 13-16 s when the scheduler interleaves long prefills
with short decodes). Full data: [../results/inference_concurrency_sweep/](../results/inference_concurrency_sweep/).

**Super-120B** merged-FT also scales (`--max-num-seqs=3`, ctx=4096):

| Workload | conc=1 | conc=2 | conc=3 | speedup at conc=3 |
|---|---:|---:|---:|---:|
| prompt=512, output=64 | 13.0 tok/s | 24.6 | 35.0 | 2.69x |
| prompt=512, output=256 | 13.7 tok/s | 26.9 | 34.1 | 2.49x |
| prompt=512, output=2048 | 13.9 tok/s | 27.5 | **40.2** | **2.89x** |
| prompt=2048, output=64 | 11.1 tok/s | 19.5 | 22.6 | 2.04x |
| prompt=2048, output=256 | 13.2 tok/s | 25.0 | 31.5 | 2.39x |
| prompt=2048, output=2048 | 13.8 tok/s | 27.3 | 35.9 | 2.60x |

![Super-120B-NVFP4 merged-FT inference scaling with concurrency](../plots/07_super_inference_concurrency.png)

Mean aggregate rises 13.1 (conc=1) → 25.1 (conc=2) → 33.2 (conc=3). Full per-trial JSONLs:
[../results/super_inference_concurrency_sweep/](../results/super_inference_concurrency_sweep/).

---

## Dequant kernel throughput

NVFP4 weight dequant runs as a fused Triton kernel (`nvfp4_lora/triton_dequant.py`, v1.2+): full
unpack + E2M1 LUT + group scale + per-tensor scale + bf16 store in one dispatch. Per dequant call,
measured on GB10 (PyTorch 2.11.0+cu130, Triton 3.6.0):

| shape | PyTorch | Triton | speedup |
|---|---:|---:|---:|
| 4096 x 4096 bf16 | 6.02 ms | 0.28 ms | 21.6x |
| 4096 x 2048 bf16 | 3.04 ms | 0.13 ms | 23.5x |
| 4096 x 1024 bf16 | 1.44 ms | 0.06 ms | 24.8x |
| 6144 x 256 bf16  | 0.47 ms | 0.03 ms | 14.5x |

End-to-end LoRA training step time on a 119B-class NVFP4 MoE (128 routed experts/layer) drops from
**984 s to 92 s (10.7x)** at bsz=1, grad_accum=8, seq_len=2048 with gradient checkpointing on. Parity
against the PyTorch path is bit-identical (`max_abs_diff = 0.0` across all tested shapes and both
modelopt and compressed-tensors formats). See [../smoke_tests/triton_dequant_parity.py](../smoke_tests/triton_dequant_parity.py).
