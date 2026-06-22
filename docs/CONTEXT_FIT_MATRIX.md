# Context-fit matrix (unified trainer)

Cross-model record of which **training sequence lengths** fit on the GB10 DGX Spark
under the unified trainer (`scripts/train_nvfp4_lora.py`), and under which
**conditions** (FLCE, optimizer, LoRA config). Companion to the Super-120B-specific
deep dive in [LONG_CONTEXT_EXPERIMENTS.md](LONG_CONTEXT_EXPERIMENTS.md) (legacy
`train/train_super_nvfp4.py` path).

**Machine:** NVIDIA GB10, unified memory. `torch.cuda.mem_get_info()` reports
**130.66 GB total** (the rounded `free -g` "121" is conservative). Treat ~125 GB as
the practical ceiling; leave margin for allocator fragmentation.

**How peaks are measured:** `--dry-run` preflight — loads exactly like a real run
(load + LoRA + gradient checkpointing + optimizer), runs one synthetic
forward+backward at `(batch, max_length)`, logs `cuda_max_allocated_gb`, exits.
Reproduce any row with the same flags + `--dry-run`.

**Standing conditions (apply to all rows unless noted):** gradient checkpointing ON
(`use_reentrant=False`), SDPA attention, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`,
batch 1, grad-accum 8, AdamW on LoRA params only. The Qwen serve must be stopped
first (it holds ~112 GB UMA).

## Key lever: `--fused-linear-ce` (FLCE)

Binds Liger `liger_kernel.transformers.model.glm4.lce_forward` onto the causal-LM
head: computes cross-entropy in sequence chunks, never materializing the full
`(seq x vocab)` logits tensor or its fp32 upcast. **Numerically exact** (same loss
and gradients; dry-run loss matched the non-FLCE path to 3e-4). Savings grow with
seq x vocab — biggest at long context / large vocab. Bind ONLY `lce_forward`, never
`apply_liger_kernel_to_glm4` (it rewrites MoE MLPs to dense SwiGLU and corrupts the
NVFP4 experts).

## Validated fit table

| Model (NVFP4) | model_type | seq | FLCE | LoRA | Peak `cuda_max_allocated` | Headroom (of 130.7) | Status | Date |
|---|---|---:|:---:|---|---:|---:|---|---|
| GLM-4.5-Air-106B-A12B | glm4_moe | 4096 | off | r32/a64 q,k,v,o | 72.77 GB | 58 GB | PASS | 2026-06-21 |
| GLM-4.5-Air-106B-A12B | glm4_moe | 8192 | off | r32/a64 q,k,v,o | 83.08 GB | 48 GB | PASS | 2026-06-21 |
| GLM-4.5-Air-106B-A12B | glm4_moe | 8192 | **on** | r32/a64 q,k,v,o | **74.15 GB** | 56 GB | PASS | 2026-06-21 |
| GLM-4.5-Air-106B-A12B | glm4_moe | 16384 | **on** | r32/a64 q,k,v,o | 85.23 GB | 45 GB | PASS | 2026-06-21 |

Notes:
- GLM-4.5-Air vocab = 151552; FLCE saves ~8.9 GB at 8192 (83.08 -> 74.15). 4096->8192
  costs only +10.3 GB. Everything up to 16384 fits comfortably; 8192 fits even without FLCE.
- Super-120B (nemotron_h) long-context results live in LONG_CONTEXT_EXPERIMENTS.md
  (legacy script, `--loss-mode chunked_frozen_ce`, synthetic-length rows): full-rank
  certified ceiling 16384; frontier crash/fit to ~32768 with reduced rank.

## Data-driven context choice

Picking max-length above what the data needs just burns compute. Measure the real
token-length distribution with the trainer's own `ChatJsonlDataset` at a large cap.

| Dataset | tok p50 | p99 | max | fully fit <=4096 | <=8192 | <=16384 |
|---|---:|---:|---:|---:|---:|---:|
| ICH_v4_1_ICH_FT_smoke train (766) | 4406 | 6748 | 8485 | 214 (28%) | 764 (100%) | 766 (100%) |
| ICH_v4_1_ICH_FT_smoke val (137) | 4524 | 5730 | 5904 | 31 (23%) | 137 (100%) | 137 (100%) |

=> For ICH v4_1, **8192 is the sweet spot**: captures 99.7% of examples fully, vs only
28% at 4096. 16384 adds ~nothing for ~2x the per-step compute.

## Training run log

Actual training runs (not dry-run probes), newest first.

| Run | Model | seq | FLCE | LoRA | Epochs | Status | Notes |
|---|---|---:|:---:|---|---:|---|---|
| glm45air_lora_ich_v4_1_8k_r32a64 | GLM-4.5-Air | 8192 | on | r32/a64 | 2 | COMPLETE (2026-06-22 03:57) | Full data (766/137 all usable). 191 updates in 15.8 h (~270 s/update; only ~8% over 4096's 249 s — dequant-bound, not token-bound). Best val **0.8433** (step 191); curve 0.9029(50) -> 0.8579(100) -> 0.8460(150) -> 0.8433(191), decelerating but improving to the end. Live peak ~74 GB matched dry-run. NVFP4_EVAL_CACHE_GB=0. |
| glm45air_lora_ich_v4_1_smoke_r32a64 | GLM-4.5-Air | 4096 | off | r32/a64 | 2 | CANCELLED @ step 113 | Pivoted to 8192 for full data coverage. Best val 0.8876 @ step 100. Artifacts deleted. |
| (smoke validation) | GLM-4.5-Air | 4096 | off | r16/a32 | 3 steps | PASS | First glm4_moe bring-up; finite loss, no NaN/OOM. Caught fused-3D meta bug. Deleted. |

Append a row per real run: context length + conditions + outcome.
