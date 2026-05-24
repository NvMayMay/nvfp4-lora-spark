# Reproducing nvfp4-lora-spark results

Exact stack used to produce the headline numbers in the README.

## Path conventions

Training and diagnostic scripts use placeholder paths of the form
`/path/to/Models/...`, `/path/to/adapters/...`, `/path/to/datasets/...`.
Edit the constants at the top of each `train/*.py` script to point at
your local layout. The serve scripts under `serve/` accept an env-var
override (`MODEL_DIR=...`), so they do not need editing as long as you
export the variable.

## Hardware

- **System**: NVIDIA DGX Spark
- **GPU**: NVIDIA GB10 (Blackwell consumer, sm_121, 128 GB unified LPDDR5x)
- **Compute capability**: 12.1
- **CUDA driver/runtime**: CUDA 13.0
- **OS**: Linux 6.17 aarch64 (Ubuntu kernel)

## Software stack (versions verified 2026-05-24)

| Component | Version |
|-----------|---------|
| Python | 3.12.3 |
| PyTorch | 2.11.0+cu130 |
| vLLM | 0.21.0 |
| transformers | 5.8.1 |
| peft | 0.19.1 |
| safetensors | 0.7.0 |
| nvidia-modelopt | 0.44.0 |
| flashinfer-python | 0.6.8.post1 |
| accelerate | 1.13.0 |
| huggingface-hub | 1.14.0 |
| causal-conv1d | 1.6.2.post1 (built from source, see below) |

## Build `causal-conv1d` from source (required for training)

The Mamba2 fast path needs `causal-conv1d` built against your CUDA
toolchain. Without it, training falls back to a Python scan that is
infeasible at any useful sequence length.

```bash
MAX_JOBS=1 pip install --no-build-isolation causal-conv1d==1.6.2.post1
```

`MAX_JOBS=1` is mandatory on Spark to prevent nvcc from being OOM-killed
during parallel compilation on the 128 GB unified pool.

## Model artifacts

| Artifact | Source | Hash |
|----------|--------|------|
| Base: `Nemotron-3-Super-120B-A12B-NVFP4` | [HuggingFace](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4) | (HF revision pinned in scripts) |
| Trained Super-FT adapter (example) | Bundled at `adapters/super_ich_v1_0/` | `sha256: 3d2ada8a624c797f764268d5da4dfb9621fc681e863c475c97ca4856112418b3` |
| Merged Super-FT NVFP4 (example) | Produced by `scripts/merge_lora_into_nvfp4.py` | (hash recorded in `merge_manifest.json` after merge) |

The training data for the Super-FT example adapter is private clinical/
regulatory text. To reproduce a similar FT, train a LoRA against the base
on your own domain corpus following the recipe in `train/train_super_nvfp4.py`.

## Reproducing the headline numbers

### Train

```bash
hf download nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4 \
    --local-dir models/Nemotron-3-Super-120B-A12B-NVFP4
python train/train_super_nvfp4.py    # edit paths at top of file
```

Measured wall: 40.7 h on a single Spark, 1 epoch over 1081 chat-format
examples, max_len=1536, batch=1 with grad_accum=4 (effective batch 4),
AdamW lr=1e-4, gradient checkpointing on. Final train loss 0.81.

### Merge LoRA into NVFP4 base

```bash
python scripts/merge_lora_into_nvfp4.py \
    --base-model-dir models/Nemotron-3-Super-120B-A12B-NVFP4 \
    --lora-adapter-dir adapters/super_ich_v1_0 \
    --output-dir models/Nemotron-3-Super-120B-A12B-NVFP4-ich-v1.0
```

Measured wall: ~25 min on Spark for all 17 shards (~90 s/shard).
Writes per-shard manifest (with source/output sha256 + per-tensor stats)
and `merge_stats.jsonl` for downstream validation.

### Validate the merge

```bash
python scripts/validate_merge.py \
    --base-model-dir models/Nemotron-3-Super-120B-A12B-NVFP4 \
    --merged-model-dir models/Nemotron-3-Super-120B-A12B-NVFP4-ich-v1.0
```

Reports: tokenizer/config integrity (must be byte-identical to base),
coverage (merged tensor count vs adapter target count), per-tensor
delta-to-quant-step audit, merge cosine similarity, no-op fraction.

### Serve Super base (no FT) via CUTLASS

```bash
./serve/run_super_base_inference_cutlass.sh
```

Measured throughput: ~12-14 tok/s, flat across prompt lengths
12-456 tokens, output lengths 32-256 tokens. See
`serve/diagnostics/bench_cutlass_eager_super_base_*.jsonl`.

### Serve Super-FT (merged) via CUTLASS

```bash
MODEL_DIR=models/Nemotron-3-Super-120B-A12B-NVFP4-ich-v1.0 \
    ./serve/run_super_ft_merged.sh
```

Same throughput as base CUTLASS (~12-14 tok/s). The FT behavior is
baked into the served weights.

### Distinguishing test (FT vs base)

```bash
# With base server running on port 8000:
python scripts/distinguish_ft.py collect \
    --url http://localhost:8000 \
    --model nemotron-3-super-a12b-nvfp4 \
    --output-jsonl /tmp/base_outputs.jsonl

# Kill base server, start FT server on port 8000:
python scripts/distinguish_ft.py collect \
    --url http://localhost:8000 \
    --model nemotron-3-super-a12b-nvfp4+ich_v1_0 \
    --output-jsonl /tmp/ft_outputs.jsonl

# Compare:
python scripts/distinguish_ft.py compare /tmp/base_outputs.jsonl /tmp/ft_outputs.jsonl
```

Visually inspect the differing prompts to confirm FT signal is present.

## Licensing and redistribution

This repository is Apache 2.0 (see [LICENSE](LICENSE)). The Nemotron-3
base models are under the [NVIDIA Nemotron Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-nemotron-open-model-license/),
which is more restrictive than standard OSS licenses.

**What we publish**:
- The training pipeline, scripts, LoRA adapter, merge script, and serve
  recipes are all Apache 2.0 (our own code/data).
- The example LoRA adapter at `adapters/super_ich_v1_0/` is Apache 2.0
  (our trained weights, derived from the base).

**What we do NOT publish**:
- The merged Super-FT NVFP4 checkpoint produced by `merge_lora_into_nvfp4.py`.
  Merged weights are a derivative work of the NVIDIA base model and fall
  under the NVIDIA Nemotron Open Model License's redistribution terms.
  To get the merged checkpoint, download the base from HuggingFace and
  apply our merge script locally.

**For commercial use**: read the NVIDIA Nemotron Open Model License
carefully; it has carve-outs that may or may not apply to your use case.

## Known divergences

Bit-for-bit reproducibility is NOT guaranteed across:

- Different versions of any package above.
- Different CUDA driver / hardware revisions.
- Different filesystem layouts (the safetensors mmap interacts with kernel
  page cache; CUDA memory shows different `cuda_free` after each load).

For reasonable numerical reproducibility (matching tok/s and FT behavior
within ~5%), the stack table above should be sufficient on any GB10
DGX Spark.
