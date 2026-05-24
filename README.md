# nvfp4-lora-spark

NVIDIA ships Nemotron-3-Super so it fits on a 128 GB GB10 box by quantizing to NVFP4. This repo solves the missing training bridge (LoRA fine-tuning directly on the NVFP4 base with on-the-fly dequant inside autograd) AND ships practical, validated serving recipes for both Nano and Super on a single DGX Spark.

| Model | Use case | Serving recipe | Throughput |
|---|---|---|---|
| Nemotron-3-Nano-30B-A3B-NVFP4 | base + dynamic LoRA via `--lora-modules` | `serve/serve_nemotron_nvfp4.sh` (marlin) | (measured separately) |
| Nemotron-3-Super-120B-A12B-NVFP4 | base inference | `serve/run_super_base_inference_cutlass.sh` (CUTLASS native FP4) | **~12-14 tok/s** |
| Nemotron-3-Super-120B-A12B-NVFP4 + LoRA | FT serving via merge-then-serve | `scripts/merge_lora_into_nvfp4.py` + `serve/run_super_ft_merged.sh` | **~12-14 tok/s** |
| Nemotron-3-Super-120B-A12B-NVFP4 + dynamic LoRA | future work | (Phase 2 upstream contribution; see [docs/PHASE2.md](docs/PHASE2.md)) | n/a today |

The serving complication for Super-FT (no dynamic LoRA via vLLM today, so we ship a merge step + a serve recipe for the merged checkpoint) is itself a documented contribution. See [serve/README.md](serve/README.md) for why this split exists, [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) for the failure-signature playbook, and [REPRODUCE.md](REPRODUCE.md) for the exact stack.

For single-box LoRA fine-tuning of Nemotron-3-Super on a 128 GB GB10 system, the released NVFP4 checkpoint is the only practical base format we found. Training a bf16 base in the cloud and shipping the adapter back works in principle but suffers because the base shifts under quantization. Full cloud training defeats the point of owning local hardware. This repo removes that constraint.

![Loss curves](plots/01_loss_curves.png)

## What works today

| Run | Base dtype | LoRA targets | Wall (h) | Peak GPU (GB) | Step time (s) | Final train loss |
|---|---|---|---:|---:|---:|---:|
| Super-120B NVFP4 | NVFP4 | up_proj, down_proj (r=8) | 40.7 | 92 | 135.6 | 0.81 |
| Nano-30B NVFP4 | NVFP4 | up_proj, down_proj (r=8) | 13.1 | 22 | 43.8 | 1.00 |
| Nano-30B BF16 (baseline) | BF16 | up_proj, down_proj (r=8) | 1.22 | 67 | 4.2 | 0.98 |

All runs: 1 epoch over 1081 chat-format examples, max_len=1536, batch_size=1 with grad_accum=4 (effective batch 4), AdamW lr=1e-4, gradient checkpointing on. Single GB10 system (NVIDIA DGX Spark, 128 GB unified memory).

The BF16 baseline row is a quantization ablation, not a published configuration: we trained an identical LoRA against the bf16 Nano base to measure how much quality the NVFP4 dequant path costs. On this 1081-example ablation, NVFP4 and BF16 LoRA reached nearly identical training loss (1.00 vs 0.98) with similar adapter norms. We did not observe a training-loss penalty. This is not a general quality guarantee; broader evals are future work.

What NVFP4 *does* cost is wall time: ~11x slower per step than BF16 LoRA on the same hardware, because the current `NVFP4LoRALinear` round-trips through HBM rather than streaming dequanted weights through tensor-core registers. See [docs/PERFORMANCE_ROADMAP.md](docs/PERFORMANCE_ROADMAP.md) for the five routes to close this gap.

## Why this exists (more detail)

The serving story for NVFP4 on GB10 is solved by vLLM marlin. The training story is not. NVFP4 weights are packed E2M1 nibbles in uint8 with separate `fp8_e4m3fn` group scales and an `fp32` per-tensor scale, as stored in these Nemotron-3 NVFP4 checkpoints. Nemotron-3-Super further mixes NVFP4 routed experts with FP8 per-tensor Mamba2 and shared-expert MLPs, alongside bf16 norms and embeddings. None of the standard adapter libraries know what to do with that layout.

Owners of NVFP4 bases on GB10 boxes have three workarounds today: train on a bf16 base in the cloud (adapter transfers imperfectly because the base shifts under quantization), rent a datacenter GPU for the whole pipeline (defeats the point of local hardware), or skip fine-tuning. This repo collapses those into "train on the NVFP4 base you already have, serve with the NVFP4 base you already have."

Concrete gaps this fills:

- No published recipe for LoRA on NVFP4-quantized bases, anywhere we could find.
- No published loader for Nemotron-3's mixed NVFP4 + FP8 + bf16 + Mamba2 weight layout, with the MTP-layer caveat (Multi-Token Prediction layers exist for vLLM speculative decoding; loading them into a training graph breaks things).
- No documented set of GB10-specific footguns: the `causal-conv1d` build dance for the Mamba2 fast path (without which long-sequence training is infeasible), the `MAX_JOBS=1` cap that prevents FlashInfer JIT from OOM-ing the 128 GB unified pool, the `--moe-backend marlin` requirement that silently degrades inference if you forget it, the FP8 shared-expert layers that silently lose their adapter unless the loader handles them.

The use cases are not hypothetical. On-prem and air-gapped deployments where data never leaves the box. Iteration speed: ~13 hours for a Nano-30B FT run, ~40 hours for Super-120B, all on local hardware you already own. Zero cloud spend for the customization cycle of a model you can already serve locally.

### Why NVFP4, not plain FP4

NVFP4 is not just "FP4". A plain E2M1 element (one sign + two exponent + one mantissa bit) represents values only up to magnitude 6 before scaling, which is nowhere near enough to cover transformer weight distributions. A single outlier in any tensor saturates the entire 4-bit space and the rest of the block degenerates. NVFP4 wraps E2M1 in a two-level scaling scheme that solves this:

- Each block of 16 weights gets its own `fp8_e4m3fn` scale (1D block-scaling, as stored in these Nemotron-3 NVFP4 checkpoints), so local variance does not saturate the 4-bit range.
- One `fp32` per-tensor scale absorbs the overall weight magnitude.

That block-scaled structure is the recent quantization advance that makes 4-bit weights practical in production. NVFP4 uses real `fp8` block scales, which give finer outlier handling than MXFP4's `ue8m0` power-of-two scales (the open OCP alternative). NVIDIA targets NVFP4 on Blackwell with native FP4 GEMM kernels; on our GB10/vLLM stack we use marlin (weight-only) because native FP4 MoE kernels for sm_121 are not what vLLM ships today.

Accuracy claims for NVFP4 against the bf16 reference are documented on every NVIDIA model card. For example, [nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4) and [nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4) each ship with an accuracy chart vs the corresponding BF16 release ([Nano BF16](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16), [Super BF16](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16)); the typical claim is sub-1% delta on standard benchmarks, with FP8 sitting in between. The NVFP4 weights themselves are produced by [NVIDIA Model Optimizer](https://github.com/NVIDIA/Model-Optimizer) (formerly TensorRT Model Optimizer), which is the canonical reference for the quantization recipe. NVIDIA's Transformer Engine also supports NVFP4 training recipes; this repo solves a different problem (attaching LoRA to already-quantized Nemotron NVFP4 checkpoint tensors without first loading a BF16 base).

The BF16 baseline row above is the local ablation: same data, same hyperparams, same code path otherwise.

## Target hardware

- **GPU**: NVIDIA GB10 (Blackwell consumer, sm_121).
- **Memory**: 128 GB unified LPDDR5x (training uses ~92 GB peak for Super-120B, ~22 GB for Nano-30B at batch=1, max_len=1536).
- **CUDA**: 13.0 (required for sm_121 support).
- **Verified on**: NVIDIA DGX Spark. Should work on any GB10 SKU with the same 128 GB unified memory budget (Asus, HP, and other OEM GB10 systems ship the same internal config).
- **Not tested**: Hopper, Ada, or datacenter Blackwell. The marlin weight-only kernel would technically run there, but if your GPU has native FP4 GEMM you would not want to use marlin over it.

On our GB10/vLLM stack, marlin is the reliable path for serving these NVFP4 MoE weights; we did not validate native FP4 MoE kernels on sm_121. That is a current-stack reality, not a claim about silicon support.

## Architecture (what is custom vs stock)

- **Custom**: `nvfp4_lora.linear.NVFP4LoRALinear` (holds the original NVFP4 tensors frozen, dequants on-the-fly to bf16 inside a custom autograd Function so the dequant does not blow memory, trains a low-rank LoRA delta in parallel) and `nvfp4_lora.loader.load_nemotron_with_nvfp4_lora` (handles the mixed-precision reality: NVFP4 routed experts, FP8 per-tensor Mamba2 + shared experts, bf16 norms and embeddings, MTP layers that vLLM uses but training should skip).
- **Stock**: the saved adapter is plain PEFT format (`base_model.model.<name>.lora_{A,B}.weight`); vLLM serves it via `--enable-lora --lora-modules`; the rest of the training loop is standard PyTorch + `transformers` + `AutoTokenizer`.

The base NVFP4 weights are frozen throughout. Gradients flow only into the LoRA `A` and `B` parameters. There is no NVFP4 weight update.

## Correctness and reproducibility

Three smoke tests under [`smoke_tests/`](smoke_tests/) check the math without GPU heavy lifting:

```bash
# 1. Dequant round-trip vs reference (NVFP4 uint8 + fp8 scales + fp32 scale -> bf16)
python smoke_tests/dequant_correctness.py

# 2. NVFP4LoRALinear forward parity:
#    output of NVFP4LoRALinear == bf16-dequantized weight @ x + (alpha/r) * B @ A @ x
python smoke_tests/linear_smoke.py

# 3. Loader: replace NVFP4 modules in a small Nemotron-3 slice, verify
#    grads flow only into LoRA params and not into the frozen base
python smoke_tests/loader_smoke.py
```

Reference environment (the exact stack the headline numbers were measured on):

- Hardware: NVIDIA DGX Spark (GB10, sm_121, 128 GB unified LPDDR5x)
- OS: Linux 6.17 aarch64
- CUDA: 13.0
- PyTorch: 2.12.x
- transformers: 5.8.1 (Nemotron-3 needs `trust_remote_code=True`)
- peft: 0.19.1 (only used by the BF16 baseline)
- safetensors: 0.5+
- huggingface-hub: 0.28+
- causal-conv1d: 1.6.2.post1, built `--no-build-isolation` against your CUDA (see Quick Start)
- vLLM: 0.21.0+ with marlin backend (separate venv from training is recommended)

For full dependency rationale see [docs/LESSONS.md](docs/LESSONS.md).

## Performance plots

![Throughput and memory](plots/04_throughput_and_memory.png)

Training loss curves for all three runs overlaid:

![Loss curves](plots/01_loss_curves.png)

Quantization ablation (Nano-NVFP4 vs Nano-BF16, identical hyperparams):

![Quant ablation training](plots/02_quant_ablation_training.png)

## Quick start

### 1. Environment

```bash
git clone https://github.com/<user>/nvfp4-lora-spark
cd nvfp4-lora-spark
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Then build `causal-conv1d` against your CUDA toolchain (~25 min on aarch64+CUDA13, faster on x86):

```bash
MAX_JOBS=1 pip install --no-build-isolation causal-conv1d==1.6.2.post1
```

Without `causal-conv1d`, Mamba2 falls back to a naive Python scan and training is effectively impossible at any useful sequence length. See [docs/LESSONS.md](docs/LESSONS.md) for why.

### 2. Get a Nemotron-3 NVFP4 base

```bash
hf download nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4 \
    --local-dir models/Nemotron-3-Nano-30B-A3B-NVFP4
```

Super-120B is the same pattern with a larger download (~70 GB on disk).

### 3. Prepare a chat-format dataset

One JSON line per example, with a `messages` field in OpenAI chat format:

```json
{"messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

The training loop applies the model's chat template, so make sure the template renders cleanly on a sample of your data before kicking off a full run. Nemotron-3 has an `enable_thinking` parameter that affects template output; the training scripts handle this.

### 4. Train

```bash
python train/train_nano_nvfp4.py
```

The current scripts hardcode paths to model, data, and output adapter directories at the top of each file. Edit those four constants for your setup. (Argparse-based version is a TODO; see [Limitations](#limitations).)

Measured wall on a single GB10 box (DGX Spark), max_len=1536, 1 epoch over 1081 examples:

| script | wall | step time |
|---|---|---|
| `train_nano_nvfp4.py` | 13.1 h | 43.8 s |
| `train_super_nvfp4.py` | 40.7 h | 135.6 s |
| `train_nano_bf16.py` (baseline) | 1.22 h | 4.2 s |

The BF16 baseline is included as a quantization ablation only; production users on GB10 should use the NVFP4 path (and accept the wall-time cost) because the bf16 Super base does not fit on this hardware. For Nano specifically, where the bf16 base does fit, BF16 LoRA is ~11x faster per step at similar adapter quality on our 1081-example test.

### 5. Serve

The serving story splits by model size on GB10. See [serve/README.md](serve/README.md) for the full picture.

**Nano-30B-NVFP4** (base or with LoRA): standard vLLM marlin recipe works.

```bash
./serve/serve_nemotron_nvfp4.sh nano adapters/your_adapter your_adapter_tag
```

Required env (so vLLM uses the marlin weight-only path for NVFP4 MoE weights on sm_121):

```
VLLM_NVFP4_GEMM_BACKEND=marlin
--moe-backend marlin
MAX_JOBS=1                   # caps FlashInfer JIT, otherwise OOMs the 128 GB unified pool
--dtype bfloat16             # activations
```

The script sets all four. Exposes both the base model (`nemotron-3-nano-nvfp4`) and the FT model (`nemotron-3-nano-nvfp4+your_adapter_tag`) over the OpenAI-compatible endpoint at `http://localhost:8000/v1/completions`.

**Super-120B-NVFP4** base inference: marlin does NOT fit on Spark (per-expert weight repack transient exceeds the 130 GB physical ceiling). Use the **VLLM_CUTLASS** native FP4 backend instead:

```bash
./serve/run_super_base_inference_cutlass.sh
```

Measured **~12-14 tok/s** on Spark (see [serve/diagnostics/bench_cutlass_eager_super_base_*.jsonl](serve/diagnostics/)). Functional and reasonably fast for the publication's inference sweep. An EMULATION fallback at [serve/run_super_base_inference.sh](serve/run_super_base_inference.sh) exists (~0.7 tok/s, 18× slower) in case CUTLASS breaks in a future vLLM release.

**Super-120B-FT (with LoRA)**: blocked in vLLM 0.21. The fast CUTLASS kernel doesn't have a LoRA path (`CutlassExpertsFp4.supports_lora() = False`); the EMULATION kernel that does claim LoRA support hits a Triton bug (`illegal memory access` in `_fused_moe_lora_expand` during warmup); Marlin OOMs. Workarounds (see [serve/README.md](serve/README.md) for details): (1) merge LoRA into NVFP4 base + requantize via NVIDIA Model Optimizer, then serve via CUTLASS at 12-14 tok/s with FT baked in; (2) custom FastAPI server using training-side `NVFP4LoRALinear`; (3) wait for upstream fixes.

## Reproducing the plots in this README

```bash
python plots/extract_train_metrics.py        # parses train logs -> train_metrics.json
python plots/make_plots.py all                # renders 7 candidate plots
```

Plots auto-skip runs that haven't started and stub-out eval-side plots until you drop in an `eval_results.json`. The expected eval schema is documented at the bottom of `plots/make_plots.py`. Derived metrics (pad-tokens/sec for training, per-user-tps and prefill-tps for inference) are computed at plot-render time from the JSON; see the docstring of `plots/make_plots.py` for formulas.

## Preliminary evals

This repo does not ship an evaluation harness. The training scripts produce a standard PEFT-format adapter; bring your own benchmark. The eval-side plots in `plots/make_plots.py` (`eval_headline_accuracy`, `eval_ft_lift`, `eval_quant_tax`) are stubs that render once you drop an `eval_results.json` in place with the documented schema.

The author has run smoke evals against a flawed domain dataset (clinical/regulatory text from an in-house corpus with known issues). Those numbers are deliberately not surfaced in this README and should be treated as not-yet-meaningful. Once cleaner eval data is in place, the eval plots will be filled in via the same plotting code.

## Not supported yet

- Multi-GPU / tensor parallelism (GB10 ships single-GPU; not a near-term need).
- Attention LoRA on Nemotron-3 (the Mamba2-attention blocks in Nemotron-3 are not the standard transformer-attention layout most LoRA tooling targets; we have not validated `q_proj`/`k_proj`/`v_proj`/`o_proj` LoRA on this architecture).
- LoRA on non-MLP FP4 modules (see Limitations on FP8 demotion).
- Other NVFP4 model families (the loader is Nemotron-3 specific).
- Native FP4 training kernels (current path goes through bf16 dequant; see [docs/PERFORMANCE_ROADMAP.md](docs/PERFORMANCE_ROADMAP.md) for the planned routes to native FP4).
- A bundled eval harness.

## Dependencies

Full inventory with rationale in [docs/LESSONS.md](docs/LESSONS.md). Quick version:

- PyTorch 2.12+ with CUDA 13.0 (GB10 requires CUDA 13)
- transformers 5.8+ with `trust_remote_code=True` (Nemotron-3 ships its own modeling code)
- peft 0.19+ (only used by `train_nano_bf16.py`; the NVFP4 path is custom)
- safetensors, accelerate, huggingface-hub
- causal-conv1d 1.6.2+ built against your CUDA (see above)
- vLLM 0.21.0+ with marlin backend, for the serve venv (not strictly required for training)

`vllm` and the training stack don't have to share a venv. The reference setup runs them separately: `qwen-peft` for training, `qwen-serve` for vLLM. Mixing in one venv works but is harder to keep clean across upgrades.

## What's in the repo

```
nvfp4_lora/                  # the library
  linear.py                  # NVFP4LoRALinear with on-the-fly dequant
  loader.py                  # mixed NVFP4 + FP8 + Mamba loader for Nemotron-3
  dequant.py                 # NVFP4 -> bf16 dequant kernel
train/                       # production training scripts
  train_super_nvfp4.py       # Super-120B-A12B NVFP4
  train_nano_nvfp4.py        # Nano-30B-A3B NVFP4
  train_nano_bf16.py         # Nano-30B-A3B BF16 (quant ablation baseline)
serve/
  serve_nemotron_nvfp4.sh    # one-line vLLM launcher, optional LoRA attach
plots/
  extract_train_metrics.py   # logs -> structured JSON
  make_plots.py              # 7 candidate plots, all or by name
smoke_tests/                 # library correctness tests, no GPU heavy lifting
  dequant_correctness.py
  linear_smoke.py
  loader_smoke.py
docs/
  LESSONS.md                 # debug history and dependency rationale
  PERFORMANCE_ROADMAP.md     # five routes to close the NVFP4-vs-bf16 throughput gap
```

## Limitations

- **GB10 only (sm_121)**: on Hopper, Ada, or datacenter Blackwell with native FP4 compute, the marlin weight-only path here is not optimal. The training code should still work, but you would not want to serve with `--moe-backend marlin` if you can use the native FP4 kernel instead.
- **Single GPU**: tensor parallelism is not tested. GB10 systems ship with one GPU, so this was not a near-term need.
- **Nemotron-3 specific**: the loader hardcodes Nemotron-3 weight-naming conventions (the `backbone.` vs `model.` prefix divergence between Nano and Super, the FP8 demotion for Mamba and shared experts, MTP layer skipping). Porting to other NVFP4 model families means updating the loader.
- **LoRA targets up_proj and down_proj only**: targeting attention or gate_proj would need verification that those layers are NVFP4 in your base model. In Super-120B, shared expert MLPs and Mamba projections are FP8, so the loader silently demotes any LoRA target on those modules to frozen (with a count printed at load time).
- **Hardcoded paths in train scripts**: model, data, and output paths sit at the top of each `train/*.py`. A small argparse layer is on the TODO list.
- **No eval harness shipped**: this repo trains and serves; bring your own benchmark. The expected JSON schema for the `plots/make_plots.py` eval plots is documented inline.
- **NVFP4 path is ~11x slower than bf16 LoRA per step** at batch=1 (measured on Nano-30B at max_len=1536). This is the cost of the current on-the-fly dequant implementation. NVFP4 is still the only viable path for Super-120B on GB10 (bf16 base doesn't fit), but for smaller models that fit at bf16 the trade is unfavorable. See [docs/PERFORMANCE_ROADMAP.md](docs/PERFORMANCE_ROADMAP.md) for the five routes to close this gap, ordered by effort-to-payoff.
- **Super-120B serving via vLLM has multiple gotchas on Spark.** Base inference works at ~12-14 tok/s via the VLLM_CUTLASS backend with `--enforce-eager --gpu-memory-utilization 0.70 --max-model-len 2048 --max-num-seqs 1`. Marlin does NOT fit (per-expert weight repack transient exceeds physical memory). The native-FP4 FLASHINFER_TRTLLM and FLASHINFER_CUTEDSL backends reject sm_121 (kernel binaries not compiled). EMULATION fits but runs at ~0.7 tok/s. **For LoRA serving**, VLLM_CUTLASS / FLASHINFER_CUTLASS kernels report no LoRA support; EMULATION + LoRA hits an upstream Triton kernel bug (illegal memory access in `fused_moe_lora_expand`); Marlin OOMs. Workaround: merge the LoRA into the base, requantize via NVIDIA Model Optimizer, then serve via CUTLASS. See [serve/README.md](serve/README.md) and [serve/diagnostics/README.md](serve/diagnostics/README.md) for the full investigation and reproducer artifacts.

## License

Apache 2.0. See [LICENSE](LICENSE).

## Citation

If you build on this, a link back is appreciated.

```bibtex
@software{nvfp4_lora_spark_2026,
  title  = {nvfp4-lora-spark: LoRA fine-tuning Nemotron-3 NVFP4 MoE on a single DGX Spark},
  year   = {2026},
  url    = {https://github.com/<user>/nvfp4-lora-spark}
}
```
