# Serving Nemotron-3 NVFP4 models on DGX Spark

vLLM serving for these NVFP4 models on Spark (GB10, sm_121, 128 GB UMA)
splits cleanly by model size. This directory is the result of a multi-day
investigation; see [DECISION_LOG D016-D020](../docs/) and
[diagnostics/README.md](diagnostics/README.md) for the long story.

## Quick summary

| Model | Base inference via vLLM | Throughput | LoRA serving via vLLM |
|-------|-------------------------|------------|------------------------|
| **Nano-30B-NVFP4** | ✅ marlin backend works | (measured separately) | ✅ standard `--enable-lora --lora-modules` |
| **Super-120B-NVFP4** | ✅ **VLLM_CUTLASS** (recommended) | **~12-14 tok/s** | ❌ kernel does not support LoRA |
| **Super-120B-NVFP4** | ✅ EMULATION (fallback) | ~0.7 tok/s | ❌ Triton kernel illegal-memory-access bug |

### Why the split

- **Nano (30 GB on disk)** fits comfortably in Spark's 128 GB unified memory
  via the marlin weight-only path. Standard recipe in
  [`serve_nemotron_nvfp4.sh`](serve_nemotron_nvfp4.sh).

- **Super (75 GB on disk)** does NOT fit via marlin: the per-expert weight
  repack inside `prepare_nvfp4_moe_layer_for_marlin` has a transient memory
  peak that exceeds Spark's 130 GB physical ceiling.

- **Super base inference**: use **VLLM_CUTLASS** (recipe in
  [`run_super_base_inference_cutlass.sh`](run_super_base_inference_cutlass.sh)).
  Measured ~12-14 tok/s on Spark. `CutlassExpertsFp4._supports_current_device()`
  accepts `is_device_capability_family(120)` which sm_121 satisfies, and the
  kernel is compiled in vLLM 0.21. `--enforce-eager` is required (CUDA graph
  capture phase consumes ~3 GiB extra; safety floor trips one capture before
  end). An EMULATION fallback at [`run_super_base_inference.sh`](run_super_base_inference.sh)
  exists for reference (~0.7 tok/s, 18× slower).

  Among the 6 NVFP4 MoE backends in vLLM 0.21, only MARLIN (OOMs Spark),
  EMULATION (slow), VLLM_CUTLASS (fast, no LoRA), and FLASHINFER_CUTLASS
  (also no LoRA) accept sm_121 at the oracle level. FLASHINFER_TRTLLM and
  FLASHINFER_CUTEDSL reject with *"kernel does not support current device cuda."*

- **Super + LoRA serving via vLLM**: BLOCKED in vLLM 0.21 on sm_121.
  - VLLM_CUTLASS oracle cleanly rejects: `kernel does not support LoRA`.
    `CutlassExpertsFp4.supports_lora() = False`.
  - EMULATION + LoRA loads weights but crashes at warmup with
    `RuntimeError: Triton Error [CUDA]: an illegal memory access was encountered`
    in `vllm/lora/ops/triton_ops/fused_moe_lora_op.py:_fused_moe_lora_expand`.
  - Marlin path doesn't fit.
  - Workarounds (in order of recommendation):
    1. **Merge LoRA into NVFP4 base + requantize** via NVIDIA Model Optimizer,
       then serve the merged model via the CUTLASS recipe at 12-14 tok/s with
       FT behavior baked in. Loses dynamic adapter swap; gains 18× throughput
       vs the dynamic-LoRA-on-EMULATION-with-future-fix.
    2. **Custom FastAPI server** built around training-side `NVFP4LoRALinear`.
       Dynamic LoRA at probably similar speed to EMULATION.
    3. **Upstream fix** for the EMULATION fused_moe_lora kernel bug.

## Files

- [`serve_nemotron_nvfp4.sh`](serve_nemotron_nvfp4.sh) - main launcher for
  Nano serving (base or +LoRA).
- [`run_super_base_inference_cutlass.sh`](run_super_base_inference_cutlass.sh) -
  **recommended** Super base inference recipe via VLLM_CUTLASS native FP4. ~12-14 tok/s.
- [`run_super_base_inference.sh`](run_super_base_inference.sh) - slower EMULATION
  fallback for Super base. ~0.7 tok/s. Kept for reference / fallback if CUTLASS
  breaks in a future vLLM release.
- [`vllm_patches/`](vllm_patches/) - monkey-patches (Marlin chunked-repack
  micro-optimization, sitecustomize for subprocess propagation).
- [`diagnostics/`](diagnostics/) - diagnostic harness, microrepro, bench
  client used during the investigation.

## Reproducing the Super + EMULATION baseline

```bash
# ~9 min startup, then idle waiting for requests
./run_super_base_inference.sh

# In another shell: smoke test
curl -s http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
      "model": "nemotron-3-super-a12b-nvfp4",
      "prompt": "The DGX Spark is",
      "max_tokens": 30,
      "temperature": 0
    }'

# Throughput bench (a few sequential requests, writes JSONL)
python diagnostics/bench_vllm.py --output-tag my_run
```
