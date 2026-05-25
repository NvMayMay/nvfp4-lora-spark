#!/usr/bin/env bash
# Serve Nemotron-3-Super-120B-A12B-NVFP4 base inference on DGX Spark via vLLM
# with the native-FP4 CUTLASS MoE backend (VLLM_CUTLASS). RECOMMENDED PATH.
#
# Measured throughput (single-stream, sequential /v1/completions):
#   ~11-14 tok/s, flat across prompt lengths 12-456 tokens
#
# vs the EMULATION fallback (see run_super_base_inference.sh): ~18x faster.
#
# Every flag below is load-bearing on Spark (GB10, sm_121, 128 GB UMA):
#   --moe-backend cutlass    -> selects VLLM_CUTLASS (CutlassExpertsFp4); the
#                                only native-FP4 MoE kernel whose oracle accepts
#                                family 120 (Blackwell consumer) AND has the
#                                kernel binary compiled and working in vLLM 0.21.
#   --enforce-eager          -> disables CUDA graph capture. Without this, the
#                                FULL decode capture (35 batch sizes) consumes
#                                ~3 GiB extra and pushes us into our safety
#                                floor near the end (last seen: 28/35 captures
#                                completed before MemAvail hit 1.99 GB).
#   --gpu-memory-utilization 0.70 -> keeps total CUDA usage well under the 130
#                                GB physical ceiling. At default 0.92, KV cache
#                                budget is 36.99 GiB which works but leaves
#                                less headroom; 0.70 is comfortable.
#   --max-model-len 2048 etc. -> conservative default. The README
#                                prompt=2048 + output=2048 cells require
#                                MAX_MODEL_LEN=4096 for apples-to-apples
#                                comparison.
#
# CAVEAT: This serves the BASE model only. vLLM 0.21's CUTLASS kernel does
# NOT support LoRA (`CutlassExpertsFp4.supports_lora() = False`). For Super-FT
# serving, options are: (a) merge LoRA into the NVFP4 base + requantize, then
# serve via this exact recipe with the merged model; or (b) use the EMULATION
# backend with the broken Triton fused_moe_lora kernel (requires upstream fix);
# or (c) custom FastAPI server with our training-side NVFP4LoRALinear.
#
# See ../docs/PERFORMANCE_ROADMAP.md and serve/diagnostics/README.md for full context.

set -euo pipefail

MODEL_DIR="${MODEL_DIR:-/path/to/Models/Nemotron-3-Super-120B-A12B-NVFP4}"
PORT="${PORT:-8000}"
HOST="${HOST:-0.0.0.0}"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MAX_JOBS=1

vllm serve "$MODEL_DIR" \
    --served-model-name nemotron-3-super-a12b-nvfp4 \
    --host "$HOST" --port "$PORT" \
    --tensor-parallel-size 1 \
    --dtype bfloat16 \
    --max-model-len 2048 \
    --max-num-batched-tokens 128 \
    --max-num-seqs 1 \
    --gpu-memory-utilization 0.70 \
    --enforce-eager \
    --moe-backend cutlass
