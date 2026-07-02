# Serving NVFP4 + runtime-LoRA (the blessed recipe)

Runtime-LoRA is the v1 serve path: the NVFP4 base loads once and the LoRA delta is
applied live in bf16, attaching the adapter at request time. Dense and
attention / shared-expert targets apply on any backend. Routed-expert MoE deltas are
BACKEND-GATED, not merge-only: they serve live on a LoRA-capable MoE backend
(`--moe-backend emulation`, or marlin), and are blocked only on the cutlass/flashinfer
fast kernels (which report `supports_lora=False`). `nybbloris inspect` tells you which
path an adapter takes, and `nybbloris serve` auto-selects emulation for routed adapters.
Merge-then-serve remains an option where you do not need request-time adapter swap.

On a DGX Spark / GB10 the **blessed serve path is a host venv**, not a
container (see runtime table below). The commands here are exact and tested.

## 1. Pick the serve runtime by the base's quant convention

| Base quant convention | Min vLLM | Where |
|---|---|---|
| compressed-tensors NVFP4 (e.g. RedHatAI 122B) | 0.19+ | NGC `vllm:26.04`/`26.05` Docker |
| ModelOpt NVFP4 (the canonical 3.6) | **0.22.1** | host venv |

`nvidia/Qwen3.6-35B-A3B-NVFP4` is ModelOpt + a multimodal wrapper, so it needs
vLLM 0.22.1. NGC images top out at 0.20.1, and an aarch64 0.22.1 container build
is an open gap - so the canonical model serves from a **host venv** carrying
vLLM 0.22.1. (Container support is tracked as a platform risk, not claimed.)

## 2. Pre-flight (cheap, no GPU)

```
nybbloris inspect --base-model-dir <base> --adapter-dir <adapter>
```

- `VERDICT PASS` -> binds and serves live.
- `NO-OP` / `NEEDS-REKEY` -> a flat PEFT adapter on a multimodal-wrapped base;
  serve with `--rekey auto` (or pre-rekey with `scripts/rekey_lora_for_vllm.py`).
- A **quantized `lm_head` crashes vLLM** (it keeps `lm_head` bf16). Fix it first:
  `nybbloris serve --fix-lm-head` (or `scripts/fix_nvfp4_lm_head.py --apply`).

Exit codes are CI-friendly: `0` PASS, `1` FAIL/EMPTY, `3` NO-OP/NEEDS-REKEY,
`4` BLOCKED-ROUTED. `nybbloris doctor` checks the env/deps before you start.

## 3. Serve

```
nybbloris serve --base-model-dir <base> \
  --adapter myft=<adapter> --rekey auto \
  --vllm /path/to/qwen-serve/bin/vllm \
  --max-model-len 16384 --gpu-memory-utilization 0.6
```

`nybbloris serve` runs the pre-flight, auto-rekeys silent no-ops, and launches
vLLM with the right flags. The equivalent raw launch it builds:

```
MAX_JOBS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
FLASHINFER_DISABLE_VERSION_CHECK=1 CUTE_DSL_ARCH=sm_121a \
VLLM_ALLOW_RUNTIME_LORA_UPDATING=True \
  <venv>/bin/python <venv>/bin/vllm serve <base> \
    --enable-lora --max-lora-rank 128 --max-loras 2 \
    --lora-modules myft=<rekeyed-adapter> \
    --max-model-len 16384 --max-num-batched-tokens 2048 --enable-chunked-prefill \
    --max-num-seqs 4 --enforce-eager \
    --gpu-memory-utilization 0.6 --kv-cache-dtype fp8
```

Invoke vLLM as `<venv>/bin/python <venv>/bin/vllm`: a copied/relocated venv keeps
a stale shebang, so run the script through its sibling interpreter.

## 4. GB10 / unified-memory gotchas (hard-won)

- **`gpu-memory-utilization` is a fraction of the SHARED pool.** On UMA, GPU
  memory IS system memory (~128 GB total). A utilization that exceeds what is
  physically free does not just fail the load - it can OOM-kill the whole box.
  Keep it conservative (0.55-0.6 leaves wide margin for a ~24 GB NVFP4 base) and
  gate first loads behind a MemAvailable floor watchdog.
- **flashinfer JIT warmup is ~15-20 min cold.** It compiles ~80 MoE kernels at
  init; `MAX_JOBS=1` serializes nvcc so a parallel-compile spike does not OOM the
  box next to the loaded weights. The cache warms after the first run.
- **EngineCore outlives the launcher.** vLLM spawns a `VLLM::EngineCore` worker
  that holds the weights/GPU. Killing `vllm serve` does NOT kill it, and its comm
  is truncated to `VLLM::EngineCor`, so `pkill -f "VLLM::EngineCore"` MISSES it
  (the worker's args do not contain that string). Kill it by process name:
  `pkill -9 EngineCor` (or kill the PID). Confirm free memory recovered before
  the next load, or it OOMs.

## 5. Verify it actually applied

```
nybbloris serve ... --verify --verify-only --val-file <chat.jsonl>
```

Static `inspect` proves an adapter BINDS; `--verify` proves it CHANGED behavior
at runtime (low char-prefix agreement vs base = diverged = applied; ~identical =
a possible silent no-op). `--verify-only` exits non-zero on a WARN, so it works
as a CI gate.
