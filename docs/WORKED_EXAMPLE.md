# Worked example: train → inspect → serve → verify (`nybbloris` CLI)

The end-to-end **runtime-LoRA** flow on a single GB10 (DGX Spark). Runtime-LoRA
applies the adapter in bf16 at serve time and attaches it to the NVFP4 base at
request time. It is the path for dense models and attention / shared-expert
targets; for routed-MoE-on-CUTLASS (where request-time LoRA is not available on
sm_121), merge-then-serve is the validated path - see the README.

The three commands form a closed loop:

- `nybbloris inspect` — *will this adapter bind?* (static, no GPU)
- `nybbloris serve --verify` — *does it actually change behavior?* (runtime)
- `nybbloris train` — fine-tune, then auto-run the inspect pre-flight

---

## 0. Pick the serve runtime for your checkpoint

The base image / venv **is** the serve runtime, and the required vLLM version is
set by the checkpoint's quantization layout (`nybbloris inspect` prints which):

| checkpoint layout | example | vLLM |
|---|---|---|
| compressed-tensors NVFP4 | `RedHatAI/Qwen3.5-122B-A10B-NVFP4` | ≥ 0.19 (NGC `vllm:26.04+`) |
| ModelOpt NVFP4, quantized MoE / `lm_head` | `nvidia/Qwen3.6-35B-A3B-NVFP4` | ≥ 0.22.1 |

The 0.22.1 path is a host venv on GB10 (an aarch64 0.22.1 container build is out
of scope); see [serve/README.md](../serve/README.md).

## 1. Inspect — the binding contract (no GPU, seconds)

```bash
nybbloris inspect --base-model-dir models/<base> --adapter-dir adapters/<adapter>
```

It classifies every target (NVFP4 / FP8 / bf16, dense vs routed-expert), resolves
the adapter keys against the **vLLM runtime module tree**, and returns a verdict:

- **PASS** — binds directly, all targets LoRA-live at serve.
- **NO-OP** — binds *only* after a re-key; the adapter as-shipped would serve the
  un-adapted base. Common when a flat-key adapter meets a multimodal-wrapped base
  (the LM lives under `language_model.model.layers.*` in vLLM). `serve --rekey
  auto` fixes it.
- **NEEDS-REKEY / BLOCKED-ROUTED / FAIL / EMPTY** — partial bind, routed-expert
  FusedMoE (can't bind dynamically), no resolution, or no targets found.

Dense FP8 attention is **live** at serve (the delta is bf16, independent of base
quant); it is frozen only by the eager *training* loader.

## 2. Train

```bash
nybbloris train \
    --model-dir models/<base> \
    --train-file train.jsonl --val-file val.jsonl \
    --target-modules down_proj,gate_proj,up_proj \
    --output-dir adapters/<adapter> \
    --max-length 8192 --lora-r 16 --epochs 1 \
    --permissive-load --allow-partial-targets
```

The family and LoRA mechanism (native NVFP4 vs PEFT/bf16) are auto-detected. When
it finishes, the **post-train serve pre-flight runs automatically** (= step 1 on
the freshly trained adapter), so a non-bindable adapter is caught immediately.

Gotchas worth knowing up front:

- `--target-modules` is **comma-separated** (`a,b,c`), not space-separated.
- **Long prompts:** `--max-length` must fit your data or every example is dropped
  (`num_samples=0`). Chat datasets with long context need `8192`, not `2048`.
- `--permissive-load` allows intentionally-absent tensors (e.g. an MTP
  speculation head); `--allow-partial-targets` allows a native suffix to be
  partly BF16 (those BF16 instances would not train natively). FP8 targets
  do train natively via `FP8LoRALinear` on the non-pooled loader, so they
  need no flag.
- **GatedDeltaNet (GDN) models** (Qwen3.5 / 3.6 hybrid attention) require
  `flash-linear-attention==0.4.2` in the training env, or the GDN forward fails.
- A few-step run is undertrained on purpose for a smoke; expect `verify` to
  report `WARN` (≈ base) until you train enough to change behavior.

## 3. Fix a quantized `lm_head` (only if inspect/serve says so)

vLLM keeps `lm_head` in bf16; a checkpoint that quantized it crashes the load.
`nybbloris serve` refuses with the remediation, or run it directly:

```bash
python scripts/fix_nvfp4_lm_head.py --model-dir models/<base>            # dry-run
python scripts/fix_nvfp4_lm_head.py --model-dir models/<base> --apply    # then write (backs up)
```

(or `serve --fix-lm-head` to auto-apply.)

## 4. Serve + verify

```bash
nybbloris serve \
    --base-model-dir models/<base> \
    --adapter run=adapters/<adapter> \
    --rekey auto --port 8001 \
    --verify --val-file val.jsonl
```

- `--rekey auto` (default): re-keys a `NO-OP` adapter to the serve layout before
  launching.
- `--verify`: after the server is ready, diffs base vs each adapter on a few
  prompts and reports **PASS** (diverged = the adapter applied) or **WARN** (≈
  base = a possible silent no-op), printing a sample so you can see the
  difference.
- `--verify-only`: run the check, stop the server, exit non-zero on `WARN` — a CI
  gate. Teardown is graceful (no orphaned GPU memory).

For a relocated/copied serve venv whose `vllm` shebang points elsewhere, pass
`--vllm /path/to/venv/bin/vllm` (the interpreter beside it is used automatically).

## 5. Container

```bash
# Default NGC base (compressed-tensors NVFP4, e.g. the 122B):
docker build -t nybbloris .
# Or build on the vLLM runtime your checkpoint needs:
docker build --build-arg VLLM_BASE=<image-with-vllm-0.22.1> -t nybbloris:serve .

docker run --gpus all --ipc=host --network host -v /models:/models nybbloris \
    serve --base-model-dir /models/<base> --adapter run=/models/<adapter> \
          --rekey auto --port 8001
```

`inspect` needs no GPU; `serve` does. Mount adapters read-write so `--rekey auto`
can write the re-keyed copy.
