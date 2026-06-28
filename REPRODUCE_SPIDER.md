# Reproduce the Spider text-to-SQL before/after

This reproduces the public result on the README front page, end to end, on a single
GB10 box: a one-epoch NVFP4 LoRA on `nvidia/Llama-3.1-8B-Instruct-NVFP4` improves
held-out gold-SQL NLL and Spider exact-set-match, served via runtime-LoRA (the adapter
is attached to the 4-bit base at request time, never merged or re-quantized).

Everything here uses a **public base + public dataset** and a **deterministic** metric
(no sampling, no database execution), so the numbers are reproducible.

## Expected result (full 1034-row dev, 2 epochs, deterministic)

| Spider dev (Llama-3.1-8B-NVFP4) | base | adapter | delta |
|---|---|---|---|
| gold-SQL NLL (lower is better) | 0.889 | 0.850 | -0.039 |
| exact-set-match | 36.8% | 52.0% | +15.3 pp |

The *delta* is the signal; the absolute exact-set-match is scorer-dependent (strict
component set-match, value-insensitive, no DB execution).

**Generalizes across families** (same recipe, swap `--model-dir`):

| Base (NVFP4) | exact-set-match | NLL |
|---|---|---|
| Llama-3.1-8B   | 36.8% -> 52.0% (+15.3 pp) | 0.889 -> 0.850 |
| Mistral-Small-3.2-24B | 24.5% -> 61.0% (+36.5 pp) | 1.37 -> 0.57 |
| Qwen3-32B      | 46.1% -> 43.4% (saturated) | 1.29 -> 0.26 (large) |

The capability lift scales with base headroom: a base that already near-saturates the strict
set-match (Qwen3-32B) shows its gain as a large NLL/calibration improvement rather than +EM.

**Runtime hot-load variant.** `DYNAMIC=1 bash scripts/repro_spider.sh` serves the bare base
with runtime LoRA updates enabled and loads the adapter via `POST /v1/load_lora_adapter`
after startup (then unloads at teardown), instead of attaching it at launch. The eval is
identical; validated to match the launch-attach behavior on Llama-3.1-8B.

## Prerequisites

- One GB10 / DGX Spark (sm_121, ~128 GB unified memory). Peak training memory is ~11 GB.
- The training and serving venvs from the [README](README.md#quickstart). The serving
  step uses the **runtime-LoRA path**, i.e. the **vLLM 0.22.1 host venv**.
- `nvcc` on `PATH` for the serve step (flashinfer JITs its kernels at first serve):
  `export PATH=/usr/local/cuda/bin:$PATH`.

## 1. Get the base model (~6 GB)

```bash
hf download nvidia/Llama-3.1-8B-Instruct-NVFP4 \
    --local-dir models/Llama-3.1-8B-Instruct-NVFP4
```

## 2. Build the Spider data

Joins each question to its DB schema (the base hallucinates columns without it) and writes
chat-format JSONL. Schemas come from `richardr1126/spider-schema` (auto-downloaded).

```bash
python scripts/prep_spider.py --out-dir data/spider
# -> data/spider/spider.train.chat.jsonl  (7000)
# -> data/spider/spider.dev.chat.jsonl    (1034)
```

## 3. Train the LoRA (~1.8 h, 1 epoch)

The `llama` family resolves to native NVFP4 LoRA automatically; `--dry-run` first if you
want to confirm target coverage and memory before committing.

```bash
python -u scripts/train_nvfp4_lora.py \
    --model-dir models/Llama-3.1-8B-Instruct-NVFP4 \
    --train-file data/spider/spider.train.chat.jsonl \
    --val-file   data/spider/spider.dev.chat.jsonl \
    --output-dir adapters/spider_llama8b_r32 \
    --max-length 2048 --epochs 1 --batch-size 1 --grad-accum 16 \
    --lora-r 32 --lora-alpha 64 --lora-dropout 0.05 \
    --target-modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
    --learning-rate 1e-4 --warmup-ratio 0.03 \
    --checkpoint-every 50 --eval-every 200 --eval-subset 64
```

The adapter lands in `adapters/spider_llama8b_r32/best` (lowest dev loss).

## 4. Serve base + adapter (runtime-LoRA)

Serve the NVFP4 base once and attach the adapter as a named LoRA. Both are queryable:
`base` (no adapter) and `myft` (adapter applied live). Use the vLLM 0.22.1 host venv.

```bash
export PATH=/usr/local/cuda/bin:$PATH        # flashinfer needs nvcc
MAX_JOBS=1 vllm serve models/Llama-3.1-8B-Instruct-NVFP4 \
    --served-model-name base --host 127.0.0.1 --port 8000 \
    --enable-lora --max-lora-rank 32 --max-loras 2 \
    --lora-modules myft=adapters/spider_llama8b_r32/best \
    --max-model-len 8192 --enforce-eager \
    --gpu-memory-utilization 0.6 --kv-cache-dtype fp8
```

First serve JITs flashinfer kernels (a few minutes). `inspect` first if you want the
binding verdict: `python -m nybbloris.cli inspect --base-model-dir models/Llama-3.1-8B-Instruct-NVFP4 --adapter-dir adapters/spider_llama8b_r32/best` (Llama is flat, so it binds directly: VERDICT PASS).

## 5. Score the before/after (deterministic)

In another shell, against the running server:

```bash
python scripts/eval_retention.py \
    --dev-file data/spider/spider.dev.chat.jsonl \
    --models base myft --n 200 --out spider_retention.json
```

`eval_retention.py` reports two deterministic metrics, base vs adapter:
- **gold-SQL NLL** - teacher-forced per-token cross-entropy of the gold SQL given the
  schema+question prompt, via vLLM `/v1/completions` echo + logprobs over the gold span.
  No decoding, fully deterministic.
- **exact-set-match** - greedy generation scored against a self-contained Spider
  component set-match (SELECT/WHERE/GROUP BY/ORDER BY/keyword bags as sets,
  value-insensitive). No database execution.

You should see the deltas in the table above (`myft` lower NLL, higher exact-set-match).

## Notes

- **Tear-down.** vLLM spawns an `EngineCore` worker that holds the GPU after `vllm serve`
  exits; on GB10 its comm is truncated, so kill it by name: `pkill -9 EngineCor`.
- **Determinism.** The NLL metric is teacher-forced (single forward pass) and fully
  deterministic; exact-set-match uses greedy decode (`temperature=0`).
- **n.** `--n 200` is a stable headline; pass `--n 1034` for the full dev set, or train
  more epochs for a larger exact-set-match lift.
- **Tekken / mistral_common models (e.g. Mistral-Small).** These repacks ship an HF
  tokenizer that mis-tokenizes versus the native tekken tokenizer the model trained on,
  which also makes the NLL eval's echo offsets unusable (symptom: every row prints
  `note: NLL unavailable ... no gold tokens selected`). `repro_spider.sh` auto-detects
  `tekken.json` and serves `--tokenizer-mode mistral` so the served tokenizer matches
  training. Override with `TOKENIZER_MODE=none` (or another mode) if needed.
- **Vision-language repacks (e.g. Mistral-Small-3.2, a Pixtral VLM).** A text-only serve
  of a VLM crashes in the image processor (`Failed to apply PixtralProcessor`) unless
  image inputs are disabled. `repro_spider.sh` auto-detects a vision config and adds
  `--limit-mm-per-prompt '{"image":0}'`. Override with `LIMIT_MM=none`.
- **The eval refuses to fake a result.** If a metric fails for every row (wrong tokenizer,
  dead server, unbound adapter) the summary records it under `skipped` / `warnings` and
  prints a loud `WARNING:` rather than silently reporting `null` NLL or `0.0` EM. An
  adapter whose NLL and EM exactly match base is flagged as a likely silent no-op (the
  LoRA did not bind at serve) -- confirm binding with `nybbloris serve --verify`.
