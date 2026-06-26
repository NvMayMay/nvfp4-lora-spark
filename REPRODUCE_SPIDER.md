# Reproduce the Spider text-to-SQL before/after

This reproduces the public result on the README front page, end to end, on a single
GB10 box: a one-epoch NVFP4 LoRA on `nvidia/Llama-3.1-8B-Instruct-NVFP4` improves
held-out gold-SQL NLL and Spider exact-set-match, served via runtime-LoRA (the adapter
is attached to the 4-bit base at request time, never merged or re-quantized).

Everything here uses a **public base + public dataset** and a **deterministic** metric
(no sampling, no database execution), so the numbers are reproducible.

## Expected result (n=200, deterministic)

| Spider dev | base | adapter | delta |
|---|---|---|---|
| gold-SQL NLL (lower is better) | 0.977 | 0.932 | -0.045 |
| exact-set-match | 34.0% | 41.5% | +7.5 pp |

One epoch, so the *delta* is the signal; the absolute exact-set-match is scorer-dependent
(this uses a strict component set-match, value-insensitive, no DB execution). More epochs
lift it further.

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
