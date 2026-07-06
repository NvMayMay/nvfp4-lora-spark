# Command reference

Every command in one place: what it does, its key flags, and a one-line example. For a
narrative run of the whole loop see [WORKED_EXAMPLE.md](WORKED_EXAMPLE.md); for the concepts
behind them see the [project README](../README.md).

Examples use `models/Llama-3.1-8B-Instruct-NVFP4` as the base and `adapters/my_run/best` as the
adapter; substitute your own paths.

---

## The `nybbloris` CLI

`pip install -e .` puts `nybbloris` on your PATH. `inspect` and `doctor` are pure-library (no
GPU); `train`/`serve` shell out to repo-relative `scripts/`, so run them from a clone.

| Command | What it does | GPU |
|---|---|---|
| [`inspect`](#nybbloris-inspect) | Predict whether an adapter binds + serves live, from config + the safetensors index only | no |
| [`train`](#nybbloris-train) | LoRA fine-tune on the 4-bit weights (family + LoRA mode auto-detected), then a post-train serve pre-flight | yes |
| [`serve`](#nybbloris-serve) | Pre-flight gate, then start the vLLM serve; `--verify` proves the adapter changed the forward | yes |
| [`doctor`](#nybbloris-doctor) | Environment pre-flight: which train/serve deps + versions are present | no |
| [`data-check`](#nybbloris-data-check) | Training-data pre-flight: mask coverage, truncation drops, length histogram | no |
| [`contamination`](#nybbloris-contamination) | Train/eval overlap report (exact-match + n-gram + `db_id`) | no |

### `nybbloris inspect`

The single most common way a 4-bit fine-tune dies is a silent no-op at serve. `inspect` reads
only `config.json` + the safetensors *index* (no weights, no GPU) and returns a verdict.

```bash
nybbloris inspect --base-model-dir models/Llama-3.1-8B-Instruct-NVFP4 \
    --adapter-dir adapters/my_run/best
```

| Flag | Meaning |
|---|---|
| `--base-model-dir` | the NVFP4 base to bind against (required) |
| `--adapter-dir` | the PEFT adapter to check (required) |
| `--json` | emit the plan as JSON on stdout (suppresses the human report) |
| `--json-out PATH` | also write the plan object to a file |

Verdicts (also exit codes, so CI can branch): `PASS` binds + serves as-is (`0`) ·
`NO-OP`/`NEEDS-REKEY` binds only after a re-key, which `serve --rekey auto` handles (`3`) ·
`BLOCKED-ROUTED` routed-expert MoE, needs `--moe-backend emulation` (`4`) · `FAIL`/`EMPTY` does
not bind (`1`). (`NO-OP` = the adapter's keys resolve to zero base modules as-is: a silent no-op
until re-keyed.)

### `nybbloris train`

LoRA fine-tune with the strategy detected from the checkpoint. All flags forward to
[`scripts/train_nvfp4_lora.py`](../scripts/train_nvfp4_lora.py) (run that with `--help` for the
full set); the common ones:

```bash
nybbloris train \
    --model-dir models/Llama-3.1-8B-Instruct-NVFP4 \
    --train-file train.jsonl --val-file val.jsonl \
    --target-modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
    --output-dir adapters/my_run --epochs 2 --max-length 2048
```

| Flag | Meaning |
|---|---|
| `--model-dir` | the NVFP4 base to train on |
| `--train-file` / `--val-file` | chat-JSONL datasets |
| `--target-modules` | comma-separated module suffixes; a suffix matching nothing is a hard error |
| `--output-dir` | writes `best/`, rotated `checkpoint_step_N/`, and `target_coverage.json` |
| `--train-target {text,vision,both}` | text backbone (default), the bf16 vision tower, or both jointly |
| `--allow-unverified-family` | train an unregistered flat causal-LM via the generic fallback |
| `--resume-from` | resume from a checkpoint dir |

### `nybbloris serve`

Gates before launch (refuses a quantized `lm_head`, refuses a wrong-base adapter via the
manifest fingerprint, auto-re-keys a silent-no-op adapter, auto-selects `--moe-backend
emulation` for routed-expert deltas), then serves.

```bash
nybbloris serve --base-model-dir models/Llama-3.1-8B-Instruct-NVFP4 \
    --adapter myft=adapters/my_run/best \
    --vllm /path/to/serve-venv/bin/vllm \
    --verify --val-file val.jsonl
```

| Flag | Meaning |
|---|---|
| `--base-model-dir` | the NVFP4 base (required) |
| `--adapter NAME=PATH` | adapter to register (repeatable); `NAME` defaults to the dir basename |
| `--vllm PATH` | the vllm entrypoint (e.g. the host serve-venv's `bin/vllm`) |
| `--rekey {auto,off}` | auto-re-key a silent-no-op adapter to the serve layout (default `auto`) |
| `--fix-lm-head` | auto-dequantize a quantized `lm_head` vLLM can't load |
| `--moe-backend` | force a MoE backend; `emulation` applies routed-expert LoRA live |
| `--verify` / `--verify-only` | run the apply-check: a base-vs-adapter logprob delta (identical logprobs prove a no-op; a moved delta proves it applies) |
| `--val-file` | the JSONL whose prompts drive `--verify` (required with it) |
| `--dry-run` | print the resolved vLLM command without launching |
| `--host` / `--port` / `--max-model-len` / `--gpu-memory-utilization` | standard serve knobs |

### `nybbloris doctor`

Prints an `OK/WARN/FAIL` table for torch / transformers / vllm / fla / nvcc.

```bash
nybbloris doctor                                        # deps + versions
nybbloris doctor --base-model-dir models/Llama-3.1-8B-Instruct-NVFP4   # also check lm_head serve-compat
```

### `nybbloris data-check`

Training-data pre-flight; forwards to [`scripts/data_check.py`](../scripts/data_check.py).

```bash
nybbloris data-check --data train.jsonl --tokenizer models/Llama-3.1-8B-Instruct-NVFP4 --max-length 2048
```

### `nybbloris contamination`

Train/eval overlap report; forwards to
[`scripts/check_contamination.py`](../scripts/check_contamination.py).

```bash
nybbloris contamination --train train.jsonl --eval val.jsonl
```

---

## Scripts

`nybbloris` wraps the common paths; these scripts cover the rest. Run any with `--help` for its
full flags. This is the curated user-facing set — the complete tree (eval harnesses, one-off
prep, internal probes) is under [`scripts/`](../scripts/).

### Train and inspect

| Script | What it does |
|---|---|
| `train_nvfp4_lora.py` | The unified multi-family LoRA trainer (`nybbloris train` wraps it). |
| `inspect_nvfp4_checkpoint.py` | Layout + trainability report on a single checkpoint (run this first when porting a family). Distinct from `nybbloris inspect`, which plans a base+adapter binding. |

### Merge (serve by baking the adapter into the base)

| Script | What it does |
|---|---|
| `merge_lora_into_nvfp4.py` | Merge a LoRA into a Nano/Super NVFP4 (ModelOpt) base and re-emit a new checkpoint. |
| `merge_lora_into_ct_nvfp4.py` | Same for a compressed-tensors NVFP4 base (RedHatAI layout, e.g. Mistral / Qwen). |
| `merge_vision_lora.py` | Merge a vision-tower LoRA into the bf16 tower, preserving the NVFP4 backbone byte-for-byte. |
| `split_both_adapter.py` | Split a `--train-target both` adapter into its LLM and vision halves. |
| `export_llm_lora.py` | Export just the LLM half of a `both` adapter for runtime-LoRA serving (no merge). |

### Serve prep

| Script | What it does |
|---|---|
| `rekey_lora_for_vllm.py` | Re-key a PEFT adapter to the vLLM serve layout (the `NEEDS-REKEY` fix; `serve --rekey auto` calls it). |
| `rekey_expert_lora_for_vllm.py` | Same for wrapped multimodal bases and expert-LoRA adapters. |
| `fix_nvfp4_lm_head.py` | Dequantize a quantized `lm_head` to bf16 so vLLM can load it (`serve --fix-lm-head` wraps it). |

### Evaluate and validate

| Script | What it does |
|---|---|
| `eval_retention.py` | Score a text before/after: gold-answer NLL + exact-set-match. |
| `eval_vision_retention.py` | Score a vision before/after: normalized exact-match on a VQA set. |
| `validate_merge.py` | Audit a merged checkpoint: per-tensor cosine, no-op fraction, non-weight-file integrity. |
| `distinguish_ft.py` | Temperature-0 distinguishing-prompt test: did the fine-tune change behaviour? |

### Data prep

| Script | What it does |
|---|---|
| `prep_spider.py` | Build the Spider text-to-SQL train/val JSONL used in the on-ramp. |
| `prepare_vision_dataset.py` | Build an image+text dataset for vision / `both` training. |
