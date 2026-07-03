# NVFP4 runtime-LoRA quality on public Spider (text-to-SQL)

Public, reproducible quality evidence: does an NVFP4-native LoRA, served via runtime-LoRA,
actually improve a task over the frozen NVFP4 base? Measured on the Spider dev set with the
repo's own `scripts/eval_retention.py` (exact-set-match + teacher-forced NLL).

## Result (2026-07-02, DGX Spark GB10 / sm_121, vLLM 0.22.1)

- Base: `Llama-3.1-8B-Instruct-NVFP4` (dense NVFP4, default MoE-free serve path).
- Adapter: `spider_llama8b_r32` (r=32 LoRA, q/k/v/o + gate/up/down), served as runtime-LoRA
  (not merged) alongside base.
- Eval: full 1034-row Spider dev set (and a 200-row subset), greedy, max_new_tokens=256.

| metric (n=1034 full dev) | base | +spider_llama8b_r32 | delta |
|---|---|---|---|
| **exact-set-match** | 0.376 | **0.597** | **+0.221 (~1.6x)** |

Subset check (n=200): 0.315 -> 0.535 (+0.220) -- consistent with the full set. Raw per-row output:
`llama8b_nvfp4_spider_dev_n1034.json` (full) and `llama8b_nvfp4_spider_dev_n200.json` (subset).

The fine-tune lifts exact-match from 37.6% to 59.7% (+22 points) on the full dev set. NLL is ~flat (marginally
higher) -- the LoRA optimizes task behaviour (valid SQL that set-matches gold), which EM captures
and teacher-forced gold-NLL does not fully reflect. Raw per-row output: `llama8b_nvfp4_spider_dev_n200.json`.

## Scope / honesty

- This measures the NVFP4-native-LoRA-vs-base lift on public data (does the fine-tune improve the
  quantized model on a real task, served as runtime-LoRA).
- Dense base, so this exercises attention+MLP runtime-LoRA, not the MoE expert-LoRA path.

## Reproduce

```bash
# serve base + adapter (dense NVFP4, runtime-LoRA):
vllm serve <Llama-3.1-8B-Instruct-NVFP4> --served-model-name base --enable-lora \
  --lora-modules spider_llama8b=<adapter> --max-lora-rank 32 --port 8004
python scripts/eval_retention.py --base-url http://127.0.0.1:8004 \
  --dev-file <spider.dev.chat.jsonl> --models base spider_llama8b --n 200 --out out.json
```
