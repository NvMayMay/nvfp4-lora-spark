"""
Throughput benchmark for the local vLLM server (port 8000).

Runs N requests sequentially, varying output length, records per-request
latency and tokens/sec. Writes results to a JSONL file and prints a summary.

Usage:
  python bench_vllm.py [--model nemotron-3-super-a12b-nvfp4]
                        [--output-tag base|lora]

                        [--cells "32,32;128,64;..."]
"""

import argparse
import json
import os
import time
from pathlib import Path

import requests

SERVER = "http://localhost:8000"

# Default sweep cells: (prompt_token_target, output_tokens)
DEFAULT_CELLS = [
    (32, 32),     # tiny: warmup
    (128, 64),    # short prompt, short output
    (128, 256),   # short prompt, medium output
    (512, 256),   # medium prompt, medium output
    (1024, 128),  # long prompt, short output
]

PROMPT_TEXTS = {
    32: "The DGX Spark is a compact AI workstation that runs",
    128: "The DGX Spark is a compact AI workstation built around the NVIDIA GB10 unified-memory chip. "
         "It runs large language models with NVFP4 quantization and supports a range of inference "
         "backends including Marlin and emulation paths. Today we are testing",
    512: ("The DGX Spark is a compact AI workstation built around the NVIDIA GB10 unified-memory chip. "
          "It runs large language models with NVFP4 quantization and supports a range of inference "
          "backends including Marlin and emulation paths. The unified memory architecture means CPU "
          "and GPU share the same physical LPDDR5x pool, which has implications for how vLLM allocates "
          "memory during model load and inference. The Nemotron-3 family from NVIDIA includes a "
          "Super-120B-A12B model that uses a hybrid architecture of Mamba state-space layers, "
          "attention layers, and Mixture-of-Experts (MoE) layers. The MoE component has 512 routed "
          "experts and a top-k of 22 active experts per token, which is unusually high. We have spent "
          "considerable effort getting this model to serve on a single DGX Spark, which has 128 GB of "
          "unified memory total. The successful configuration uses the EMULATION MoE backend with "
          "enforce_eager mode and gpu_memory_utilization of 0.70. Today we are testing") * 1,
    1024: None,  # built below
}
PROMPT_TEXTS[1024] = PROMPT_TEXTS[512] + " " + PROMPT_TEXTS[512]


def fit_prompt(target_tokens: int) -> str:
    """Pick the prompt text closest to target_tokens."""
    keys = sorted(PROMPT_TEXTS.keys())
    for k in keys:
        if k >= target_tokens:
            return PROMPT_TEXTS[k]
    return PROMPT_TEXTS[keys[-1]]


def run_completion(model: str, prompt: str, max_tokens: int, timeout: int = 600):
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    t0 = time.time()
    r = requests.post(f"{SERVER}/v1/completions", json=payload, timeout=timeout)
    t1 = time.time()
    r.raise_for_status()
    data = r.json()
    return {
        "latency_s": t1 - t0,
        "usage": data.get("usage", {}),
        "completion_text_preview": data["choices"][0]["text"][:100],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="nemotron-3-super-a12b-nvfp4")
    ap.add_argument("--output-tag", default="base", help="tag included in result filename")
    ap.add_argument(
        "--cells",
        default=None,
        help="semicolon-separated prompt_len,output_len pairs, e.g. 32,32;128,64",
    )
    args = ap.parse_args()

    if args.cells:
        cells = []
        for pair in args.cells.split(";"):
            p, o = pair.split(",")
            cells.append((int(p), int(o)))
    else:
        cells = DEFAULT_CELLS

    # Health check
    h = requests.get(f"{SERVER}/health", timeout=10)
    print(f"health: HTTP {h.status_code}")
    m = requests.get(f"{SERVER}/v1/models", timeout=10).json()
    print(f"served models: {[mm['id'] for mm in m.get('data', [])]}")
    print()

    ts = time.strftime("%Y%m%d_%H%M%S")
    log_path = Path(__file__).parent / f"bench_{args.output_tag}_{ts}.jsonl"
    log = open(log_path, "w", buffering=1)
    print(f"Writing results to {log_path}")
    print()

    print(f"{'prompt_len':>10}  {'output_len':>10}  {'latency_s':>10}  {'tok/s':>8}  preview")
    print("-" * 100)
    for prompt_len_target, output_len in cells:
        prompt = fit_prompt(prompt_len_target)
        try:
            res = run_completion(args.model, prompt, output_len)
            usage = res["usage"]
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", output_len)
            total = res["latency_s"]
            tok_per_s = completion_tokens / total if total > 0 else 0
            print(
                f"{prompt_tokens:>10}  {completion_tokens:>10}  "
                f"{total:>10.2f}  {tok_per_s:>8.2f}  "
                f"{res['completion_text_preview'][:60]!r}"
            )
            log.write(json.dumps({
                "ts": time.time(),
                "prompt_target": prompt_len_target,
                "prompt_tokens_actual": prompt_tokens,
                "output_target": output_len,
                "completion_tokens_actual": completion_tokens,
                "latency_s": total,
                "tok_per_s": tok_per_s,
                "preview": res["completion_text_preview"],
            }) + "\n")
        except Exception as e:
            print(f"{prompt_len_target:>10}  {output_len:>10}  ERROR: {e!r}")
            log.write(json.dumps({
                "ts": time.time(),
                "prompt_target": prompt_len_target,
                "output_target": output_len,
                "error": repr(e),
            }) + "\n")
    log.close()
    print()
    print(f"Results saved to {log_path}")


if __name__ == "__main__":
    main()
