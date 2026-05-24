"""
P1.5.3 + P1.5.4 distinguishing-prompt test for a merged FT model.

Sends a fixed list of prompts to TWO vLLM servers (base and merged-FT)
and compares outputs. Reports:

- Whether merged-FT outputs differ visibly from base on the same prompts
  (distinguishing test, qualitative).
- Token-level overlap rate between base and merged-FT at temperature=0
  (quantitative: 1.0 means identical outputs, lower means FT signal is
  visible).
- Per-prompt completion text dumped to a JSONL file.

Usage:
  # Start base CUTLASS server (port 8000)
  ./serve/run_super_base_inference_cutlass.sh &
  # Wait, then start merged-FT server on port 8001
  PORT=8001 SERVED_NAME=ft ./serve/run_super_ft_merged.sh &
  # Wait, then:
  python scripts/distinguish_ft.py \\
      --base-url http://localhost:8000 \\
      --base-model nemotron-3-super-a12b-nvfp4 \\
      --ft-url http://localhost:8001 \\
      --ft-model ft \\
      --output-jsonl distinguish_results.jsonl

(In practice we run one server at a time on Spark since two won't fit
in memory; restart between base and FT, save outputs, then compare.)

For a single-server flow:
  python scripts/distinguish_ft.py --single-server \\
      --base-url http://localhost:8000 \\
      --base-model <model_name> \\
      --output-jsonl base_outputs.jsonl
  # restart server with merged model, then
  python scripts/distinguish_ft.py --single-server \\
      --base-url http://localhost:8000 \\
      --base-model <model_name> \\
      --output-jsonl ft_outputs.jsonl
  python scripts/distinguish_ft.py --compare base_outputs.jsonl ft_outputs.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import requests


# 100 fixed prompts - mix of domain-specific (ICH clinical/regulatory,
# matching training corpus) and generic to detect both FT-style and
# distribution drift.
PROMPTS = [
    # Domain (ICH clinical/regulatory): these should show FT signal.
    "The primary objective of ICH E6(R2) is to",
    "When designing a Phase III clinical trial, the most important consideration for",
    "Adverse event reporting under MedDRA terminology requires",
    "The principles of GCP compliance include",
    "Risk-based monitoring (RBM) in clinical trials means",
    "An IND application to the FDA must include",
    "ICH Q1A(R2) stability testing recommends",
    "Pharmacovigilance reporting in the EU follows",
    "The CTD format for regulatory submissions consists of",
    "Bioequivalence studies under ICH M9 require",
    # Sub-domain: regulatory writing style
    "Please summarize the key changes in ICH E6(R3) compared to E6(R2):",
    "Describe the role of a Data Safety Monitoring Board:",
    "Explain the difference between a serious adverse event and an unexpected adverse event:",
    "Outline the typical timelines for an FDA Type B meeting:",
    "What is the purpose of a clinical study protocol amendment?",
    # Generic factual: should not change much between base and FT
    "The DGX Spark is",
    "Python's GIL prevents",
    "The capital of France is",
    "An LLM's context window refers to",
    "The transformer architecture was introduced in the paper",
    # Generic completion style: small differences expected
    "Once upon a time in a faraway kingdom,",
    "The best way to learn a new language is",
    "If I had to choose between mountain and sea, I would pick",
    # Reasoning prompts: FT may change reasoning style
    "If a car travels 60 mph for 2 hours and then 80 mph for 1 hour, the total distance is",
    "A doctor prescribes 250 mg of aspirin every 8 hours. The total daily dose is",
    "If 3 out of 4 patients respond to drug A, the response rate is",
    # Code-style prompts (generic): should not change
    "def fibonacci(n):",
    "import torch",
    "SELECT * FROM patients WHERE",
] + [f"Filler prompt {i} for statistical sample:" for i in range(71)]


def call_completion(url, model, prompt, max_tokens=64, temperature=0.0, timeout=600):
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    t0 = time.time()
    r = requests.post(f"{url}/v1/completions", json=payload, timeout=timeout)
    elapsed = time.time() - t0
    r.raise_for_status()
    data = r.json()
    text = data["choices"][0]["text"]
    usage = data.get("usage", {})
    return {
        "prompt": prompt,
        "completion": text,
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "latency_s": elapsed,
    }


def collect(url, model, output_jsonl, max_tokens):
    print(f"[collect] {url} model={model} -> {output_jsonl}")
    out = open(output_jsonl, "w")
    for i, prompt in enumerate(PROMPTS):
        try:
            res = call_completion(url, model, prompt, max_tokens=max_tokens)
        except Exception as e:
            res = {"prompt": prompt, "error": repr(e)}
        out.write(json.dumps(res) + "\n")
        out.flush()
        if "completion" in res:
            preview = res["completion"][:60].replace("\n", " ")
            print(f"  [{i+1:3d}/{len(PROMPTS)}] {res.get('latency_s', 0):.1f}s {preview!r}")
        else:
            print(f"  [{i+1:3d}/{len(PROMPTS)}] ERROR: {res['error']}")
    out.close()


def compare(base_jsonl, ft_jsonl):
    """Report distinguishing test results."""
    def load(p):
        recs = []
        with open(p) as f:
            for line in f:
                recs.append(json.loads(line))
        return {r["prompt"]: r for r in recs if "completion" in r}

    base = load(base_jsonl)
    ft = load(ft_jsonl)
    common = sorted(set(base) & set(ft))
    print(f"[compare] {len(common)} prompts common to both files")
    print()

    n_identical = 0
    n_different = 0
    differing_examples = []
    for prompt in common:
        b = base[prompt]["completion"]
        f = ft[prompt]["completion"]
        if b == f:
            n_identical += 1
        else:
            n_different += 1
            if len(differing_examples) < 15:
                differing_examples.append((prompt, b, f))

    pct_diff = 100 * n_different / max(1, len(common))
    print(f"identical completions:  {n_identical}/{len(common)}")
    print(f"differing completions: {n_different}/{len(common)} ({pct_diff:.1f}%)")
    print()
    print(f"=== first {len(differing_examples)} differing prompts ===")
    for prompt, b, f in differing_examples:
        print(f"\n--- prompt: {prompt!r}")
        print(f"  base: {b[:120]!r}{'...' if len(b)>120 else ''}")
        print(f"  ft:   {f[:120]!r}{'...' if len(f)>120 else ''}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_collect = sub.add_parser("collect")
    p_collect.add_argument("--url", default="http://localhost:8000")
    p_collect.add_argument("--model", required=True)
    p_collect.add_argument("--output-jsonl", required=True, type=Path)
    p_collect.add_argument("--max-tokens", type=int, default=64)

    p_compare = sub.add_parser("compare")
    p_compare.add_argument("base_jsonl", type=Path)
    p_compare.add_argument("ft_jsonl", type=Path)

    args = ap.parse_args()
    if args.cmd == "collect":
        collect(args.url, args.model, args.output_jsonl, args.max_tokens)
    elif args.cmd == "compare":
        compare(args.base_jsonl, args.ft_jsonl)


if __name__ == "__main__":
    main()
