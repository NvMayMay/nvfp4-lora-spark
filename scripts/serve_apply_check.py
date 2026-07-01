"""Decisive runtime-LoRA APPLY proof: prompt-echo logprob delta, base vs adapter.

Why this exists (and why it is separate from scripts/serve_parity_vllm.py):
`serve_parity_vllm.py` compares GENERATED text (a behavioural / quality proxy).
That is advisory only: vLLM greedy decode is nondeterministic, and on saturated
prompts base and adapter emit identical text even when the adapter IS applied --
or, worse, identical text when it is NOT applied (a silent no-op). This bit us on
Qwen3.5-122B: a rekeyed expert adapter LOADED and vLLM reported READY, but the
expert LoRA delta was never applied (a wrapped-model key-path mismatch left the
MoE buffers zero); greedy generation looked fine while the forward pass was
byte-identical to base.

This script asks the decisive question instead: does the adapter change the
forward pass AT ALL? It scores the SAME prompt tokens under base and under the
adapter via `/v1/completions` with `echo=True, logprobs=0` (the per-token
logprobs of the prompt itself reflect the forward pass, adapter included), and
compares them. Identical logprobs => the adapter is a NO-OP, regardless of what
greedy generation shows. This is the runtime-apply gate the binding contract's
static `inspect` cannot provide (static binding proves key resolution, not that
the delta executes).

Exit code: 0 = APPLIES (max per-token |delta| > threshold), 1 = NO-OP, 2 = error.
Intended as a CI/release gate and as the engine behind `nybbloris serve --verify`.
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request


def prompt_echo_logprobs(base_url: str, model: str, prompt: str, timeout: float = 1800.0):
    """Return the per-token logprobs of `prompt` itself under `model` (echo path)."""
    url = base_url.rstrip("/") + "/v1/completions"
    body = json.dumps({
        "model": model, "prompt": prompt, "max_tokens": 1,
        "temperature": 0, "echo": True, "logprobs": 0,
    }).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    r = json.loads(urllib.request.urlopen(req, timeout=timeout).read())
    lp = r["choices"][0]["logprobs"]["token_logprobs"]
    # The first echoed token has no preceding context -> logprob is null; drop nulls.
    return [x for x in lp if isinstance(x, (int, float))]


def apply_verdict(base_lp, adapter_lp, threshold: float = 1e-4) -> dict:
    """Pure verdict: does the adapter move the prompt logprobs?

    Compares element-wise over the common prefix (both score the same prompt, so
    lengths match in practice; min() guards a tokenizer edge). Returns the deltas
    and an `applies` bool. NO tokens scored, or max |delta| <= threshold, => no-op.
    """
    n = min(len(base_lp), len(adapter_lp))
    if n == 0:
        return {"n": 0, "sum_base": 0.0, "sum_adapter": 0.0, "sum_delta": 0.0,
                "max_abs_delta": 0.0, "applies": False}
    sum_b = float(sum(base_lp[:n]))
    sum_a = float(sum(adapter_lp[:n]))
    max_abs = max(abs(adapter_lp[i] - base_lp[i]) for i in range(n))
    return {
        "n": n,
        "sum_base": sum_b,
        "sum_adapter": sum_a,
        "sum_delta": sum_a - sum_b,
        "max_abs_delta": float(max_abs),
        "applies": bool(max_abs > threshold),
    }


DEFAULT_PROMPT = (
    "-- SQLite\nSELECT name, age FROM singer WHERE country = 'France' "
    "ORDER BY age DESC LIMIT 5;\nSELECT COUNT(*) FROM concert WHERE year > 2010;"
)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--base-model", required=True,
                    help="served model name of the BASE (no adapter)")
    ap.add_argument("--adapter-model", required=True,
                    help="served model name of the ADAPTER (a --lora-modules name)")
    ap.add_argument("--prompt", default=None, help="probe prompt (default: a SQL-ish probe)")
    ap.add_argument("--prompt-file", default=None, help="read the probe prompt from this file")
    ap.add_argument("--threshold", type=float, default=1e-4,
                    help="max per-token |logprob delta| above which the adapter counts as applied")
    ap.add_argument("--timeout", type=float, default=1800.0)
    ap.add_argument("--out", default=None, help="also write the verdict JSON here")
    args = ap.parse_args(argv)

    prompt = args.prompt or DEFAULT_PROMPT
    if args.prompt_file:
        prompt = open(args.prompt_file).read()

    try:
        base_lp = prompt_echo_logprobs(args.base_url, args.base_model, prompt, args.timeout)
        adap_lp = prompt_echo_logprobs(args.base_url, args.adapter_model, prompt, args.timeout)
    except Exception as e:  # noqa: BLE001
        print(f"[apply-check] request failed: {type(e).__name__}: {e}", file=sys.stderr)
        return 2

    v = apply_verdict(base_lp, adap_lp, args.threshold)
    v["base_model"] = args.base_model
    v["adapter_model"] = args.adapter_model
    v["threshold"] = args.threshold
    v["verdict"] = "APPLIES" if v["applies"] else "NO-OP"

    print(f"tokens scored: base={len(base_lp)} adapter={len(adap_lp)}")
    print(f"sum logprob: base={v['sum_base']:.4f}  adapter={v['sum_adapter']:.4f}  "
          f"(delta={v['sum_delta']:+.4f})")
    print(f"max per-token |delta|={v['max_abs_delta']:.4e}  (threshold={args.threshold:.1e})")
    print(f"VERDICT: {v['verdict']}"
          + ("" if v["applies"] else " -- adapter loaded but NOT applied (logprobs identical)"))

    if args.out:
        with open(args.out, "w") as f:
            json.dump(v, f, indent=2)

    return 0 if v["applies"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
