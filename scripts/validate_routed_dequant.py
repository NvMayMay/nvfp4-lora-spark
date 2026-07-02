"""Validate the routed-only NVFP4 emulation dequant patch (serve/vllm_patches/
nvfp4_emulation_routed_dequant.py) against a running vLLM serve.

The patch dequantizes only the routed experts per forward on the emulation MoE backend
(the only LoRA-capable NVFP4 MoE path on sm_121). It must be EXACT (same math, fewer
experts materialized) and FASTER. This script proves both, the decisive way: prompt-echo
logprobs (the forward pass itself), not generated text.

Procedure (two arms, same base + adapter, `--moe-backend emulation`):
  1. serve WITHOUT the patch  -> `probe --arm off --out off.json`
  2. serve WITH VLLM_PATCH_ROUTED_DEQUANT=1 (PYTHONPATH=serve/vllm_patches) -> `probe --arm on --out on.json`
  3. `compare off.json on.json`

`compare` asserts: (a) base != adapter in BOTH arms (LoRA fires), (b) adapter_off == adapter_on
and base_off == base_on within tol (patch is numerically exact -- bit-exact parity subsumes any
greedy-EM check), and reports the decode speedup. Exit 0 = PASS, 1 = FAIL.

Measured 2026-07-02 on Nemotron-3-Nano-30B-A3B-NVFP4 + an expert-LoRA (up_proj/down_proj):
parity max|delta|=0 (bit-exact) base AND adapter; decode 2.56 -> 12.86 tok/s (5.0x), no fallbacks.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from serve_apply_check import prompt_echo_logprobs  # noqa: E402

DEFAULT_PROBE = ("Question: Write a SQL query to list the names of all employees in the "
                 "Sales department earning more than 50000.\nAnswer:")


def _timed_generate(base_url, model, prompt, max_tokens, timeout=1800.0):
    url = base_url.rstrip("/") + "/v1/completions"
    body = json.dumps({"model": model, "prompt": prompt, "max_tokens": max_tokens,
                       "temperature": 0, "logprobs": 0}).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    t0 = time.time()
    r = json.loads(urllib.request.urlopen(req, timeout=timeout).read())
    dt = time.time() - t0
    n = r["usage"]["completion_tokens"]
    return {"seconds": dt, "completion_tokens": n, "tok_per_s": (n / dt if dt else 0.0)}


def _maxdelta(a, b):
    n = min(len(a), len(b))
    return (max(abs(a[i] - b[i]) for i in range(n)) if n else float("nan"), n)


def cmd_probe(a):
    out = {"arm": a.arm, "probe": a.prompt, "gen_tokens": a.gen_tokens}
    out["base_lp"] = prompt_echo_logprobs(a.base_url, a.base_model, a.prompt)
    out["adapter_lp"] = prompt_echo_logprobs(a.base_url, a.adapter_model, a.prompt)
    out["base_gen"] = _timed_generate(a.base_url, a.base_model, a.prompt, a.gen_tokens)
    out["adapter_gen"] = _timed_generate(a.base_url, a.adapter_model, a.prompt, a.gen_tokens)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=2)
    fire, n = _maxdelta(out["base_lp"], out["adapter_lp"])
    print(f"[{a.arm}] LoRA delta max|base-adapter|={fire:.4g} over {n} toks; "
          f"adapter gen {out['adapter_gen']['tok_per_s']:.3f} tok/s -> {a.out}")
    return 0


def cmd_compare(a):
    off, on = json.load(open(a.off)), json.load(open(a.on))
    off_fire, n1 = _maxdelta(off["base_lp"], off["adapter_lp"])
    on_fire, n2 = _maxdelta(on["base_lp"], on["adapter_lp"])
    par_ad, na = _maxdelta(off["adapter_lp"], on["adapter_lp"])
    par_ba, nb = _maxdelta(off["base_lp"], on["base_lp"])
    o = off["adapter_gen"]["tok_per_s"]
    n = on["adapter_gen"]["tok_per_s"]
    print(f"[1] LoRA fires  OFF max|d|={off_fire:.4g}  ON max|d|={on_fire:.4g}")
    print(f"[2] parity      adapter OFF-vs-ON max|d|={par_ad:.4g}  base max|d|={par_ba:.4g}")
    print(f"[3] speedup     adapter {o:.3f} -> {n:.3f} tok/s = {n/o:.2f}x" if o else "[3] no timing")
    ok = (off_fire > a.fire_tol and on_fire > a.fire_tol
          and par_ad < a.par_tol and par_ba < a.par_tol)
    print(f"VERDICT: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("probe")
    p.add_argument("--base-url", default="http://127.0.0.1:8001")
    p.add_argument("--base-model", default="base")
    p.add_argument("--adapter-model", default="canary")
    p.add_argument("--arm", required=True)
    p.add_argument("--prompt", default=DEFAULT_PROBE)
    p.add_argument("--gen-tokens", type=int, default=64)
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_probe)
    c = sub.add_parser("compare")
    c.add_argument("off")
    c.add_argument("on")
    c.add_argument("--par-tol", type=float, default=1e-3)
    c.add_argument("--fire-tol", type=float, default=1e-4)
    c.set_defaults(func=cmd_compare)
    a = ap.parse_args(argv)
    return a.func(a)


if __name__ == "__main__":
    raise SystemExit(main())
