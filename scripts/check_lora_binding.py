#!/usr/bin/env python3
"""Binding contract: will every LoRA target actually bind and apply when the
adapter is served in vLLM?

Two silent-no-op classes, caught from key names + the base index alone (no
weights, no GPU):
  * KEY mismatch -- the adapter's module paths don't match the SERVE engine's
    runtime module tree, so vLLM binds them to nothing and serves the un-adapted
    base. This is resolved against the VLLM_BUILD naming (e.g. a multimodal
    *ForConditionalGeneration wrapper exposes the LM as `language_model.model.*`,
    NOT the on-disk `model.language_model.*`), so it predicts the REAL serve
    binding (MEASURED §7h: the on-disk-keyed check returned the inverse).
  * ROUTED-expert FusedMoE -- supports_lora=False at serve; those deltas can't
    bind dynamically (merge-for-serve or skip). Dense targets (attention / MLP /
    shared_expert) serve LIVE regardless of NVFP4 / FP8 / bf16 quant -- the delta
    is applied in bf16 independently of the base weight (FP8 is frozen only by the
    eager TRAIN loader, never by serve).

This is the binding-focused CLI; the full serve plan is `nybbloris inspect`. All
logic lives in nybbloris/plan.py (single source of truth); this just renders it.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root for the nybbloris package
from nybbloris.plan import serve_plan  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model-dir", required=True, type=Path)
    ap.add_argument("--adapter-dir", required=True, type=Path)
    args = ap.parse_args()

    plan = serve_plan(args.base_model_dir, args.adapter_dir)
    b, bi, t = plan["base"], plan["binding"], plan["targets"]
    n = plan["adapter"]["n_targets"]
    fp8 = t["fp8_dense_live"]

    print(f"[binding] adapter targets {n} modules; base arch {b['arch']} "
          f"({'multimodal-wrapped' if b['wrapped'] else 'causal-LM'})")
    print(f"[binding] vLLM binds against {b['serve_naming']}")
    print(f"[binding] naive (no re-key): {bi['naive_resolve']}/{n} resolve")
    print(f"[binding] best re-key '{bi['rekey']}': {bi['resolved']}/{n} resolve")
    print(f"[quant]   by weight type: {t['by_quant']}")
    live_note = f"  (incl. {fp8} dense-FP8 served live; frozen only by the eager TRAIN loader)" if fp8 else ""
    blocked_note = f"  |  BLOCKED routed-MoE: {t['blocked_routed']}/{n}" if t["blocked_routed"] else ""
    print(f"[quant]   LoRA-LIVE (served): {t['live']}/{n}{live_note}{blocked_note}")

    v = plan["verdict"]
    miss = bi["resolved"] - bi["naive_resolve"]
    if v == "EMPTY":
        print(f"[binding] VERDICT: EMPTY -- no LoRA targets found in {plan['adapter']['dir']} "
              f"(no adapter_model*.safetensors, or no lora_A/B weights). Check the adapter path.")
    elif v == "FAIL":
        print(f"[binding] VERDICT: FAIL -- {t['unresolved']} target(s) unresolved even after re-key. "
              f"e.g. {bi['unresolved'][:5]}")
    elif v == "NO-OP":
        print(f"[binding] VERDICT: NO-OP -- all {n} targets bind ONLY via the '{bi['rekey']}' re-key; the "
              f"adapter as-shipped resolves {bi['naive_resolve']}/{n} and would serve the un-adapted base. "
              f"Re-key it (scripts/rekey_lora_for_vllm.py) before serving.")
    elif v == "BLOCKED-ROUTED":
        print(f"[binding] VERDICT: BLOCKED-ROUTED -- {t['blocked_routed']}/{n} targets are routed-expert "
              f"FusedMoE, BACKEND-GATED at serve. They serve LIVE on a LoRA-capable MoE backend "
              f"(emulation -- the validated one-box path -- or marlin); they are blocked ONLY on the "
              f"cutlass/flashinfer fast backends (supports_lora=False). Serve with --moe-backend emulation, "
              f"or merge-for-serve. NOT a dead end: do not read this as 'merge-only'.")
    elif v == "NEEDS-REKEY":
        pct = 100 * miss // n
        print(f"[binding] VERDICT: NEEDS-REKEY -- binds only with the '{bi['rekey']}' re-key; a naive load "
              f"would silently apply {miss}/{n} = {pct}% NOTHING.")
    else:
        print(f"[binding] VERDICT: PASS -- all {n} targets bind directly and are LoRA-live at serve."
              + (f" ({fp8} are dense-FP8: live at serve, frozen only when training via the eager loader.)"
                 if fp8 else ""))


if __name__ == "__main__":
    main()
