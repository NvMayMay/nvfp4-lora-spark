#!/usr/bin/env python3
"""Binding contract: does every LoRA target resolve to a base module, AND is its
weight a quant type that can take LoRA at serve time?

Two silent-no-op classes, caught from key names + the base index alone (no weights,
no GPU): KEY mismatch (a naive load binds ZERO and serves the un-adapted base) and
QUANT-type freeze (FP8 weights are served frozen, deltas dropped, unless
allow_fp8_targets). This is the binding-focused report; the full serve plan is
`nybbloris inspect` (nybbloris.plan.serve_plan). Shared primitives live in
nybbloris/plan.py (single source of truth).
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root for the nybbloris package
from nybbloris.plan import REKEYS, adapter_modules, classify  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model-dir", required=True, type=Path)
    ap.add_argument("--adapter-dir", required=True, type=Path)
    ap.add_argument("--allow-fp8-targets", action="store_true",
                    help="count FP8 targets as live (only valid if the serve loader enables FP8 LoRA)")
    args = ap.parse_args()

    base_keys = set(json.load(open(args.base_model_dir / "model.safetensors.index.json"))["weight_map"])
    mods = adapter_modules(args.adapter_dir)
    print(f"[binding] adapter targets {len(mods)} modules; base index has {len(base_keys)} tensors")

    best = None
    for name, fn in REKEYS:
        n = sum(1 for m in mods if classify(fn(m), base_keys) is not None)
        print(f"  re-key '{name}': {n}/{len(mods)} resolve")
        if best is None or n > best[1]:
            best = (name, n, fn)
    name, bound, fn = best
    naive = sum(1 for m in mods if classify(m, base_keys) is not None)

    kinds = Counter()
    unresolved = []
    for m in mods:
        c = classify(fn(m), base_keys)
        unresolved.append(m) if c is None else kinds.__setitem__(c, kinds[c] + 1)
    frozen = 0 if args.allow_fp8_targets else kinds["FP8"]
    live = len(mods) - len(unresolved) - frozen

    print(f"\n[binding] naive (no re-key): {naive}/{len(mods)} resolve")
    print(f"[binding] best re-key '{name}': {bound}/{len(mods)} resolve")
    print(f"[quant]   by weight type: {dict(kinds)}")
    print(f"[quant]   LoRA-LIVE: {live}/{len(mods)}  |  served FROZEN (FP8, deltas dropped): {frozen}"
          + ("  [--allow-fp8-targets]" if args.allow_fp8_targets else ""))

    if unresolved:
        print(f"[binding] VERDICT: FAIL -- {len(unresolved)} target(s) unresolved even after re-key. "
              f"e.g. {unresolved[:5]}")
    elif frozen:
        pct = 100 * frozen // len(mods)
        rk = f" (only with the '{name}' re-key; a naive load would no-op {len(mods) - naive})" if naive < bound else ""
        print(f"[binding] VERDICT: PARTIAL -- all {len(mods)} bind by key{rk}, but {frozen}/{len(mods)} = "
              f"{pct}% resolve to FP8 weights and serve FROZEN (their deltas are silently dropped). "
              f"Enable allow_fp8_targets, or target NVFP4 modules.")
    elif naive < bound:
        pct = 100 * (len(mods) - naive) // len(mods)
        print(f"[binding] VERDICT: PASS *only* with the '{name}' re-key. A naive load would "
              f"silently apply {len(mods) - naive}/{len(mods)} = {pct}% NOTHING.")
    else:
        print(f"[binding] VERDICT: PASS -- all {len(mods)} targets bind and are LoRA-live.")


if __name__ == "__main__":
    main()
