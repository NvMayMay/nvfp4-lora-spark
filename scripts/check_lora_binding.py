#!/usr/bin/env python3
"""Binding contract: does every LoRA target resolve to a base module, AND is its
weight a quant type that can actually take LoRA at serve time?

Two silent-no-op classes, both caught from key names + the base index alone (no
weights, no GPU) -- a cheap pre-flight gate before train / merge / serve:

  1) KEY mismatch -- the adapter's module paths don't match the base (e.g. a
     multimodal base nests the LM under `language_model.` while the adapter keys
     omit it); a naive load binds ZERO and silently serves the un-adapted base.

  2) QUANT-type mismatch -- a target resolves to an FP8-quantized weight. The
     NVFP4-LoRA runtime path applies the delta on the NVFP4 dequant; FP8 weights
     are demoted to frozen (no LoRA) unless allow_fp8_targets. So an adapter
     trained on a bf16 base can bind by key yet serve FROZEN on the quantized
     base, silently dropping those deltas. We report which targets are LIVE
     (NVFP4 / bf16) vs served FROZEN (FP8).

Quant type is read from the scale tensors present in the index:
  - weight_packed / weight_global_scale  -> NVFP4 (compressed-tensors)
  - weight_scale_2                        -> NVFP4 (ModelOpt)
  - input_scale (and none of the above)  -> FP8 static-scaled
  - weight only                          -> bf16 / unquantized
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import struct
from collections import Counter
from pathlib import Path


def adapter_modules(adapter_dir: Path):
    """Target module paths (lora suffix stripped) from the adapter header."""
    mods = set()
    for f in sorted(glob.glob(str(adapter_dir / "adapter_model*.safetensors"))):
        with open(f, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            hdr = json.loads(fh.read(n))
        for k in hdr:
            if k == "__metadata__" or not re.search(r"\.lora_[AB]\.weight$", k):
                continue
            t = k[len("base_model.model."):] if k.startswith("base_model.model.") else k
            mods.add(re.sub(r"\.lora_[AB]\.weight$", "", t))
    return sorted(mods)


# Candidate re-key transforms (extend as new base layouts appear).
REKEYS = [
    ("identity", lambda k: k),
    ("language_model", lambda k: k.replace("model.layers.", "model.language_model.layers.", 1)),
    ("language_model-prefix", lambda k: k.replace("model.", "model.language_model.", 1)),
]


def classify(base_path: str, base_keys: set):
    """None (unbound) or NVFP4 / FP8 / BF16 for a resolved target module."""
    has_w = f"{base_path}.weight" in base_keys
    has_wp = f"{base_path}.weight_packed" in base_keys
    if not (has_w or has_wp):
        return None
    if (has_wp
            or f"{base_path}.weight_global_scale" in base_keys
            or f"{base_path}.weight_scale_2" in base_keys):
        return "NVFP4"            # LoRA-live
    if f"{base_path}.input_scale" in base_keys:
        return "FP8"              # demoted to frozen on the NVFP4-LoRA path
    return "BF16"                 # unquantized -> LoRA-live


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
