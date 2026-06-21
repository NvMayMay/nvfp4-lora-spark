#!/usr/bin/env python3
"""§8 binding check: does every LoRA target actually resolve to a base module?

Catches the silent-no-op / "swathes of LoRA filters not applied" class: when the
adapter's module paths don't match the base (e.g. a multimodal base nests the LM
under `language_model.` while the adapter keys omit it), a naive load binds ZERO
modules and silently serves the un-adapted base. Reads key names only (no weights,
no GPU) -- a cheap pre-flight gate before any train/merge/serve.
"""
from __future__ import annotations
import argparse
import glob
import json
import re
import struct
from pathlib import Path


def adapter_modules(adapter_dir: Path):
    mods = set()
    for f in sorted(glob.glob(str(adapter_dir / "adapter_model*.safetensors"))):
        with open(f, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            hdr = json.loads(fh.read(n))
        for k in hdr:
            if k == "__metadata__" or not re.search(r"\.lora_[AB]\.weight$", k):
                continue
            t = k[len("base_model.model."):] if k.startswith("base_model.model.") else k
            mods.add(re.sub(r"\.lora_[AB]\.weight$", ".weight", t))
    return sorted(mods)


# Candidate re-key transforms (extend as new base layouts appear).
REKEYS = [
    ("identity", lambda k: k),
    ("language_model", lambda k: k.replace("model.layers.", "model.language_model.layers.", 1)),
    ("language_model-prefix", lambda k: k.replace("model.", "model.language_model.", 1)),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model-dir", required=True, type=Path)
    ap.add_argument("--adapter-dir", required=True, type=Path)
    args = ap.parse_args()

    base_keys = set(json.load(open(args.base_model_dir / "model.safetensors.index.json"))["weight_map"])
    mods = adapter_modules(args.adapter_dir)
    print(f"[binding] adapter targets {len(mods)} modules; base index has {len(base_keys)} tensors")

    best = None
    for name, fn in REKEYS:
        bound = sum(1 for m in mods if fn(m) in base_keys)
        print(f"  re-key '{name}': {bound}/{len(mods)} bound")
        if best is None or bound > best[1]:
            best = (name, bound, fn)
    name, bound, fn = best
    naive = sum(1 for m in mods if m in base_keys)

    print(f"\n[binding] naive (no re-key): {naive}/{len(mods)} bound")
    print(f"[binding] best re-key '{name}': {bound}/{len(mods)} bound")
    if naive == len(mods):
        print("[binding] VERDICT: PASS -- all targets bind directly.")
    elif bound == len(mods):
        pct = 100 * (len(mods) - naive) // len(mods)
        print(f"[binding] VERDICT: PASS *only* with the '{name}' re-key. A NAIVE load would "
              f"silently apply {len(mods) - naive}/{len(mods)} = {pct}% NOTHING "
              f"(the 'swathes of LoRA filters not applied' silent no-op).")
    else:
        miss = [m for m in mods if fn(m) not in base_keys][:5]
        print(f"[binding] VERDICT: FAIL -- {len(mods) - bound} targets unresolved even after re-key. e.g. {miss}")


if __name__ == "__main__":
    main()
