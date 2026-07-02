#!/usr/bin/env python
"""Offline copy of the serve-time language_model re-key.

Rewrites a PEFT adapter's flat text-only keys
    base_model.model.model.layers.N...
to the multimodal ConditionalGeneration layout
    base_model.model.language_model.model.layers.N...
so the adapter binds against the vLLM Qwen3_5MoeForConditionalGeneration module
tree WITHOUT relying on the runtime monkeypatch. This is exactly the transform
attention_only_lora_cutlass_moe._remap_key applies at load; producing it offline
lets us cross-check (orig + runtime-remap) == (offline-rekey) entirely in vLLM.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from safetensors.torch import load_file, save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nvfp4_lora.adapter_keys import wrapped_remap_safetensors_key as remap_key  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    ind, outd = Path(args.in_dir), Path(args.out_dir)
    outd.mkdir(parents=True, exist_ok=True)

    st = load_file(str(ind / "adapter_model.safetensors"))
    new, n = {}, 0
    for k, v in st.items():
        nk = remap_key(k)
        if nk != k:
            n += 1
        new[nk] = v
    save_file(new, str(outd / "adapter_model.safetensors"))

    for f in (
        "adapter_config.json",
        "chat_template.jinja",
        "tokenizer_config.json",
        "tokenizer.json",
    ):
        if (ind / f).exists():
            shutil.copy(ind / f, outd / f)
    print(f"remapped {n}/{len(st)} keys -> {outd}")


if __name__ == "__main__":
    main()
