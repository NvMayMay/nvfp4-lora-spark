#!/usr/bin/env python3
"""Make a ModelOpt/compressed-tensors NVFP4 checkpoint vLLM-loadable by
dequantizing a quantized `lm_head` to bf16 and dropping its scale tensors.

vLLM keeps `lm_head` in bf16 by class, so a checkpoint that quantized it (NVFP4
`lm_head.weight` + `weight_scale` + `weight_scale_2` [+ `input_scale`]) crashes
vLLM at load with "no module or parameter named lm_head.input_scale". This
rewrites, in place, the single shard that holds the `lm_head.*` tensors:
`lm_head.weight` (packed uint8) -> bf16, and the scale tensors removed; the
index weight_map drops the scale keys. vLLM reads shard CONTENTS, not just the
index, so the scales must leave the shard, not merely the map (notebook §7h).

Backups: `<shard>.bak` and `model.safetensors.index.json.orig`.
Idempotent: a no-op if `lm_head` is already bf16. Default is --dry-run (prints
the plan, writes nothing); pass --apply to execute.

Encodes the manual fix proven on nvidia/Qwen3.6-35B-A3B-NVFP4. The destructive
--apply path mirrors that proven sequence but is not re-runnable for regression
here (no un-fixed checkpoint remains); always --dry-run and keep the backups.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from nybbloris.plan import lm_head_status  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True, type=Path)
    ap.add_argument("--apply", action="store_true", help="write changes (default: dry-run)")
    args = ap.parse_args()
    md = args.model_dir

    st = lm_head_status(md)
    print(f"[lm_head] {st['note']}")
    print(f"[lm_head] scale keys present: {st['scale_keys']}")
    if not st["quantized"]:
        print("[lm_head] already bf16 / vLLM-loadable -- nothing to do.")
        return 0

    idx_path = md / "model.safetensors.index.json"
    idx = json.load(open(idx_path))
    wm = idx["weight_map"]
    lh_keys = sorted(k for k in wm if k.startswith("lm_head."))
    shards = {wm[k] for k in lh_keys}
    if len(shards) != 1:
        print(f"[lm_head] ERROR: lm_head tensors span multiple shards {shards}; "
              "this fixer assumes they are co-located. Resolve manually.")
        return 2
    shard_name = wm["lm_head.weight"]
    shard_path = md / shard_name
    drop = [k for k in lh_keys if k != "lm_head.weight"]
    print(f"[lm_head] shard: {shard_name}  | lm_head tensors: {lh_keys}")
    print(f"[lm_head] plan: dequant lm_head.weight NVFP4 -> bf16; drop {drop} from shard + index")

    if not args.apply:
        print("[lm_head] DRY-RUN: pass --apply to write (backups: <shard>.bak, index.json.orig).")
        return 0

    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file
    from nvfp4_lora.dequant import dequantize_nvfp4_weight

    with safe_open(str(shard_path), "pt") as f:
        shard = {k: f.get_tensor(k) for k in f.keys()}
    w, ws, ws2 = shard["lm_head.weight"], shard["lm_head.weight_scale"], shard["lm_head.weight_scale_2"]
    out_f, in_f = w.shape[0], w.shape[1] * 2
    print(f"[lm_head] dequant (out={out_f}, in={in_f}) ...")
    w_bf16 = dequantize_nvfp4_weight(w, ws, ws2, group_size=16,
                                     out_dtype=torch.bfloat16, format="modelopt").contiguous()

    new_shard = {k: v for k, v in shard.items()
                 if not (k.startswith("lm_head.") and k != "lm_head.weight")}
    new_shard["lm_head.weight"] = w_bf16
    print(f"[lm_head] shard tensors: {len(shard)} -> {len(new_shard)}")

    if not (md / "model.safetensors.index.json.orig").exists():
        shutil.copy2(idx_path, md / "model.safetensors.index.json.orig")
    bak = shard_path.with_suffix(shard_path.suffix + ".bak")
    if not bak.exists():
        shutil.copy2(shard_path, bak)

    save_file(new_shard, str(shard_path), metadata={"format": "pt"})
    for k in drop:
        wm.pop(k, None)
    json.dump(idx, open(idx_path, "w"), indent=2)

    st2 = lm_head_status(md)
    print(f"[lm_head] DONE: {st2['note']}")
    return 0 if not st2["quantized"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
