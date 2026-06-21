#!/usr/bin/env python3
"""Measure LoRA delta magnitude relative to the base weights (delta_to_base_ratio).

Predicts merge-for-serve viability on a 4-bit base: NVFP4 re-quant adds ~9%
per-element noise, so a delta well below that floor gets rounded away on merge
(measured: a rank-8 / 0.85%-delta adapter lost ~98% of its effect through merge,
see measured_evidence.md). Same per-element metric the merge tool logs, so it is
directly comparable to that 0.85%.

Supports a bf16 base directly; an NVFP4 base is dequantized via ModelOpt.
"""
from __future__ import annotations
import argparse
import glob
import json
import re
import sys
from pathlib import Path

import torch
from safetensors import safe_open


def adapter_base_key(akey: str) -> str:
    t = akey
    if t.startswith("base_model.model."):
        t = t[len("base_model.model."):]
    return re.sub(r"\.lora_[AB]\.weight$", ".weight", t)


def get_base_weight(model_dir: Path, key: str, weight_map: dict, dev):
    shard = model_dir / weight_map[key]
    with safe_open(shard, framework="pt") as sf:
        W = sf.get_tensor(key)
        if key.endswith(".weight") and (key.replace(".weight", ".weight_scale_2") in weight_map):
            # NVFP4 base: dequant via ModelOpt
            from modelopt.torch.quantization.qtensor.nvfp4_tensor import NVFP4QTensor
            gs = get_raw(model_dir, key.replace(".weight", ".weight_scale"), weight_map)
            pts = get_raw(model_dir, key.replace(".weight", ".weight_scale_2"), weight_map)
            qt = NVFP4QTensor((W.shape[0], W.shape[1] * 2), torch.bfloat16, W.to(dev))
            return qt.dequantize(scale=gs.to(dev), double_scale=pts.to(dev), block_sizes={-1: 16}).float()
    return W.to(dev, torch.float32)


def get_raw(model_dir: Path, key: str, weight_map: dict):
    with safe_open(model_dir / weight_map[key], framework="pt") as sf:
        return sf.get_tensor(key)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model-dir", required=True, type=Path)
    ap.add_argument("--adapter-dir", required=True, type=Path)
    ap.add_argument("--n", type=int, default=0, help="sample N modules evenly (0=all)")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    dev = torch.device(args.device)

    cfg = json.load(open(args.adapter_dir / "adapter_config.json"))
    scale = cfg["lora_alpha"] / cfg["r"]
    print(f"[delta] r={cfg['r']} alpha={cfg['lora_alpha']} scale={scale}")

    AB: dict[str, dict] = {}
    for f in sorted(glob.glob(str(args.adapter_dir / "adapter_model*.safetensors"))):
        with safe_open(f, framework="pt") as sf:
            for k in sf.keys():
                m = re.search(r"\.lora_([AB])\.weight$", k)
                if not m:
                    continue
                AB.setdefault(adapter_base_key(k), {})[m.group(1)] = sf.get_tensor(k)
    mods = [k for k, v in AB.items() if "A" in v and "B" in v]
    if args.n:
        mods = mods[:: max(1, len(mods) // args.n)][:args.n]
    print(f"[delta] {len(mods)} modules to measure")

    weight_map = json.load(open(args.base_model_dir / "model.safetensors.index.json"))["weight_map"]
    is_nvfp4 = any(k.endswith(".weight_scale_2") for k in weight_map)

    def resolve(bk):
        # Handle the multimodal `language_model.` re-key: adapter keys may omit the
        # `language_model.` segment the base nests the LM under.
        for c in (bk,
                  bk.replace("model.layers.", "model.language_model.layers.", 1),
                  bk.replace("model.", "model.language_model.", 1)):
            if c in weight_map:
                return c
        return None

    results = []
    misses = 0
    for bk in mods:
        rk = resolve(bk)
        if rk is None:
            misses += 1
            if misses <= 3:
                print(f"  MISS base key: {bk}")
            continue
        W = get_base_weight(args.base_model_dir, rk, weight_map, dev)
        A = AB[bk]["A"].to(dev, torch.float32)
        B = AB[bk]["B"].to(dev, torch.float32)
        delta = scale * (B @ A)
        if delta.shape != W.shape:
            print(f"  SHAPE mismatch {bk}: delta{tuple(delta.shape)} W{tuple(W.shape)}")
            continue
        fro = (delta.norm() / W.norm()).item()
        elem = (delta.abs().mean() / W.abs().mean()).item()
        results.append((elem, fro, bk))

    if not results:
        print("[delta] no modules measured (all missed?) -- check key mapping")
        return
    results.sort()
    elems = [r[0] for r in results]
    fros = [r[1] for r in results]

    def pct(x, p):
        return x[min(len(x) - 1, int(p * len(x)))]

    print(f"\n[delta] measured {len(results)} modules (base: {'NVFP4' if is_nvfp4 else 'bf16'}), misses={misses}")
    print(f"  delta_to_base PER-ELEM (== merge tool's metric): "
          f"p10={pct(elems,.1):.4f} p50={pct(elems,.5):.4f} p90={pct(elems,.9):.4f} max={elems[-1]:.4f}")
    print(f"  delta_to_base FROBENIUS: "
          f"p10={pct(fros,.1):.4f} p50={pct(sorted(fros),.5):.4f} p90={pct(sorted(fros),.9):.4f} max={max(fros):.4f}")
    print(f"  --> NVFP4 re-quant per-elem noise floor ~0.09. r=8 smoke was 0.0085 (lost ~98%).")
    print(f"  smallest: {[(round(r[0],4), r[2].split('layers.')[-1]) for r in results[:3]]}")
    print(f"  largest : {[(round(r[0],4), r[2].split('layers.')[-1]) for r in results[-3:]]}")


if __name__ == "__main__":
    main()
