#!/usr/bin/env python3
"""Check that the repo's Triton NVFP4 dequant agrees with ModelOpt's dequant.

The merge tool dequantizes/re-quantizes with ModelOpt's NVFP4QTensor, but the
training/serving forward path dequantizes with the repo's Triton kernel
(nvfp4_lora.dequant.dequantize_nvfp4_weight). If those two disagree on the same
NVFP4 bytes+scales, then a merged model (written by ModelOpt) read back by Triton
is systematically off on every weight -- which would inflate any merged-vs-
reference forward divergence as a TOOLING artifact rather than a real merge cost.

Compares both kernels on a sample of expert weights from the given model dir.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import torch
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from nvfp4_lora.dequant import dequantize_nvfp4_weight  # noqa: E402


def modelopt_dequant(packed, gs, pts):
    from modelopt.torch.quantization.qtensor.nvfp4_tensor import NVFP4QTensor
    qt = NVFP4QTensor((packed.shape[0], packed.shape[1] * 2), torch.bfloat16, packed)
    return qt.dequantize(scale=gs, double_scale=pts, block_sizes={-1: 16})


def main():
    model_dir = Path(sys.argv[1])
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    device = "cuda"
    weight_map = json.load(open(model_dir / "model.safetensors.index.json"))["weight_map"]
    keys = [k for k in weight_map if ".experts." in k and k.endswith(".up_proj.weight")][:n]
    print(f"[dequant-check] {model_dir.name}: comparing Triton vs ModelOpt on {len(keys)} expert weights")

    def get(key):
        with safe_open(model_dir / weight_map[key], framework="pt") as f:
            return f.get_tensor(key)

    cosines, rels = [], []
    for k in keys:
        sk = k.replace(".weight", ".weight_scale")
        s2 = k.replace(".weight", ".weight_scale_2")
        w = get(k).to(device)
        gs = get(sk).to(device)
        pts = get(s2).to(device)
        dm = modelopt_dequant(w, gs, pts).float()
        dt = dequantize_nvfp4_weight(w, gs, pts).float()
        cos = torch.nn.functional.cosine_similarity(dm.flatten(), dt.flatten(), dim=0).item()
        rel = ((dm - dt).abs() / (dm.abs() + 1e-6)).mean().item()
        maxd = (dm - dt).abs().max().item()
        cosines.append(cos)
        rels.append(rel)
        print(f"  {k.split('backbone.')[-1]}: cos={cos:.6f} mean_rel={rel:.4f} maxabs={maxd:.4f} "
              f"| modelopt_absmean={dm.abs().mean():.4f} triton_absmean={dt.abs().mean():.4f}")

    print(f"\n[dequant-check] cosine: min={min(cosines):.6f} mean={sum(cosines)/len(cosines):.6f}")
    print(f"[dequant-check] mean_rel: max={max(rels):.4f} mean={sum(rels)/len(rels):.4f}")
    if min(cosines) > 0.9999 and max(rels) < 1e-3:
        print("[dequant-check] VERDICT: Triton == ModelOpt dequant (consistent).")
    else:
        print("[dequant-check] VERDICT: Triton != ModelOpt dequant -- INCONSISTENT "
              "(explains/inflates merged-vs-reference forward divergence).")


if __name__ == "__main__":
    main()
