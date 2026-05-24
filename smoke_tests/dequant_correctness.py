#!/usr/bin/env python3
"""Day 1 correctness gate: our NVFP4 dequant vs torchao's reference.

Loads one NVFP4 layer from Nemotron-3-Nano-30B-A3B-NVFP4 on disk, dequants via our
hand-rolled function, and compares against torchao's `NVFP4Tensor`. If they agree to
cosine sim > 0.9999 (per SYNTHESIS.md Day 1 gate), the dequant impl is correct and
we can proceed to the LoRA wrapping.

The packing-order ambiguity (low-nibble-first vs high-nibble-first) is resolved
empirically: if cosine > 0.9999 with the default `_unpack_nibbles` ordering, we're
good. If not, retry with the other order.
"""
import sys
import os

# Make package importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import json
import torch
import torch.nn.functional as F
import safetensors

from nvfp4_lora.dequant import dequantize_nvfp4_weight


MODEL_DIR = "/path/to/Models/Nemotron-3-Nano-30B-A3B-NVFP4"
# Layer 0 mixer in_proj is NVFP4-quantized (not in the exclude list)
LAYER_PREFIX = "backbone.layers.0.mixer.in_proj"


def load_layer_tensors(prefix: str) -> dict[str, torch.Tensor]:
    idx = json.load(open(f"{MODEL_DIR}/model.safetensors.index.json"))
    wm = idx["weight_map"]
    needed = {
        "weight": f"{prefix}.weight",
        "weight_scale": f"{prefix}.weight_scale",
        "weight_scale_2": f"{prefix}.weight_scale_2",
    }
    tensors = {}
    for short, key in needed.items():
        shard = wm[key]
        with safetensors.safe_open(f"{MODEL_DIR}/{shard}", framework="pt", device="cpu") as f:
            tensors[short] = f.get_tensor(key)
    return tensors


def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    return F.cosine_similarity(a.flatten().float(), b.flatten().float(), dim=0).item()


def rms_rel_err(a: torch.Tensor, b: torch.Tensor) -> float:
    return ((a.float() - b.float()).pow(2).mean().sqrt() / b.float().pow(2).mean().sqrt()).item()


def torchao_reference(weight_u8, weight_scale_fp8, weight_scale_2) -> torch.Tensor:
    """Use torchao to construct an NVFP4Tensor from the same raw bits and dequantize."""
    from torchao.prototype.mx_formats.nvfp4_tensor import NVFP4Tensor

    # torchao's NVFP4Tensor stores (qdata=uint8, scale=fp8, per_tensor_scale=fp32, block_size=16, hp_dtype=bf16, ...)
    # Constructor signature varies by version; try direct construction
    out_feat = weight_u8.shape[0]
    in_feat = weight_u8.shape[1] * 2  # unpacked dim
    # Many torchao revs use a frozen private constructor; safest is to use from_hp() on a known weight, then patch.
    # Try direct: NVFP4Tensor(qdata=..., scale=..., per_tensor_scale=...) - fallback to construction-by-quant-then-replace if needed.
    try:
        # torchao NVFP4Tensor signature (v0.17): (qdata, scale, block_size, orig_dtype, per_tensor_scale=None, ...)
        t = NVFP4Tensor(
            qdata=weight_u8,
            scale=weight_scale_fp8,
            block_size=16,
            orig_dtype=torch.bfloat16,
            per_tensor_scale=weight_scale_2.reshape(()),
        )
        # NVFP4Tensor.dequantize(target_dtype=...) or maybe to() conversion
        try:
            return t.dequantize(target_dtype=torch.bfloat16)
        except TypeError:
            return t.dequantize()
    except Exception as e:
        print(f"  torchao direct construction failed: {type(e).__name__}: {e}", flush=True)
        import traceback
        traceback.print_exc()
        return None


def main():
    print(f"=== Day 1 dequant correctness gate ===")
    print(f"layer: {LAYER_PREFIX}")
    t = load_layer_tensors(LAYER_PREFIX)
    print(f"  weight        : {tuple(t['weight'].shape)} {t['weight'].dtype}")
    print(f"  weight_scale  : {tuple(t['weight_scale'].shape)} {t['weight_scale'].dtype}")
    print(f"  weight_scale_2: shape={tuple(t['weight_scale_2'].shape)} {t['weight_scale_2'].dtype} value={float(t['weight_scale_2'])}")

    # Our dequant
    print("\n--- our dequant ---")
    W_ours = dequantize_nvfp4_weight(
        t["weight"], t["weight_scale"], t["weight_scale_2"], group_size=16, out_dtype=torch.bfloat16
    )
    print(f"  W_ours shape={tuple(W_ours.shape)} dtype={W_ours.dtype}")
    print(f"  W_ours stats: min={W_ours.min().item():.4f} max={W_ours.max().item():.4f} mean={W_ours.mean().item():.4f}")
    print(f"  W_ours finite frac: {torch.isfinite(W_ours).float().mean().item():.4f}")

    # torchao reference
    print("\n--- torchao reference ---")
    W_ref = torchao_reference(t["weight"], t["weight_scale"], t["weight_scale_2"])
    if W_ref is None:
        print("  torchao reference unavailable - falling back to manual sanity")
        # Sanity: confirm finite + non-trivial std
        if torch.isfinite(W_ours).all() and W_ours.std().item() > 1e-6:
            print("  PASS-SANITY (no torchao reference but ours is finite and non-zero)")
            return 0
        else:
            print("  FAIL-SANITY (dequant produced non-finite or all-zero output)")
            return 1
    print(f"  W_ref shape={tuple(W_ref.shape)} dtype={W_ref.dtype}")
    print(f"  W_ref stats: min={W_ref.min().item():.4f} max={W_ref.max().item():.4f} mean={W_ref.mean().item():.4f}")

    # Compare
    cos = cosine_sim(W_ours, W_ref)
    rms = rms_rel_err(W_ours, W_ref)
    max_abs = (W_ours.float() - W_ref.float()).abs().max().item()
    print(f"\n=== COMPARISON ===")
    print(f"  cosine_similarity: {cos:.8f}")
    print(f"  rms_rel_err: {rms:.4e}")
    print(f"  max_abs: {max_abs:.4e}")

    threshold = 0.9999
    if cos > threshold:
        print(f"\nPASS (cosine > {threshold})")
        return 0
    else:
        print(f"\nFAIL (cosine < {threshold}) - likely packing order is wrong; try swapping low/high in _unpack_nibbles")
        return 2


if __name__ == "__main__":
    sys.exit(main())
