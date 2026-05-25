#!/usr/bin/env python3
"""Day 1 second-half smoke: NVFP4LoRALinear forward + backward sanity.

Verifies:
1. Construction from real Nano on-disk tensors works.
2. Forward produces finite output of the right shape.
3. Forward output matches a "dequant then F.linear" reference (since the base path
   is exactly that).
4. Backward through the base path produces non-zero dx.
5. LoRA params receive gradients; base buffers do not.
6. With LoRA r=0 (frozen-only), backward still works and lora params don't exist.
"""
from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json


LAYER_PREFIX = "backbone.layers.0.mixer.in_proj"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run NVFP4LoRALinear forward and backward smoke checks on one Nano layer.",
    )
    parser.add_argument(
        "--model-dir",
        default=os.environ.get("NVFP4_SMOKE_MODEL_DIR"),
        help="Path to Nemotron-3-Nano-30B-A3B-NVFP4. Can also be set via NVFP4_SMOKE_MODEL_DIR.",
    )
    args = parser.parse_args()
    if not args.model_dir:
        parser.print_usage(sys.stderr)
        print(
            "error: provide --model-dir /path/to/Nemotron-3-Nano-30B-A3B-NVFP4 "
            "or set NVFP4_SMOKE_MODEL_DIR",
            file=sys.stderr,
        )
        raise SystemExit(2)
    if not os.path.exists(os.path.join(args.model_dir, "model.safetensors.index.json")):
        print(
            f"error: no model.safetensors.index.json under {args.model_dir}; check --model-dir",
            file=sys.stderr,
        )
        raise SystemExit(2)
    return args


def load_tensors(model_dir: str, prefix: str) -> dict[str, torch.Tensor]:
    idx = json.load(open(f"{model_dir}/model.safetensors.index.json"))
    wm = idx["weight_map"]
    record = {}
    for suffix in ("weight", "weight_scale", "weight_scale_2"):
        key = f"{prefix}.{suffix}"
        shard = wm[key]
        with safetensors.safe_open(f"{model_dir}/{shard}", framework="pt", device="cpu") as f:
            record[key] = f.get_tensor(key)
    return record


def main():
    args = parse_args()
    global torch, F, safetensors, dequantize_nvfp4_weight, NVFP4LoRALinear
    import torch
    import torch.nn.functional as F
    import safetensors
    from nvfp4_lora.dequant import dequantize_nvfp4_weight
    from nvfp4_lora.linear import NVFP4LoRALinear

    print("=== Day 1 NVFP4LoRALinear smoke ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    record = load_tensors(args.model_dir, LAYER_PREFIX)
    weight_u8 = record[f"{LAYER_PREFIX}.weight"]
    out_feat = weight_u8.shape[0]      # 10304
    in_feat = weight_u8.shape[1] * 2   # 2688

    # Build module with LoRA r=8
    print(f"\n--- construction (in={in_feat}, out={out_feat}, r=8) ---")
    mod = NVFP4LoRALinear.from_safetensors_record(
        record, prefix=LAYER_PREFIX,
        in_features=in_feat, out_features=out_feat,
        r=8, lora_alpha=16, device=device, dtype=torch.bfloat16,
    )
    print(f"  {mod}")
    print(f"  trainable params: {sum(p.numel() for p in mod.parameters() if p.requires_grad)}")
    print(f"  buffer params:    {sum(b.numel() for b in mod.buffers())}")

    # Forward
    print("\n--- forward ---")
    torch.manual_seed(0)
    x = torch.randn(2, 64, in_feat, dtype=torch.bfloat16, device=device, requires_grad=True)
    y = mod(x)
    print(f"  x shape={tuple(x.shape)} y shape={tuple(y.shape)}")
    assert y.shape == (2, 64, out_feat), f"unexpected output shape {y.shape}"
    assert torch.isfinite(y).all(), "non-finite output"
    print(f"  y finite, y stats: min={y.min().item():.4f} max={y.max().item():.4f} mean={y.mean().item():.4f} std={y.std().item():.4f}")

    # Reference: with LoRA r=0 (lora delta starts at zero with B init), forward should equal F.linear(x, W_bf16)
    print("\n--- reference vs forward (LoRA at init is no-op since B=0) ---")
    W_bf16 = dequantize_nvfp4_weight(
        weight_u8, record[f"{LAYER_PREFIX}.weight_scale"], record[f"{LAYER_PREFIX}.weight_scale_2"],
        out_dtype=torch.bfloat16,
    ).to(device)
    y_ref = F.linear(x.detach(), W_bf16)
    cos = F.cosine_similarity(y.detach().flatten().float(), y_ref.flatten().float(), dim=0).item()
    rms_rel = ((y.detach() - y_ref).float().pow(2).mean().sqrt() / y_ref.float().pow(2).mean().sqrt()).item()
    print(f"  cos={cos:.8f} rms_rel={rms_rel:.4e}")
    assert cos > 0.9999, f"forward diverged from reference: cos={cos}"
    print(f"  PASS cos > 0.9999")

    # Backward
    print("\n--- backward (LoRA grads) ---")
    loss = y.sum()
    loss.backward()
    assert mod.lora_A.grad is not None, "lora_A.grad is None"
    assert mod.lora_B.grad is not None, "lora_B.grad is None"
    assert torch.isfinite(mod.lora_A.grad).all(), "lora_A.grad has non-finite"
    assert torch.isfinite(mod.lora_B.grad).all(), "lora_B.grad has non-finite"
    # lora_A.grad nonzero only if x is nonzero AND lora_B is nonzero (it's zero-init). So lora_A.grad SHOULD be all-zero at init!
    # lora_B.grad nonzero because x flows through lora_A which is Kaiming-init.
    print(f"  lora_A.grad nonzero count: {(mod.lora_A.grad != 0).sum().item()} / {mod.lora_A.grad.numel()}")
    print(f"  lora_B.grad nonzero count: {(mod.lora_B.grad != 0).sum().item()} / {mod.lora_B.grad.numel()}")
    print(f"  lora_B.grad std: {mod.lora_B.grad.std().item():.4e}")
    assert mod.lora_B.grad.abs().sum().item() > 0, "lora_B.grad is exactly zero - LoRA path not flowing gradient"

    # dx grad
    assert x.grad is not None, "x.grad is None"
    assert torch.isfinite(x.grad).all(), "x.grad non-finite"
    print(f"  x.grad shape={tuple(x.grad.shape)} std={x.grad.std().item():.4e}")
    assert x.grad.std().item() > 0, "x.grad is all zero - backward through frozen base broken"

    # No grad on buffers
    for name, buf in mod.named_buffers():
        assert not buf.requires_grad, f"buffer {name} has requires_grad=True"
    print(f"  PASS: lora_A grad zero at init (correct), lora_B grad nonzero, x.grad nonzero, no buffer grads")

    # Test r=0 (frozen-only) path
    print("\n--- frozen-only (r=0) path ---")
    mod_frozen = NVFP4LoRALinear.from_safetensors_record(
        record, prefix=LAYER_PREFIX,
        in_features=in_feat, out_features=out_feat,
        r=0, device=device, dtype=torch.bfloat16,
    )
    assert sum(p.numel() for p in mod_frozen.parameters() if p.requires_grad) == 0
    x2 = torch.randn(1, 16, in_feat, dtype=torch.bfloat16, device=device, requires_grad=True)
    y2 = mod_frozen(x2)
    y2.sum().backward()
    assert x2.grad is not None and torch.isfinite(x2.grad).all() and x2.grad.std().item() > 0
    print(f"  PASS: r=0 frozen-only path has 0 trainable params and still produces x.grad")

    print("\n=== Day 1 NVFP4LoRALinear smoke PASS ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
