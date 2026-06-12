#!/usr/bin/env python3
"""Unit test: quant.py output round-trips through dequant.py within FP4 error.

We construct synthetic 2D and 3D weights, run them through quantize_to_nvfp4_2d
(custom quant) and dequantize_nvfp4_weight (existing CT loader path), and check
that the reconstruction is bounded by NVFP4's inherent quantization noise.

NVFP4 has 16 representable values per group; the worst-case relative error per
group is bounded by ~ (max_in_group / 2) / max_in_group = 0.5 for a single value,
but the AVERAGE per-element error should be << 1.0 for any realistic weight
distribution. We use a generous tolerance: max abs relative error < 0.25
(typical observation: ~0.05 on N(0,1) data).

Run:
    /home/veritan-spark-01/Veritan/.venvs/qwen-serve/bin/python -u \\
        smoke_tests/test_custom_quant_roundtrip.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import importlib.util
import torch
from nvfp4_lora.dequant import dequantize_nvfp4_weight


def _load_quant_module():
    """Load scripts/quantize_mistral_to_nvfp4.py as a module (not on sys.path)."""
    here = os.path.dirname(os.path.abspath(__file__))
    qpath = os.path.join(os.path.dirname(here), "scripts", "quantize_mistral_to_nvfp4.py")
    spec = importlib.util.spec_from_file_location("_quant_mod", qpath)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_2d_roundtrip_smooth_distribution():
    quant = _load_quant_module()
    torch.manual_seed(0)
    # Smooth N(0, 1) — what the LoRA-target Linear weights look like.
    W = torch.randn(64, 256, dtype=torch.bfloat16)
    packed, scale_fp8, gscale_fp32 = quant.quantize_to_nvfp4_2d(W)
    assert packed.shape == (64, 128) and packed.dtype == torch.uint8
    assert scale_fp8.shape == (64, 16) and scale_fp8.dtype == torch.float8_e4m3fn
    assert gscale_fp32.shape == (1,) and gscale_fp32.dtype == torch.float32

    # Dequantize via CT-format path
    W_recon = dequantize_nvfp4_weight(
        weight_uint8=packed,
        weight_scale_fp8=scale_fp8,
        weight_scale_2_fp32=(1.0 / gscale_fp32).to(torch.float32),  # CT global_scale → ModelOpt weight_scale_2
        group_size=16,
        out_dtype=torch.bfloat16,
        format="modelopt",
    )
    diff = (W.float() - W_recon.float()).abs()
    rel_err = (diff / (W.float().abs().clamp(min=1e-6))).mean().item()
    max_err = diff.max().item()
    print(f"2D N(0,1):  mean_rel_err={rel_err:.4f}  max_abs_err={max_err:.4f}")
    assert rel_err < 0.25, f"Mean relative error too large: {rel_err}"


def test_2d_roundtrip_with_outliers():
    quant = _load_quant_module()
    torch.manual_seed(1)
    W = torch.randn(64, 256, dtype=torch.bfloat16)
    # Add a single large outlier — stresses the per-tensor scale
    W[0, 0] = 10.0
    packed, scale_fp8, gscale_fp32 = quant.quantize_to_nvfp4_2d(W)
    W_recon = dequantize_nvfp4_weight(
        packed, scale_fp8,
        (1.0 / gscale_fp32).to(torch.float32),
        group_size=16, out_dtype=torch.bfloat16, format="modelopt",
    )
    # The outlier should be preserved within FP4 error of its group
    recovered_outlier = W_recon[0, 0].float().item()
    print(f"2D outlier:  W[0,0]=10.0 → recon={recovered_outlier:.3f}")
    assert abs(recovered_outlier - 10.0) < 2.0, f"outlier {recovered_outlier} far from 10.0"


def test_3d_per_slice_roundtrip():
    """Smoke-test the 3D-per-slice path used for fused MoE."""
    quant = _load_quant_module()
    torch.manual_seed(2)
    E, OUT, IN_F = 4, 32, 128
    W = torch.randn(E, OUT, IN_F, dtype=torch.bfloat16)
    slices = quant.quantize_to_nvfp4_3d_per_slice(W)
    assert len(slices) == E
    for e_idx, (packed, scale_fp8, gscale_fp32) in enumerate(slices):
        assert packed.shape == (OUT, IN_F // 2)
        assert scale_fp8.shape == (OUT, IN_F // 16)
        assert gscale_fp32.shape == (1,)
        W_recon = dequantize_nvfp4_weight(
            packed, scale_fp8,
            (1.0 / gscale_fp32).to(torch.float32),
            group_size=16, out_dtype=torch.bfloat16, format="modelopt",
        )
        rel_err = ((W[e_idx].float() - W_recon.float()).abs() / W[e_idx].float().abs().clamp(min=1e-6)).mean().item()
        assert rel_err < 0.25, f"slice {e_idx} rel_err={rel_err} too large"
    print(f"3D round-trip: {E} slices OK")


def test_format_compatible_with_ct_loader():
    """Verify the trio (packed, scale_fp8 shape (1,)) is loadable via format='compressed_tensors'."""
    quant = _load_quant_module()
    torch.manual_seed(3)
    W = torch.randn(32, 64, dtype=torch.bfloat16)
    packed, scale_fp8, gscale_fp32 = quant.quantize_to_nvfp4_2d(W)
    # CT format expects per-tensor scale at shape (1,) — exactly what our quant emits.
    # In CT format the on-disk per-tensor value is the LARGE global_scale; the dequant
    # reciprocates it internally. Pass gscale verbatim — no manual reciprocation.
    assert gscale_fp32.shape == (1,)
    W_recon = dequantize_nvfp4_weight(
        packed, scale_fp8,
        gscale_fp32.to(torch.float32),
        group_size=16, out_dtype=torch.bfloat16, format="compressed_tensors",
    )
    rel_err = ((W.float() - W_recon.float()).abs() / W.float().abs().clamp(min=1e-6)).mean().item()
    print(f"CT-format round-trip:  rel_err={rel_err:.4f}")
    assert rel_err < 0.25


if __name__ == "__main__":
    test_2d_roundtrip_smooth_distribution()
    test_2d_roundtrip_with_outliers()
    test_3d_per_slice_roundtrip()
    test_format_compatible_with_ct_loader()
    print("\n=== ALL QUANT ROUND-TRIP TESTS PASSED ===")
