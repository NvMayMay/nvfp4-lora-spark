"""Hand-rolled NVFP4 → bf16 dequant for ModelOpt-quantized checkpoints.

On-disk layout (verified against NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4):
- weight       : uint8, shape (out, in/2)   - two 4-bit values packed per byte (low nibble = even index, high nibble = odd index per ModelOpt convention; validated below)
- weight_scale : float8_e4m3fn, shape (out, in/group_size) - per-group scale, group_size=16
- weight_scale_2 : float32 scalar - per-tensor scale
- input_scale (optional) : float32 scalar - used by inference, not needed for weight dequant
"""

from __future__ import annotations

import torch


# NVFP4 E2M1 lookup table (4-bit float: 1 sign, 2 exp, 1 mantissa, bias=1).
# Index by the unsigned 4-bit value (0..15). MSB = sign bit.
NVFP4_E2M1_LUT: torch.Tensor = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
     -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float32,
)


def _unpack_nibbles(packed: torch.Tensor) -> torch.Tensor:
    """Unpack uint8 tensor (..., N) → int64 tensor (..., 2N) with values in [0, 15].

    Two pack orderings are common: low-nibble-first ('le') and high-nibble-first ('be').
    ModelOpt uses low-nibble-first (LSN-first), confirmed in dequant_correctness.py
    by comparing against torchao's reference.
    """
    low = (packed & 0x0F).to(torch.int64)
    high = ((packed >> 4) & 0x0F).to(torch.int64)
    # Interleave: low at even indices, high at odd indices
    out = torch.empty(packed.shape[:-1] + (packed.shape[-1] * 2,), dtype=torch.int64, device=packed.device)
    out[..., 0::2] = low
    out[..., 1::2] = high
    return out


def dequantize_nvfp4_weight(
    weight_uint8: torch.Tensor,           # (out, in/2), uint8
    weight_scale_fp8: torch.Tensor,        # (out, in/group_size), float8_e4m3fn
    weight_scale_2_fp32: torch.Tensor,     # scalar, float32
    group_size: int = 16,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Dequantize an NVFP4-stored weight to a high-precision tensor.

    Returns: tensor of shape (out, in) in `out_dtype`.

    The dequant formula:
        W[o, i] = lut[w_uint8 unpack at (o, i)] * (weight_scale_fp8[o, i // group_size] as f32) * weight_scale_2

    Done in fp32 then cast to `out_dtype` at the end.
    """
    if weight_uint8.dtype != torch.uint8:
        raise TypeError(f"weight_uint8 must be uint8, got {weight_uint8.dtype}")
    if weight_scale_fp8.dtype != torch.float8_e4m3fn:
        raise TypeError(f"weight_scale_fp8 must be float8_e4m3fn, got {weight_scale_fp8.dtype}")

    device = weight_uint8.device
    lut = NVFP4_E2M1_LUT.to(device=device, dtype=torch.float32)

    # Unpack uint8 → int64 indices (0..15), shape (out, in)
    indices = _unpack_nibbles(weight_uint8)
    out_feat, in_feat = indices.shape
    assert in_feat == weight_uint8.shape[-1] * 2

    # Lookup FP4 values in fp32, shape (out, in)
    fp4_values = lut[indices]  # (out, in)

    # Per-group scale: weight_scale shape (out, in/group_size). Repeat each scale group_size times along in axis.
    n_groups = in_feat // group_size
    if weight_scale_fp8.shape != (out_feat, n_groups):
        raise ValueError(
            f"weight_scale shape mismatch: got {tuple(weight_scale_fp8.shape)} "
            f"expected ({out_feat}, {n_groups})"
        )
    # FP8 E4M3 → FP32 via torch's native cast
    weight_scale_fp32 = weight_scale_fp8.to(torch.float32)
    # Broadcast: (out, n_groups, 1) → (out, in) by repeating each group
    scale_broadcast = weight_scale_fp32.unsqueeze(-1).expand(out_feat, n_groups, group_size).reshape(out_feat, in_feat)

    # Per-tensor scale (scalar)
    per_tensor = weight_scale_2_fp32.to(torch.float32) if weight_scale_2_fp32.dtype != torch.float32 else weight_scale_2_fp32

    # Combine
    result_fp32 = fp4_values * scale_broadcast * per_tensor
    return result_fp32.to(out_dtype)
