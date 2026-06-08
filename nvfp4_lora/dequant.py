"""Hand-rolled NVFP4 -> bf16 dequant for ModelOpt and compressed-tensors checkpoints.

ModelOpt on-disk layout:
- weight       : uint8, shape (out, in/2) - two 4-bit values packed per byte
- weight_scale : float8_e4m3fn, shape (out, in/group_size) - per-group scale
- weight_scale_2 : float32 scalar - per-tensor scale
- input_scale (optional) : float32 scalar - used by inference, not needed for weight dequant

compressed-tensors on-disk layout:
- weight_packed        : uint8, shape (out, in/2) - same low-nibble-first packing
- weight_scale         : float8_e4m3fn, shape (out, in/group_size)
- weight_global_scale  : float32, shape (1,) - per-tensor scale
- input_global_scale   : float32, shape (1,) - used by inference, not needed for weight dequant
"""

from __future__ import annotations

from typing import Literal

import torch


NVFP4Format = Literal["modelopt", "compressed_tensors"]


# NVFP4 E2M1 lookup table (4-bit float: 1 sign, 2 exp, 1 mantissa, bias=1).
# Index by the unsigned 4-bit value (0..15). MSB = sign bit.
NVFP4_E2M1_LUT: torch.Tensor = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
     -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float32,
)


def format_for_record(safetensors_keys: set[str], prefix: str) -> NVFP4Format:
    """Return the NVFP4 checkpoint format used by one safetensors module prefix."""
    has_ct = f"{prefix}.weight_packed" in safetensors_keys
    has_modelopt = f"{prefix}.weight" in safetensors_keys
    if has_ct == has_modelopt:
        if has_ct:
            raise ValueError(f"{prefix} has both ModelOpt and compressed-tensors weight keys")
        raise ValueError(f"{prefix} has neither ModelOpt nor compressed-tensors weight keys")
    return "compressed_tensors" if has_ct else "modelopt"


def _unpack_nibbles(packed: torch.Tensor) -> torch.Tensor:
    """Unpack uint8 tensor (..., N) -> int64 tensor (..., 2N) with values in [0, 15].

    ModelOpt and compressed-tensors NVFP4 both use low-nibble-first packing.
    """
    low = (packed & 0x0F).to(torch.int64)
    high = ((packed >> 4) & 0x0F).to(torch.int64)
    out = torch.empty(packed.shape[:-1] + (packed.shape[-1] * 2,), dtype=torch.int64, device=packed.device)
    out[..., 0::2] = low
    out[..., 1::2] = high
    return out


def dequantize_nvfp4_weight(
    weight_uint8: torch.Tensor,           # (out, in/2), uint8
    weight_scale_fp8: torch.Tensor,        # (out, in/group_size), float8_e4m3fn
    weight_scale_2_fp32: torch.Tensor,     # scalar or shape-(1,), float32
    group_size: int = 16,
    out_dtype: torch.dtype = torch.bfloat16,
    out: torch.Tensor | None = None,
    format: NVFP4Format = "modelopt",
) -> torch.Tensor:
    """Dequantize an NVFP4-stored weight to a high-precision tensor.

    Returns: tensor of shape (out, in) in `out_dtype`.

    The dequant formula:
        W[o, i] = lut[w_uint8 unpack at (o, i)] * (weight_scale_fp8[o, i // group_size] as f32) * weight_scale_2

    Done in fp32 then cast to `out_dtype` at the end. If `out` is provided,
    the final cast result is written into that tensor in-place and returned.
    """
    if format not in ("modelopt", "compressed_tensors"):
        raise ValueError(f"format must be 'modelopt' or 'compressed_tensors', got {format!r}")
    if weight_uint8.dtype != torch.uint8:
        raise TypeError(f"weight_uint8 must be uint8, got {weight_uint8.dtype}")
    if weight_uint8.ndim != 2:
        raise ValueError(f"weight_uint8 must have shape (out, in/2), got {tuple(weight_uint8.shape)}")
    if weight_scale_fp8.dtype != torch.float8_e4m3fn:
        raise TypeError(f"weight_scale_fp8 must be float8_e4m3fn, got {weight_scale_fp8.dtype}")
    if weight_scale_fp8.ndim != 2:
        raise ValueError(f"weight_scale_fp8 must have shape (out, in/group_size), got {tuple(weight_scale_fp8.shape)}")
    if weight_scale_2_fp32.dtype != torch.float32:
        raise TypeError(f"weight_scale_2_fp32 must be float32, got {weight_scale_2_fp32.dtype}")

    # Accept either scalar () (ModelOpt) or shape-(1,) (compressed-tensors) — the math
    # below reshapes to scalar either way. Relaxed from the strict per-format check so
    # cross-format call sites (e.g. NVFP4LoRALinear's eval-mode dequant path) work
    # without each having to thread `format=` through explicitly.
    if tuple(weight_scale_2_fp32.shape) not in ((), (1,)):
        raise ValueError(
            f"per-tensor scale must be scalar () or shape (1,); got {tuple(weight_scale_2_fp32.shape)}"
        )

    device = weight_uint8.device
    out_feat = weight_uint8.shape[0]
    in_feat = weight_uint8.shape[1] * 2

    if in_feat % group_size != 0:
        raise ValueError(f"unpacked input dim {in_feat} is not divisible by group_size={group_size}")
    n_groups = in_feat // group_size
    if weight_scale_fp8.shape != (out_feat, n_groups):
        raise ValueError(
            f"weight_scale shape mismatch: got {tuple(weight_scale_fp8.shape)} "
            f"expected ({out_feat}, {n_groups})"
        )

    # Compute effective per-tensor scale once. Both formats end up as a single
    # fp32 multiplier the kernel applies on top of group_scale * fp4_val.
    per_tensor_raw = weight_scale_2_fp32.reshape(()).to(torch.float32)
    if format == "modelopt":
        per_tensor_effective = per_tensor_raw
    else:
        per_tensor_effective = 1.0 / per_tensor_raw.clamp(min=1e-30)

    # Triton fast path — 18-28x faster than the eager PyTorch chain below.
    # Falls back to PyTorch when Triton isn't importable or the tensor lives on CPU.
    if device.type == "cuda":
        from .triton_dequant import triton_available, triton_dequant_nvfp4
        if triton_available() and out_dtype == torch.bfloat16:
            weight_scale_fp32 = weight_scale_fp8.to(torch.float32)
            return triton_dequant_nvfp4(
                weight_uint8,
                weight_scale_fp32,
                per_tensor_effective.reshape(1),
                group_size=group_size,
                out=out,
            )

    lut = NVFP4_E2M1_LUT.to(device=device, dtype=torch.float32)

    # Unpack uint8 -> int64 indices (0..15), shape (out, in)
    indices = _unpack_nibbles(weight_uint8)
    assert indices.shape == (out_feat, in_feat)

    # Lookup FP4 values in fp32, shape (out, in)
    fp4_values = lut[indices]
    weight_scale_fp32 = weight_scale_fp8.to(torch.float32)

    result_fp32 = (
        fp4_values.view(out_feat, n_groups, group_size)
        * weight_scale_fp32.unsqueeze(-1)
        * per_tensor_effective
    )
    result = result_fp32.reshape(out_feat, in_feat)
    if out is not None:
        expected_shape = (out_feat, in_feat)
        if tuple(out.shape) != expected_shape or out.dtype != out_dtype:
            raise ValueError(
                "out shape/dtype mismatch: "
                f"got shape={tuple(out.shape)} dtype={out.dtype}, "
                f"expected shape={expected_shape} dtype={out_dtype}"
            )
        out.copy_(result)
        return out
    return result.to(out_dtype)
