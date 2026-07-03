"""bf16 -> NVFP4 weight quantization, pure torch, CPU and CUDA capable.

This is the importable home for the quantization logic that used to live in two
scripts:
  - scripts/quantize_mistral_to_nvfp4.py  (compressed-tensors "ct" layout)
  - scripts/merge_lora_into_nvfp4.py       (ModelOpt layout, via the modelopt lib)

The two on-disk layouts differ, and so do the numerical conventions the two
sources used. Both are preserved here behind the `layout=` switch so this is a
faithful port, not a redesign. See nvfp4_lora/dequant.py for the inverse.

Output key naming (the caller builds the full tensor keys):
  ct       -> weight_packed (uint8, (out, in/2)),
              weight_scale (float8_e4m3fn, (out, in/group_size)),
              weight_global_scale (float32, (1,))   [a DIVISOR: large]
  modelopt -> weight (uint8, (out, in/2)),
              weight_scale (float8_e4m3fn, (out, in/group_size)),
              weight_scale_2 (float32, (1,))         [a MULTIPLIER: small]

Layout differences (why the two paths are NOT bit-identical)
------------------------------------------------------------
Both layouts compute the SAME per-group fp8 scale and the SAME effective
divisor for a weight value (per_group_max * FP8_MAX / per_tensor_max, cast to
fp8). The stored per-tensor scalar is just a reciprocal of the other:
    ct.weight_global_scale == 1.0 / modelopt.weight_scale_2
so `dequantize_nvfp4_weight` reciprocates the ct value internally (format=
"compressed_tensors") and uses the modelopt value directly (format="modelopt").

They differ only in the fp4 ROUNDING rule and in two edge behaviours:
  1. Rounding at the exact midpoint between two E2M1 grid values:
       ct       rounds to nearest by absolute distance, ties resolved to the
                LOWER-magnitude grid value (argmin over the LUT).
       modelopt rounds half to EVEN at the odd E2M1 bounds [0.75, 1.75, 2.5]
                (searchsorted bounds + the modelopt tie bump).
     On random N(0,1) weights this makes ~0.2% of nibbles differ; the dequant
     values then differ by at most one E2M1 step at that group's scale.
  2. An all-zero group: ct clamps its scale to 1e-30; modelopt sets it to 1.0.
     Either way the dequant of an all-zero group is 0.
  3. ct clamps the scaled weight to [-6, 6] before rounding; modelopt lets
     searchsorted map anything above the top bound to the top grid value (6).
     These agree for every finite input.

So dequant(ct-quant(W)) and dequant(modelopt-quant(W)) agree everywhere except
the measure-zero midpoint set; a values-already-on-grid input round-trips
bit-identically through EITHER layout (see tests/test_quantize_cpu.py).
"""
from __future__ import annotations

from typing import Literal

import torch


NVFP4Layout = Literal["ct", "modelopt"]

GROUP_SIZE = 16
FP4_MAX = 6.0           # E2M1 max representable magnitude
FP8_E4M3_MAX = 448.0    # float8_e4m3fn max value

# Positive half of the E2M1 grid (LUT indices 0..7). The sign bit (bit 3) is
# ORed on for negatives; this matches nvfp4_lora.dequant.NVFP4_E2M1_LUT.
NVFP4_LUT_POSITIVE: torch.Tensor = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32
)

# Upper bounds between adjacent positive E2M1 magnitudes, used by the ModelOpt
# searchsorted rounding. e2m1_bounds[i] is the midpoint between value i and i+1.
# The odd-indexed bounds [0.75, 1.75, 2.5] are the ones where modelopt rounds
# half to even by bumping the ordinal up by one on an exact hit.
_E2M1_BOUNDS: torch.Tensor = torch.tensor(
    [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], dtype=torch.float32
)
_E2M1_ODD_BOUNDS: torch.Tensor = _E2M1_BOUNDS[[1, 3, 5]]


def _round_to_e2m1_nibbles_ct(scaled: torch.Tensor) -> torch.Tensor:
    """Map scaled weights to unsigned 4-bit E2M1 nibbles, ct rounding.

    Nearest grid value by absolute distance; ties resolved to the lower LUT
    index by argmin. `scaled` is expected pre-clamped to [-FP4_MAX, FP4_MAX].
    Returns int64 nibble indices in [0, 15] with the sign bit at bit 3.
    """
    lut = NVFP4_LUT_POSITIVE.to(scaled.device)
    abs_vals = scaled.abs()
    abs_diff = (abs_vals.unsqueeze(-1) - lut).abs()      # (..., 8)
    abs_indices = abs_diff.argmin(dim=-1)                # in [0, 7]
    sign = scaled.signbit().long() << 3                  # bit 3
    return abs_indices + sign                            # in [0, 15]


def _round_to_e2m1_nibbles_modelopt(scaled: torch.Tensor) -> torch.Tensor:
    """Map scaled weights to unsigned 4-bit E2M1 nibbles, ModelOpt rounding.

    Ports modelopt NVFP4QTensor._cast_fp4 exactly: searchsorted on the E2M1
    bounds, plus a +1 tie bump at the odd bounds [0.75, 1.75, 2.5] to get
    round-half-to-even. Returns int64 nibble indices in [0, 15]; the sign bit
    is at bit 3.
    """
    bounds = _E2M1_BOUNDS.to(scaled.device)
    odd_bounds = _E2M1_ODD_BOUNDS.to(scaled.device)
    sign_bit = (scaled < 0).to(torch.int64)
    abs_vals = scaled.abs()
    ordinal = torch.searchsorted(bounds, abs_vals, out_int32=True).to(torch.int64)
    equals_odd = torch.any(abs_vals.unsqueeze(-1) == odd_bounds, dim=-1).to(torch.int64)
    return (sign_bit << 3) + ordinal + equals_odd


def _pack_nibbles(indices: torch.Tensor, out_feat: int, in_feat: int) -> torch.Tensor:
    """Pack nibble indices (out, in) into uint8 (out, in/2), low nibble first.

    Matches nvfp4_lora.dequant._unpack_nibbles (and both source scripts): the
    even input position is the low nibble, the odd position the high nibble.
    """
    indices_flat = indices.reshape(out_feat, in_feat)
    indices_pairs = indices_flat.reshape(out_feat, in_feat // 2, 2)
    packed = (indices_pairs[..., 0] | (indices_pairs[..., 1] << 4)).to(torch.uint8)
    return packed


def quantize_nvfp4_2d(
    weight: torch.Tensor,
    *,
    layout: NVFP4Layout = "ct",
    group_size: int = GROUP_SIZE,
    per_tensor_max_override: float | None = None,
) -> dict[str, torch.Tensor]:
    """Quantize a 2D weight (out, in) to NVFP4.

    Args:
      weight: 2D tensor (out, in). Any float dtype; math runs in fp32.
      layout: "ct" (compressed-tensors) or "modelopt". Controls the stored
        per-tensor scalar convention and the fp4 rounding rule (see the module
        docstring for the exact numerical difference).
      group_size: block size along the input axis (must divide `in`).
      per_tensor_max_override: externally supplied per-tensor abs-max. Used when
        several slices must share one per-tensor scale so their stored scalars
        stay bit-identical (fused gate_up_proj experts; vLLM-fused q/k/v).

    Returns a dict of CPU tensors keyed by the layout's suffixes:
      ct       -> {"weight_packed", "weight_scale", "weight_global_scale"}
      modelopt -> {"weight", "weight_scale", "weight_scale_2"}
    """
    if layout not in ("ct", "modelopt"):
        raise ValueError(f"layout must be 'ct' or 'modelopt', got {layout!r}")
    if weight.ndim != 2:
        raise ValueError(f"Expected 2D tensor, got shape {tuple(weight.shape)}")
    out_feat, in_feat = weight.shape
    if in_feat % group_size != 0:
        raise ValueError(f"in_feat={in_feat} not divisible by group_size={group_size}")

    device = weight.device
    w_fp32 = weight.to(dtype=torch.float32)
    n_groups = in_feat // group_size
    w_grouped = w_fp32.reshape(out_feat, n_groups, group_size)

    per_group_max = w_grouped.abs().amax(dim=-1)                       # (out, n_groups)

    # The two layouts share the same per-group fp8 scale and the same effective
    # rounding divisor, but each source computed those in a slightly different
    # order. fp32 rounding is order-sensitive, so each branch reproduces its
    # source's exact arithmetic to stay bit-for-bit identical to it.
    if layout == "ct":
        return _quantize_ct(
            w_grouped, per_group_max, w_fp32, out_feat, in_feat,
            per_tensor_max_override, device,
        )
    return _quantize_modelopt(
        w_grouped, per_group_max, w_fp32, out_feat, in_feat,
        per_tensor_max_override, device,
    )


def _quantize_ct(
    w_grouped: torch.Tensor,
    per_group_max: torch.Tensor,
    w_fp32: torch.Tensor,
    out_feat: int,
    in_feat: int,
    per_tensor_max_override: float | None,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Compressed-tensors NVFP4 quant, bit-exact port of the mistral script."""
    if per_tensor_max_override is not None:
        per_tensor_max = torch.tensor(
            per_tensor_max_override, dtype=torch.float32, device=device
        ).clamp(min=1e-30)
    else:
        per_tensor_max = w_fp32.abs().amax().clamp(min=1e-30)

    # CT convention (compressed_tensors.compressors.nvfp4.base):
    #   global_scale = FP8_MAX * FP4_MAX / per_tensor_max          (a large divisor)
    #   scale_fp32   = per_group_max * FP8_MAX / per_tensor_max     (<= FP8_MAX, stored in fp8)
    #   during quant: q = round(W / (scale_fp8 / global_scale))
    global_scale_fp32 = torch.tensor(
        (FP8_E4M3_MAX * FP4_MAX) / per_tensor_max.item(),
        dtype=torch.float32, device=device,
    )
    scale_fp32 = per_group_max * (FP8_E4M3_MAX / per_tensor_max)
    scale_fp32 = scale_fp32.clamp(min=1e-30)

    # fp8 cast is lossy; recompute the post-cast effective scale so the rounding
    # sees the same effective scale a future dequant will see.
    scale_fp8 = scale_fp32.to(torch.float8_e4m3fn)
    effective_scale = scale_fp8.to(torch.float32) / global_scale_fp32
    effective_scale = effective_scale.clamp(min=1e-30)

    w_scaled = w_grouped / effective_scale.unsqueeze(-1)
    w_scaled = w_scaled.clamp(-FP4_MAX, FP4_MAX)
    indices = _round_to_e2m1_nibbles_ct(w_scaled)
    packed = _pack_nibbles(indices, out_feat, in_feat)

    return {
        "weight_packed": packed.cpu(),
        "weight_scale": scale_fp8.cpu(),
        "weight_global_scale": global_scale_fp32.reshape(1).cpu(),
    }


def _quantize_modelopt(
    w_grouped: torch.Tensor,
    per_group_max: torch.Tensor,
    w_fp32: torch.Tensor,
    out_feat: int,
    in_feat: int,
    per_tensor_max_override: float | None,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """ModelOpt NVFP4 quant, bit-exact port of NVFP4QTensor.quantize."""
    # ModelOpt convention (modelopt NVFP4QTensor):
    #   weight_scale_2 = amax(W) / (FP4_MAX * FP8_MAX)              (a small multiplier)
    #   weight_scale   = per_group_max / (FP4_MAX * weight_scale_2) then fp8-cast
    #   during quant: q = round(W / (weight_scale_fp8 * weight_scale_2))
    if per_tensor_max_override is not None:
        per_tensor_max = torch.tensor(
            per_tensor_max_override, dtype=torch.float32, device=device
        )
    else:
        per_tensor_max = w_fp32.abs().amax().to(torch.float32)
    weight_scale_2 = per_tensor_max / (FP4_MAX * FP8_E4M3_MAX)

    scale_fp32 = per_group_max.to(torch.float32) / (FP4_MAX * weight_scale_2)
    # ModelOpt zeroes-out safety net: an all-zero group's scale is set to 1.0.
    scale_fp32 = torch.where(
        scale_fp32 == 0, torch.ones_like(scale_fp32), scale_fp32
    )
    scale_fp8 = scale_fp32.to(torch.float8_e4m3fn)

    effective_scale = scale_fp8.to(torch.float32) * weight_scale_2
    w_scaled = w_grouped / effective_scale.unsqueeze(-1)
    indices = _round_to_e2m1_nibbles_modelopt(w_scaled)
    packed = _pack_nibbles(indices, out_feat, in_feat)

    return {
        "weight": packed.cpu(),
        "weight_scale": scale_fp8.cpu(),
        "weight_scale_2": weight_scale_2.reshape(1).to(torch.float32).cpu(),
    }


def quantize_nvfp4_3d_per_slice(
    weight: torch.Tensor,
    *,
    layout: NVFP4Layout = "ct",
    group_size: int = GROUP_SIZE,
) -> list[dict[str, torch.Tensor]]:
    """Quantize each 2D slice of a 3D weight (E, out, in) independently.

    Returns a list of dicts, one per slice, each shaped like the return of
    `quantize_nvfp4_2d`. Every slice gets its own per-tensor scale; the caller
    places each dict under its per-slice keys (see the quantize script for the
    fused-MoE expert splitting).
    """
    if weight.ndim != 3:
        raise ValueError(f"Expected 3D tensor, got shape {tuple(weight.shape)}")
    return [
        quantize_nvfp4_2d(weight[e].contiguous(), layout=layout, group_size=group_size)
        for e in range(weight.shape[0])
    ]
