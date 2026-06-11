"""Triton fused NVFP4 -> bf16 dequant kernel.

Single-pass implementation that replaces the 8-10 eager PyTorch kernel dispatches
of `dequantize_nvfp4_weight` with one Triton call. The 16x int64 intermediate
tensor in `_unpack_nibbles` is eliminated by keeping the nibble in a register.

Measured on Blackwell B10 (GB10 Spark) at PyTorch 2.11.0+cu130 / Triton 3.6.0:
    gate_up (4096x4096 BF16 out)  PyTorch 9.75 ms  Triton 0.516 ms  18.9x
    down    (4096x2048 BF16 out)  PyTorch 5.37 ms  Triton 0.189 ms  28.5x

The dispatcher in `dequant.py` falls back to the PyTorch path when:
- Triton is not importable
- The input tensor is on CPU
- The shapes do not satisfy the kernel's contiguity requirements

The kernel is autograd-agnostic: it dequantizes a frozen NVFP4 weight to a
bf16 buffer. Caller is responsible for the gradient path (the custom autograd
Functions in linear.py / experts.py save the packed weight and recompute on
backward as before).
"""
from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
    _TRITON_AVAILABLE = True
except ImportError:
    _TRITON_AVAILABLE = False


if _TRITON_AVAILABLE:

    @triton.jit
    def _nvfp4_dequant_kernel(
        packed_ptr,           # uint8, shape (n_rows, n_cols_packed)
        scale_ptr,            # fp32, shape (n_rows, n_groups)
        gscale_ptr,           # fp32, shape (1,) — effective per-tensor scale
        out_ptr,              # bf16, shape (n_rows, n_cols_packed * 2)
        n_rows,
        n_cols_packed,
        n_groups,
        group_size: tl.constexpr,
        BLOCK_ROW: tl.constexpr,
        BLOCK_COL: tl.constexpr,
    ):
        # Each program handles BLOCK_ROW rows x BLOCK_COL output columns.
        row_start = tl.program_id(0) * BLOCK_ROW
        col_start = tl.program_id(1) * BLOCK_COL

        rows = row_start + tl.arange(0, BLOCK_ROW)            # (BLOCK_ROW,)
        cols_out = col_start + tl.arange(0, BLOCK_COL)        # (BLOCK_COL,)
        cols_packed = cols_out // 2                            # which packed byte
        n_cols_out = n_cols_packed * 2

        row_mask = rows < n_rows
        col_mask_out = cols_out < n_cols_out
        col_mask_packed = cols_packed < n_cols_packed

        # Gather packed bytes (one byte produces two output values).
        packed_offs = rows[:, None] * n_cols_packed + cols_packed[None, :]
        packed_mask = row_mask[:, None] & col_mask_packed[None, :]
        packed_vals = tl.load(packed_ptr + packed_offs, mask=packed_mask, other=0).to(tl.int32)

        # NVFP4 packing is low-nibble-first: even output cols read bits 0-3,
        # odd output cols read bits 4-7. (cols_out % 2 == 0 -> low nibble.)
        is_high = (cols_out % 2 == 1)[None, :]                                  # (1, BLOCK_COL)
        nibbles = tl.where(is_high, (packed_vals >> 4) & 0x0F, packed_vals & 0x0F)

        # E2M1 decode: nibble layout is [sign | exp(2) | mantissa(1)] (MSB=sign).
        sign_bit = (nibbles >> 3) & 1
        magnitude_idx = nibbles & 0x7
        # Inline LUT for the 8 magnitudes: {0,0.5,1.0,1.5,2.0,3.0,4.0,6.0}.
        # Triton compiles the nested tl.where chain to a register-resident
        # selector — no memory traffic, no separate gather kernel.
        mag = tl.where(
            magnitude_idx == 0, 0.0,
            tl.where(magnitude_idx == 1, 0.5,
            tl.where(magnitude_idx == 2, 1.0,
            tl.where(magnitude_idx == 3, 1.5,
            tl.where(magnitude_idx == 4, 2.0,
            tl.where(magnitude_idx == 5, 3.0,
            tl.where(magnitude_idx == 6, 4.0, 6.0)))))))
        mag = mag.to(tl.float32)
        fp4_val = tl.where(sign_bit > 0, -mag, mag)

        # Per-group scale (one fp32 per group_size output cols).
        group_idx = cols_out // group_size
        scale_offs = rows[:, None] * n_groups + group_idx[None, :]
        scale_mask = row_mask[:, None] & (group_idx[None, :] < n_groups)
        group_scale = tl.load(scale_ptr + scale_offs, mask=scale_mask, other=1.0)

        # Per-tensor scale (already pre-reciprocated by the dispatcher for CT format).
        gscale = tl.load(gscale_ptr)
        out_val = fp4_val * group_scale * gscale

        out_offs = rows[:, None] * n_cols_out + cols_out[None, :]
        out_mask = row_mask[:, None] & col_mask_out[None, :]
        tl.store(out_ptr + out_offs, out_val.to(tl.bfloat16), mask=out_mask)

    @triton.jit
    def _nvfp4_dequant_kernel_batched(
        packed_ptr,           # uint8, shape (num_experts, n_rows, n_cols_packed), FULL buffer
        scale_ptr,            # fp32, shape (K, n_rows, n_groups), selected experts only
        gscale_ptr,           # fp32, shape (K,), effective per-selected-expert scale
        expert_map_ptr,       # int64, shape (K,), selected expert index per batch slot
        out_ptr,              # bf16, shape (K, n_rows, n_cols_packed * 2)
        n_rows,
        n_cols_packed,
        n_groups,
        group_size: tl.constexpr,
        BLOCK_ROW: tl.constexpr,
        BLOCK_COL: tl.constexpr,
    ):
        # Same tile math as _nvfp4_dequant_kernel; axis 2 of the grid walks the
        # selected experts. The expert map indirection lets the kernel read the
        # packed nibbles straight out of the full (num_experts, ...) module
        # buffer without a gathered copy. The fp8 scales need an fp32 cast
        # anyway, so the dispatcher pre-gathers those K experts; scale, gscale
        # and out are therefore indexed by the batch slot, not the expert id.
        slot = tl.program_id(2)
        expert = tl.load(expert_map_ptr + slot)
        row_start = tl.program_id(0) * BLOCK_ROW
        col_start = tl.program_id(1) * BLOCK_COL

        rows = row_start + tl.arange(0, BLOCK_ROW)            # (BLOCK_ROW,)
        cols_out = col_start + tl.arange(0, BLOCK_COL)        # (BLOCK_COL,)
        cols_packed = cols_out // 2                            # which packed byte
        n_cols_out = n_cols_packed * 2

        row_mask = rows < n_rows
        col_mask_out = cols_out < n_cols_out
        col_mask_packed = cols_packed < n_cols_packed

        packed_base = expert.to(tl.int64) * n_rows * n_cols_packed
        packed_offs = packed_base + rows[:, None] * n_cols_packed + cols_packed[None, :]
        packed_mask = row_mask[:, None] & col_mask_packed[None, :]
        packed_vals = tl.load(packed_ptr + packed_offs, mask=packed_mask, other=0).to(tl.int32)

        is_high = (cols_out % 2 == 1)[None, :]                                  # (1, BLOCK_COL)
        nibbles = tl.where(is_high, (packed_vals >> 4) & 0x0F, packed_vals & 0x0F)

        sign_bit = (nibbles >> 3) & 1
        magnitude_idx = nibbles & 0x7
        mag = tl.where(
            magnitude_idx == 0, 0.0,
            tl.where(magnitude_idx == 1, 0.5,
            tl.where(magnitude_idx == 2, 1.0,
            tl.where(magnitude_idx == 3, 1.5,
            tl.where(magnitude_idx == 4, 2.0,
            tl.where(magnitude_idx == 5, 3.0,
            tl.where(magnitude_idx == 6, 4.0, 6.0)))))))
        mag = mag.to(tl.float32)
        fp4_val = tl.where(sign_bit > 0, -mag, mag)

        group_idx = cols_out // group_size
        scale_base = slot.to(tl.int64) * n_rows * n_groups
        scale_offs = scale_base + rows[:, None] * n_groups + group_idx[None, :]
        scale_mask = row_mask[:, None] & (group_idx[None, :] < n_groups)
        group_scale = tl.load(scale_ptr + scale_offs, mask=scale_mask, other=1.0)

        gscale = tl.load(gscale_ptr + slot)
        out_val = fp4_val * group_scale * gscale

        out_base = slot.to(tl.int64) * n_rows * n_cols_out
        out_offs = out_base + rows[:, None] * n_cols_out + cols_out[None, :]
        out_mask = row_mask[:, None] & col_mask_out[None, :]
        tl.store(out_ptr + out_offs, out_val.to(tl.bfloat16), mask=out_mask)


def triton_available() -> bool:
    return _TRITON_AVAILABLE


def triton_dequant_nvfp4(
    weight_uint8: torch.Tensor,                # (out, in/2), uint8, CUDA
    weight_scale_fp32: torch.Tensor,           # (out, in/group_size), float32, CUDA
    effective_global_scale_fp32: torch.Tensor, # shape (1,) or scalar, float32, CUDA
    *,
    group_size: int = 16,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run the Triton dequant. Caller pre-computes the effective per-tensor scale.

    For ModelOpt format pass weight_scale_2 verbatim. For compressed-tensors pass
    `1.0 / weight_global_scale.clamp(min=1e-30)` — same reciprocation that the
    eager PyTorch path does internally. This keeps the kernel format-agnostic.

    Returns a bf16 tensor of shape (out, in). If `out` is provided, writes into it.
    """
    if not _TRITON_AVAILABLE:
        raise RuntimeError("Triton is not available; use the PyTorch dequant fallback.")
    if weight_uint8.device.type != "cuda":
        raise ValueError(f"Triton dequant requires CUDA tensors, got {weight_uint8.device}")

    n_rows, n_cols_packed = weight_uint8.shape
    n_cols_out = n_cols_packed * 2
    if n_cols_out % group_size != 0:
        raise ValueError(f"output dim {n_cols_out} not divisible by group_size={group_size}")
    n_groups = n_cols_out // group_size

    # Normalize gscale to a 1-element fp32 tensor (Triton tl.load expects a pointer).
    gscale = effective_global_scale_fp32.reshape(-1)[:1].contiguous()

    if out is None:
        out = torch.empty(n_rows, n_cols_out, dtype=torch.bfloat16, device=weight_uint8.device)
    else:
        if out.shape != (n_rows, n_cols_out):
            raise ValueError(f"out shape mismatch: {tuple(out.shape)} vs expected {(n_rows, n_cols_out)}")
        if out.dtype != torch.bfloat16:
            raise ValueError(f"out must be bfloat16, got {out.dtype}")
        if out.device != weight_uint8.device:
            raise ValueError("out must be on same device as weight_uint8")

    # Tile sizes — these match the benchmarked configuration on GB10. If the
    # input rows < BLOCK_ROW or cols < BLOCK_COL the mask path covers the tail.
    BLOCK_ROW = 16
    BLOCK_COL = 128
    grid = (triton.cdiv(n_rows, BLOCK_ROW), triton.cdiv(n_cols_out, BLOCK_COL))
    _nvfp4_dequant_kernel[grid](
        weight_uint8.contiguous(),
        weight_scale_fp32.contiguous(),
        gscale,
        out,
        n_rows,
        n_cols_packed,
        n_groups,
        group_size=group_size,
        BLOCK_ROW=BLOCK_ROW,
        BLOCK_COL=BLOCK_COL,
    )
    return out


def triton_dequant_nvfp4_batched(
    weight_uint8: torch.Tensor,                # (num_experts, out, in/2), uint8, CUDA, FULL buffer
    weight_scale_fp32: torch.Tensor,           # (K, out, in/group_size), float32, CUDA, selected experts
    effective_global_scale_fp32: torch.Tensor, # (K,) or (K, 1), float32, CUDA
    *,
    group_size: int = 16,
    out: torch.Tensor | None = None,
    expert_idx: torch.Tensor | None = None,    # (K,) int indices into dim 0 of weight_uint8
) -> torch.Tensor:
    """Batched Triton dequant over K selected experts in one launch.

    Grid axis 2 walks the K batch slots; each slot resolves its expert id via
    `expert_idx` and reads packed nibbles directly from the full 3D buffer (no
    gathered uint8 copy). `weight_scale_fp32` and the global scales are
    expected pre-gathered to the K selected experts because the fp8 -> fp32
    cast forces a copy regardless. As with the 2D entry point the caller
    pre-computes the effective per-tensor scale (reciprocated for the
    compressed-tensors format).

    Returns a bf16 tensor of shape (K, out, in). If `out` is provided, writes
    into it.
    """
    if not _TRITON_AVAILABLE:
        raise RuntimeError("Triton is not available; use the PyTorch dequant fallback.")
    if weight_uint8.device.type != "cuda":
        raise ValueError(f"Triton dequant requires CUDA tensors, got {weight_uint8.device}")
    if weight_uint8.ndim != 3:
        raise ValueError(f"weight_uint8 must be 3D (num_experts, out, in/2), got {tuple(weight_uint8.shape)}")

    num_experts, n_rows, n_cols_packed = weight_uint8.shape
    n_cols_out = n_cols_packed * 2
    if n_cols_out % group_size != 0:
        raise ValueError(f"output dim {n_cols_out} not divisible by group_size={group_size}")
    n_groups = n_cols_out // group_size

    if expert_idx is None:
        expert_map = torch.arange(num_experts, dtype=torch.int64, device=weight_uint8.device)
    else:
        expert_map = expert_idx.reshape(-1).to(dtype=torch.int64, device=weight_uint8.device).contiguous()
    n_selected = expert_map.numel()

    if weight_scale_fp32.shape != (n_selected, n_rows, n_groups):
        raise ValueError(
            f"weight_scale_fp32 shape mismatch: got {tuple(weight_scale_fp32.shape)} "
            f"expected {(n_selected, n_rows, n_groups)}"
        )
    gscale = effective_global_scale_fp32.reshape(-1).contiguous()
    if gscale.numel() != n_selected:
        raise ValueError(
            f"effective_global_scale_fp32 must have {n_selected} elements, got {gscale.numel()}"
        )

    if out is None:
        out = torch.empty(n_selected, n_rows, n_cols_out, dtype=torch.bfloat16, device=weight_uint8.device)
    else:
        if out.shape != (n_selected, n_rows, n_cols_out):
            raise ValueError(
                f"out shape mismatch: {tuple(out.shape)} vs expected {(n_selected, n_rows, n_cols_out)}"
            )
        if out.dtype != torch.bfloat16:
            raise ValueError(f"out must be bfloat16, got {out.dtype}")
        if out.device != weight_uint8.device:
            raise ValueError("out must be on same device as weight_uint8")

    # Same tile sizes as the 2D kernel; grid axis 2 is the expert batch slot.
    BLOCK_ROW = 16
    BLOCK_COL = 128
    grid = (triton.cdiv(n_rows, BLOCK_ROW), triton.cdiv(n_cols_out, BLOCK_COL), n_selected)
    _nvfp4_dequant_kernel_batched[grid](
        weight_uint8.contiguous(),
        weight_scale_fp32.contiguous(),
        gscale,
        expert_map,
        out,
        n_rows,
        n_cols_packed,
        n_groups,
        group_size=group_size,
        BLOCK_ROW=BLOCK_ROW,
        BLOCK_COL=BLOCK_COL,
    )
    return out
