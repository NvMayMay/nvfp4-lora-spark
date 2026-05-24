"""NVFP4LoRALinear - frozen NVFP4 base weight + trainable bf16 LoRA delta.

Implements the architecture both Opus and GPT-5.5 subagents converged on (see
`Research/nvfp4_lora_spark/agent_outputs/SYNTHESIS.md`):

- Base weight: NVFP4-quantized, stored as (qdata, scales, per_tensor_scale) - never trained.
- LoRA path: bf16 `lora_A` (r, in) and `lora_B` (out, r), trainable.
- Forward: y = (dequant(W) @ x.T).T + scale * (x @ A.T) @ B.T
  Equivalent: y = x @ dequant(W).T + scale * x @ A.T @ B.T
- Backward through W: standard F.linear backward auto-handles dx = dy @ W; we wrap
  the dequant in a custom autograd.Function so W_bf16 is recomputed in backward
  rather than saved, keeping memory ~weight-storage-only across the autograd graph.
- LoRA gradient: standard, since lora_A and lora_B are normal nn.Parameter.

This is `nn.Module` with a `Linear`-compatible interface, NOT a subclass of nn.Linear
(per the v6.2 PROPOSAL §5.3.1 lesson - nn.Linear assumes `.weight` is a trainable Parameter,
which is wrong here).
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dequant import dequantize_nvfp4_weight


class _DequantLinear(torch.autograd.Function):
    """Custom autograd that dequants on every forward and recomputes on every backward.

    Does NOT save the dequantized bf16 weight across the graph. Saves only the small
    quantized buffers + scales (typically ~1/4 the size of bf16).

    The frozen base weight produces no gradient; only `x` gets a grad via `dx = dy @ W`.
    """

    @staticmethod
    def forward(ctx, x, weight_uint8, weight_scale_fp8, weight_scale_2_fp32, group_size: int):
        ctx.save_for_backward(x, weight_uint8, weight_scale_fp8, weight_scale_2_fp32)
        ctx.group_size = group_size
        W_bf16 = dequantize_nvfp4_weight(
            weight_uint8, weight_scale_fp8, weight_scale_2_fp32,
            group_size=group_size, out_dtype=x.dtype,
        )
        # F.linear computes x @ W.T (no bias handled here; caller adds it)
        return F.linear(x, W_bf16, bias=None)

    @staticmethod
    def backward(ctx, grad_output):
        x, weight_uint8, weight_scale_fp8, weight_scale_2_fp32 = ctx.saved_tensors
        group_size = ctx.group_size

        grad_x = None
        if ctx.needs_input_grad[0]:
            # Recompute W_bf16 inside backward - never saved
            W_bf16 = dequantize_nvfp4_weight(
                weight_uint8, weight_scale_fp8, weight_scale_2_fp32,
                group_size=group_size, out_dtype=grad_output.dtype,
            )
            # dx = dy @ W (since y = x @ W.T, dy/dx = W)
            grad_x = grad_output @ W_bf16

        # No grad for the frozen base weight or scales
        return grad_x, None, None, None, None


class NVFP4LoRALinear(nn.Module):
    """nn.Linear-compatible interface with frozen NVFP4 base + trainable bf16 LoRA delta.

    Args:
        in_features, out_features: standard Linear dims (in_features is the unpacked dim).
        weight_uint8: (out, in/2) uint8 - the on-disk packed NVFP4 weight.
        weight_scale_fp8: (out, in/group_size) float8_e4m3fn.
        weight_scale_2_fp32: scalar float32.
        group_size: NVFP4 group size, default 16.
        bias: optional bias tensor (trainable; if you want frozen bias, set requires_grad=False after construction).
        r: LoRA rank; if 0, LoRA path is disabled (frozen linear only).
        lora_alpha: LoRA scaling factor; effective scale = lora_alpha / r.
        lora_dropout: dropout applied to the LoRA path input (not the base path).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        weight_uint8: torch.Tensor,
        weight_scale_fp8: torch.Tensor,
        weight_scale_2_fp32: torch.Tensor,
        group_size: int = 16,
        bias: Optional[torch.Tensor] = None,
        r: int = 0,
        lora_alpha: int = 0,
        lora_dropout: float = 0.0,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size
        self.r = r
        self.lora_alpha = lora_alpha
        self.lora_scale = (lora_alpha / r) if r > 0 else 0.0
        self.dtype = dtype

        # Frozen NVFP4 base (register as buffers, not Parameters - never trained, never saved by optimizer)
        device = device or weight_uint8.device
        self.register_buffer("weight_uint8", weight_uint8.to(device=device).contiguous())
        self.register_buffer("weight_scale_fp8", weight_scale_fp8.to(device=device).contiguous())
        self.register_buffer("weight_scale_2_fp32", weight_scale_2_fp32.to(device=device).contiguous())

        if bias is not None:
            self.bias = nn.Parameter(bias.to(device=device, dtype=dtype), requires_grad=False)
        else:
            self.bias = None

        # LoRA params
        if r > 0:
            self.lora_A = nn.Parameter(torch.empty(r, in_features, device=device, dtype=dtype))
            self.lora_B = nn.Parameter(torch.zeros(out_features, r, device=device, dtype=dtype))
            # Kaiming init for A; B starts at zero so LoRA delta starts at zero (standard PEFT init)
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            self.lora_dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0 else nn.Identity()
        else:
            self.lora_A = None
            self.lora_B = None
            self.lora_dropout = nn.Identity()

        # Eval-mode bf16 weight cache (populated lazily on first eval forward; cleared on train())
        # Materializes the dequantized base weight once for fast `F.linear` reuse. Skips the per-forward
        # custom-autograd dequant cost when we don't need gradients.
        self._eval_weight: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Base path: training uses custom autograd (dequant recomputed in backward, no bf16 shadow saved);
        # eval uses a lazily-materialized bf16 weight cache for fast `F.linear`.
        if self.training:
            y = _DequantLinear.apply(
                x, self.weight_uint8, self.weight_scale_fp8, self.weight_scale_2_fp32, self.group_size
            )
        else:
            if self._eval_weight is None or self._eval_weight.dtype != x.dtype:
                self._eval_weight = dequantize_nvfp4_weight(
                    self.weight_uint8, self.weight_scale_fp8, self.weight_scale_2_fp32,
                    group_size=self.group_size, out_dtype=x.dtype,
                )
            y = F.linear(x, self._eval_weight, bias=None)
        # LoRA delta
        if self.r > 0:
            lora_out = F.linear(self.lora_dropout(x), self.lora_A)  # (..., r)
            lora_out = F.linear(lora_out, self.lora_B)              # (..., out_features)
            y = y + self.lora_scale * lora_out
        if self.bias is not None:
            y = y + self.bias
        return y

    def train(self, mode: bool = True):
        # Clear the eval-mode bf16 cache when switching back to training so we don't leak the shadow.
        if mode:
            self._eval_weight = None
        return super().train(mode)

    @property
    def weight(self) -> torch.Tensor:
        """Compatibility shim for code that queries `.weight.dtype` / `.shape` / `.device`.

        Returns a META tensor (zero memory) with the correct shape and dtype. Any code
        that tries to USE this for compute will fail loudly - which is correct, since
        the actual NVFP4 storage lives in `self.weight_uint8` + scales, dequanted lazily
        in forward via `_DequantLinear`.

        Specifically relevant for Nemotron-3's modeling_nemotron_h.py:855 which does
        `expert.down_proj.weight.dtype` for autocast-dtype detection.
        """
        # Meta tensor: correct metadata, zero memory
        return torch.empty(
            self.out_features, self.in_features,
            dtype=self.dtype, device="meta",
        )

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"r={self.r}, lora_alpha={self.lora_alpha}, group_size={self.group_size}, "
            f"bias={self.bias is not None}"
        )

    @classmethod
    def from_safetensors_record(
        cls,
        record: dict[str, torch.Tensor],
        prefix: str,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 0,
        lora_dropout: float = 0.0,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.bfloat16,
    ) -> "NVFP4LoRALinear":
        """Build from a dict of tensors loaded from a safetensors shard.

        Expected keys (relative to `prefix`):
            {prefix}.weight (uint8)
            {prefix}.weight_scale (float8_e4m3fn)
            {prefix}.weight_scale_2 (float32 scalar)
            {prefix}.bias (optional)
        """
        bias = record.get(f"{prefix}.bias")
        return cls(
            in_features=in_features,
            out_features=out_features,
            weight_uint8=record[f"{prefix}.weight"],
            weight_scale_fp8=record[f"{prefix}.weight_scale"],
            weight_scale_2_fp32=record[f"{prefix}.weight_scale_2"],
            group_size=16,
            bias=bias,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            device=device,
            dtype=dtype,
        )
