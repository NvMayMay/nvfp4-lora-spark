#!/usr/bin/env python3
"""CPU-only Pre-M1b smoke tests for the NVFP4 dequant workspace path."""
from __future__ import annotations

import os
import sys

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nvfp4_lora.dequant import dequantize_nvfp4_weight
from nvfp4_lora.linear import NVFP4LoRALinear, _DequantLinear
from nvfp4_lora.loader import _assign_dequant_workspaces, _verify_dequant_workspaces


def _synthetic_nvfp4_tensors(out_features: int = 3, in_features: int = 32):
    assert in_features % 16 == 0
    assert in_features % 2 == 0
    torch.manual_seed(1234)
    weight = torch.randint(0, 256, (out_features, in_features // 2), dtype=torch.uint8)
    scale_fp32 = torch.rand(out_features, in_features // 16, dtype=torch.float32) + 0.25
    scale = scale_fp32.to(torch.float8_e4m3fn)
    scale_2 = torch.tensor(0.75, dtype=torch.float32)
    return weight, scale, scale_2


def _make_linear(out_features: int = 4, in_features: int = 32, dtype: torch.dtype = torch.bfloat16):
    weight, scale, scale_2 = _synthetic_nvfp4_tensors(out_features, in_features)
    mod = NVFP4LoRALinear(
        in_features=in_features,
        out_features=out_features,
        weight_uint8=weight,
        weight_scale_fp8=scale,
        weight_scale_2_fp32=scale_2,
        group_size=16,
        r=0,
        device=torch.device("cpu"),
        dtype=dtype,
    )
    return mod


def test_dequant_out_bit_parity_cpu():
    weight, scale, scale_2 = _synthetic_nvfp4_tensors()
    expected = dequantize_nvfp4_weight(weight, scale, scale_2, group_size=16, out_dtype=torch.bfloat16)

    preallocated = torch.empty_like(expected)
    actual = dequantize_nvfp4_weight(
        weight,
        scale,
        scale_2,
        group_size=16,
        out_dtype=torch.bfloat16,
        out=preallocated,
    )

    assert actual is preallocated
    assert torch.allclose(actual, expected, atol=0, rtol=0)


def test_dequant_out_shape_and_dtype_mismatch_raises():
    weight, scale, scale_2 = _synthetic_nvfp4_tensors()

    wrong_shape = torch.empty(2, 32, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="out shape/dtype mismatch"):
        dequantize_nvfp4_weight(
            weight,
            scale,
            scale_2,
            group_size=16,
            out_dtype=torch.bfloat16,
            out=wrong_shape,
        )

    wrong_dtype = torch.empty(3, 32, dtype=torch.float32)
    with pytest.raises(ValueError, match="out shape/dtype mismatch"):
        dequantize_nvfp4_weight(
            weight,
            scale,
            scale_2,
            group_size=16,
            out_dtype=torch.bfloat16,
            out=wrong_dtype,
        )


def test_shared_workspace_detect_anomaly_backward_cpu():
    torch.manual_seed(5678)
    mod_a = _make_linear()
    mod_b = _make_linear()
    workspace = torch.empty(4, 32, dtype=torch.bfloat16, requires_grad=False)
    mod_a.w_bf16_workspace = workspace
    mod_b.w_bf16_workspace = workspace

    x_a = torch.randn(2, 32, dtype=torch.bfloat16, requires_grad=True)
    x_b = torch.randn(2, 32, dtype=torch.bfloat16, requires_grad=True)

    y_a = _DequantLinear.apply(
        x_a,
        mod_a.weight_uint8,
        mod_a.weight_scale_fp8,
        mod_a.weight_scale_2_fp32,
        mod_a.group_size,
        mod_a.w_bf16_workspace,
    )
    y_b = _DequantLinear.apply(
        x_b,
        mod_b.weight_uint8,
        mod_b.weight_scale_fp8,
        mod_b.weight_scale_2_fp32,
        mod_b.group_size,
        mod_b.w_bf16_workspace,
    )

    with torch.autograd.set_detect_anomaly(True):
        (y_a.float().sum() + y_b.float().sum()).backward()

    assert x_a.grad is not None
    assert x_b.grad is not None


def test_loader_workspace_requires_grad_invariant_cpu():
    model = nn.Sequential(_make_linear(), _make_linear())
    pool = _assign_dequant_workspaces(model, device=torch.device("cpu"), dtype=torch.bfloat16)

    assert len(pool) == 1
    first = model[0].w_bf16_workspace
    second = model[1].w_bf16_workspace
    assert first is second
    assert first.requires_grad is False
    _verify_dequant_workspaces(model)

    model[0].w_bf16_workspace = torch.empty(4, 32, dtype=torch.bfloat16, requires_grad=True)
    with pytest.raises(AssertionError, match="dequant workspace requires grad"):
        _verify_dequant_workspaces(model)
