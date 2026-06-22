"""FP8LoRALinear: frozen FP8 base + trainable bf16 LoRA delta (CPU-only).

Validates the module that lets a NATIVE run adapt FP8 targets (the 3.6's attention)
instead of freezing them. Pure CPU: tiny random tensors, fp8 cast on CPU; checks the
forward math (base + LoRA), the zero-init delta, that only the LoRA params train, and
per-output-channel scale broadcasting.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from nvfp4_lora.linear import FP8LoRALinear


def _make(out_f=8, in_f=6, r=4, alpha=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    w_fp8 = torch.randn(out_f, in_f, generator=g).to(torch.float8_e4m3fn)
    scale = torch.tensor(0.5, dtype=torch.float32)
    return w_fp8, scale, out_f, in_f, r, alpha


def test_base_only_matches_dequant():
    w_fp8, scale, o, i, _, _ = _make()
    m = FP8LoRALinear(i, o, w_fp8, scale, r=0, dtype=torch.float32)
    x = torch.randn(3, i)
    W = w_fp8.to(torch.float32) * scale
    assert torch.allclose(m(x), F.linear(x, W), atol=1e-5)


def test_lora_delta_zero_at_init():
    # lora_B initializes to zero -> the delta is zero -> forward == base path.
    w_fp8, scale, o, i, r, a = _make()
    m = FP8LoRALinear(i, o, w_fp8, scale, r=r, lora_alpha=a, dtype=torch.float32)
    x = torch.randn(2, i)
    W = w_fp8.to(torch.float32) * scale
    assert torch.allclose(m(x), F.linear(x, W), atol=1e-5)


def test_lora_delta_applied():
    w_fp8, scale, o, i, r, a = _make()
    m = FP8LoRALinear(i, o, w_fp8, scale, r=r, lora_alpha=a, dtype=torch.float32)
    with torch.no_grad():
        m.lora_B.copy_(torch.randn(o, r))
    x = torch.randn(2, i)
    base = F.linear(x, w_fp8.to(torch.float32) * scale)
    delta = (a / r) * (x @ m.lora_A.T @ m.lora_B.T)
    assert torch.allclose(m(x), base + delta, atol=1e-5)


def test_only_lora_gets_grad():
    w_fp8, scale, o, i, r, a = _make()
    m = FP8LoRALinear(i, o, w_fp8, scale, r=r, lora_alpha=a, dtype=torch.float32)
    with torch.no_grad():
        m.lora_B.copy_(torch.randn(o, r) * 0.1)
    x = torch.randn(2, i, requires_grad=True)
    m(x).sum().backward()
    assert m.lora_A.grad is not None and m.lora_B.grad is not None
    assert m.lora_A.grad.abs().sum() > 0          # A receives grad (B is nonzero)
    assert not isinstance(m.weight_fp8, torch.nn.Parameter)  # frozen base, never trained
    assert x.grad is not None                     # base path passes grad to the input


def test_per_channel_scale_broadcasts():
    w_fp8, _, o, i, _, _ = _make()
    scale = torch.rand(o) + 0.5  # per-output-channel vector
    m = FP8LoRALinear(i, o, w_fp8, scale, r=0, dtype=torch.float32)
    x = torch.randn(2, i)
    W = w_fp8.to(torch.float32) * scale.reshape(o, 1)
    assert torch.allclose(m(x), F.linear(x, W), atol=1e-5)


def test_weight_property_is_meta():
    w_fp8, scale, o, i, _, _ = _make()
    m = FP8LoRALinear(i, o, w_fp8, scale, r=0)
    assert m.weight.device.type == "meta"
    assert tuple(m.weight.shape) == (o, i)
