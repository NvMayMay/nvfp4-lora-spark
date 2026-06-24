"""BF16LoRALinear: frozen bf16 base + trainable bf16 LoRA delta (CPU-only).

The simplest of the LoRALinear family (no dequant): validates the forward math
(base + LoRA), zero-init delta, and that only the LoRA params train (the base is a
frozen buffer). This is what lets a genuinely-BF16 target co-train natively alongside
NVFP4/FP8 targets in one adapter.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from nvfp4_lora.linear import BF16LoRALinear


def _make(out_f=8, in_f=6, r=4, alpha=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(out_f, in_f, generator=g), out_f, in_f, r, alpha


def test_base_only_matches_linear():
    w, o, i, _, _ = _make()
    m = BF16LoRALinear(i, o, w, r=0, dtype=torch.float32)
    x = torch.randn(3, i)
    assert torch.allclose(m(x), F.linear(x, w), atol=1e-6)


def test_lora_delta_zero_at_init():
    w, o, i, r, a = _make()
    m = BF16LoRALinear(i, o, w, r=r, lora_alpha=a, dtype=torch.float32)
    x = torch.randn(2, i)
    assert torch.allclose(m(x), F.linear(x, w), atol=1e-6)


def test_lora_delta_applied():
    w, o, i, r, a = _make()
    m = BF16LoRALinear(i, o, w, r=r, lora_alpha=a, dtype=torch.float32)
    with torch.no_grad():
        m.lora_B.copy_(torch.randn(o, r))
    x = torch.randn(2, i)
    base = F.linear(x, w)
    delta = (a / r) * (x @ m.lora_A.T @ m.lora_B.T)
    assert torch.allclose(m(x), base + delta, atol=1e-5)


def test_only_lora_gets_grad():
    w, o, i, r, a = _make()
    m = BF16LoRALinear(i, o, w, r=r, lora_alpha=a, dtype=torch.float32)
    with torch.no_grad():
        m.lora_B.copy_(torch.randn(o, r) * 0.1)
    x = torch.randn(2, i, requires_grad=True)
    m(x).sum().backward()
    assert m.lora_A.grad is not None and m.lora_B.grad is not None
    assert m.lora_A.grad.abs().sum() > 0
    assert not isinstance(m.weight, torch.nn.Parameter)  # frozen base = buffer, never trained
    assert m.weight.requires_grad is False
    assert x.grad is not None
