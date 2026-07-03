"""Train-time bf16 dequant cache (P1-3): numerically identical to the recompute path.

The opt-in cache materializes the frozen dequantized base weight once and reuses it with a plain
F.linear, instead of re-dequantizing every step (_DequantLinear). Because the base weight is frozen,
autograd gives the same dx = dy @ W and no grad to W -- so the cached path must match the recompute
path bit-for-bit on the forward AND on every gradient. Also checks the budget lifecycle.
"""
from __future__ import annotations

import torch

from nvfp4_lora import linear as lin
from nvfp4_lora.linear import NVFP4LoRALinear, set_train_dequant_cache_gb
from test_numerical_oracle import _make_nvfp4_base


def _build(seed=7):
    out_f, in_f, gs, r, alpha = 6, 32, 16, 4, 8
    w8, scales, scale2 = _make_nvfp4_base(out_f, in_f, gs, seed=seed)
    m = NVFP4LoRALinear(in_f, out_f, w8, scales, scale2,
                        group_size=gs, r=r, lora_alpha=alpha, dtype=torch.float32)
    m.w_bf16_workspace = torch.empty(out_f, in_f, dtype=torch.float32)
    m.train()
    with torch.no_grad():
        m.lora_B.copy_(torch.randn(out_f, r) * 0.1)  # non-zero so A receives grad
    return m, in_f


def _fwd_bwd(m, in_f, x_seed=11):
    for p in (m.lora_A, m.lora_B):
        if p.grad is not None:
            p.grad = None
    g = torch.Generator().manual_seed(x_seed)
    x = torch.randn(2, in_f, generator=g, requires_grad=True)
    y = m(x)
    y.sum().backward()
    return y.detach().clone(), x.grad.clone(), m.lora_A.grad.clone(), m.lora_B.grad.clone()


def test_cache_matches_recompute_forward_and_grads():
    try:
        m, in_f = _build()
        set_train_dequant_cache_gb(0)                 # recompute path
        y0, gx0, ga0, gb0 = _fwd_bwd(m, in_f)
        assert m._train_weight is None

        set_train_dequant_cache_gb(1.0)               # cache path (1 GB >> tiny weight)
        y1, gx1, ga1, gb1 = _fwd_bwd(m, in_f)
        assert m._train_weight is not None            # cache populated
        assert m._train_weight.requires_grad is False  # frozen

        torch.testing.assert_close(y1, y0, rtol=0, atol=0)   # bit-identical forward
        torch.testing.assert_close(gx1, gx0, rtol=0, atol=0)
        torch.testing.assert_close(ga1, ga0, rtol=0, atol=0)
        torch.testing.assert_close(gb1, gb0, rtol=0, atol=0)
    finally:
        set_train_dequant_cache_gb(0)


def test_budget_exhaustion_falls_back_to_recompute():
    try:
        m, in_f = _build()
        # A budget far smaller than one weight (6*32*2 = 384 bytes) -> cannot allocate -> recompute.
        lin._TRAIN_CACHE_LIMIT_BYTES = 1
        lin._TRAIN_CACHE_BYTES_RESERVED = 0
        assert m._train_cached_weight(torch.float32) is None
        assert m._train_weight is None
    finally:
        set_train_dequant_cache_gb(0)


def test_disabled_by_default_returns_none():
    try:
        m, in_f = _build()
        set_train_dequant_cache_gb(0)
        assert m._train_cached_weight(torch.float32) is None
    finally:
        set_train_dequant_cache_gb(0)
