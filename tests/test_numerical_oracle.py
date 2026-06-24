"""M2 numerical-correctness oracle for the LoRALinear family (CPU-only).

The dequant FORWARD already has a known-answer test (test_dequant_cpu.py) and the
LoRALinear forwards are checked behaviorally (test_{bf16,fp8}_lora_linear.py). This
file closes the deepest correctness gap: the CUSTOM AUTOGRAD BACKWARD of the two
dequant autograd.Functions, and the apply-the-delta math, proven against an
INDEPENDENT reference rather than the module's own forward.

Three groups of checks:

1. Backward correctness of `_DequantLinear` / `_FP8DequantLinear`:
   - `torch.autograd.gradcheck` in float64. Both Functions are exactly LINEAR in
     `x` (`y = x @ dequant(W).T`), so the analytic VJP `grad_x = grad_output @ W`
     must agree with central finite differences to float64 tolerance. gradcheck is
     valid here even though the base is quantized: only `x` carries `requires_grad`,
     the quant/fp8 buffers are non-differentiable inputs (return None), and the
     dequant is a fixed affine map applied at the input dtype. We additionally assert
     the analytic `grad_x` equals an independent `grad_output @ dequant(W)` and that
     the frozen buffers receive NO gradient (their `needs_input_grad`/returned grads
     are None).

2. Logits parity per quant path (NVFP4 / FP8 / BF16): a tiny module of each variant
   with a KNOWN lora_A/lora_B must equal an independent reference
       F.linear(x, dequant(W)) + (alpha/r) * (x @ A.T @ B.T)
   where dequant(W) is computed WITHOUT the module's forward. Tolerances: NVFP4 is
   looser (bf16 LoRA params + group-scaled E2M1 base), FP8 / BF16 tight.

3. Bind-but-zero-effect is detectable: lora_B == 0 gives base-identical output, and a
   non-zero lora_B gives a measurably different output, so a silent no-op adapter
   (the failure mode the binding contract exists to catch) cannot pass.
"""
from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from nvfp4_lora.dequant import dequantize_nvfp4_weight
from nvfp4_lora.linear import (
    _DequantLinear,
    _FP8DequantLinear,
    NVFP4LoRALinear,
    FP8LoRALinear,
    BF16LoRALinear,
)

CPU = torch.device("cpu")


# --------------------------------------------------------------------------- #
# Helpers: build small, valid quantized base buffers on CPU.
# --------------------------------------------------------------------------- #
def _make_nvfp4_base(out_f, in_f, group_size=16, seed=0):
    """Return (weight_uint8, weight_scale_fp8, weight_scale_2) for a tiny NVFP4 base.

    in_f must be a multiple of group_size and of 2 (two nibbles per byte). Values are
    arbitrary-but-valid nibble indices [0,15] and small positive group scales; the
    point is exercising real dequant arithmetic, not a specific weight.
    """
    assert in_f % group_size == 0 and in_f % 2 == 0
    g = torch.Generator().manual_seed(seed)
    weight_uint8 = torch.randint(0, 256, (out_f, in_f // 2), dtype=torch.uint8, generator=g)
    n_groups = in_f // group_size
    # Small positive scales representable in fp8_e4m3 (0.5 .. 2.0).
    scales = (torch.rand(out_f, n_groups, generator=g) * 1.5 + 0.5).to(torch.float8_e4m3fn)
    scale2 = torch.tensor(0.75, dtype=torch.float32)
    return weight_uint8, scales, scale2


def _ref_nvfp4_W(weight_uint8, scales, scale2, group_size, out_dtype, fmt="modelopt"):
    """Independent dequant reference: we reuse the library dequant (itself known-answer
    tested in test_dequant_cpu.py) but NEVER the LoRALinear forward path.

    `fmt` selects the checkpoint format so the compressed-tensors reciprocal-scale
    semantics (per_tensor_effective = 1 / scale2) are exercised here too."""
    return dequantize_nvfp4_weight(
        weight_uint8, scales, scale2,
        group_size=group_size, out_dtype=out_dtype, format=fmt,
    )


def _make_fp8_base(out_f, in_f, seed=0):
    g = torch.Generator().manual_seed(seed)
    w_fp8 = (torch.randn(out_f, in_f, generator=g) * 0.5).to(torch.float8_e4m3fn)
    scale = torch.tensor(0.5, dtype=torch.float32)
    return w_fp8, scale


def _make_fp8_base_per_channel(out_f, in_f, seed=0):
    """FP8 base with a PER-OUTPUT-CHANNEL scale vector (shape (out_f,)).

    FP8LoRALinear / _FP8DequantLinear support a per-output-channel scale that
    broadcasts over the input dim (normalized to (out,1)); the scalar helper above
    only exercises the per-tensor case."""
    g = torch.Generator().manual_seed(seed)
    w_fp8 = (torch.randn(out_f, in_f, generator=g) * 0.5).to(torch.float8_e4m3fn)
    # Distinct positive per-channel scales (0.25 .. 1.0) so a dropped/transposed
    # broadcast would change the result.
    scale = (torch.rand(out_f, generator=g) * 0.75 + 0.25).to(torch.float32)
    return w_fp8, scale


# =========================================================================== #
# 1. Backward correctness of the custom autograd Functions.
# =========================================================================== #
def test_dequant_linear_gradcheck_float64():
    """gradcheck `_DequantLinear` in float64: analytic grad_x vs central differences."""
    out_f, in_f, group_size = 4, 16, 16
    weight_uint8, scales, scale2 = _make_nvfp4_base(out_f, in_f, group_size, seed=1)
    workspace = torch.empty(out_f, in_f, dtype=torch.float64)
    x = torch.randn(3, in_f, dtype=torch.float64, requires_grad=True)

    def fn(x_):
        return _DequantLinear.apply(
            x_, weight_uint8, scales, scale2, group_size, workspace, "modelopt"
        )

    # Only x requires grad; the quant buffers are non-differentiable (return None).
    assert torch.autograd.gradcheck(fn, (x,), eps=1e-6, atol=1e-7, rtol=1e-5)


def test_dequant_linear_grad_x_equals_analytic_reference():
    """grad_x must equal grad_output @ dequant(W); frozen buffers get no grad."""
    out_f, in_f, group_size = 5, 32, 16
    weight_uint8, scales, scale2 = _make_nvfp4_base(out_f, in_f, group_size, seed=2)
    workspace = torch.empty(out_f, in_f, dtype=torch.float64)
    x = torch.randn(3, in_f, dtype=torch.float64, requires_grad=True)

    y = _DequantLinear.apply(x, weight_uint8, scales, scale2, group_size, workspace, "modelopt")
    grad_output = torch.randn_like(y)
    y.backward(grad_output)

    W = _ref_nvfp4_W(weight_uint8, scales, scale2, group_size, torch.float64)
    expected = grad_output @ W
    assert torch.allclose(x.grad, expected, atol=1e-10, rtol=1e-8)

    # Frozen base/quant buffers must never accumulate a gradient.
    for buf in (weight_uint8, scales, scale2, workspace):
        assert buf.grad is None
        assert getattr(buf, "requires_grad", False) is False


def test_dequant_linear_gradcheck_float64_compressed_tensors():
    """gradcheck `_DequantLinear` with format='compressed_tensors'.

    The CT path uses per_tensor_effective = 1/scale2 (reciprocal) in dequant; the
    custom backward must thread `ctx.format` through so its recomputed W matches the
    forward W. A backward that dropped/misused `ctx.format` (e.g. defaulted to
    'modelopt') would recompute a *different* W and the analytic grad_x would no
    longer match central differences -> gradcheck fails."""
    out_f, in_f, group_size = 4, 16, 16
    weight_uint8, scales, scale2 = _make_nvfp4_base(out_f, in_f, group_size, seed=11)
    workspace = torch.empty(out_f, in_f, dtype=torch.float64)
    x = torch.randn(3, in_f, dtype=torch.float64, requires_grad=True)

    def fn(x_):
        return _DequantLinear.apply(
            x_, weight_uint8, scales, scale2, group_size, workspace, "compressed_tensors"
        )

    assert torch.autograd.gradcheck(fn, (x,), eps=1e-6, atol=1e-7, rtol=1e-5)


def test_dequant_linear_grad_x_carries_ct_format_through_backward():
    """grad_x must equal grad_output @ dequant_CT(W); a backward that ignored the CT
    format would instead match the modelopt dequant and fail this assertion."""
    out_f, in_f, group_size = 5, 32, 16
    weight_uint8, scales, scale2 = _make_nvfp4_base(out_f, in_f, group_size, seed=12)
    workspace = torch.empty(out_f, in_f, dtype=torch.float64)
    x = torch.randn(3, in_f, dtype=torch.float64, requires_grad=True)

    y = _DequantLinear.apply(
        x, weight_uint8, scales, scale2, group_size, workspace, "compressed_tensors"
    )
    grad_output = torch.randn_like(y)
    y.backward(grad_output)

    W_ct = _ref_nvfp4_W(weight_uint8, scales, scale2, group_size, torch.float64, fmt="compressed_tensors")
    expected = grad_output @ W_ct
    assert torch.allclose(x.grad, expected, atol=1e-10, rtol=1e-8)

    # And the CT dequant must DIFFER from the modelopt dequant for this base (scale2 != 1),
    # so the test above is genuinely format-sensitive rather than vacuously true.
    W_mo = _ref_nvfp4_W(weight_uint8, scales, scale2, group_size, torch.float64, fmt="modelopt")
    assert not torch.allclose(W_ct, W_mo)


def test_fp8_dequant_linear_gradcheck_float64():
    """gradcheck `_FP8DequantLinear` in float64."""
    out_f, in_f = 4, 6
    # Build the fp8 base directly so casting noise is fixed before gradcheck runs.
    w_fp8, scale = _make_fp8_base(out_f, in_f, seed=3)
    x = torch.randn(3, in_f, dtype=torch.float64, requires_grad=True)

    def fn(x_):
        return _FP8DequantLinear.apply(x_, w_fp8, scale)

    assert torch.autograd.gradcheck(fn, (x,), eps=1e-6, atol=1e-7, rtol=1e-5)


def test_fp8_dequant_linear_gradcheck_float64_per_channel_scale():
    """gradcheck `_FP8DequantLinear` with a PER-OUTPUT-CHANNEL scale (shape (out,1)).

    FP8LoRALinear normalizes a per-channel scale vector to (out,1) and broadcasts it
    over the input dim. This exercises that broadcast through the custom backward
    (the analytic grad_x = grad_output @ (w_fp8 * scale) must agree with central
    differences when scale varies per output row)."""
    out_f, in_f = 5, 7
    w_fp8, scale_vec = _make_fp8_base_per_channel(out_f, in_f, seed=13)
    # Same normalization FP8LoRALinear applies: (out,) -> (out,1).
    scale = scale_vec.reshape(out_f, 1)
    x = torch.randn(3, in_f, dtype=torch.float64, requires_grad=True)

    def fn(x_):
        return _FP8DequantLinear.apply(x_, w_fp8, scale)

    assert torch.autograd.gradcheck(fn, (x,), eps=1e-6, atol=1e-7, rtol=1e-5)


def test_fp8_dequant_linear_grad_x_per_channel_scale_reference():
    """Per-channel-scale grad_x must equal grad_output @ (w_fp8 * scale_per_channel)."""
    out_f, in_f = 6, 9
    w_fp8, scale_vec = _make_fp8_base_per_channel(out_f, in_f, seed=14)
    scale = scale_vec.reshape(out_f, 1)
    x = torch.randn(4, in_f, dtype=torch.float64, requires_grad=True)

    y = _FP8DequantLinear.apply(x, w_fp8, scale)
    grad_output = torch.randn_like(y)
    y.backward(grad_output)

    W = w_fp8.to(torch.float64) * scale.to(torch.float64)  # broadcast (out,1) over in
    expected = grad_output @ W
    assert torch.allclose(x.grad, expected, atol=1e-10, rtol=1e-8)
    for buf in (w_fp8, scale):
        assert buf.grad is None
        assert getattr(buf, "requires_grad", False) is False


def test_fp8_dequant_linear_grad_x_equals_analytic_reference():
    out_f, in_f = 5, 7
    w_fp8, scale = _make_fp8_base(out_f, in_f, seed=4)
    x = torch.randn(4, in_f, dtype=torch.float64, requires_grad=True)

    y = _FP8DequantLinear.apply(x, w_fp8, scale)
    grad_output = torch.randn_like(y)
    y.backward(grad_output)

    W = w_fp8.to(torch.float64) * scale.to(torch.float64)  # independent dequant
    expected = grad_output @ W
    assert torch.allclose(x.grad, expected, atol=1e-10, rtol=1e-8)

    for buf in (w_fp8, scale):
        assert buf.grad is None
        assert getattr(buf, "requires_grad", False) is False


def test_lora_module_backward_only_trains_adapter():
    """End-to-end through NVFP4LoRALinear (train mode): A/B get grad, base does not,
    and grad flows to x via the custom backward."""
    out_f, in_f, group_size, r, alpha = 6, 32, 16, 4, 8
    weight_uint8, scales, scale2 = _make_nvfp4_base(out_f, in_f, group_size, seed=5)
    m = NVFP4LoRALinear(
        in_f, out_f, weight_uint8, scales, scale2,
        group_size=group_size, r=r, lora_alpha=alpha, dtype=torch.float32,
    )
    m.w_bf16_workspace = torch.empty(out_f, in_f, dtype=torch.float32)
    m.train()
    with torch.no_grad():
        m.lora_B.copy_(torch.randn(out_f, r) * 0.1)  # non-zero so A receives grad

    x = torch.randn(2, in_f, requires_grad=True)
    m(x).sum().backward()

    assert m.lora_A.grad is not None and m.lora_A.grad.abs().sum() > 0
    assert m.lora_B.grad is not None and m.lora_B.grad.abs().sum() > 0
    assert x.grad is not None
    # Frozen base buffers carry no grad.
    assert m.weight_uint8.grad is None
    assert not isinstance(m.weight_uint8, torch.nn.Parameter)


# =========================================================================== #
# 2. Logits parity per quant path vs an INDEPENDENT reference.
# =========================================================================== #
def _ref_lora_logits(x, W, A, B, scale):
    """Independent oracle: base + (alpha/r) * x A^T B^T. Does NOT call the module."""
    return F.linear(x, W) + scale * (x @ A.T @ B.T)


def test_nvfp4_logits_parity():
    out_f, in_f, group_size, r, alpha = 6, 32, 16, 4, 8
    weight_uint8, scales, scale2 = _make_nvfp4_base(out_f, in_f, group_size, seed=6)
    m = NVFP4LoRALinear(
        in_f, out_f, weight_uint8, scales, scale2,
        group_size=group_size, r=r, lora_alpha=alpha, dtype=torch.float32,
    )
    g = torch.Generator().manual_seed(60)
    with torch.no_grad():
        m.lora_A.copy_(torch.randn(r, in_f, generator=g) * 0.2)
        m.lora_B.copy_(torch.randn(out_f, r, generator=g) * 0.2)
    m.eval()  # uses the eval bf16 cache path; reference is independent regardless

    x = torch.randn(3, in_f)
    W = _ref_nvfp4_W(weight_uint8, scales, scale2, group_size, torch.float32)
    ref = _ref_lora_logits(x, W, m.lora_A.detach(), m.lora_B.detach(), alpha / r)
    # NVFP4 looser: bf16-derived LoRA params + group-scaled E2M1 base. Here all fp32,
    # so the only slack is float32 matmul ordering -> a modest absolute tolerance.
    assert torch.allclose(m(x), ref, atol=1e-4, rtol=1e-4)


def test_nvfp4_logits_parity_compressed_tensors():
    """Same parity check on a compressed-tensors module. The module must dequant its
    base with the CT reciprocal-scale semantics; the independent reference does too.
    A module that misused ctx.format / the CT scale would diverge from the reference."""
    out_f, in_f, group_size, r, alpha = 6, 32, 16, 4, 8
    weight_uint8, scales, scale2 = _make_nvfp4_base(out_f, in_f, group_size, seed=61)
    m = NVFP4LoRALinear(
        in_f, out_f, weight_uint8, scales, scale2,
        group_size=group_size, r=r, lora_alpha=alpha, dtype=torch.float32,
        format="compressed_tensors",
    )
    g = torch.Generator().manual_seed(610)
    with torch.no_grad():
        m.lora_A.copy_(torch.randn(r, in_f, generator=g) * 0.2)
        m.lora_B.copy_(torch.randn(out_f, r, generator=g) * 0.2)
    m.eval()  # exercises the eval bf16 cache path under the CT format

    x = torch.randn(3, in_f)
    W = _ref_nvfp4_W(weight_uint8, scales, scale2, group_size, torch.float32, fmt="compressed_tensors")
    ref = _ref_lora_logits(x, W, m.lora_A.detach(), m.lora_B.detach(), alpha / r)
    assert torch.allclose(m(x), ref, atol=1e-4, rtol=1e-4)

    # Also check a train-mode forward (uses the _DequantLinear custom Function, which
    # must thread ctx.format) matches the same CT reference.
    m.train()
    m.w_bf16_workspace = torch.empty(out_f, in_f, dtype=torch.float32)
    assert torch.allclose(m(x), ref, atol=1e-4, rtol=1e-4)


def test_fp8_logits_parity():
    out_f, in_f, r, alpha = 6, 8, 4, 8
    w_fp8, scale = _make_fp8_base(out_f, in_f, seed=7)
    m = FP8LoRALinear(in_f, out_f, w_fp8, scale, r=r, lora_alpha=alpha, dtype=torch.float32)
    g = torch.Generator().manual_seed(70)
    with torch.no_grad():
        m.lora_A.copy_(torch.randn(r, in_f, generator=g) * 0.2)
        m.lora_B.copy_(torch.randn(out_f, r, generator=g) * 0.2)

    x = torch.randn(3, in_f)
    W = w_fp8.to(torch.float32) * scale  # independent dequant
    ref = _ref_lora_logits(x, W, m.lora_A.detach(), m.lora_B.detach(), alpha / r)
    assert torch.allclose(m(x), ref, atol=1e-5, rtol=1e-5)


def test_fp8_logits_parity_per_channel_scale():
    """FP8 parity with a per-output-channel scale vector: the module normalizes (out,)
    -> (out,1) and broadcasts over the input dim; the reference uses the same."""
    out_f, in_f, r, alpha = 6, 8, 4, 8
    w_fp8, scale_vec = _make_fp8_base_per_channel(out_f, in_f, seed=71)
    m = FP8LoRALinear(in_f, out_f, w_fp8, scale_vec, r=r, lora_alpha=alpha, dtype=torch.float32)
    g = torch.Generator().manual_seed(710)
    with torch.no_grad():
        m.lora_A.copy_(torch.randn(r, in_f, generator=g) * 0.2)
        m.lora_B.copy_(torch.randn(out_f, r, generator=g) * 0.2)

    x = torch.randn(3, in_f)
    W = w_fp8.to(torch.float32) * scale_vec.reshape(out_f, 1)  # independent dequant
    ref = _ref_lora_logits(x, W, m.lora_A.detach(), m.lora_B.detach(), alpha / r)
    assert torch.allclose(m(x), ref, atol=1e-5, rtol=1e-5)


def test_bf16_logits_parity():
    out_f, in_f, r, alpha = 6, 8, 4, 8
    g = torch.Generator().manual_seed(80)
    w = torch.randn(out_f, in_f, generator=g)
    m = BF16LoRALinear(in_f, out_f, w, r=r, lora_alpha=alpha, dtype=torch.float32)
    with torch.no_grad():
        m.lora_A.copy_(torch.randn(r, in_f, generator=g) * 0.2)
        m.lora_B.copy_(torch.randn(out_f, r, generator=g) * 0.2)

    x = torch.randn(3, in_f)
    ref = _ref_lora_logits(x, w, m.lora_A.detach(), m.lora_B.detach(), alpha / r)
    assert torch.allclose(m(x), ref, atol=1e-5, rtol=1e-5)


# =========================================================================== #
# 3. Bind-but-zero-effect must be detectable.
# =========================================================================== #
@pytest.mark.parametrize("variant", ["nvfp4", "fp8", "bf16"])
def test_zero_B_is_base_identical_and_nonzero_B_differs(variant):
    """lora_B == 0 -> output exactly equals the base path; a non-zero lora_B -> a
    measurably different output. Guards against a silently-bound no-op adapter."""
    out_f, in_f, group_size, r, alpha = 6, 32, 16, 4, 8

    if variant == "nvfp4":
        weight_uint8, scales, scale2 = _make_nvfp4_base(out_f, in_f, group_size, seed=9)
        m = NVFP4LoRALinear(
            in_f, out_f, weight_uint8, scales, scale2,
            group_size=group_size, r=r, lora_alpha=alpha, dtype=torch.float32,
        )
        base_W = _ref_nvfp4_W(weight_uint8, scales, scale2, group_size, torch.float32)
        atol = 1e-4
    elif variant == "fp8":
        w_fp8, scale = _make_fp8_base(out_f, in_f, seed=9)
        m = FP8LoRALinear(in_f, out_f, w_fp8, scale, r=r, lora_alpha=alpha, dtype=torch.float32)
        base_W = w_fp8.to(torch.float32) * scale
        atol = 1e-5
    else:  # bf16
        g = torch.Generator().manual_seed(9)
        w = torch.randn(out_f, in_f, generator=g)
        m = BF16LoRALinear(in_f, out_f, w, r=r, lora_alpha=alpha, dtype=torch.float32)
        base_W = w
        atol = 1e-5

    m.eval()
    x = torch.randn(3, in_f)
    base = F.linear(x, base_W)

    # lora_B is zero-initialized -> bound but no effect -> base-identical.
    assert torch.allclose(m.lora_B, torch.zeros_like(m.lora_B))
    assert torch.allclose(m(x), base, atol=atol)

    # A non-zero lora_B must visibly move the output.
    with torch.no_grad():
        m.lora_A.copy_(torch.randn_like(m.lora_A) + 0.5)  # ensure A is non-zero too
        m.lora_B.copy_(torch.randn_like(m.lora_B) + 0.5)
    out = m(x)
    diff = (out - base).abs().max().item()
    assert diff > 1e-3, f"non-zero lora_B produced a no-op output (max diff {diff})"
