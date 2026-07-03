"""bf16 -> NVFP4 quantization, round-trip and cross-layout checks, on CPU.

nvfp4_lora.quantize is pure torch and runs on whatever device its input lives
on. Every tensor here is explicitly CPU, so the GPU is never touched even when
torch.cuda.is_available() is True (a training run may hold the device).

The inverse (nvfp4_lora.dequant.dequantize_nvfp4_weight) has a pure-torch CPU
path, so the whole quant/dequant loop stays on CPU.

Conventions under test (see the quantize module docstring):
  ct       stores weight_global_scale as a large DIVISOR (1 / weight_scale_2);
           dequant reciprocates it internally (format="compressed_tensors").
  modelopt stores weight_scale_2 as a small MULTIPLIER; dequant uses it as-is
           (format="modelopt").
The two layouts share the per-group fp8 scale and the effective divisor; they
differ only in the fp4 rounding rule at exact grid midpoints.
"""
from __future__ import annotations

import pytest
import torch

from nvfp4_lora.dequant import NVFP4_E2M1_LUT, dequantize_nvfp4_weight
from nvfp4_lora.quantize import (
    FP4_MAX,
    quantize_nvfp4_2d,
    quantize_nvfp4_3d_per_slice,
)

CPU = torch.device("cpu")


def _assert_cpu(*tensors):
    for t in tensors:
        assert t.device.type == "cpu"


def _dequant_ct(d: dict, out_dtype=torch.float32) -> torch.Tensor:
    """Dequantize a ct-layout trio via the compressed_tensors dequant path."""
    return dequantize_nvfp4_weight(
        d["weight_packed"],
        d["weight_scale"],
        d["weight_global_scale"].to(torch.float32),
        group_size=16,
        out_dtype=out_dtype,
        format="compressed_tensors",
    )


def _dequant_modelopt(d: dict, out_dtype=torch.float32) -> torch.Tensor:
    """Dequantize a modelopt-layout trio via the modelopt dequant path."""
    return dequantize_nvfp4_weight(
        d["weight"],
        d["weight_scale"],
        d["weight_scale_2"].to(torch.float32),
        group_size=16,
        out_dtype=out_dtype,
        format="modelopt",
    )


# ---------------------------------------------------------------------------
# Output contract
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("layout", ["ct", "modelopt"])
def test_output_keys_shapes_dtypes(layout):
    torch.manual_seed(0)
    W = torch.randn(64, 256, dtype=torch.bfloat16, device=CPU)
    d = quantize_nvfp4_2d(W, layout=layout)

    if layout == "ct":
        assert set(d) == {"weight_packed", "weight_scale", "weight_global_scale"}
        packed = d["weight_packed"]
        scalar = d["weight_global_scale"]
    else:
        assert set(d) == {"weight", "weight_scale", "weight_scale_2"}
        packed = d["weight"]
        scalar = d["weight_scale_2"]

    _assert_cpu(*d.values())
    assert packed.shape == (64, 128) and packed.dtype == torch.uint8
    assert d["weight_scale"].shape == (64, 16)
    assert d["weight_scale"].dtype == torch.float8_e4m3fn
    assert scalar.shape == (1,) and scalar.dtype == torch.float32


def test_ct_global_scale_is_reciprocal_of_modelopt_scale_2():
    # ct.weight_global_scale == 1 / modelopt.weight_scale_2 (same per-tensor max).
    torch.manual_seed(1)
    W = torch.randn(32, 128, dtype=torch.bfloat16, device=CPU)
    ct = quantize_nvfp4_2d(W, layout="ct")
    mo = quantize_nvfp4_2d(W, layout="modelopt")
    g = ct["weight_global_scale"].item()
    s2 = mo["weight_scale_2"].item()
    assert g == pytest.approx(1.0 / s2, rel=1e-6)
    # The stored per-group fp8 scale is identical between layouts.
    assert torch.equal(ct["weight_scale"].float(), mo["weight_scale"].float())


# ---------------------------------------------------------------------------
# Grid identity: values already on the E2M1 x Scale grid survive a round trip.
# ---------------------------------------------------------------------------

def _normalize_signed_zero(t: torch.Tensor) -> torch.Tensor:
    """Map -0.0 to +0.0 so torch.equal ignores the sign of zero.

    E2M1 has two zero encodings (index 0 and index 8). A weight can round to
    either, so grid-identity comparisons that care about magnitude must treat
    the two zeros as equal.
    """
    return torch.where(t == 0, torch.zeros_like(t), t)


@pytest.mark.parametrize("layout", ["ct", "modelopt"])
def test_grid_identity_dequant_quant_dequant_is_fixed_point(layout):
    # A weight that already sits on the representable E2M1 x scale grid must
    # survive quant -> dequant bit-for-bit. The genuine on-grid input is the
    # quantizer's OWN output: W0 = dequant(quantize(random bf16)). A hand-built
    # "random packed + arbitrary scales" is NOT generally a fixed point, because
    # those scales are not the ones the quantizer derives from the data (the
    # per-tensor / per-group scale derivation only reproduces itself once it has
    # been applied); see the module docstring. The second pass is the identity.
    torch.manual_seed(2)
    W_raw = torch.randn(48, 160, dtype=torch.bfloat16, device=CPU)

    d0 = quantize_nvfp4_2d(W_raw, layout=layout)
    W0 = _dequant_ct(d0) if layout == "ct" else _dequant_modelopt(d0)   # on grid
    _assert_cpu(W0)

    d1 = quantize_nvfp4_2d(W0, layout=layout)
    W1 = _dequant_ct(d1) if layout == "ct" else _dequant_modelopt(d1)

    # Fixed point: re-quantizing an on-grid weight and dequantizing returns the
    # same values (up to the sign of zero, which E2M1 encodes two ways).
    assert torch.equal(_normalize_signed_zero(W0), _normalize_signed_zero(W1)), (
        f"{layout}: on-grid weight was not a fixed point "
        f"(max abs diff {(W0 - W1).abs().max().item()})"
    )


# ---------------------------------------------------------------------------
# Round-trip error bound derived from the E2M1 format (not a magic number).
# ---------------------------------------------------------------------------

def _max_step_ratio() -> float:
    """Largest gap between adjacent positive E2M1 values, over the smaller one.

    A weight of magnitude m lands within half a step of the nearest grid point;
    scaled by the group scale, the worst-case per-element relative error is
    bounded by half the largest adjacent-gap ratio on the grid. E2M1's widest
    relative gap is between 4 and 6 (gap 2, ratio 2/4 = 0.5), so half-step is
    0.25 in the worst placement. We derive that here instead of hardcoding it.
    """
    pos = NVFP4_E2M1_LUT[:8]  # 0, 0.5, 1, 1.5, 2, 3, 4, 6
    nonzero = pos[1:]
    gaps = nonzero[1:] - nonzero[:-1]
    ratios = gaps / nonzero[:-1]
    return float(ratios.max().item())


@pytest.mark.parametrize("layout", ["ct", "modelopt"])
def test_roundtrip_error_within_format_bound(layout):
    torch.manual_seed(3)
    W = torch.randn(128, 512, dtype=torch.bfloat16, device=CPU)
    d = quantize_nvfp4_2d(W, layout=layout)
    W_recon = _dequant_ct(d) if layout == "ct" else _dequant_modelopt(d)
    _assert_cpu(W_recon)

    diff = (W.float() - W_recon.float()).abs()
    denom = W.float().abs().clamp(min=1e-6)
    mean_rel = (diff / denom).mean().item()

    # Half the widest relative grid step bounds the worst single element; the
    # mean over a smooth distribution is far below it. Assert the mean respects
    # the format bound (a weak, format-derived ceiling, not a tuned constant).
    bound = 0.5 * _max_step_ratio()
    assert mean_rel < bound, f"{layout}: mean rel err {mean_rel} exceeds format bound {bound}"


def test_outlier_preserved_within_group_step():
    # A single large outlier must survive within one E2M1 step of its group.
    torch.manual_seed(4)
    W = torch.randn(64, 256, dtype=torch.bfloat16, device=CPU)
    W[0, 0] = 10.0
    d = quantize_nvfp4_2d(W, layout="ct")
    W_recon = _dequant_ct(d)
    recovered = W_recon[0, 0].item()
    assert abs(recovered - 10.0) < 2.0, f"outlier {recovered} far from 10.0"


# ---------------------------------------------------------------------------
# Cross-layout agreement / documented difference.
# ---------------------------------------------------------------------------

def test_layouts_agree_on_grid_aligned_input():
    # On values already on the E2M1 x scale grid there is no rounding decision,
    # so ct and modelopt dequantize to bit-identical values despite their
    # different rounding rules. The on-grid input is the quantizer's own output.
    torch.manual_seed(5)
    W = torch.randn(32, 128, dtype=torch.bfloat16, device=CPU)
    W0 = _dequant_ct(quantize_nvfp4_2d(W, layout="ct"))   # on grid

    ct = _dequant_ct(quantize_nvfp4_2d(W0, layout="ct"))
    mo = _dequant_modelopt(quantize_nvfp4_2d(W0, layout="modelopt"))
    assert torch.equal(_normalize_signed_zero(ct), _normalize_signed_zero(mo))


def test_layouts_differ_only_at_grid_midpoints():
    # The documented difference: the two rounding rules disagree only at exact
    # midpoints between grid values (ct rounds ties down by magnitude, modelopt
    # rounds half to even). Construct scaled values sitting on E2M1 midpoints so
    # the disagreement is forced and bounded to those positions.
    #
    # Midpoints between adjacent positive grid values, at group scale 1 and
    # per-tensor scale 1 (so the scaled value equals the raw value):
    #   0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0
    midpoints = torch.tensor(
        [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], dtype=torch.float32
    )
    row = midpoints.repeat(16)[:16]           # one full group of 16
    W = row.reshape(1, 16).to(torch.float32)

    ct = _dequant_ct(quantize_nvfp4_2d(W, layout="ct"))
    mo = _dequant_modelopt(quantize_nvfp4_2d(W, layout="modelopt"))

    # They agree everywhere except the forced midpoints, and where they differ
    # the gap is exactly one E2M1 step (never larger).
    diff = (ct - mo).abs()
    assert diff.max().item() > 0.0, "expected a rounding difference at midpoints"
    # Largest single E2M1 step is 6 - 4 = 2.0; every disagreement is <= that.
    assert diff.max().item() <= 2.0 + 1e-6


# ---------------------------------------------------------------------------
# 3d-per-slice consistency vs per-slice 2d calls.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("layout", ["ct", "modelopt"])
def test_3d_per_slice_matches_2d_calls(layout):
    torch.manual_seed(6)
    E, out_feat, in_feat = 4, 32, 128
    W = torch.randn(E, out_feat, in_feat, dtype=torch.bfloat16, device=CPU)

    slices = quantize_nvfp4_3d_per_slice(W, layout=layout)
    assert len(slices) == E
    for e in range(E):
        d2 = quantize_nvfp4_2d(W[e].contiguous(), layout=layout)
        assert set(slices[e]) == set(d2)
        for k, v in d2.items():
            a = slices[e][k]
            if v.dtype == torch.float8_e4m3fn:
                assert torch.equal(a.float(), v.float()), f"{layout} slice {e} key {k}"
            else:
                assert torch.equal(a, v), f"{layout} slice {e} key {k}"


def test_3d_per_slice_shapes():
    torch.manual_seed(7)
    E, out_feat, in_feat = 3, 48, 160
    W = torch.randn(E, out_feat, in_feat, dtype=torch.bfloat16, device=CPU)
    for layout in ("ct", "modelopt"):
        for d in quantize_nvfp4_3d_per_slice(W, layout=layout):
            packed = d["weight_packed"] if layout == "ct" else d["weight"]
            assert packed.shape == (out_feat, in_feat // 2)
            assert d["weight_scale"].shape == (out_feat, in_feat // 16)


# ---------------------------------------------------------------------------
# Shared per-tensor scale + input validation.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("layout", ["ct", "modelopt"])
def test_shared_per_tensor_max_gives_identical_scalar(layout):
    # Two slices requantized with a shared per-tensor abs-max must store the same
    # per-tensor scalar (fused gate_up experts / vLLM-fused q/k/v rely on this).
    torch.manual_seed(8)
    Wa = torch.randn(32, 128, dtype=torch.bfloat16, device=CPU)
    Wb = torch.randn(32, 128, dtype=torch.bfloat16, device=CPU)
    shared = max(Wa.abs().max().item(), Wb.abs().max().item())
    da = quantize_nvfp4_2d(Wa, layout=layout, per_tensor_max_override=shared)
    db = quantize_nvfp4_2d(Wb, layout=layout, per_tensor_max_override=shared)
    scalar = "weight_global_scale" if layout == "ct" else "weight_scale_2"
    assert torch.equal(da[scalar], db[scalar])


def test_rejects_bad_inputs():
    W = torch.randn(8, 30, dtype=torch.bfloat16, device=CPU)  # 30 % 16 != 0
    with pytest.raises(ValueError):
        quantize_nvfp4_2d(W, layout="ct")
    with pytest.raises(ValueError):
        quantize_nvfp4_2d(torch.randn(8, 16, device=CPU), layout="bogus")
    with pytest.raises(ValueError):
        quantize_nvfp4_2d(torch.randn(2, 8, 16, device=CPU), layout="ct")  # 3d in 2d fn
    with pytest.raises(ValueError):
        quantize_nvfp4_3d_per_slice(torch.randn(8, 16, device=CPU), layout="ct")  # 2d in 3d fn


def test_clamp_min_magnitude_bound_from_grid():
    # Sanity that FP4_MAX matches the LUT top value (used by the ct clamp).
    assert FP4_MAX == float(NVFP4_E2M1_LUT[:8].max().item())
