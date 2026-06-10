"""NVFP4 -> high-precision dequant, known-answer, on CPU.

nvfp4_lora.dequant.dequantize_nvfp4_weight has a pure-torch path that runs whenever
the input tensors are NOT on a CUDA device (the Triton fast path is gated on
device.type == "cuda"). All tensors here are explicitly CPU, so the GPU is never
touched even though torch.cuda.is_available() may be True on this box.

The E2M1 4-bit lookup table is the ground truth:
    index: 0    1    2   3    4   5   6   7    8    9   10    11   12   13   14   15
    value: 0   0.5  1  1.5   2   3   4   6   -0  -0.5 -1  -1.5  -2   -3   -4   -6
"""
from __future__ import annotations

import pytest
import torch

from nvfp4_lora.dequant import (
    NVFP4_E2M1_LUT,
    dequantize_nvfp4_weight,
    _unpack_nibbles,
)

CPU = torch.device("cpu")


def _assert_cpu(*tensors):
    for t in tensors:
        assert t.device.type == "cpu"


def test_unpack_nibbles_low_first():
    # byte = (high << 4) | low ; unpack must interleave [low0, high0, low1, high1, ...]
    packed = torch.tensor([[0x21, 0xF0]], dtype=torch.uint8, device=CPU)  # (low,high)=(1,2),(0,15)
    out = _unpack_nibbles(packed)
    assert out[0].tolist() == [1, 2, 0, 15]


def _make_packed(lows, highs):
    bytes_ = [(h << 4) | l for l, h in zip(lows, highs)]
    return torch.tensor([bytes_], dtype=torch.uint8, device=CPU)


def test_dequant_identity_scales_matches_lut():
    # group_size 16 -> one group over 16 unpacked values. scale=1 everywhere means the
    # dequantized weight is exactly the LUT value for each nibble.
    lows = [1, 3, 5, 7, 0, 2, 4, 6]
    highs = [2, 4, 6, 0, 9, 11, 13, 15]
    packed = _make_packed(lows, highs)
    scale = torch.ones((1, 1), dtype=torch.float8_e4m3fn, device=CPU)
    scale2 = torch.tensor(1.0, dtype=torch.float32, device=CPU)
    _assert_cpu(packed, scale, scale2)

    out = dequantize_nvfp4_weight(packed, scale, scale2, group_size=16,
                                  out_dtype=torch.float32, format="modelopt")
    _assert_cpu(out)
    assert out.shape == (1, 16)

    expected_idx = []
    for l, h in zip(lows, highs):
        expected_idx += [l, h]
    expected = NVFP4_E2M1_LUT[torch.tensor(expected_idx)]
    assert torch.allclose(out[0], expected)


def test_dequant_applies_group_and_per_tensor_scale():
    lows = [1, 3, 5, 7, 0, 2, 4, 6]
    highs = [2, 4, 6, 0, 9, 11, 13, 15]
    packed = _make_packed(lows, highs)
    group_scale = torch.full((1, 1), 3.0, dtype=torch.float8_e4m3fn, device=CPU)
    per_tensor = torch.tensor(2.0, dtype=torch.float32, device=CPU)

    out = dequantize_nvfp4_weight(packed, group_scale, per_tensor, group_size=16,
                                  out_dtype=torch.float32, format="modelopt")
    expected_idx = []
    for l, h in zip(lows, highs):
        expected_idx += [l, h]
    expected = NVFP4_E2M1_LUT[torch.tensor(expected_idx)] * 3.0 * 2.0
    assert torch.allclose(out[0], expected)


def test_dequant_compressed_tensors_inverts_per_tensor_scale():
    # In compressed-tensors format the per-tensor scale is applied as 1 / scale2,
    # and the scalar may be supplied as a shape-(1,) tensor.
    lows = [2, 4, 6, 1, 3, 5, 7, 0]
    highs = [1, 1, 1, 1, 1, 1, 1, 1]
    packed = _make_packed(lows, highs)
    group_scale = torch.ones((1, 1), dtype=torch.float8_e4m3fn, device=CPU)
    scale2 = torch.tensor([0.5], dtype=torch.float32, device=CPU)  # 1/0.5 == 2.0

    out = dequantize_nvfp4_weight(packed, group_scale, scale2, group_size=16,
                                  out_dtype=torch.float32, format="compressed_tensors")
    expected_idx = []
    for l, h in zip(lows, highs):
        expected_idx += [l, h]
    expected = NVFP4_E2M1_LUT[torch.tensor(expected_idx)] * 2.0
    assert torch.allclose(out[0], expected)


def test_dequant_multi_group_and_multi_row():
    # 2 rows x 32 in-features = 2 groups per row. Distinct group scales per group/row.
    packed = torch.zeros((2, 16), dtype=torch.uint8, device=CPU)
    # nibble value 2 -> LUT 1.0 in every position (byte 0x22 packs low=2, high=2)
    packed[:] = 0x22
    group_scale = torch.tensor(
        [[2.0, 4.0], [8.0, 1.0]], dtype=torch.float8_e4m3fn, device=CPU
    )
    per_tensor = torch.tensor(1.0, dtype=torch.float32, device=CPU)
    out = dequantize_nvfp4_weight(packed, group_scale, per_tensor, group_size=16,
                                  out_dtype=torch.float32, format="modelopt")
    assert out.shape == (2, 32)
    # row 0 group 0 -> 1.0 * 2.0; group 1 -> 1.0 * 4.0; row 1 -> 8.0 and 1.0
    assert torch.allclose(out[0, :16], torch.full((16,), 2.0))
    assert torch.allclose(out[0, 16:], torch.full((16,), 4.0))
    assert torch.allclose(out[1, :16], torch.full((16,), 8.0))
    assert torch.allclose(out[1, 16:], torch.full((16,), 1.0))


def test_dequant_into_out_buffer():
    lows = [1, 2, 3, 4, 5, 6, 7, 0]
    highs = [0] * 8
    packed = _make_packed(lows, highs)
    group_scale = torch.ones((1, 1), dtype=torch.float8_e4m3fn, device=CPU)
    per_tensor = torch.tensor(1.0, dtype=torch.float32, device=CPU)
    out_buf = torch.empty((1, 16), dtype=torch.float32, device=CPU)
    ret = dequantize_nvfp4_weight(packed, group_scale, per_tensor, group_size=16,
                                  out_dtype=torch.float32, out=out_buf, format="modelopt")
    assert ret is out_buf
    assert out_buf.device.type == "cpu"


def test_dequant_rejects_bad_dtypes():
    packed = torch.zeros((1, 8), dtype=torch.uint8, device=CPU)
    scale = torch.ones((1, 1), dtype=torch.float8_e4m3fn, device=CPU)
    scale2 = torch.tensor(1.0, dtype=torch.float32, device=CPU)
    with pytest.raises(ValueError):
        dequantize_nvfp4_weight(packed, scale, scale2, group_size=16, format="bogus")
    with pytest.raises(TypeError):
        # weight must be uint8
        dequantize_nvfp4_weight(packed.float(), scale, scale2, group_size=16)
