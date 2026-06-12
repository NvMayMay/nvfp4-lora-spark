#!/usr/bin/env python3
"""Phase 0.1 smoke tests for compressed-tensors NVFP4 decode and key sniffing."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import safetensors
import torch

from nvfp4_lora.dequant import dequantize_nvfp4_weight, format_for_record
from nvfp4_lora.loader import list_quantized_modules


MODEL_DIR = Path("/home/veritan-spark-01/Veritan/Models/Qwen3.5-122B-A10B-NVFP4")
CT_PREFIX = "model.language_model.layers.3.mlp.experts.0.gate_proj"


def _require_qwen35_model() -> Path:
    idx_path = MODEL_DIR / "model.safetensors.index.json"
    if not idx_path.exists():
        pytest.skip(f"Qwen3.5 NVFP4 model not found at {MODEL_DIR}")
    return MODEL_DIR


def _load_tensors(model_dir: Path, prefix: str) -> dict[str, torch.Tensor]:
    idx = json.loads((model_dir / "model.safetensors.index.json").read_text())
    wm = idx["weight_map"]
    needed = {
        "weight_packed": f"{prefix}.weight_packed",
        "weight_scale": f"{prefix}.weight_scale",
        "weight_global_scale": f"{prefix}.weight_global_scale",
        "input_global_scale": f"{prefix}.input_global_scale",
    }
    tensors = {}
    for short, key in needed.items():
        shard = wm[key]
        with safetensors.safe_open(str(model_dir / shard), framework="pt", device="cpu") as f:
            tensors[short] = f.get_tensor(key)
    return tensors


def test_ct_dequant_qwen35_expert_tensor_bit_contract():
    model_dir = _require_qwen35_model()
    tensors = _load_tensors(model_dir, CT_PREFIX)

    out = dequantize_nvfp4_weight(
        tensors["weight_packed"],
        tensors["weight_scale"],
        tensors["weight_global_scale"],
        format="compressed_tensors",
    )

    assert tuple(out.shape) == (1024, 3072)
    assert out.dtype == torch.bfloat16
    assert torch.isfinite(out.float()).all()


def test_ct_and_modelopt_dequant_are_bit_identical_for_same_bits():
    weight = torch.tensor(
        [
            [0x10, 0x32, 0x54, 0x76, 0x98, 0xBA, 0xDC, 0xFE],
            [0x01, 0x23, 0x45, 0x67, 0x89, 0xAB, 0xCD, 0xEF],
        ],
        dtype=torch.uint8,
    )
    scale_values = torch.tensor([[1.0], [0.5]], dtype=torch.float32)
    scale_fp8 = scale_values.to(torch.float8_e4m3fn)
    modelopt_scale2 = torch.tensor(0.25, dtype=torch.float32)
    ct_scale2 = torch.tensor([0.25], dtype=torch.float32)

    modelopt = dequantize_nvfp4_weight(
        weight,
        scale_fp8,
        modelopt_scale2,
        group_size=16,
        format="modelopt",
    )
    ct = dequantize_nvfp4_weight(
        weight,
        scale_fp8,
        ct_scale2,
        group_size=16,
        format="compressed_tensors",
    )

    assert torch.equal(modelopt, ct)


def test_format_for_record_detects_formats_and_rejects_ambiguous_keys():
    prefix = "model.layers.0.self_attn.q_proj"

    assert format_for_record(
        {
            f"{prefix}.weight_packed",
            f"{prefix}.weight_scale",
            f"{prefix}.weight_global_scale",
            f"{prefix}.input_global_scale",
        },
        prefix,
    ) == "compressed_tensors"

    assert format_for_record(
        {
            f"{prefix}.weight",
            f"{prefix}.weight_scale",
            f"{prefix}.weight_scale_2",
        },
        prefix,
    ) == "modelopt"

    with pytest.raises(ValueError):
        format_for_record({f"{prefix}.weight_scale"}, prefix)

    with pytest.raises(ValueError):
        format_for_record({f"{prefix}.weight", f"{prefix}.weight_packed", f"{prefix}.weight_scale"}, prefix)


def test_qwen35_list_quantized_modules_finds_ct_weight_packed_modules():
    model_dir = _require_qwen35_model()

    modules = list_quantized_modules(model_dir)

    assert len(modules) > 0
    assert CT_PREFIX in modules
