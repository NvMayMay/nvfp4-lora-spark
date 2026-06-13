"""NVFP4Experts3D storage modes: ModelOpt vs compressed-tensors keys, fused vs
split gate/up per-tensor scales.

Builds tiny synthetic per-expert NVFP4 checkpoints (random packed nibbles +
random fp8 block scales + per-projection per-tensor scales), assembles them
through the real assembly path, and checks the module forward (both the
grouped and the legacy per-expert path) against a reference implementation
that dequantizes each projection independently with dequantize_nvfp4_weight.
Everything runs on CPU in float32, so parity is tight.
"""
from __future__ import annotations

import json

import pytest
import torch
from safetensors.torch import save_file

from nvfp4_lora.dequant import dequantize_nvfp4_weight
from nvfp4_lora.experts import (
    NVFP4Experts3D,
    assemble_nvfp4_experts3d_batched,
    detect_moe_expert_storage,
)

E, H, I, GS = 4, 32, 16, 16  # experts, hidden, intermediate, group size
PREFIX = "model.layers.0.mlp.experts"


def _rand_proj(gen, out_f, in_f):
    packed = torch.randint(0, 256, (out_f, in_f // 2), dtype=torch.uint8, generator=gen)
    scale = (torch.rand(out_f, in_f // GS, generator=gen) * 0.5 + 0.25).to(torch.float8_e4m3fn)
    return packed, scale


def _build_checkpoint(tmp_path, fmt: str, equal_gate_up: bool):
    """Write a one-shard synthetic checkpoint; return (weight_map, refs) where
    refs[proj][e] is the reference dense float32 weight."""
    gen = torch.Generator().manual_seed(0)
    w_key, s_key, g_key = (
        ("weight", "weight_scale", "weight_scale_2") if fmt == "modelopt"
        else ("weight_packed", "weight_scale", "weight_global_scale")
    )
    tensors: dict[str, torch.Tensor] = {}
    refs: dict[str, list[torch.Tensor]] = {"gate_proj": [], "up_proj": [], "down_proj": []}
    for e in range(E):
        for proj, (out_f, in_f) in (("gate_proj", (I, H)), ("up_proj", (I, H)), ("down_proj", (H, I))):
            packed, scale = _rand_proj(gen, out_f, in_f)
            # Distinct per-projection per-tensor scales unless equality is requested
            # (gate/up only; down always has its own).
            if fmt == "modelopt":
                g = 0.011 if (equal_gate_up and proj in ("gate_proj", "up_proj")) else (
                    0.007 + 0.003 * e + (0.002 if proj == "up_proj" else 0.0))
            else:  # CT stores a divisor
                g = 90.0 if (equal_gate_up and proj in ("gate_proj", "up_proj")) else (
                    80.0 + 10.0 * e + (5.0 if proj == "up_proj" else 0.0))
            gscale = torch.tensor([g], dtype=torch.float32)
            base = f"{PREFIX}.{e}.{proj}"
            tensors[f"{base}.{w_key}"] = packed
            tensors[f"{base}.{s_key}"] = scale
            tensors[f"{base}.{g_key}"] = gscale
            refs[proj].append(
                dequantize_nvfp4_weight(
                    packed, scale, gscale, group_size=GS,
                    out_dtype=torch.float32, format=fmt,
                )
            )
    shard = "model-00001-of-00001.safetensors"
    save_file(tensors, str(tmp_path / shard))
    weight_map = {k: shard for k in tensors}
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": weight_map})
    )
    return weight_map, refs


def _ref_forward(x, top_k_index, top_k_weights, refs, act):
    out = torch.zeros_like(x)
    for t in range(x.shape[0]):
        for pos in range(top_k_index.shape[1]):
            e = int(top_k_index[t, pos])
            h = act(x[t] @ refs["gate_proj"][e].T) * (x[t] @ refs["up_proj"][e].T)
            out[t] += (h @ refs["down_proj"][e].T) * top_k_weights[t, pos]
    return out


def _routing(gen, n_tokens=6, top_k=2):
    idx = torch.randint(0, E, (n_tokens, top_k), generator=gen)
    w = torch.rand(n_tokens, top_k, generator=gen)
    return idx, w / w.sum(-1, keepdim=True)


@pytest.mark.parametrize("fmt", ["compressed_tensors", "modelopt"])
@pytest.mark.parametrize("split", [False, True])
@pytest.mark.parametrize("grouped", [True, False])
def test_forward_parity(tmp_path, fmt, split, grouped):
    # equal scales when fused (required); unequal when split (the point of it)
    weight_map, refs = _build_checkpoint(tmp_path, fmt, equal_gate_up=not split)
    module = NVFP4Experts3D(E, H, I, group_size=GS, quant_format=fmt,
                            split_gate_up_scales=split)
    assemble_nvfp4_experts3d_batched(module, PREFIX, tmp_path, weight_map)
    module.grouped_experts = grouped

    gen = torch.Generator().manual_seed(1)
    x = torch.randn(6, H, generator=gen)
    idx, w = _routing(gen)
    got = module(x, idx, w)
    want = _ref_forward(x, idx, w, refs, module.act_fn)
    torch.testing.assert_close(got, want, rtol=1e-4, atol=1e-4)


def test_fused_assembly_rejects_unequal_scales(tmp_path):
    weight_map, _ = _build_checkpoint(tmp_path, "compressed_tensors", equal_gate_up=False)
    module = NVFP4Experts3D(E, H, I, group_size=GS)  # fused default
    with pytest.raises(ValueError) as exc:
        assemble_nvfp4_experts3d_batched(module, PREFIX, tmp_path, weight_map)
    assert "split_gate_up_scales" in str(exc.value)


@pytest.mark.parametrize("fmt,equal,expected_split", [
    ("compressed_tensors", True, False),
    ("compressed_tensors", False, True),
    ("modelopt", True, False),
    ("modelopt", False, True),
])
def test_detect_moe_expert_storage(tmp_path, fmt, equal, expected_split):
    weight_map, _ = _build_checkpoint(tmp_path, fmt, equal_gate_up=equal)
    info = detect_moe_expert_storage(tmp_path, weight_map)
    assert info == {"quant_format": fmt, "split_gate_up_scales": expected_split}


def test_detect_returns_none_without_expert_keys(tmp_path):
    save_file({"model.norm.weight": torch.ones(4)}, str(tmp_path / "a.safetensors"))
    assert detect_moe_expert_storage(tmp_path, {"model.norm.weight": "a.safetensors"}) is None


def test_split_buffers_match_fused_layout(tmp_path):
    # Same checkpoint assembled both ways (equal scales so fused is legal):
    # the split buffers must hold exactly the halves of the fused buffer.
    weight_map, _ = _build_checkpoint(tmp_path, "compressed_tensors", equal_gate_up=True)
    fused = NVFP4Experts3D(E, H, I, group_size=GS)
    split = NVFP4Experts3D(E, H, I, group_size=GS, split_gate_up_scales=True)
    assemble_nvfp4_experts3d_batched(fused, PREFIX, tmp_path, weight_map)
    assemble_nvfp4_experts3d_batched(split, PREFIX, tmp_path, weight_map)
    assert torch.equal(split.gate_packed, fused.gate_up_packed[:, :I])
    assert torch.equal(split.up_packed, fused.gate_up_packed[:, I:])
    assert torch.equal(split.gate_global_scale, fused.gate_up_global_scale)
    assert torch.equal(split.up_global_scale, fused.gate_up_global_scale)
