from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from nvfp4_lora.dequant import dequantize_nvfp4_weight
from nvfp4_lora.experts import NVFP4Experts3D


def _pack_low_nibble_first(values: torch.Tensor) -> torch.Tensor:
    values = values.to(torch.uint8)
    return values[..., 0::2] | (values[..., 1::2] << 4)


def _constant_pack(out_features: int, in_features: int, nibble: int) -> torch.Tensor:
    unpacked = torch.full((out_features, in_features), nibble, dtype=torch.uint8)
    return _pack_low_nibble_first(unpacked)


def _synthetic_ct_weight(out_features: int, in_features: int, group_size: int, offset: int = 0):
    unpacked = (torch.arange(out_features * in_features, dtype=torch.int64).reshape(out_features, in_features) + offset) % 16
    packed = _pack_low_nibble_first(unpacked)
    scale = torch.ones(out_features, in_features // group_size, dtype=torch.float8_e4m3fn)
    gscale = torch.ones(1, dtype=torch.float32)
    return packed, scale, gscale


class ReferenceExperts(nn.Module):
    def __init__(self, gate_up_proj: torch.Tensor, down_proj: torch.Tensor, act_fn: nn.Module):
        super().__init__()
        self.num_experts = gate_up_proj.shape[0]
        self.gate_up_proj = nn.Parameter(gate_up_proj.clone(), requires_grad=False)
        self.down_proj = nn.Parameter(down_proj.clone(), requires_grad=False)
        self.act_fn = act_fn

    def forward(self, hidden_states, top_k_index, top_k_weights):
        final_hidden_states = torch.zeros_like(hidden_states)
        with torch.no_grad():
            expert_mask = torch.nn.functional.one_hot(top_k_index, num_classes=self.num_experts)
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
        for expert_idx in expert_hit:
            expert_idx = expert_idx[0]
            if expert_idx == self.num_experts:
                continue
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx]
            gate, up = F.linear(current_state, self.gate_up_proj[expert_idx]).chunk(2, dim=-1)
            current_hidden_states = self.act_fn(gate) * up
            current_hidden_states = F.linear(current_hidden_states, self.down_proj[expert_idx])
            current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))
        return final_hidden_states


def _fill_module_with_synthetic_weights(
    module: NVFP4Experts3D,
    gate_up_parts: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
    down_parts: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
):
    with torch.no_grad():
        for expert_idx, (gate_pack, gate_scale, gate_gscale, up_pack, up_scale, up_gscale) in enumerate(gate_up_parts):
            module.gate_up_packed[expert_idx, : module.intermediate_dim].copy_(gate_pack)
            module.gate_up_packed[expert_idx, module.intermediate_dim :].copy_(up_pack)
            module.gate_up_scale[expert_idx, : module.intermediate_dim].copy_(gate_scale)
            module.gate_up_scale[expert_idx, module.intermediate_dim :].copy_(up_scale)
            assert torch.equal(gate_gscale, up_gscale)
            module.gate_up_global_scale[expert_idx].copy_(gate_gscale)

        for expert_idx, (down_pack, down_scale, down_gscale) in enumerate(down_parts):
            module.down_packed[expert_idx].copy_(down_pack)
            module.down_scale[expert_idx].copy_(down_scale)
            module.down_global_scale[expert_idx].copy_(down_gscale)


def test_construction_shapes_and_frozen_buffers_cpu():
    module = NVFP4Experts3D(
        num_experts=4,
        hidden_dim=64,
        intermediate_dim=32,
        group_size=16,
        act_fn=nn.SiLU(),
    )

    assert module.gate_up_packed.shape == (4, 64, 32)
    assert module.gate_up_scale.shape == (4, 64, 4)
    assert module.gate_up_global_scale.shape == (4, 1)
    assert module.down_packed.shape == (4, 64, 16)
    assert module.down_scale.shape == (4, 64, 2)
    assert module.down_global_scale.shape == (4, 1)

    assert len(list(module.parameters())) == 0
    for buffer in module.buffers():
        assert buffer.requires_grad is False


def test_per_expert_dequant_bit_correctness_synthetic_ct():
    hidden_dim = 64
    intermediate_dim = 32
    group_size = 16

    module = NVFP4Experts3D(
        num_experts=2,
        hidden_dim=hidden_dim,
        intermediate_dim=intermediate_dim,
        group_size=group_size,
        act_fn=nn.SiLU(),
    )

    gate_up_parts = []
    down_parts = []
    refs_gate_up = []
    refs_down = []

    for expert_idx in range(2):
        gate = _synthetic_ct_weight(intermediate_dim, hidden_dim, group_size, offset=expert_idx)
        up = _synthetic_ct_weight(intermediate_dim, hidden_dim, group_size, offset=expert_idx + 3)
        down = _synthetic_ct_weight(hidden_dim, intermediate_dim, group_size, offset=expert_idx + 7)

        gate_up_parts.append((*gate, *up))
        down_parts.append(down)

        ref_gate = dequantize_nvfp4_weight(*gate, group_size=group_size, out_dtype=torch.bfloat16, format="compressed_tensors")
        ref_up = dequantize_nvfp4_weight(*up, group_size=group_size, out_dtype=torch.bfloat16, format="compressed_tensors")
        ref_down = dequantize_nvfp4_weight(*down, group_size=group_size, out_dtype=torch.bfloat16, format="compressed_tensors")
        refs_gate_up.append(torch.cat([ref_gate, ref_up], dim=0))
        refs_down.append(ref_down)

    _fill_module_with_synthetic_weights(module, gate_up_parts, down_parts)

    for expert_idx in range(2):
        got_gate_up = dequantize_nvfp4_weight(
            module.gate_up_packed[expert_idx].contiguous(),
            module.gate_up_scale[expert_idx].contiguous(),
            module.gate_up_global_scale[expert_idx],
            group_size=group_size,
            out_dtype=torch.bfloat16,
            format="compressed_tensors",
        )
        got_down = dequantize_nvfp4_weight(
            module.down_packed[expert_idx].contiguous(),
            module.down_scale[expert_idx].contiguous(),
            module.down_global_scale[expert_idx],
            group_size=group_size,
            out_dtype=torch.bfloat16,
            format="compressed_tensors",
        )

        assert torch.equal(got_gate_up, refs_gate_up[expert_idx])
        assert torch.equal(got_down, refs_down[expert_idx])


def test_forward_parity_vs_reference_bf16_moe():
    torch.manual_seed(0)
    num_experts = 3
    hidden_dim = 64
    intermediate_dim = 32
    group_size = 16

    nvfp4 = NVFP4Experts3D(
        num_experts=num_experts,
        hidden_dim=hidden_dim,
        intermediate_dim=intermediate_dim,
        group_size=group_size,
        act_fn=nn.SiLU(),
    )

    gate_up_parts = []
    down_parts = []
    gate_up_bf16 = []
    down_bf16 = []

    for expert_idx in range(num_experts):
        gate = _synthetic_ct_weight(intermediate_dim, hidden_dim, group_size, offset=expert_idx)
        up = _synthetic_ct_weight(intermediate_dim, hidden_dim, group_size, offset=expert_idx + 5)
        down = _synthetic_ct_weight(hidden_dim, intermediate_dim, group_size, offset=expert_idx + 9)
        gate_up_parts.append((*gate, *up))
        down_parts.append(down)

        gate_w = dequantize_nvfp4_weight(*gate, group_size=group_size, out_dtype=torch.bfloat16, format="compressed_tensors")
        up_w = dequantize_nvfp4_weight(*up, group_size=group_size, out_dtype=torch.bfloat16, format="compressed_tensors")
        down_w = dequantize_nvfp4_weight(*down, group_size=group_size, out_dtype=torch.bfloat16, format="compressed_tensors")
        gate_up_bf16.append(torch.cat([gate_w, up_w], dim=0))
        down_bf16.append(down_w)

    _fill_module_with_synthetic_weights(nvfp4, gate_up_parts, down_parts)
    ref = ReferenceExperts(torch.stack(gate_up_bf16), torch.stack(down_bf16), nn.SiLU())

    hidden_states = torch.randn(11, hidden_dim, dtype=torch.bfloat16)
    top_k_index = torch.tensor(
        [[0, 1], [1, 2], [2, 0], [0, 2], [1, 0], [2, 1], [0, 1], [1, 2], [2, 0], [0, 2], [1, 0]],
        dtype=torch.long,
    )
    top_k_weights = torch.rand(11, 2, dtype=torch.bfloat16)

    got = nvfp4(hidden_states, top_k_index, top_k_weights)
    expected = ref(hidden_states, top_k_index, top_k_weights)

    assert torch.allclose(got, expected, atol=1e-2, rtol=1e-2)


def test_backward_signal_flows_to_hidden_states_only():
    torch.manual_seed(1)
    num_experts = 2
    hidden_dim = 64
    intermediate_dim = 32
    group_size = 16

    module = NVFP4Experts3D(
        num_experts=num_experts,
        hidden_dim=hidden_dim,
        intermediate_dim=intermediate_dim,
        group_size=group_size,
        act_fn=nn.SiLU(),
    )

    gate_up_parts = []
    down_parts = []
    for expert_idx in range(num_experts):
        # Use random uint8 packs (not arange % 16) — the arange-based synthetic data
        # produces W with row-sum exactly zero (each row = two full cycles of the
        # E2M1 LUT) which combined with constant `gate*up` cross-product zeros out
        # forward output and therefore the backward signal. Random packs avoid that
        # pathology while still exercising the real dequant path.
        def _rand_pack(out_f, in_f, seed):
            g = torch.Generator().manual_seed(seed)
            return (
                torch.randint(0, 256, (out_f, in_f // 2), generator=g, dtype=torch.uint8),
                torch.ones(out_f, in_f // group_size, dtype=torch.float8_e4m3fn),
                torch.ones(1, dtype=torch.float32),
            )

        gate = _rand_pack(intermediate_dim, hidden_dim, seed=expert_idx * 7 + 1)
        up = _rand_pack(intermediate_dim, hidden_dim, seed=expert_idx * 7 + 2)
        down = _rand_pack(hidden_dim, intermediate_dim, seed=expert_idx * 7 + 3)
        gate_up_parts.append((*gate, *up))
        down_parts.append(down)
    _fill_module_with_synthetic_weights(module, gate_up_parts, down_parts)

    hidden_states = torch.randn(6, hidden_dim, dtype=torch.bfloat16).requires_grad_(True)
    top_k_index = torch.tensor([[0, 1], [1, 0], [0, 1], [1, 0], [0, 1], [1, 0]], dtype=torch.long)
    top_k_weights = torch.rand(6, 2, dtype=torch.bfloat16)

    loss = module(hidden_states, top_k_index, top_k_weights).float().sum()
    loss.backward()

    assert hidden_states.grad is not None
    assert torch.isfinite(hidden_states.grad).all()
    assert hidden_states.grad.abs().sum() > 0

    for buffer in module.buffers():
        assert buffer.grad is None


def test_gate_up_packing_order_gate_first_then_up():
    hidden_dim = 16
    intermediate_dim = 16
    group_size = 16

    module = NVFP4Experts3D(
        num_experts=1,
        hidden_dim=hidden_dim,
        intermediate_dim=intermediate_dim,
        group_size=group_size,
        act_fn=nn.Identity(),
    )

    gate_pack = _constant_pack(intermediate_dim, hidden_dim, 2)
    up_pack = _constant_pack(intermediate_dim, hidden_dim, 4)
    down_pack = _constant_pack(hidden_dim, intermediate_dim, 1)
    scale_gate_up = torch.ones(intermediate_dim, hidden_dim // group_size, dtype=torch.float8_e4m3fn)
    scale_down = torch.ones(hidden_dim, intermediate_dim // group_size, dtype=torch.float8_e4m3fn)
    gscale = torch.ones(1, dtype=torch.float32)

    _fill_module_with_synthetic_weights(
        module,
        [(gate_pack, scale_gate_up, gscale, up_pack, scale_gate_up, gscale)],
        [(down_pack, scale_down, gscale)],
    )

    hidden_states = torch.zeros(1, hidden_dim, dtype=torch.bfloat16)
    hidden_states[0, 0] = 1
    top_k_index = torch.tensor([[0]], dtype=torch.long)
    top_k_weights = torch.ones(1, 1, dtype=torch.bfloat16)

    got = module(hidden_states, top_k_index, top_k_weights)

    gate_w = dequantize_nvfp4_weight(
        gate_pack, scale_gate_up, gscale, group_size=group_size, out_dtype=torch.bfloat16, format="compressed_tensors"
    )
    up_w = dequantize_nvfp4_weight(
        up_pack, scale_gate_up, gscale, group_size=group_size, out_dtype=torch.bfloat16, format="compressed_tensors"
    )
    down_w = dequantize_nvfp4_weight(
        down_pack, scale_down, gscale, group_size=group_size, out_dtype=torch.bfloat16, format="compressed_tensors"
    )
    expected = F.linear(F.linear(hidden_states, gate_w) * F.linear(hidden_states, up_w), down_w)

    swapped_expected = F.linear(F.linear(hidden_states, up_w) * F.linear(hidden_states, gate_w), down_w)
    assert torch.equal(got, expected)
    assert torch.equal(got, swapped_expected)

    gate_pack_asymmetric = _constant_pack(intermediate_dim, hidden_dim, 1)
    up_pack_asymmetric = _constant_pack(intermediate_dim, hidden_dim, 4)
    down_pack_asymmetric = _constant_pack(hidden_dim, intermediate_dim, 1)

    with torch.no_grad():
        module.gate_up_packed[0, :intermediate_dim].copy_(gate_pack_asymmetric)
        module.gate_up_packed[0, intermediate_dim:].copy_(up_pack_asymmetric)
        module.down_packed[0].copy_(down_pack_asymmetric)

    got_ordered = module(hidden_states, top_k_index, top_k_weights)

    gate_w = dequantize_nvfp4_weight(
        gate_pack_asymmetric,
        scale_gate_up,
        gscale,
        group_size=group_size,
        out_dtype=torch.bfloat16,
        format="compressed_tensors",
    )
    up_w = dequantize_nvfp4_weight(
        up_pack_asymmetric,
        scale_gate_up,
        gscale,
        group_size=group_size,
        out_dtype=torch.bfloat16,
        format="compressed_tensors",
    )
    down_w = dequantize_nvfp4_weight(
        down_pack_asymmetric,
        scale_down,
        gscale,
        group_size=group_size,
        out_dtype=torch.bfloat16,
        format="compressed_tensors",
    )

    expected_ordered = F.linear(F.linear(hidden_states, gate_w) * F.linear(hidden_states, up_w), down_w)
    expected_if_storage_were_swapped = F.linear(F.linear(hidden_states, up_w) * F.linear(hidden_states, gate_w), down_w)

    assert torch.equal(got_ordered, expected_ordered)
    assert torch.equal(got_ordered, expected_if_storage_were_swapped)
