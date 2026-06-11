"""Parity test: grouped vs legacy per-expert NVFP4Experts3D forward/backward.

Builds a small NVFP4Experts3D (8 experts, hidden 64, intermediate 32,
group_size 16) with random valid packed/scale data and random top-k routing,
then checks that the grouped path matches the legacy per-expert loop within
bf16 rounding tolerance on both the forward output and the input gradient
(expert weights are frozen; only grad w.r.t. activations exists).

On CPU the dequant dispatcher never reaches Triton, so this exercises the
batched torch fallback. The device parametrization lets the same test run on
GPU later; set NVFP4_TEST_FORCE_CPU=1 to skip CUDA without ever querying the
CUDA runtime (needed when another process owns the GPU).

Run:
    cd /path/to/nvfp4-lora-spark
    NVFP4_TEST_FORCE_CPU=1 /home/veritan-spark-01/Veritan/.venvs/qwen-serve/bin/python \
        -m pytest smoke_tests/test_experts3d_grouped_parity.py -x -q
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import torch

from nvfp4_lora.experts import NVFP4Experts3D

NUM_EXPERTS = 8
HIDDEN_DIM = 64
INTERMEDIATE_DIM = 32
GROUP_SIZE = 16
NUM_TOKENS = 64
TOP_K = 2

# Weights are bit-identical between the two paths (same dequant math, no
# reductions); the only divergence is bmm-vs-F.linear reduction order and the
# bf16 accumulation order of index_add_, both at bf16 rounding level. The
# synthetic gscales below keep outputs O(1) so these tolerances are meaningful
# (a deliberate cross-expert gscale swap fails this check).
RTOL = 2e-2
ATOL = 1e-2

DEVICES = ["cpu", "cuda"]


def _cuda_available() -> bool:
    # NVFP4_TEST_FORCE_CPU=1 short-circuits BEFORE torch.cuda.is_available so
    # the test never touches the CUDA runtime on a box where another process
    # owns the GPU.
    if os.environ.get("NVFP4_TEST_FORCE_CPU", "0") == "1":
        return False
    return torch.cuda.is_available()


def _skip_if_no_device(device: str) -> None:
    if device == "cuda" and not _cuda_available():
        pytest.skip("CUDA not available (or NVFP4_TEST_FORCE_CPU=1)")


def _make_module(device: str, seed: int = 0) -> NVFP4Experts3D:
    module = NVFP4Experts3D(
        num_experts=NUM_EXPERTS,
        hidden_dim=HIDDEN_DIM,
        intermediate_dim=INTERMEDIATE_DIM,
        group_size=GROUP_SIZE,
        device=device,
    )
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for packed, scale, gscale in (
            (module.gate_up_packed, module.gate_up_scale, module.gate_up_global_scale),
            (module.down_packed, module.down_scale, module.down_global_scale),
        ):
            packed.copy_(torch.randint(0, 256, packed.shape, dtype=torch.uint8, generator=g))
            scale_raw = torch.randn(scale.shape, dtype=torch.float32, generator=g) * 0.5
            scale.copy_(scale_raw.clamp(-448.0, 448.0).to(torch.float8_e4m3fn))
            # compressed-tensors global scale is reciprocated at dequant time;
            # use distinct positive per-expert values around 1.0 (so outputs
            # stay O(1) and the tolerances above have teeth) to make any
            # per-expert mixup in the batched path show up as a parity failure.
            gscale.copy_(torch.rand(gscale.shape, dtype=torch.float32, generator=g) * 1.5 + 0.5)
    return module


def _make_routing(device: str, seed: int = 1, routable_experts: int = NUM_EXPERTS):
    """Random top-k routing over the first `routable_experts` experts."""
    g = torch.Generator().manual_seed(seed)
    hidden = torch.randn(NUM_TOKENS, HIDDEN_DIM, dtype=torch.float32, generator=g)
    hidden = hidden.to(device=device, dtype=torch.bfloat16)
    scores = torch.randn(NUM_TOKENS, routable_experts, dtype=torch.float32, generator=g)
    top_vals, top_idx = scores.topk(TOP_K, dim=-1)
    top_w = torch.softmax(top_vals, dim=-1)
    grad_out = torch.randn(NUM_TOKENS, HIDDEN_DIM, dtype=torch.float32, generator=g)
    return (
        hidden,
        top_idx.to(device),
        top_w.to(device=device, dtype=torch.bfloat16),
        grad_out.to(device=device, dtype=torch.bfloat16),
    )


def _run_path(module, grouped: bool, hidden, top_idx, top_w, grad_out):
    module.grouped_experts = grouped
    x = hidden.clone().detach().requires_grad_(True)
    out = module(x, top_idx, top_w)
    out.backward(grad_out)
    return out.detach(), x.grad.detach()


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("expert_batch_size", [1, 3, 8])
def test_grouped_matches_per_expert(device, expert_batch_size):
    _skip_if_no_device(device)
    module = _make_module(device)
    module.expert_batch_size = expert_batch_size
    hidden, top_idx, top_w, grad_out = _make_routing(device)

    out_ref, grad_ref = _run_path(module, False, hidden, top_idx, top_w, grad_out)
    out_grp, grad_grp = _run_path(module, True, hidden, top_idx, top_w, grad_out)

    torch.testing.assert_close(out_grp.float(), out_ref.float(), rtol=RTOL, atol=ATOL)
    torch.testing.assert_close(grad_grp.float(), grad_ref.float(), rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("device", DEVICES)
def test_grouped_matches_per_expert_with_unhit_experts(device):
    """Skewed routing: half the experts get zero tokens, so the hit-expert list
    is sparse and the last K-batch is ragged."""
    _skip_if_no_device(device)
    module = _make_module(device, seed=2)
    module.expert_batch_size = 3
    hidden, top_idx, top_w, grad_out = _make_routing(
        device, seed=3, routable_experts=NUM_EXPERTS // 2
    )

    out_ref, grad_ref = _run_path(module, False, hidden, top_idx, top_w, grad_out)
    out_grp, grad_grp = _run_path(module, True, hidden, top_idx, top_w, grad_out)

    torch.testing.assert_close(out_grp.float(), out_ref.float(), rtol=RTOL, atol=ATOL)
    torch.testing.assert_close(grad_grp.float(), grad_ref.float(), rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("device", DEVICES)
def test_grouped_routing_weight_gradient(device):
    """top_k_weights must keep receiving gradient through the grouped path
    (the router is trainable even though the experts are frozen)."""
    _skip_if_no_device(device)
    module = _make_module(device)
    hidden, top_idx, top_w, grad_out = _make_routing(device)

    grads = {}
    for grouped in (False, True):
        module.grouped_experts = grouped
        w = top_w.clone().detach().requires_grad_(True)
        out = module(hidden.clone().detach(), top_idx, w)
        out.backward(grad_out)
        grads[grouped] = w.grad.detach()

    torch.testing.assert_close(grads[True].float(), grads[False].float(), rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("format", ["modelopt", "compressed_tensors"])
@pytest.mark.parametrize("use_expert_idx", [False, True])
def test_batched_dequant_matches_per_expert(device, format, use_expert_idx):
    """dequantize_nvfp4_weight_batched must reproduce per-expert
    dequantize_nvfp4_weight exactly (same fp32 math, no reductions). On CUDA
    this additionally covers the batched Triton kernel against the 2D one."""
    _skip_if_no_device(device)
    from nvfp4_lora.dequant import dequantize_nvfp4_weight, dequantize_nvfp4_weight_batched

    out_feat, in_feat = 32, 64
    g = torch.Generator().manual_seed(7)
    packed = torch.randint(
        0, 256, (NUM_EXPERTS, out_feat, in_feat // 2), dtype=torch.uint8, generator=g
    ).to(device)
    scale_raw = torch.randn(
        NUM_EXPERTS, out_feat, in_feat // GROUP_SIZE, dtype=torch.float32, generator=g
    ) * 0.5
    scale = scale_raw.clamp(-448.0, 448.0).to(torch.float8_e4m3fn).to(device)
    gscale = (torch.rand(NUM_EXPERTS, 1, dtype=torch.float32, generator=g) * 1.5 + 0.5).to(device)

    expert_idx = torch.tensor([5, 0, 3], device=device) if use_expert_idx else None
    batched = dequantize_nvfp4_weight_batched(
        packed, scale, gscale, group_size=GROUP_SIZE, format=format, expert_idx=expert_idx
    )

    selected = expert_idx.tolist() if use_expert_idx else range(NUM_EXPERTS)
    for slot, e in enumerate(selected):
        ref = dequantize_nvfp4_weight(
            packed[e], scale[e], gscale[e], group_size=GROUP_SIZE, format=format
        )
        torch.testing.assert_close(batched[slot].float(), ref.float(), rtol=0.0, atol=0.0)


def test_env_var_selects_per_expert_path(monkeypatch):
    monkeypatch.setenv("NVFP4_GROUPED_EXPERTS", "0")
    module = _make_module("cpu")
    assert module.grouped_experts is False
    monkeypatch.setenv("NVFP4_GROUPED_EXPERTS", "1")
    module = _make_module("cpu")
    assert module.grouped_experts is True
    monkeypatch.delenv("NVFP4_GROUPED_EXPERTS")
    module = _make_module("cpu")
    assert module.grouped_experts is True
