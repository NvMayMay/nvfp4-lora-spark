"""Multi-K-batch grouped-forward regression for NVFP4Experts3D._forward_grouped.

The storage-modes parity test only ever routes to <=4 experts, and the grouped
path defaults to expert_batch_size (K) = 8, so the K-batch loop always runs a
SINGLE iteration: the multi-chunk `out_chunks` concat + the per-batch padding
tail are never exercised. A cross-batch expert->token scramble (wrong scatter
of the second/third K-batch back into sorted order) would pass green there.

This test forces >1 K-batch two ways -- a small expert_batch_size AND more than
8 routed experts -- with deliberately UNEVEN per-expert token counts (so every
K-batch has a nontrivial padding tail), and checks BOTH:

  * grouped forward output, and
  * lora_A / lora_B gradients

against a fully INDEPENDENT per-token reference (a plain Python loop over
(token, top_k_pos) pairs that dequantizes each projection once and adds the
LoRA delta by hand). The reference never calls the module's own per-expert
path (`_forward_per_expert` / `_gate_up_acted`), so a bug shared between the two
module code paths cannot hide.

CPU / float32 only; honors tests/conftest.py (no CUDA allocation).
"""
from __future__ import annotations

import math

import pytest
import torch
from safetensors.torch import save_file

from nvfp4_lora.dequant import dequantize_nvfp4_weight
from nvfp4_lora.experts import NVFP4Experts3D, assemble_nvfp4_experts3d_batched

# Wider than the storage-modes fixture: 12 experts so top-k routing can hit
# >8 experts and force multiple K-batches even at the default K=8.
E, H, I, GS = 12, 32, 16, 16
PREFIX = "model.layers.0.mlp.experts"


def _rand_proj(gen, out_f, in_f):
    packed = torch.randint(0, 256, (out_f, in_f // 2), dtype=torch.uint8, generator=gen)
    scale = (torch.rand(out_f, in_f // GS, generator=gen) * 0.5 + 0.25).to(torch.float8_e4m3fn)
    return packed, scale


def _build_checkpoint(tmp_path, fmt="compressed_tensors"):
    """Fused-storage synthetic checkpoint with EQUAL gate/up per-tensor scales
    (fused mode is the validated fast path). Returns (weight_map, refs) with
    refs[proj][e] the reference dense float32 weight."""
    gen = torch.Generator().manual_seed(7)
    w_key, s_key, g_key = ("weight_packed", "weight_scale", "weight_global_scale")
    tensors: dict[str, torch.Tensor] = {}
    refs: dict[str, list[torch.Tensor]] = {"gate_proj": [], "up_proj": [], "down_proj": []}
    for e in range(E):
        for proj, (out_f, in_f) in (("gate_proj", (I, H)), ("up_proj", (I, H)), ("down_proj", (H, I))):
            packed, scale = _rand_proj(gen, out_f, in_f)
            # gate/up share one per-tensor scale per expert (fused requirement);
            # down carries its own. Vary per expert so experts are distinct.
            g = (90.0 - 2.0 * e) if proj in ("gate_proj", "up_proj") else (70.0 + 3.0 * e)
            gscale = torch.tensor([g], dtype=torch.float32)
            base = f"{PREFIX}.{e}.{proj}"
            tensors[f"{base}.{w_key}"] = packed
            tensors[f"{base}.{s_key}"] = scale
            tensors[f"{base}.{g_key}"] = gscale
            refs[proj].append(
                dequantize_nvfp4_weight(packed, scale, gscale, group_size=GS,
                                        out_dtype=torch.float32, format=fmt)
            )
    shard = "model-00001-of-00001.safetensors"
    save_file(tensors, str(tmp_path / shard))
    weight_map = {k: shard for k in tensors}
    return weight_map, refs


def _uneven_routing(n_tokens, top_k):
    """Deterministic routing chosen so per-expert hit counts are UNEVEN and >8
    distinct experts are hit -- guaranteeing padding tails and (for the default
    K=8) multiple K-batches. Every expert index used stays in [0, E)."""
    # Hand-built assignments: expert 0 hit many times, others a mix, several
    # experts (>8 distinct) touched, with clearly unequal per-expert counts.
    rows = [
        [0, 1],
        [0, 2],
        [0, 3],
        [4, 0],
        [5, 6],
        [7, 0],
        [8, 9],
        [10, 0],
        [11, 1],
        [2, 0],
    ]
    assert len(rows) == n_tokens and all(len(r) == top_k for r in rows)
    idx = torch.tensor(rows, dtype=torch.long)
    gen = torch.Generator().manual_seed(3)
    w = torch.rand(n_tokens, top_k, generator=gen)
    return idx, w / w.sum(-1, keepdim=True)


def _ref_forward_and_grad(x, idx, w, refs, module):
    """Independent per-token reference: plain loop over (token, pos) pairs.

    Computes the forward output AND autograd grads for the module's LoRA
    parameters, WITHOUT touching any module forward path. LoRA math mirrors the
    documented recipe: delta = scale * (dropout(x) @ A_e^T) @ B_e^T, added to the
    dequantized base projection. Dropout is a no-op here (module in eval / p=0).
    """
    scale = module.lora_scale
    A_gu, B_gu = module.lora_A_gate_up, module.lora_B_gate_up
    A_dn, B_dn = module.lora_A_down, module.lora_B_down
    act = module.act_fn

    out = torch.zeros_like(x)
    for t in range(x.shape[0]):
        xt = x[t]
        for pos in range(idx.shape[1]):
            e = int(idx[t, pos])
            # gate_up base (fused: [gate, up] along output axis)
            gate_base = xt @ refs["gate_proj"][e].T
            up_base = xt @ refs["up_proj"][e].T
            gate_up = torch.cat([gate_base, up_base], dim=-1)
            # gate_up LoRA delta
            gu_delta = scale * ((xt @ A_gu[e].T) @ B_gu[e].T)
            gate_up = gate_up + gu_delta
            gate, up = gate_up.chunk(2, dim=-1)
            h = act(gate) * up
            # down base + LoRA delta
            down = h @ refs["down_proj"][e].T
            dn_delta = scale * ((h @ A_dn[e].T) @ B_dn[e].T)
            down = down + dn_delta
            out[t] = out[t] + down * w[t, pos]
    return out


def _run_case(tmp_path, expert_batch_size):
    weight_map, refs = _build_checkpoint(tmp_path)
    n_tokens, top_k = 10, 2

    # Two identical modules with identical (random) LoRA params so their grads
    # are directly comparable: the grouped module under test and a reference-
    # only module whose params feed the independent loop.
    def _make():
        gen = torch.Generator().manual_seed(11)
        m = NVFP4Experts3D(E, H, I, group_size=GS, quant_format="compressed_tensors",
                           lora_r=4, lora_alpha=8, lora_dropout=0.0,
                           lora_dtype=torch.float32)
        assemble_nvfp4_experts3d_batched(m, PREFIX, tmp_path, weight_map)
        m.eval()  # kill any dropout stochasticity (also p=0 already)
        # Randomize B (zero-init by default) so lora_B grads are nonzero and the
        # forward LoRA delta is nonzero; A already Kaiming-init'd. Fixed seed so
        # both modules get identical params.
        with torch.no_grad():
            for name, p in m.named_parameters():
                p.copy_(torch.randn(p.shape, generator=gen, dtype=p.dtype) * 0.05)
        return m

    grouped = _make()
    grouped.grouped_experts = True
    grouped.expert_batch_size = expert_batch_size

    ref = _make()  # same params (same seed)

    xgen = torch.Generator().manual_seed(21)
    x = torch.randn(n_tokens, H, generator=xgen, dtype=torch.float32)
    idx, w = _uneven_routing(n_tokens, top_k)

    # Sanity: this routing actually forces >1 K-batch and uneven counts.
    counts = torch.bincount(idx.reshape(-1), minlength=E)
    hit = int((counts > 0).sum())
    assert hit > 1
    n_batches = math.ceil(hit / expert_batch_size)
    assert n_batches > 1, f"expected >1 K-batch, got {n_batches} (hit={hit}, K={expert_batch_size})"
    nz = counts[counts > 0]
    assert int(nz.min()) != int(nz.max()), "per-expert counts must be uneven"

    # ---- forward parity ----
    x_g = x.clone().requires_grad_(False)
    got = grouped(x_g, idx, w)
    want = _ref_forward_and_grad(x, idx, w, refs, ref)
    torch.testing.assert_close(got, want, rtol=1e-4, atol=1e-4)

    # ---- gradient parity (lora_A / lora_B for gate_up and down) ----
    grouped.zero_grad(set_to_none=True)
    ref.zero_grad(set_to_none=True)
    got2 = grouped(x.clone(), idx, w)
    want2 = _ref_forward_and_grad(x, idx, w, refs, ref)
    # Same scalar objective on both graphs.
    target = torch.randn(got2.shape, generator=torch.Generator().manual_seed(31))
    torch.nn.functional.mse_loss(got2, target).backward()
    torch.nn.functional.mse_loss(want2, target).backward()

    for name in ("lora_A_gate_up", "lora_B_gate_up", "lora_A_down", "lora_B_down"):
        g_grad = getattr(grouped, name).grad
        r_grad = getattr(ref, name).grad
        assert g_grad is not None, f"{name} got no grad in grouped path"
        assert r_grad is not None, f"{name} got no grad in reference"
        torch.testing.assert_close(
            g_grad, r_grad, rtol=1e-4, atol=1e-4,
            msg=f"{name} grad mismatch (multi-K-batch scatter/reassembly bug?)",
        )


@pytest.mark.parametrize("expert_batch_size", [2, 3, 8])
def test_grouped_multibatch_forward_and_grad(tmp_path, expert_batch_size):
    # K=2,3 force many small K-batches over the 9 hit experts; K=8 exercises the
    # boundary where 9 hit experts still spill into a 2nd (single-expert) batch,
    # so the multi-chunk concat runs with a tiny trailing chunk.
    _run_case(tmp_path, expert_batch_size)
