"""Per-expert LoRA on NVFP4Experts3D (train-side, Phase 1).

CPU-only. Exercises the opt-in per-expert LoRA path added to the frozen fused-3D
NVFP4 MoE container: wiring, zero-init no-op, trainable-only-the-adapter,
grouped-vs-per-expert parity, and a float64 gradcheck on the delta.

NOTE (scope): this validates the TRAIN-side math only. Runtime serving of the
expert delta (vLLM FusedMoE3DWithLoRA via the marlin backend) and train<->serve
numerical parity are separate, GPU-gated steps (see docs/plans/expert_lora_scope.md).
The frozen base buffers are left at their zero-init values here, which makes the
base expert output zero and isolates the LoRA delta as the only signal -- exactly
what we want to test for the adapter math.
"""
from __future__ import annotations

import torch
import pytest

from nvfp4_lora.experts import NVFP4Experts3D


def _routing(n_tokens: int, num_experts: int, top_k: int, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    idx = torch.stack(
        [torch.randperm(num_experts, generator=g)[:top_k] for _ in range(n_tokens)]
    )  # (n_tokens, top_k), distinct experts per token
    w = torch.rand(n_tokens, top_k, generator=g, dtype=torch.float32)
    return idx, w


def _mod(num_experts=6, hidden=16, inter=32, group_size=16, lora_r=0, lora_alpha=0,
         lora_dropout=0.0, split=False, dtype=torch.float32):
    m = NVFP4Experts3D(
        num_experts=num_experts, hidden_dim=hidden, intermediate_dim=inter,
        group_size=group_size, lora_r=lora_r, lora_alpha=lora_alpha,
        lora_dropout=lora_dropout, split_gate_up_scales=split, lora_dtype=dtype,
    )
    return m.eval()


def test_r0_creates_no_lora_params():
    m = _mod(lora_r=0)
    assert m.lora_A_gate_up is None and m.lora_B_down is None
    trainable = [n for n, p in m.named_parameters() if p.requires_grad]
    assert trainable == [], f"r=0 should have zero trainable params, got {trainable}"


def test_r_creates_expected_shapes():
    E, h, i, r = 6, 16, 32, 4
    m = _mod(num_experts=E, hidden=h, inter=i, lora_r=r, lora_alpha=8)
    assert m.lora_A_gate_up.shape == (E, r, h)
    assert m.lora_B_gate_up.shape == (E, 2 * i, r)
    assert m.lora_A_down.shape == (E, r, i)
    assert m.lora_B_down.shape == (E, h, r)
    # B zero-init (delta starts at zero); A nonzero (Kaiming)
    assert torch.count_nonzero(m.lora_B_gate_up) == 0
    assert torch.count_nonzero(m.lora_B_down) == 0
    assert torch.count_nonzero(m.lora_A_gate_up) > 0
    # frozen base weights are buffers, never trainable Parameters
    pnames = {n for n, _ in m.named_parameters()}
    assert "gate_up_packed" not in pnames and "down_packed" not in pnames
    assert {"lora_A_gate_up", "lora_B_gate_up", "lora_A_down", "lora_B_down"} <= pnames


@pytest.mark.parametrize("grouped", ["1", "0"])
@pytest.mark.parametrize("split", [False, True])
def test_zero_init_is_exact_noop(monkeypatch, grouped, split):
    """With B zero-init, the LoRA module output must equal the r=0 baseline,
    across both forward paths AND both split_gate_up_scales modes."""
    monkeypatch.setenv("NVFP4_GROUPED_EXPERTS", grouped)
    idx, w = _routing(8, 6, 2)
    x = torch.randn(8, 16)
    base = _mod(lora_r=0, split=split); base.grouped_experts = grouped != "0"
    lora = _mod(lora_r=4, lora_alpha=8, split=split); lora.grouped_experts = grouped != "0"
    yb = base(x, idx, w)
    yl = lora(x, idx, w)
    assert torch.allclose(yb, yl, atol=1e-6), "zero-init LoRA changed the output"


def test_alpha_defaults_to_2r_not_dead():
    """lora_r>0 with lora_alpha=0 must NOT yield a dead (scale=0) adapter."""
    m = _mod(lora_r=4, lora_alpha=0)
    assert m.lora_alpha == 8 and m.lora_scale == 2.0


def test_kaiming_init_uses_correct_fan_in():
    """Per-expert init: A std must match fan_in=in_dim, not the 3D-tensor fan_in=r*in
    (which would shrink the init by ~sqrt(r))."""
    import math as _m
    in_dim, r = 64, 8
    m = _mod(hidden=in_dim, lora_r=r, lora_alpha=16)
    a = m.lora_A_gate_up.detach()  # (E, r, in_dim)
    std = a.std().item()
    correct = 1.0 / _m.sqrt(3 * in_dim)          # uniform(-1/sqrt(in),..) std
    buggy = 1.0 / _m.sqrt(3 * r * in_dim)        # if fan_in were r*in
    assert abs(std - correct) < abs(std - buggy), f"std={std} closer to buggy fan_in"


@pytest.mark.parametrize("grouped", ["1", "0"])
def test_dropout_active_in_train_identity_in_eval(monkeypatch, grouped):
    monkeypatch.setenv("NVFP4_GROUPED_EXPERTS", grouped)
    idx, w = _routing(8, 6, 2)
    x = torch.randn(8, 16)
    m = _mod(lora_r=4, lora_alpha=8, lora_dropout=0.5)
    m.grouped_experts = grouped != "0"
    with torch.no_grad():
        m.lora_B_gate_up.normal_(0, 0.1); m.lora_B_down.normal_(0, 0.1)
    m.eval()
    e1, e2 = m(x, idx, w), m(x, idx, w)
    assert torch.allclose(e1, e2), "eval must be deterministic (dropout=identity)"
    m.train()
    torch.manual_seed(1); t1 = m(x, idx, w)
    torch.manual_seed(2); t2 = m(x, idx, w)
    assert not torch.allclose(t1, t2), "train-mode dropout should perturb the delta"


@pytest.mark.parametrize("grouped", ["1", "0"])
def test_backward_trains_only_adapter(monkeypatch, grouped):
    monkeypatch.setenv("NVFP4_GROUPED_EXPERTS", grouped)
    idx, w = _routing(8, 6, 2)
    x = torch.randn(8, 16)
    m = _mod(lora_r=4, lora_alpha=8); m.grouped_experts = grouped != "0"
    m.train()
    # break the zero-init so a gradient can flow to B as well as A
    with torch.no_grad():
        m.lora_B_gate_up.normal_(0, 0.02)
        m.lora_B_down.normal_(0, 0.02)
    y = m(x, idx, w)
    y.pow(2).sum().backward()
    for name in ("lora_A_gate_up", "lora_B_gate_up", "lora_A_down", "lora_B_down"):
        g = getattr(m, name).grad
        assert g is not None and torch.count_nonzero(g) > 0, f"{name} got no gradient"
    # frozen base buffers carry no grad
    for name in ("gate_up_packed", "down_packed", "gate_up_scale", "down_scale"):
        assert getattr(m, name).grad is None


@pytest.mark.parametrize("split", [False, True])
def test_grouped_matches_per_expert(monkeypatch, split):
    """Grouped and per-expert paths must agree (routing math is identical),
    in both fused and split gate_up storage modes."""
    idx, w = _routing(12, 6, 2, seed=3)
    x = torch.randn(12, 16)
    # shared adapter weights: build once (grouped), copy into a per-expert clone
    monkeypatch.setenv("NVFP4_GROUPED_EXPERTS", "1")
    mg = _mod(lora_r=4, lora_alpha=8, split=split); mg.grouped_experts = True
    with torch.no_grad():
        mg.lora_B_gate_up.normal_(0, 0.05); mg.lora_B_down.normal_(0, 0.05)
    mp = _mod(lora_r=4, lora_alpha=8, split=split); mp.grouped_experts = False
    mp.load_state_dict(mg.state_dict())
    yg = mg(x, idx, w)
    yp = mp(x, idx, w)
    assert torch.allclose(yg, yp, atol=1e-5), (yg - yp).abs().max()


class _Tok:
    def save_pretrained(self, dest):  # minimal tokenizer stub for the save path
        pass


def test_adapter_save_load_roundtrip(train_mod, tmp_path):
    """The native-mode adapter save/load must round-trip per-expert LoRA tensors
    and record the expert_lora block in adapter_config.json."""
    import json
    import torch.nn as nn

    class Wrap(nn.Module):
        def __init__(self):
            super().__init__()
            self.block = _mod(num_experts=6, hidden=16, inter=32, lora_r=4, lora_alpha=8)

    src = Wrap()
    with torch.no_grad():
        src.block.lora_B_gate_up.normal_(0, 0.05)
        src.block.lora_B_down.normal_(0, 0.05)
        src.block.lora_A_gate_up.normal_(0, 0.05)

    dest = tmp_path / "adapter"
    train_mod._save_adapter_atomic(
        src, _Tok(), dest, lambda *a, **k: None,
        lora_mode="native", base_model_dir="x",
        lora_r=8, lora_alpha=16, lora_dropout=0.0, target_suffixes=["q_proj"],
    )
    # config records the expert_lora extension
    cfg = json.loads((dest / "adapter_config.json").read_text())
    assert cfg.get("expert_lora", {}).get("r") == 4
    assert cfg["expert_lora"]["projections"] == ["gate_up", "down"]

    # fresh model (zero adapter) loads the saved tensors back exactly
    dst = Wrap()
    train_mod._load_adapter_weights(dst, dest, "native", lambda *a, **k: None)
    for proj in ("gate_up", "down"):
        for ab in ("A", "B"):
            s = getattr(src.block, f"lora_{ab}_{proj}")
            d = getattr(dst.block, f"lora_{ab}_{proj}")
            assert torch.equal(s.detach(), d.detach()), f"{ab}_{proj} did not round-trip"


def test_resume_without_expert_flag_raises(train_mod, tmp_path):
    """If the adapter has expert tensors but the model was built without expert LoRA,
    loading must fail loud (not silently drop the trained expert delta)."""
    import torch.nn as nn

    class WrapE(nn.Module):
        def __init__(self, r):
            super().__init__()
            self.block = _mod(lora_r=r, lora_alpha=8)

    src = WrapE(4)
    with torch.no_grad():
        src.block.lora_B_gate_up.normal_(0, 0.05)
    dest = tmp_path / "adapter"
    train_mod._save_adapter_atomic(
        src, _Tok(), dest, lambda *a, **k: None,
        lora_mode="native", base_model_dir="x",
        lora_r=8, lora_alpha=16, lora_dropout=0.0, target_suffixes=["q_proj"],
    )
    no_expert = WrapE(0)  # model without expert LoRA
    with pytest.raises(RuntimeError, match="per-expert LoRA"):
        train_mod._load_adapter_weights(no_expert, dest, "native", lambda *a, **k: None)


def test_partial_expert_adapter_raises(train_mod, tmp_path):
    """A partial expert block (some-but-not-all tensors) must raise, not zero-fill."""
    import torch.nn as nn
    from safetensors.torch import load_file, save_file

    class WrapE(nn.Module):
        def __init__(self):
            super().__init__()
            self.block = _mod(lora_r=4, lora_alpha=8)

    src = WrapE()
    dest = tmp_path / "adapter"
    train_mod._save_adapter_atomic(
        src, _Tok(), dest, lambda *a, **k: None,
        lora_mode="native", base_model_dir="x",
        lora_r=8, lora_alpha=16, lora_dropout=0.0, target_suffixes=["q_proj"],
    )
    # drop one expert tensor to simulate a corrupt/partial adapter
    p = dest / "adapter_model.safetensors"
    sd = load_file(str(p))
    del sd["base_model.model.block.experts.down.lora_B"]
    save_file(sd, str(p))
    dst = WrapE()
    with pytest.raises(RuntimeError, match="partial load"):
        train_mod._load_adapter_weights(dst, dest, "native", lambda *a, **k: None)


def test_lora_delta_gradcheck():
    """float64 gradcheck on the batched per-expert delta w.r.t. A and B."""
    m = _mod(lora_r=3, lora_alpha=6, dtype=torch.float64)
    torch.manual_seed(0)
    k, L, in_dim, out_dim, E, r = 2, 5, 16, 7, 6, 3
    x = torch.randn(k, L, in_dim, dtype=torch.float64)
    expert_idx = torch.tensor([1, 4])
    A = torch.randn(E, r, in_dim, dtype=torch.float64, requires_grad=True)
    B = torch.randn(E, out_dim, r, dtype=torch.float64, requires_grad=True)

    def f(A_, B_):
        return m._lora_delta_grouped(x, A_, B_, expert_idx)

    assert torch.autograd.gradcheck(f, (A, B), atol=1e-6)
