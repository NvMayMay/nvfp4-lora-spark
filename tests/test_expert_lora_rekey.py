"""P0 correctness floor for expert-LoRA serving: rekey round-trip + delta-equivalence.

The rekey (scripts/rekey_expert_lora_for_vllm.py) converts the trainer's native STACKED
expert-LoRA (per block: gate_up A (E,r,h)/B (E,2i,r), down A (E,r,i)/B (E,h,r)) into vLLM's
PER-EXPERT format (gate_proj/up_proj/down_proj, gate_up un-fused, A shared by gate+up).

These CPU tests are the cheap correctness floor before any GPU value experiment: a wrong
rekey would load fine but apply the wrong delta (a silent-correctness bug). We assert
(1) the round-trip produces the expected keys/shapes, and (2) the rekeyed per-expert tensors
reconstruct the SAME LoRA delta as the native stacked tensors (catastrophic-mismatch guard:
exact up to float round-trip, not just "close").
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

_REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "rekey_expert_lora_for_vllm", _REPO / "scripts" / "rekey_expert_lora_for_vllm.py"
)
_rk = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_rk)


def _write_native(d: Path, E=4, r=2, h=8, i=6, n_layers=2, dtype=torch.float32):
    d.mkdir(parents=True, exist_ok=True)
    g = torch.Generator().manual_seed(0)
    sd = {}
    for L in range(1, n_layers + 1):
        blk = f"model.layers.{L}.mlp.experts"
        sd[f"base_model.model.{blk}.experts.gate_up.lora_A"] = torch.randn(E, r, h, generator=g, dtype=dtype)
        sd[f"base_model.model.{blk}.experts.gate_up.lora_B"] = torch.randn(E, 2 * i, r, generator=g, dtype=dtype)
        sd[f"base_model.model.{blk}.experts.down.lora_A"] = torch.randn(E, r, i, generator=g, dtype=dtype)
        sd[f"base_model.model.{blk}.experts.down.lora_B"] = torch.randn(E, h, r, generator=g, dtype=dtype)
    save_file(sd, str(d / "adapter_model.safetensors"))
    json.dump(
        {"peft_type": "LORA", "r": r, "lora_alpha": 2 * r, "target_modules": [],
         "expert_lora": {"r": r, "lora_alpha": 2 * r, "projections": ["gate_up", "down"]}},
        open(d / "adapter_config.json", "w"),
    )
    return E, r, h, i, n_layers


def test_rekey_roundtrip_keys_and_shapes(tmp_path):
    E, r, h, i, nL = _write_native(tmp_path / "native")
    rep = _rk.rekey(tmp_path / "native", tmp_path / "vllm")
    assert rep["blocks"] == nL and rep["experts_per_block"] == [E]
    sd = load_file(str(tmp_path / "vllm" / "adapter_model.safetensors"))
    # every layer x expert x {gate,up,down}_proj x {A,B} present, correct shapes
    assert len(sd) == nL * E * 6
    for L in range(1, nL + 1):
        for e in range(E):
            b = f"base_model.model.model.layers.{L}.mlp.experts.{e}"
            assert tuple(sd[f"{b}.gate_proj.lora_A.weight"].shape) == (r, h)
            assert tuple(sd[f"{b}.gate_proj.lora_B.weight"].shape) == (i, r)
            assert tuple(sd[f"{b}.up_proj.lora_A.weight"].shape) == (r, h)
            assert tuple(sd[f"{b}.up_proj.lora_B.weight"].shape) == (i, r)
            assert tuple(sd[f"{b}.down_proj.lora_A.weight"].shape) == (r, i)
            assert tuple(sd[f"{b}.down_proj.lora_B.weight"].shape) == (h, r)
    # config records the per-expert target modules; expert_lora block consumed
    cfg = json.load(open(tmp_path / "vllm" / "adapter_config.json"))
    assert {"gate_proj", "up_proj", "down_proj"} <= set(cfg["target_modules"])
    assert "expert_lora" not in cfg


def test_rekey_preserves_delta(tmp_path):
    """The rekeyed per-expert tensors must reconstruct the SAME LoRA delta as the native
    stacked tensors -- exact up to float round-trip. This is the catastrophic-mismatch guard:
    a transpose/index/half-split bug in the rekey would show here, not in a fuzzy serve check.
    """
    E, r, h, i, nL = _write_native(tmp_path / "native")
    nat = load_file(str(tmp_path / "native" / "adapter_model.safetensors"))
    _rk.rekey(tmp_path / "native", tmp_path / "vllm")
    vll = load_file(str(tmp_path / "vllm" / "adapter_model.safetensors"))

    torch.manual_seed(1)
    x_h = torch.randn(5, h)   # input to gate_up (hidden)
    x_i = torch.randn(5, i)   # input to down (intermediate)
    for L in range(1, nL + 1):
        blk = f"model.layers.{L}.mlp.experts"
        gA = nat[f"base_model.model.{blk}.experts.gate_up.lora_A"]
        gB = nat[f"base_model.model.{blk}.experts.gate_up.lora_B"]
        dA = nat[f"base_model.model.{blk}.experts.down.lora_A"]
        dB = nat[f"base_model.model.{blk}.experts.down.lora_B"]
        for e in range(E):
            # native fused gate_up delta on x_h: (x @ A^T) @ B^T  -> (.., 2i)
            fused = (x_h @ gA[e].T) @ gB[e].T
            b = f"base_model.model.model.layers.{L}.mlp.experts.{e}"
            gp = (x_h @ vll[f"{b}.gate_proj.lora_A.weight"].T) @ vll[f"{b}.gate_proj.lora_B.weight"].T
            up = (x_h @ vll[f"{b}.up_proj.lora_A.weight"].T) @ vll[f"{b}.up_proj.lora_B.weight"].T
            recon = torch.cat([gp, up], dim=-1)
            assert torch.allclose(fused, recon, atol=1e-6), f"gate_up delta mismatch L{L} e{e}"
            # down delta
            dn_nat = (x_i @ dA[e].T) @ dB[e].T
            dn_rk = (x_i @ vll[f"{b}.down_proj.lora_A.weight"].T) @ vll[f"{b}.down_proj.lora_B.weight"].T
            assert torch.allclose(dn_nat, dn_rk, atol=1e-6), f"down delta mismatch L{L} e{e}"
