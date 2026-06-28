#!/usr/bin/env python3
"""Rekey a native nvfp4-lora-spark expert-LoRA adapter into vLLM's per-expert format.

The trainer saves per-expert LoRA in a memory-efficient STACKED layout (one tensor per
projection per MoE block, stacked over the E experts):

    base_model.model.{block}.experts.gate_up.lora_A   (E, r, hidden)
    base_model.model.{block}.experts.gate_up.lora_B   (E, 2*intermediate, r)
    base_model.model.{block}.experts.down.lora_A      (E, r, intermediate)
    base_model.model.{block}.experts.down.lora_B      (E, hidden, r)

vLLM's fused-MoE LoRA loader expects the standard PEFT PER-EXPERT layout (one A/B per
expert per projection, gate_up un-fused into gate_proj + up_proj):

    base_model.model.{block}.{e}.gate_proj.lora_A.weight  (r, hidden)
    base_model.model.{block}.{e}.gate_proj.lora_B.weight  (intermediate, r)
    base_model.model.{block}.{e}.up_proj.lora_A.weight    (r, hidden)
    base_model.model.{block}.{e}.up_proj.lora_B.weight    (intermediate, r)
    base_model.model.{block}.{e}.down_proj.lora_A.weight  (r, intermediate)
    base_model.model.{block}.{e}.down_proj.lora_B.weight  (hidden, r)

The fused gate_up shares ONE A across gate and up; the un-fused gate_proj/up_proj each
reuse that A and take their half of the stacked B (gate = B[:intermediate], up =
B[intermediate:]) -- mathematically identical to the fused delta. (`{block}` is the
NVFP4Experts3D module path, e.g. `model.layers.3.mlp.experts`.)

Usage:
    python scripts/rekey_expert_lora_for_vllm.py --in <native_adapter_dir> --out <vllm_adapter_dir>
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


def rekey(in_dir: Path, out_dir: Path) -> dict:
    in_dir, out_dir = Path(in_dir), Path(out_dir)
    sd = load_file(str(in_dir / "adapter_model.safetensors"))
    cfg = json.loads((in_dir / "adapter_config.json").read_text())

    # Collect the stacked expert tensors, grouped by block.
    # key: base_model.model.{block}.experts.{proj}.lora_{A,B}
    pat = re.compile(r"^(?P<prefix>base_model\.model\.)(?P<block>.+)\.experts\.(?P<proj>gate_up|down)\.lora_(?P<ab>[AB])$")
    blocks: dict[str, dict] = {}
    passthrough: dict[str, torch.Tensor] = {}
    for k, v in sd.items():
        m = pat.match(k)
        if not m:
            passthrough[k] = v  # attention / dense LoRA already in vLLM-compatible form
            continue
        b = blocks.setdefault(m["block"], {})
        b[f"{m['proj']}.{m['ab']}"] = v

    out: dict[str, torch.Tensor] = dict(passthrough)
    n_experts_seen = set()
    for block, t in blocks.items():
        gu_A, gu_B = t["gate_up.A"], t["gate_up.B"]      # (E,r,h), (E,2i,r)
        dn_A, dn_B = t["down.A"], t["down.B"]            # (E,r,i), (E,h,r)
        E = gu_A.shape[0]
        two_i = gu_B.shape[1]
        assert two_i % 2 == 0, f"gate_up out dim {two_i} not even for block {block}"
        i = two_i // 2
        n_experts_seen.add(E)
        for e in range(E):
            base = f"base_model.model.{block}.{e}"
            # gate_proj and up_proj share A; B is the gate / up half of the fused B.
            # .clone() (not just .contiguous()): gate_proj/up_proj share the SAME fused A,
            # and the B halves are views of one stacked tensor; safetensors refuses tensors
            # that share storage, so materialize independent copies.
            out[f"{base}.gate_proj.lora_A.weight"] = gu_A[e].clone()
            out[f"{base}.gate_proj.lora_B.weight"] = gu_B[e, :i].clone()
            out[f"{base}.up_proj.lora_A.weight"] = gu_A[e].clone()
            out[f"{base}.up_proj.lora_B.weight"] = gu_B[e, i:].clone()
            out[f"{base}.down_proj.lora_A.weight"] = dn_A[e].clone()
            out[f"{base}.down_proj.lora_B.weight"] = dn_B[e].clone()

    out_dir.mkdir(parents=True, exist_ok=True)
    save_file(out, str(out_dir / "adapter_model.safetensors"))

    # vLLM/PEFT config: target the per-expert projection module names. Keep r/alpha.
    el = cfg.get("expert_lora", {})
    tgt = set(cfg.get("target_modules", []))
    tgt.update(["gate_proj", "up_proj", "down_proj"])
    new_cfg = dict(cfg)
    new_cfg["target_modules"] = sorted(tgt)
    new_cfg.pop("expert_lora", None)  # consumed; now expressed as standard per-expert keys
    if el.get("r"):
        new_cfg["r"] = el["r"]
        new_cfg["lora_alpha"] = el.get("lora_alpha", new_cfg.get("lora_alpha"))
    (out_dir / "adapter_config.json").write_text(json.dumps(new_cfg, indent=2))
    # copy tokenizer/aux files if present (vLLM tolerates their absence)
    for aux in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "chat_template.jinja"):
        p = in_dir / aux
        if p.exists():
            (out_dir / aux).write_bytes(p.read_bytes())

    report = {
        "blocks": len(blocks),
        "experts_per_block": sorted(n_experts_seen),
        "out_tensors": len(out),
        "passthrough_tensors": len(passthrough),
        "r": new_cfg.get("r"),
    }
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_dir", required=True, help="native expert-LoRA adapter dir")
    ap.add_argument("--out", dest="out_dir", required=True, help="output vLLM per-expert adapter dir")
    args = ap.parse_args()
    rep = rekey(args.in_dir, args.out_dir)
    print(json.dumps(rep, indent=2))
    print(f"[write] {args.out_dir}/adapter_model.safetensors")


if __name__ == "__main__":
    main()
