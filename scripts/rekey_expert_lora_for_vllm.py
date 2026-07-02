#!/usr/bin/env python3
"""Rekey native nvfp4-lora-spark expert-LoRA adapters for vLLM serving.

The trainer saves per-expert LoRA in a memory-efficient STACKED layout (one tensor per
projection per MoE block, stacked over the E experts):

    base_model.model.{block}.experts.gate_up.lora_A   (E, r, hidden)
    base_model.model.{block}.experts.gate_up.lora_B   (E, 2*intermediate, r)
    base_model.model.{block}.experts.down.lora_A      (E, r, intermediate)
    base_model.model.{block}.experts.down.lora_B      (E, hidden, r)

vLLM has two routed-MoE LoRA disk layouts in the installed loader:

1. ``per-expert`` (GLM-4.5-Air path): one A/B per expert per projection,
   gate_up un-fused into gate_proj + up_proj:

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

2. ``fused-3d`` (Qwen3.5 MoE path): the routed expert module itself is a single
   FusedMoE target named ``experts``.  vLLM expects the PEFT 3D fused form:

    base_model.model.{block}.base_layer.lora_A.weight  (E*r, hidden)        # gate_up
    base_model.model.{block}.base_layer.lora_B.weight  (2*intermediate, E*r)
    base_model.model.{block}.lora_A.weight             (E*r, intermediate)  # down
    base_model.model.{block}.lora_B.weight             (hidden, E*r)

The B tensors are flattened as ``tensor.permute(1, 2, 0).reshape(out, E*r)`` so
vLLM's loader can reshape them back to ``(E, out, r)``.

Usage:
    python scripts/rekey_expert_lora_for_vllm.py --in <native_adapter_dir> --out <vllm_adapter_dir>
    python scripts/rekey_expert_lora_for_vllm.py --target-format fused-3d --in <native> --out <vllm>
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nvfp4_lora.adapter_keys import wrapped_remap_safetensors_key  # noqa: E402


QWEN35_MODEL_TYPES = {"qwen3_5_moe", "qwen3_5_moe_text"}


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _model_types_from_config(config: dict) -> set[str]:
    model_types = set()
    if isinstance(config.get("model_type"), str):
        model_types.add(config["model_type"])
    text_config = config.get("text_config")
    if isinstance(text_config, dict) and isinstance(text_config.get("model_type"), str):
        model_types.add(text_config["model_type"])
    return model_types


def _resolve_target_format(
    requested: str, adapter_cfg: dict, base_model: Path | None
) -> str:
    if requested != "auto":
        return requested

    candidate_dirs = []
    if base_model is not None:
        candidate_dirs.append(Path(base_model))
    cfg_base = adapter_cfg.get("base_model_name_or_path")
    if cfg_base:
        candidate_dirs.append(Path(cfg_base))

    for candidate in candidate_dirs:
        config_path = candidate / "config.json"
        if not config_path.exists():
            continue
        model_types = _model_types_from_config(_read_json(config_path))
        if model_types & QWEN35_MODEL_TYPES:
            return "fused-3d"

    return "per-expert"


# Wrapped multimodal bases (e.g. Qwen3.5-122B `...ForConditionalGeneration`)
# expose their decoder under `language_model.` at serve time, so vLLM's runtime
# module path is `language_model.model.layers.N...`. A native/PEFT adapter trained
# on the text decoder carries flat `base_model.model.model.layers.N...` keys; on a
# wrapped base those resolve to a module vLLM never builds, so the adapter loads
# but silently NO-OPs (the MoE 3D->2D converter finds neither expert key and leaves
# the stacked buffers zero). This is the same `language_model` re-key the attention
# path applies in scripts/rekey_lora_for_vllm.py; expert keys need it too. Both use
# the single source of truth nvfp4_lora.adapter_keys.wrapped_remap_safetensors_key.


def _candidate_base_dirs(adapter_cfg: dict, base_model: Path | None) -> list[Path]:
    dirs = []
    if base_model is not None:
        dirs.append(Path(base_model))
    cfg_base = adapter_cfg.get("base_model_name_or_path")
    if cfg_base:
        dirs.append(Path(cfg_base))
    return dirs


def _base_is_wrapped(adapter_cfg: dict, base_model: Path | None) -> bool:
    """True if the base is a multimodal `...ForConditionalGeneration` wrapper whose
    decoder lives under `language_model.` (so expert keys need the wrapped re-key)."""
    for candidate in _candidate_base_dirs(adapter_cfg, base_model):
        config_path = candidate / "config.json"
        if not config_path.exists():
            continue
        cfg = _read_json(config_path)
        archs = cfg.get("architectures") or []
        if any(isinstance(a, str) and a.endswith("ConditionalGeneration") for a in archs):
            return True
        if "vision_config" in cfg or "text_config" in cfg:
            return True
    return False


def _resolve_wrapped(requested: str, adapter_cfg: dict, base_model: Path | None) -> bool:
    if requested == "yes":
        return True
    if requested == "no":
        return False
    return _base_is_wrapped(adapter_cfg, base_model)


def _flatten_lora_a(t: torch.Tensor) -> torch.Tensor:
    # (E, r, in) -> (E*r, in), expert-major.
    return t.reshape(t.shape[0] * t.shape[1], t.shape[2]).clone()


def _flatten_lora_b(t: torch.Tensor) -> torch.Tensor:
    # (E, out, r) -> (out, E*r), matching vLLM's
    # reshape(out, -1, E).permute(2, 0, 1) load path.
    return t.permute(1, 2, 0).reshape(t.shape[1], t.shape[0] * t.shape[2]).clone()


def _rekey_block_per_expert(
    out: dict[str, torch.Tensor],
    block: str,
    gu_A: torch.Tensor,
    gu_B: torch.Tensor,
    dn_A: torch.Tensor,
    dn_B: torch.Tensor,
) -> None:
    E = gu_A.shape[0]
    two_i = gu_B.shape[1]
    assert two_i % 2 == 0, f"gate_up out dim {two_i} not even for block {block}"
    i = two_i // 2
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


def _rekey_block_fused_3d(
    out: dict[str, torch.Tensor],
    block: str,
    gu_A: torch.Tensor,
    gu_B: torch.Tensor,
    dn_A: torch.Tensor,
    dn_B: torch.Tensor,
) -> None:
    base = f"base_model.model.{block}"
    out[f"{base}.base_layer.lora_A.weight"] = _flatten_lora_a(gu_A)
    out[f"{base}.base_layer.lora_B.weight"] = _flatten_lora_b(gu_B)
    out[f"{base}.lora_A.weight"] = _flatten_lora_a(dn_A)
    out[f"{base}.lora_B.weight"] = _flatten_lora_b(dn_B)


def rekey(
    in_dir: Path,
    out_dir: Path,
    *,
    target_format: str = "auto",
    base_model: Path | None = None,
    wrapped: str = "auto",
) -> dict:
    in_dir, out_dir = Path(in_dir), Path(out_dir)
    sd = load_file(str(in_dir / "adapter_model.safetensors"))
    cfg = _read_json(in_dir / "adapter_config.json")
    target_format = _resolve_target_format(target_format, cfg, base_model)
    is_wrapped = _resolve_wrapped(wrapped, cfg, base_model)

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
        n_experts_seen.add(E)
        if target_format == "per-expert":
            _rekey_block_per_expert(out, block, gu_A, gu_B, dn_A, dn_B)
        elif target_format == "fused-3d":
            _rekey_block_fused_3d(out, block, gu_A, gu_B, dn_A, dn_B)
        else:
            raise ValueError(f"unknown target_format {target_format!r}")

    # Wrapped multimodal base: re-key the decoder path to vLLM's runtime
    # `language_model.model.layers.N...` (applies to BOTH the emitted expert keys
    # and any passthrough attention keys), else the adapter loads but NO-OPs.
    wrapped_remapped = 0
    if is_wrapped:
        remapped: dict[str, torch.Tensor] = {}
        for k, v in out.items():
            nk = wrapped_remap_safetensors_key(k)
            if nk != k:
                wrapped_remapped += 1
            remapped[nk] = v
        out = remapped

    out_dir.mkdir(parents=True, exist_ok=True)
    save_file(out, str(out_dir / "adapter_model.safetensors"))

    # vLLM/PEFT config: target the emitted MoE module names. Keep r/alpha.
    el = cfg.get("expert_lora", {})
    tgt = set(cfg.get("target_modules", []))
    if target_format == "per-expert":
        tgt.update(["gate_proj", "up_proj", "down_proj"])
    else:
        tgt.add("experts")
    new_cfg = dict(cfg)
    new_cfg["target_modules"] = sorted(tgt)
    new_cfg.pop("expert_lora", None)  # consumed; now expressed as vLLM-loadable keys
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
        "target_format": target_format,
        "wrapped_base": is_wrapped,
        "wrapped_keys_remapped": wrapped_remapped,
    }
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_dir", required=True, help="native expert-LoRA adapter dir")
    ap.add_argument("--out", dest="out_dir", required=True, help="output vLLM adapter dir")
    ap.add_argument(
        "--target-format",
        choices=("auto", "per-expert", "fused-3d"),
        default="auto",
        help="vLLM MoE LoRA disk layout; auto uses fused-3d for qwen3_5_moe, else per-expert",
    )
    ap.add_argument(
        "--base-model",
        type=Path,
        default=None,
        help="optional base model dir for auto target-format / wrapped detection",
    )
    ap.add_argument(
        "--wrapped",
        choices=("auto", "yes", "no"),
        default="auto",
        help="re-key the decoder to vLLM's wrapped `language_model.model.layers` path; "
             "auto detects a multimodal `...ForConditionalGeneration` base (e.g. Qwen3.5-122B)",
    )
    args = ap.parse_args()
    rep = rekey(
        args.in_dir,
        args.out_dir,
        target_format=args.target_format,
        base_model=args.base_model,
        wrapped=args.wrapped,
    )
    print(json.dumps(rep, indent=2))
    print(f"[write] {args.out_dir}/adapter_model.safetensors")


if __name__ == "__main__":
    main()
