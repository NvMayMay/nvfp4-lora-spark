#!/usr/bin/env python3
"""nybbloris inspect: a pre-flight serve plan for (NVFP4 base + LoRA adapter).

Given a base model dir and an adapter dir, this reports BEFORE you serve:
  - BINDING: which adapter targets resolve to base modules, and under which re-key.
    A naive mismatch is the silent "swathes of LoRA filters not applied" no-op.
  - LIVENESS by weight quant type: NVFP4 / bf16 targets are LoRA-live; FP8 targets
    are served FROZEN (deltas dropped) unless the loader enables allow_fp8_targets.
  - KINDS: attention / shared_expert (dense, servable) vs routed_expert (NOT
    servable on the CUTLASS NVFP4 MoE path; a v1 non-goal).
  - SERVE ENGINE: the base quant method maps to a minimum vLLM (from measured
    findings), or eager.
  - a single PLAN verdict and an inspectable JSON plan object.

Reads key names + config only (no weights, no GPU). This is the v1 "inspect"
surface prototype; the binding/quant logic is shared with check_lora_binding.py.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from check_lora_binding import REKEYS, adapter_modules, classify  # single source of truth

# Minimum-vLLM compatibility, from THIS repo's measured findings (extend as tested).
ENGINE_NOTES = {
    "compressed-tensors": "vLLM 0.19+ (NGC vllm:26.04+), CUTLASS NVFP4 MoE backend. Measured: serves.",
    "modelopt": ("vLLM >= 0.22.1 for the MoE expert-scale load. NGC 0.19/0.20 build the "
                 "modelopt_mixed MoE unquantized and fail with KeyError experts.w2_input_scale. "
                 "Eager (this repo's loader) serves now."),
}


def kind_of(path: str) -> str:
    if "shared_expert" in path:
        return "shared_expert"
    if re.search(r"experts\.\d+", path) or ".experts." in path:
        return "routed_expert"
    if "self_attn" in path or "linear_attn" in path:
        return "attention"
    if ".mlp." in path:
        return "mlp"
    return "other"


def read_json(p):
    try:
        return json.load(open(p))
    except Exception:  # noqa: BLE001
        return {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model-dir", required=True, type=Path)
    ap.add_argument("--adapter-dir", required=True, type=Path)
    ap.add_argument("--allow-fp8-targets", action="store_true",
                    help="count FP8 targets as live (only if the serve loader enables FP8 LoRA)")
    ap.add_argument("--json-out", type=Path, default=None)
    args = ap.parse_args()

    cfg = read_json(args.base_model_dir / "config.json")
    qcfg = cfg.get("quantization_config") or {}
    quant_method = qcfg.get("quant_method") or read_json(
        args.base_model_dir / "hf_quant_config.json").get("quant_method") or "unknown"
    arch = (cfg.get("architectures") or ["?"])[0]
    acfg = read_json(args.adapter_dir / "adapter_config.json")

    base_keys = set(json.load(open(args.base_model_dir / "model.safetensors.index.json"))["weight_map"])
    mods = adapter_modules(args.adapter_dir)
    if not mods:
        print("[inspect] no LoRA targets found in adapter; nothing to plan.")
        return

    def resolves(fn):
        return sum(1 for m in mods if classify(fn(m), base_keys) is not None)

    rekey_name, fn = max(REKEYS, key=lambda nf: resolves(nf[1]))
    naive = resolves(dict(REKEYS)["identity"]) if "identity" in dict(REKEYS) else \
        sum(1 for m in mods if classify(m, base_keys) is not None)
    resolved = resolves(fn)

    by_quant, by_kind = Counter(), Counter()
    unresolved = []
    live = frozen = blocked = 0
    for m in mods:
        q = classify(fn(m), base_keys)
        k = kind_of(m)
        by_kind[k] += 1
        if q is None:
            unresolved.append(m)
            continue
        by_quant[q] += 1
        if k == "routed_expert":
            blocked += 1
        elif q == "FP8" and not args.allow_fp8_targets:
            frozen += 1
        else:
            live += 1

    engine_note = ENGINE_NOTES.get(quant_method, "unknown quant method; verify serve compatibility.")
    verdict = ("FAIL" if unresolved else "BLOCKED-ROUTED" if blocked
               else "PARTIAL" if frozen else "PASS")

    plan = {
        "base": {"dir": str(args.base_model_dir), "arch": arch, "model_type": cfg.get("model_type"),
                 "quant_method": quant_method, "serve_engine_note": engine_note},
        "adapter": {"dir": str(args.adapter_dir), "r": acfg.get("r"), "alpha": acfg.get("lora_alpha"),
                    "n_targets": len(mods)},
        "binding": {"rekey": rekey_name, "naive_resolve": naive, "resolved": resolved,
                    "unresolved": unresolved[:10]},
        "targets": {"by_kind": dict(by_kind), "by_quant": dict(by_quant),
                    "live": live, "frozen_fp8": frozen, "blocked_routed": blocked,
                    "unresolved": len(unresolved)},
        "verdict": verdict,
    }

    n = len(mods)
    print("=== nybbloris inspect: serve plan ===")
    print(f"base    : {args.base_model_dir.name}  (arch {arch}, quant {quant_method})")
    print(f"adapter : {args.adapter_dir.name}  (r={acfg.get('r')}, alpha={acfg.get('lora_alpha')}, {n} targets)")
    print()
    rk = "directly" if naive == resolved else \
        f"via the '{rekey_name}' re-key  (a naive load resolves {naive} = silent no-op risk)"
    print(f"binding : {resolved}/{n} targets resolve {rk}")
    if unresolved:
        print(f"          UNRESOLVED {len(unresolved)}: e.g. {unresolved[:4]}")
    print(f"kinds   : {dict(by_kind)}")
    print(f"quant   : {dict(by_quant)}")
    print(f"  LoRA-LIVE        : {live}/{n}")
    print(f"  FROZEN (FP8)     : {frozen}/{n}" + ("  [--allow-fp8-targets]" if args.allow_fp8_targets else ""))
    print(f"  BLOCKED (routed) : {blocked}/{n}")
    print()
    print(f"engine  : {engine_note}")
    print()
    tail = ""
    if frozen:
        tail += f" {frozen} FP8 deltas FROZEN (dropped)."
    if blocked:
        tail += f" {blocked} routed-expert deltas BLOCKED (not servable; merge-for-serve or skip)."
    if naive < resolved:
        tail += f" Requires the '{rekey_name}' re-key."
    print(f"PLAN    : runtime-LoRA serves {live}/{n} deltas.{tail}  VERDICT: {verdict}")

    if args.json_out:
        json.dump(plan, open(args.json_out, "w"), indent=2)
        print(f"\n[inspect] wrote plan object -> {args.json_out}")


if __name__ == "__main__":
    main()
