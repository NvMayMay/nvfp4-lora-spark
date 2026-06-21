"""nybbloris.plan -- pre-flight serve plan for (NVFP4 base + LoRA adapter).

The binding + quant-liveness analysis behind `nybbloris inspect`, as a library:
``serve_plan(base, adapter) -> dict``. Reads key names + config only (no weights,
no GPU, no torch) so it runs anywhere as a cheap gate before train / merge / serve.

Two silent-no-op classes are caught here:
  * KEY mismatch -- adapter module paths don't match the base layout (e.g. a
    multimodal base nests the LM under ``language_model.``); a naive load binds
    ZERO and serves the un-adapted base.
  * QUANT-type freeze -- a target resolves to an FP8 weight, which the NVFP4-LoRA
    runtime path serves FROZEN (delta dropped) unless allow_fp8_targets; an
    adapter trained on a bf16 base can bind by key yet silently lose those deltas.
"""
from __future__ import annotations

import glob
import json
import re
import struct
from collections import Counter
from pathlib import Path

__all__ = ["serve_plan", "render_plan", "adapter_modules", "classify", "REKEYS"]

# Candidate re-key transforms (extend as new base layouts appear).
REKEYS = [
    ("identity", lambda k: k),
    ("language_model", lambda k: k.replace("model.layers.", "model.language_model.layers.", 1)),
    ("language_model-prefix", lambda k: k.replace("model.", "model.language_model.", 1)),
]

# Minimum-vLLM compatibility, from this project's MEASURED findings (extend as tested).
ENGINE_NOTES = {
    "compressed-tensors": "vLLM 0.19+ (NGC vllm:26.04+), CUTLASS NVFP4 MoE backend. Measured: serves.",
    "modelopt": ("vLLM >= 0.22.1 for the MoE expert-scale load. NGC 0.19/0.20 build the "
                 "modelopt_mixed MoE unquantized and fail with KeyError experts.w2_input_scale. "
                 "Eager (the nvfp4_lora loader) serves now."),
}


def adapter_modules(adapter_dir):
    """Target module paths (lora suffix stripped) from the adapter header."""
    adapter_dir = Path(adapter_dir)
    mods = set()
    for f in sorted(glob.glob(str(adapter_dir / "adapter_model*.safetensors"))):
        with open(f, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            hdr = json.loads(fh.read(n))
        for k in hdr:
            if k == "__metadata__" or not re.search(r"\.lora_[AB]\.weight$", k):
                continue
            t = k[len("base_model.model."):] if k.startswith("base_model.model.") else k
            mods.add(re.sub(r"\.lora_[AB]\.weight$", "", t))
    return sorted(mods)


def classify(base_path, base_keys):
    """None (unbound) or NVFP4 / FP8 / BF16 for a resolved target module."""
    has_w = f"{base_path}.weight" in base_keys
    has_wp = f"{base_path}.weight_packed" in base_keys
    if not (has_w or has_wp):
        return None
    if (has_wp
            or f"{base_path}.weight_global_scale" in base_keys
            or f"{base_path}.weight_scale_2" in base_keys):
        return "NVFP4"
    if f"{base_path}.input_scale" in base_keys:
        return "FP8"
    return "BF16"


def _kind(path):
    if "shared_expert" in path:
        return "shared_expert"
    if re.search(r"experts\.\d+", path) or ".experts." in path:
        return "routed_expert"
    if "self_attn" in path or "linear_attn" in path:
        return "attention"
    if ".mlp." in path:
        return "mlp"
    return "other"


def _read_json(p):
    try:
        return json.load(open(p))
    except Exception:  # noqa: BLE001
        return {}


def serve_plan(base_model_dir, adapter_dir, allow_fp8_targets=False):
    """Return the inspectable serve-plan object for a base + adapter."""
    base_model_dir, adapter_dir = Path(base_model_dir), Path(adapter_dir)
    cfg = _read_json(base_model_dir / "config.json")
    qcfg = cfg.get("quantization_config") or {}
    quant_method = (qcfg.get("quant_method")
                    or _read_json(base_model_dir / "hf_quant_config.json").get("quant_method")
                    or "unknown")
    arch = (cfg.get("architectures") or ["?"])[0]
    acfg = _read_json(adapter_dir / "adapter_config.json")
    base_keys = set(json.load(open(base_model_dir / "model.safetensors.index.json"))["weight_map"])
    mods = adapter_modules(adapter_dir)

    def resolves(fn):
        return sum(1 for m in mods if classify(fn(m), base_keys) is not None)

    rekey_name, fn = max(REKEYS, key=lambda nf: resolves(nf[1])) if mods else ("identity", lambda k: k)
    naive = sum(1 for m in mods if classify(m, base_keys) is not None)
    resolved = resolves(fn)

    by_quant, by_kind, unresolved = Counter(), Counter(), []
    live = frozen = blocked = 0
    for m in mods:
        q = classify(fn(m), base_keys)
        k = _kind(m)
        by_kind[k] += 1
        if q is None:
            unresolved.append(m)
            continue
        by_quant[q] += 1
        if k == "routed_expert":
            blocked += 1
        elif q == "FP8" and not allow_fp8_targets:
            frozen += 1
        else:
            live += 1

    verdict = ("FAIL" if unresolved else "BLOCKED-ROUTED" if blocked
               else "PARTIAL" if frozen else "PASS")
    return {
        "base": {"dir": str(base_model_dir), "arch": arch, "model_type": cfg.get("model_type"),
                 "quant_method": quant_method,
                 "serve_engine_note": ENGINE_NOTES.get(
                     quant_method, "unknown quant method; verify serve compatibility.")},
        "adapter": {"dir": str(adapter_dir), "r": acfg.get("r"), "alpha": acfg.get("lora_alpha"),
                    "n_targets": len(mods)},
        "binding": {"rekey": rekey_name, "naive_resolve": naive, "resolved": resolved,
                    "unresolved": unresolved[:10]},
        "targets": {"by_kind": dict(by_kind), "by_quant": dict(by_quant),
                    "live": live, "frozen_fp8": frozen, "blocked_routed": blocked,
                    "unresolved": len(unresolved)},
        "verdict": verdict,
    }


def render_plan(plan):
    """Human-readable rendering of a serve_plan() object."""
    b, a, bi, t = plan["base"], plan["adapter"], plan["binding"], plan["targets"]
    n = a["n_targets"]
    out = ["=== nybbloris inspect: serve plan ===",
           f"base    : {Path(b['dir']).name}  (arch {b['arch']}, quant {b['quant_method']})",
           f"adapter : {Path(a['dir']).name}  (r={a['r']}, alpha={a['alpha']}, {n} targets)",
           ""]
    rk = ("directly" if bi["naive_resolve"] == bi["resolved"]
          else f"via the '{bi['rekey']}' re-key  (a naive load resolves {bi['naive_resolve']} = silent no-op risk)")
    out.append(f"binding : {bi['resolved']}/{n} targets resolve {rk}")
    if bi["unresolved"]:
        out.append(f"          UNRESOLVED {t['unresolved']}: e.g. {bi['unresolved'][:4]}")
    out += [f"kinds   : {t['by_kind']}",
            f"quant   : {t['by_quant']}",
            f"  LoRA-LIVE        : {t['live']}/{n}",
            f"  FROZEN (FP8)     : {t['frozen_fp8']}/{n}",
            f"  BLOCKED (routed) : {t['blocked_routed']}/{n}",
            "",
            f"engine  : {b['serve_engine_note']}",
            ""]
    tail = ""
    if t["frozen_fp8"]:
        tail += f" {t['frozen_fp8']} FP8 deltas FROZEN (dropped)."
    if t["blocked_routed"]:
        tail += f" {t['blocked_routed']} routed-expert deltas BLOCKED (merge-for-serve or skip)."
    if bi["naive_resolve"] < bi["resolved"]:
        tail += f" Requires the '{bi['rekey']}' re-key."
    out.append(f"PLAN    : runtime-LoRA serves {t['live']}/{n} deltas.{tail}  VERDICT: {plan['verdict']}")
    return "\n".join(out)
