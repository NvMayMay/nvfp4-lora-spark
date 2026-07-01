"""nybbloris.plan -- pre-flight serve plan for (NVFP4 base + LoRA adapter).

The binding + quant-liveness analysis behind `nybbloris inspect`, as a library:
``serve_plan(base, adapter) -> dict``. Reads key names + config only (no weights,
no GPU, no torch) so it runs anywhere as a cheap gate before train / merge / serve.

Two silent-no-op classes are caught here:
  * KEY mismatch -- adapter module paths don't match the base layout (e.g. a
    multimodal base nests the LM under ``language_model.``); a naive load binds
    ZERO and serves the un-adapted base.
  * QUANT-type freeze -- historically a target resolving to an FP8 weight was
    served FROZEN by the NVFP4-LoRA runtime path; serve no longer freezes dense
    FP8 (the delta applies in bf16 independently of the base weight's quant).
"""
from __future__ import annotations

import glob
import json
import re
import struct
from collections import Counter
from pathlib import Path

__all__ = ["serve_plan", "render_plan", "adapter_modules", "classify", "REKEYS", "lm_head_status"]

# Candidate re-key transforms mapping an adapter's module path into the SERVE
# engine's runtime module tree (extend as new layouts appear). For a multimodal
# *ForConditionalGeneration wrapper, vLLM exposes the LM as
# `language_model.model.layers.*`, so a flat `model.layers.*` adapter must be
# re-keyed to bind -- a naive load otherwise no-ops (MEASURED §7h).
REKEYS = [
    ("identity", lambda k: k),
    ("language_model", lambda k: k.replace("model.layers.", "language_model.model.layers.", 1)),
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
    # routed experts, per-expert (`experts.3.*`) OR fused-3D (`...mlp.experts` /
    # `...mlp.experts.base_layer`, no expert index) -- both are FusedMoE at serve.
    if re.search(r"experts\.\d+", path) or ".experts." in path or path.endswith(".experts"):
        return "routed_expert"
    if "self_attn" in path or "linear_attn" in path:
        return "attention"
    if ".mlp." in path:
        return "mlp"
    return "other"


_FUSED_EXPERT_RE = re.compile(r"^(?P<prefix>.+\.experts)(?:\.base_layer)?$")


def _classify_fused_expert(path, base_keys):
    """Resolve a FUSED-3D routed-expert adapter target against the per-expert base.

    A fused adapter targets the whole MoE block as `X.mlp.experts` (down) and
    `X.mlp.experts.base_layer` (gate_up); there is no dense `X.mlp.experts.weight`
    in the checkpoint -- the routed experts live per-expert as
    `X.mlp.experts.{e}.{gate,up,down}_proj.*`. Resolve the fused target against
    expert 0's projections so the contract can SEE that it binds (and therefore
    distinguish a flat-vs-wrapped fused-MoE no-op from a clean bind -- the class
    that silently no-op'd on Qwen3.5-122B). Returns None for non-fused paths.
    """
    m = _FUSED_EXPERT_RE.match(path)
    if not m:
        return None
    prefix = m.group("prefix")
    for proj in ("gate_proj", "up_proj", "down_proj"):
        q = classify(f"{prefix}.0.{proj}", base_keys)
        if q is not None:
            return q
    return None


def _resolve(path, base_keys):
    """Quant type of a resolved adapter target: dense OR fused-3D routed-expert."""
    q = _classify_fused_expert(path, base_keys)
    return q if q is not None else classify(path, base_keys)


def _read_json(p):
    try:
        return json.load(open(p))
    except Exception:  # noqa: BLE001
        return {}


def lm_head_status(base_model_dir):
    """Checkpoint-compat pre-flight: is the `lm_head` quantized in a way vLLM can't load?

    vLLM keeps `lm_head` in bf16 by class, so a ModelOpt/compressed-tensors checkpoint
    that quantized `lm_head` (NVFP4 weight + scales) crashes vLLM at load with
    "no module or parameter named lm_head.input_scale" (MEASURED §7h / notebook).
    A quantized head shows scale tensors (`lm_head.weight_scale{,_2}` / `.input_scale`)
    in the index; a bf16 head has only `lm_head.weight`. Returns a small dict so a
    serve pre-flight can refuse + remediate (dequant the head to bf16, drop its scales).
    """
    base_model_dir = Path(base_model_dir)
    idx = base_model_dir / "model.safetensors.index.json"
    if not idx.exists():
        return {"present": False, "quantized": False, "scale_keys": [], "note": "no index.json"}
    wm = _read_json(idx).get("weight_map", {})
    lh = sorted(k for k in wm if k.startswith("lm_head."))
    scales = [k for k in lh if k.endswith((".weight_scale", ".weight_scale_2", ".input_scale",
                                           ".weight_global_scale", ".weight_packed"))]
    return {
        "present": any(k == "lm_head.weight" or k.endswith(".weight_packed") for k in lh),
        "quantized": bool(scales),
        "scale_keys": scales,
        "note": ("lm_head is quantized -> vLLM cannot load it; dequantize to bf16 and drop its "
                 "scale tensors first (scripts/fix_nvfp4_lm_head.py)." if scales
                 else "lm_head is bf16 (vLLM-loadable)."),
    }


def serve_plan(base_model_dir, adapter_dir):
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

    # VLLM_BUILD naming: resolve against the SERVE engine's runtime module tree, not
    # the on-disk index -- the two diverge for multimodal *ForConditionalGeneration
    # wrappers, where the checkpoint nests the LM as `model.language_model.layers.*`
    # but vLLM exposes it as `language_model.model.layers.*`. Resolving against the
    # on-disk keys returned the INVERSE of the real serve binding (MEASURED §7h: it
    # blessed the flat adapter, which no-ops, and failed the re-keyed one, which
    # binds). Detect the layout from the keys (robust; no fragile arch-string match)
    # and rebuild the set into the names vLLM actually binds LoRA against.
    #   Version note: this targets vLLM >= 0.22.1 (the canonical serve engine).
    #   vLLM 0.20.x auto-remapped flat LoRA keys via its hf_to_vllm_mapper, so a
    #   flat adapter happened to bind there; 0.22.1 dropped that, requiring the
    #   runtime-tree names. Predicting the stricter behavior is the SAFE direction
    #   -- a re-keyed adapter binds on both, a flat one silently no-ops on 0.22.1.
    wrapped = any(k.startswith("model.language_model.layers.") for k in base_keys)
    if wrapped:
        serve_keys = {(k.replace("model.language_model.", "language_model.model.", 1)
                       if k.startswith("model.language_model.") else k) for k in base_keys}
    else:
        serve_keys = base_keys

    def resolves(fn):
        return sum(1 for m in mods if _resolve(fn(m), serve_keys) is not None)

    rekey_name, fn = max(REKEYS, key=lambda nf: resolves(nf[1])) if mods else ("identity", lambda k: k)
    naive = sum(1 for m in mods if _resolve(m, serve_keys) is not None)
    resolved = resolves(fn)

    # Quant-liveness at SERVE (MEASURED §7h, vLLM 0.22.1): the LoRA delta is applied
    # in bf16 independently of the base weight's quant, so DENSE targets serve LIVE
    # whether NVFP4, FP8, or bf16. ROUTED-expert FusedMoE is BACKEND-GATED (counted in
    # blocked_routed): it serves LIVE on a LoRA-capable MoE backend (the emulation
    # backend -- our validated one-box path, proven on GLM-4.5-Air -- or marlin), and is
    # blocked ONLY on the cutlass/flashinfer fast backends (supports_lora=False). The
    # name "blocked_routed" is historical; it means "needs a LoRA-capable MoE backend",
    # NOT "merge-only". A backend-parameterized verdict (live on emulation/marlin) is a
    # follow-up; this static check cannot see the runtime backend choice. FP8 is frozen
    # only by the eager TRAIN loader, never by serve -- so dense-FP8 is counted live.
    by_quant, by_kind, unresolved = Counter(), Counter(), []
    live = blocked = fp8_dense = 0
    for m in mods:
        q = _resolve(fn(m), serve_keys)
        k = _kind(m)
        by_kind[k] += 1
        if q is None:
            unresolved.append(m)
            continue
        by_quant[q] += 1
        if k == "routed_expert":
            blocked += 1
        else:
            live += 1
            if q == "FP8":
                fp8_dense += 1

    needs_rekey = resolved > 0 and naive < resolved
    # NO-OP / NEEDS-REKEY rank ABOVE BLOCKED-ROUTED on purpose: a routed-expert
    # adapter that also needs a re-key (e.g. a fused-3D MoE adapter carrying the flat
    # `model.layers.*` path against a wrapped multimodal base) binds NOTHING as
    # shipped -- that silent no-op is the more urgent, actionable fact than the
    # backend-gating, and stating BLOCKED-ROUTED there would hide it. This is exactly
    # the Qwen3.5-122B fused-MoE case: flat keys on a `language_model.`-wrapped base
    # -> naive resolves 0 -> NO-OP, re-key first. (No existing routed case needs a
    # re-key, so this ordering does not change their verdicts.)
    verdict = ("EMPTY" if not mods else
               "FAIL" if unresolved else
               "NO-OP" if naive == 0 and resolved > 0 else
               "NEEDS-REKEY" if needs_rekey else
               "BLOCKED-ROUTED" if blocked else
               "PASS")
    return {
        "base": {"dir": str(base_model_dir), "arch": arch, "model_type": cfg.get("model_type"),
                 "quant_method": quant_method, "wrapped": wrapped,
                 "serve_naming": ("language_model.model.layers.* (multimodal wrapper)" if wrapped
                                  else "model.layers.* (causal-LM)"),
                 "serve_engine_note": ENGINE_NOTES.get(
                     quant_method, "unknown quant method; verify serve compatibility.")},
        "adapter": {"dir": str(adapter_dir), "r": acfg.get("r"), "alpha": acfg.get("lora_alpha"),
                    "n_targets": len(mods)},
        "binding": {"rekey": rekey_name, "naive_resolve": naive, "resolved": resolved,
                    "needs_rekey": needs_rekey, "unresolved": unresolved[:10]},
        "targets": {"by_kind": dict(by_kind), "by_quant": dict(by_quant),
                    "live": live, "fp8_dense_live": fp8_dense, "blocked_routed": blocked,
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
           f"naming  : vLLM binds against {b['serve_naming']}",
           ""]
    rk = ("directly (a naive load binds them)" if not bi["needs_rekey"]
          else (f"ONLY via the '{bi['rekey']}' re-key -- a naive load binds "
                f"{bi['naive_resolve']}/{n} = SILENT NO-OP"))
    out.append(f"binding : {bi['resolved']}/{n} targets resolve {rk}")
    if bi["unresolved"]:
        out.append(f"          UNRESOLVED {t['unresolved']}: e.g. {bi['unresolved'][:4]}")
    live_line = f"  LoRA-LIVE (served)   : {t['live']}/{n}"
    if t["fp8_dense_live"]:
        live_line += (f"  (incl. {t['fp8_dense_live']} dense-FP8: served live in vLLM; "
                      "frozen only by the eager TRAIN loader)")
    out += [f"kinds   : {t['by_kind']}",
            f"quant   : {t['by_quant']}",
            live_line,
            f"  BLOCKED (routed-MoE) : {t['blocked_routed']}/{n}",
            "",
            f"engine  : {b['serve_engine_note']}",
            ""]
    tail = ""
    if plan["verdict"] == "NO-OP":
        tail = (f" Adapter as-shipped binds NOTHING in vLLM -- re-key to the "
                f"'{bi['rekey']}' layout first (scripts/rekey_lora_for_vllm.py).")
    elif bi["needs_rekey"]:
        tail = f" Requires the '{bi['rekey']}' re-key (else a naive load no-ops)."
    if t["blocked_routed"]:
        tail += (f" {t['blocked_routed']} routed-expert deltas BACKEND-GATED: live on a LoRA-capable MoE "
                 f"backend (emulation/marlin), blocked only on cutlass/flashinfer. Serve --moe-backend "
                 f"emulation, or merge-for-serve.")
    out.append(f"PLAN    : runtime-LoRA serves {t['live']}/{n} deltas.{tail}  VERDICT: {plan['verdict']}")
    return "\n".join(out)
